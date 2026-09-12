"""Data-support, distant-collision and Jacobian diagnostics (private outputs)."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .extract_cache import _add_transformed_values
from .value_rope_audit import settings, MODELS, save_json


def run(args):
    cfg,vocab,tags=settings();out=args.output;dest=out/'results'
    fields=[f['db_column'] for f in cfg['data']['numeric_fields']]
    columns=fields+['sample_id','rock_group','rock_origin','rock_type','canonical_ref_id','data_source']
    with sqlite3.connect(f'file:{args.sqlite_path}?mode=ro',uri=True) as conn:
        frame=pd.read_sql_query('SELECT '+','.join('"'+c+'"' for c in columns)+' FROM v_high_quality_dataset',conn)
    transformed=_add_transformed_values(frame,cfg,vocab)
    summary=[]
    for tag,col in zip(tags,fields):
        raw=frame[col].dropna();z=transformed[f'z__{tag}'].dropna()
        summary.append(dict(element=tag,raw_present=len(raw),tokens_present=len(z),
                            missing_fraction=1-len(z)/len(frame),raw_median=raw.median(),
                            z_min=z.min(),z_p01=z.quantile(.01),z_median=z.median(),z_p99=z.quantile(.99),
                            z_max=z.max(),clipped_tokens=int((abs(z)>=12).sum())))
    pd.DataFrame(summary).to_csv(dest/'full_data_numeric_summary.csv',index=False)
    classes=[]
    for field in ['rock_group','rock_origin','rock_type']:
        for label,n in frame[field].fillna('[missing]').value_counts().items():
            classes.append(dict(field=field,label=label,count=n,fraction=n/len(frame)))
    pd.DataFrame(classes).to_csv(dest/'full_data_class_counts.csv',index=False)
    counts=transformed.num_numeric_tokens
    save_json(dest/'full_data_summary.json',dict(samples=len(frame),numeric_fields=len(tags),
              median_numeric_tokens=float(counts.median()),mean_numeric_tokens=float(counts.mean()),
              min_numeric_tokens=int(counts.min()),max_numeric_tokens=int(counts.max()),
              rock_type_present=int(frame.rock_type.notna().sum()),rock_type_classes=int(frame.rock_type.nunique()),
              references=int(frame.canonical_ref_id.nunique()),sources=int(frame.data_source.nunique()),
              distinct_reporting_panels=int(transformed[[f'z__{t}' for t in tags]].notna().drop_duplicates().shape[0]),
              clipped_numeric_tokens=sum(r['clipped_tokens'] for r in summary)))
    idx=pd.read_csv(out/'stimulus_index.csv',low_memory=False);rows=[];ranks=[];axes=[];frequencies=[]
    for name in MODELS:
        result=np.load(out/f'{name}_stimuli.npz');latent=result['mean']
        if 'rope' in name:
            for tag,g in idx[idx.family.eq('sweep')&idx.scope.eq('wide')].groupby('element',sort=False):
                for key in ('q1','k1'):
                    vector=result[key][g.index[0]].reshape(12,32,2).astype(float)
                    energy=(vector**2).sum(axis=(0,2));energy/=energy.sum()
                    for j,e in enumerate(energy):
                        frequencies.append(dict(model=name,element=tag,projection=key,frequency_index=j,energy_fraction=e))
        for case,g in idx[idx.family.eq('sweep')&idx.scope.eq('wide')].groupby('case',sort=False):
            g=g.sort_values('t');tag=g.iloc[0].element;s=summary[tags.index(tag)]
            for support,low,high in [('full_observed',s['z_min'],s['z_max']),('p01_p99',s['z_p01'],s['z_p99'])]:
                selected=g[g.t.between(low,high)]
                if len(selected)<2:continue
                x=latent[selected.index].astype(float);t=selected.t.to_numpy()
                distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)
                allowed=(abs(t[:,None]-t[None,:])>=4)&np.triu(np.ones((len(t),len(t)),bool))
                local=np.median(np.linalg.norm(np.diff(x,axis=0),axis=1))
                rows.append(dict(model=name,case=case,element=tag,support=support,
                                 candidate_pairs=int(allowed.sum()),
                                 near_collision_pairs=int(np.sum(allowed&(distance<local))),
                                 min_ratio=float(distance[allowed].min()/local) if allowed.any() else np.nan))
        # Jacobian spectra at the three complete synthetic backgrounds.
        derivative=idx[idx.family.eq('derivative')&idx.natural.eq(False)]
        for sample,g in derivative.groupby('sample',sort=False):
            vectors=[]
            for tag,gg in g.groupby('element',sort=False):
                plus=latent[gg.index[np.isclose(gg.h,.1)][0]].astype(float)
                minus=latent[gg.index[np.isclose(gg.h,-.1)][0]].astype(float)
                vectors.append((plus-minus)/.2)
            singular=np.linalg.svd(np.stack(vectors),compute_uv=False)
            energy=singular**2;energy/=energy.sum()
            ranks.append(dict(model=name,background=g.iloc[0].background,elements=len(vectors),
                              participation_rank=float(1/(energy**2).sum()),
                              rank90=int(np.searchsorted(np.cumsum(energy),.9)+1),top1_energy=float(energy[0])))
        meta=pd.read_csv(out/'metadata.csv',low_memory=False);ev=np.load(out/'global_inputs.npz')['evaluation']
        xy=np.load(out/f'{name}_pca2_evaluation.npy')
        for component in range(2):
            for field in ['num_numeric_tokens','sio2','mg_number','log_sr_y']:
                valid=meta.iloc[ev][field].notna().to_numpy()
                axes.append(dict(model=name,component=component+1,variable=field,
                                 spearman_rho=float(spearmanr(xy[valid,component],meta.iloc[ev][field].to_numpy()[valid]).statistic)))
    pd.DataFrame(rows).to_csv(dest/'support_collisions.csv',index=False)
    pd.DataFrame(ranks).to_csv(dest/'jacobian_spectra.csv',index=False)
    pd.DataFrame(axes).to_csv(dest/'pca_axis_associations.csv',index=False)
    pd.DataFrame(frequencies).to_csv(dest/'rotary_frequency_energy.csv',index=False)
    # One declared median background, identical values and panel in both models.
    case='median_intermediate:SiO2:wide';g=idx[idx.case.eq(case)].sort_values('t')
    fig,axs=plt.subplots(1,3,figsize=(15,4));z=g.t.to_numpy()
    s=summary[tags.index('SiO2')]
    for ax,name in zip(axs[:2],['v2','rope']):
        x=np.load(out/f'{name}_stimuli.npz')['mean'][g.index].astype(float)
        distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)
        local=np.median(np.linalg.norm(np.diff(x,axis=0),axis=1))
        im=ax.imshow(distance/local,origin='lower',extent=[-12,12,-12,12],vmin=0,vmax=5,cmap='magma_r')
        ax.set(title=name+': distance / median local step',xlabel='SiO2 z',ylabel='SiO2 z')
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((s['z_min'],s['z_min']),s['z_max']-s['z_min'],s['z_max']-s['z_min'],
                               fill=False,edgecolor='#00cbcc',linewidth=1.4))
        axs[2].plot((z[1:]+z[:-1])/2,np.linalg.norm(np.diff(x,axis=0),axis=1)/.25,label=name)
    axs[2].axvspan(s['z_p01'],s['z_p99'],color='#77bbbb',alpha=.2,label='Observed 1–99%')
    axs[2].set(title='Wide-sweep speed',xlabel='SiO2 z',ylabel='Latent distance per z');axs[2].legend()
    fig.subplots_adjust(wspace=.55,top=.85)
    colorbar_axis=fig.add_axes([axs[1].get_position().x1+.016,.22,.008,.52])
    fig.colorbar(im,cax=colorbar_axis)
    fig.suptitle('Phase stress test; cyan box = full-database SiO2 support')
    fig.savefig(out/'figures/phase_collision.png',dpi=170);plt.close(fig)
    print('Data and support diagnostics completed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('stage3_5_element2vec_v2/outputs/value_rope_z/audit'))
    p.add_argument('--sqlite-path',type=Path,required=True);run(p.parse_args())


if __name__=='__main__':main()
