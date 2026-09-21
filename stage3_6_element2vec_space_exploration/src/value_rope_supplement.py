"""Full label-masked extraction and diagnostic companion to the paired audit."""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path

import argparse
import copy
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .catalogs import EVOLUTION_SEQUENCES
from .value_rope_audit import settings, infer, tensor_batch, save_json, MODELS
from .value_rope_metrics import projections, transformed, quality, neighbors, path_metrics, cosine
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model


def vote(nn,labels):
    classes,coded=np.unique(np.asarray(labels).astype(str),return_inverse=True)
    counts=np.stack([np.bincount(row,minlength=len(classes)) for row in coded[nn]])
    counts[:,np.isin(classes,['nan','None',''])]=0
    predicted=classes[counts.argmax(1)].astype(object)
    predicted[counts.sum(1)==0]='[unknown]'
    return predicted.astype(str)


def classification_metrics(truth,pred):
    truth=np.asarray(truth).astype(str)
    valid=~np.isin(truth,['nan','None',''])
    return dict(knn_accuracy=accuracy_score(truth[valid],pred[valid]),
                knn_balanced_accuracy=balanced_accuracy_score(truth[valid],pred[valid]))


@torch.inference_mode()
def extract_extra(args):
    cfg,vocab,tags=settings();meta=pd.read_csv(args.output/'metadata.csv',low_memory=False)
    z=np.load(args.output/'global_inputs.npz')['z'];checks=[]
    # True trace ratio shifts use log10 concentrations / their distinct IQRs.
    stats=vocab.transform_stats.set_index('db_column')
    rows=[];values=[]
    ix=pd.read_csv(args.output/'stimulus_index.csv',low_memory=False)
    zstim=np.load(args.output/'stimulus_z.npy')
    base_indices=ix[ix.family.eq('shift')&ix.kind.eq('contrast_Rb_Sr')&ix.h.eq(0)].index
    rb,sr=tags.index('Rb_ppm'),tags.index('Sr_ppm')
    for i in base_indices:
        r=ix.iloc[i];base=zstim[i]
        for kind in ('single_log_Rb','log_ratio_Rb_Sr','common_trace_log'):
            direction=np.zeros(len(tags))
            if kind=='single_log_Rb':direction[rb]=1/float(stats.loc['rb_ppm','iqr'])
            elif kind=='log_ratio_Rb_Sr':
                direction[rb]=.5/float(stats.loc['rb_ppm','iqr']);direction[sr]=-.5/float(stats.loc['sr_ppm','iqr'])
            else:
                for j,f in enumerate(cfg['data']['numeric_fields']):
                    if f['log_transform'] and np.isfinite(base[j]):direction[j]=1/max(float(stats.loc[f['db_column'],'iqr']),1e-6)
            for h in (0.,.025,.1,.25,.5):
                raw=base+h*direction
                rows.append(dict(sample=r['sample'],background=r.background,kind=kind,h=h,
                                 input_norm=float(np.linalg.norm(h*direction)),clipped=int(np.sum(abs(raw)>12))))
                values.append(np.clip(raw,-12,12))
    extra=np.stack(values);pd.DataFrame(rows).to_csv(args.output/'ratio_inputs.csv',index=False)
    np.save(args.output/'ratio_z.npy',extra)
    for name in MODELS:
        torch.manual_seed(42)
        ck=torch.load(args.rope_checkpoint if 'rope' in name else args.v2_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model=build_model(ck['config'],vocab)
        if not name.startswith('random'):model.load_state_dict(ck['model_state_dict'])
        model.to(args.device).eval();del ck
        print(f'Full masked-label / ratio companion: {name}',flush=True)
        result=infer(model,z,vocab,args.device,128,labels=meta)
        np.savez(args.output/f'{name}_masked_full.npz',**result)
        result=infer(model,extra,vocab,args.device,128)
        np.savez(args.output/f'{name}_ratio.npz',**result)
        if 'rope' in name:
            batch=tensor_batch(z[:16],vocab,args.device)
            token=model.make_token_inputs(batch)
            token=torch.cat([model.cls_token.expand(len(token),-1,-1),token],1)
            valid=torch.cat([torch.ones(len(token),1,dtype=torch.bool,device=args.device),batch['attention_mask']],1)
            pos=model.make_rope_positions(batch)
            original=model.transformer_encoder(token,valid,pos)
            both=model.transformer_encoder(token,valid,pos+1.)
            numericpos=pos.clone();numericpos[:,1:]+=batch['attention_mask']
            actual=model.transformer_encoder(token,valid,numericpos)
            checks.append(dict(model=name,all_positions_including_cls_relative_error=float((both-original).norm()/original.norm()),
                               numeric_only_shift_relative_change=float((actual-original).norm()/original.norm())))
        del model,result;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    pd.DataFrame(checks).to_csv(args.output/'translation_anchor_control.csv',index=False)


def cpu_analysis(args):
    out=args.output;dest=out/'results';dest.mkdir(exist_ok=True)
    inputs=np.load(out/'global_inputs.npz');fit,ev=inputs['fit'],inputs['evaluation']
    meta=pd.read_csv(out/'metadata.csv',low_memory=False);ix=pd.read_csv(out/'stimulus_index.csv',low_memory=False)
    ratio_ix=pd.read_csv(out/'ratio_inputs.csv')
    qr=[];dr=[];process=[];ratio=[];pcrows=[];clsrows=[]
    for name in MODELS:
        masked=np.load(out/f'{name}_masked_full.npz')['mean'];g=np.load(out/f'{name}_global.npz');global_x=g['mean']
        for mode,x in [('numeric',global_x),('masked_labels',masked),('cls_numeric',g['cls'])]:
            pca,spaces=projections(x,fit);center=pca.mean_
            for space,basis in spaces.items():
                xx=transformed(x[ev],basis,center)
                q,nn=quality(xx,meta.iloc[ev]);pred=vote(nn,meta.iloc[ev].rock_type)
                q.update(classification_metrics(meta.iloc[ev].rock_type,pred))
                qr.append(dict(model=name,mode=mode,space=space,**q))
                if mode=='numeric':
                    for seqno,seq in enumerate(EVOLUTION_SEQUENCES):
                        seq=[c for c in seq if (meta.rock_type==c).sum()>=20]
                        centroids=np.stack([x[meta.rock_type.eq(c)].mean(0) for c in seq])
                        centroids=transformed(centroids,basis,center)
                        process.append(dict(model=name,space=space,sequence=seqno,
                                            classes=len(seq),**path_metrics(centroids,np.arange(len(seq)))))
            pcrows.append(dict(model=name,mode=mode,pc1=pca.explained_variance_ratio_[0],
                               pc3=pca.explained_variance_ratio_[:3].sum(),pc32=pca.explained_variance_ratio_[:32].sum(),
                               pc64=pca.explained_variance_ratio_.sum()))
        radius=np.sqrt(np.mean(np.sum((global_x[fit]-global_x[fit].mean(0))**2,axis=1)))
        latent=np.load(out/f'{name}_stimuli.npz')['mean']
        reference=global_x[ev]/np.maximum(np.linalg.norm(global_x[ev],axis=1,keepdims=True),1e-12)
        for rate,records in ix[ix.family.eq('dropout')].groupby('rate'):
            query=latent[records.index];query=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
            sim=query@reference.T
            for row,pos in enumerate(records.global_pos.to_numpy(int)):sim[row,np.flatnonzero(ev==pos)]=-np.inf
            nn=np.argpartition(-sim,14,axis=1)[:,:15];pred=vote(nn,meta.iloc[ev].rock_type)
            truth=meta.iloc[records.global_pos.to_numpy(int)].rock_type
            dr.append(dict(model=name,rate=rate,**classification_metrics(truth,pred)))
        x=np.load(out/f'{name}_ratio.npz')['mean']
        for (sample,kind),records in ratio_ix.groupby(['sample','kind'],sort=False):
            records=records.sort_values('h');base=x[records.index[0]]
            for i,r in records.iloc[1:].iterrows():
                ratio.append(dict(model=name,sample=sample,kind=kind,h=r.h,clipped=r.clipped,
                                  normalized_sensitivity=float(np.linalg.norm(x[i]-base)/(r.input_norm*radius))))
        print(f'{name}: supplemental quality and process replay completed',flush=True)
    for key,rows in [('quality_modes',qr),('dropout_knn',dr),('original_differentiation_centroids',process),
                     ('raw_ratio_shifts',ratio),('pca_modes',pcrows)]:pd.DataFrame(rows).to_csv(dest/f'{key}.csv',index=False)
    training(args)


def training(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output;result=out/'results';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    logpaths={'v2':args.v2_checkpoint.parent.parent/'logs',
              'rope':args.rope_checkpoint.parent.parent/'logs'}
    summaries=[];probes=[]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    keys=['total_loss','numeric_value_loss','numeric_tag_loss','rock_group_loss','rock_origin_loss','rock_type_loss']
    for name,path in logpaths.items():
        train=pd.read_csv(path/'train_log.csv');probe=pd.read_csv(path/'probe_log.csv')
        assert train.step.is_unique and np.array_equal(train.step,np.arange(1,13201))
        assert np.isfinite(train[keys].to_numpy()).all()
        summaries.append(dict(model=name,steps=int(train.step.max()),last_epoch=float(train.epoch.iloc[-1]),
                              train_last500_mean=float(train.total_loss.tail(500).mean()),
                              train_last500_std=float(train.total_loss.tail(500).std()),
                              final_lr=float(train.learning_rate.iloc[-1]),nan_losses=0,
                              effective_batches=train.effective_batch_size.value_counts().to_dict()))
        for ax,key in zip(axes.flat,keys):
            ax.plot(probe.step,probe['probe_'+key],label=name);ax.set(title=key,xlabel='Optimizer step',yscale='log')
        for key in keys:
            last=probe['probe_'+key].iloc[-1]
            probes.append(dict(model=name,metric=key,last_logged_step=int(probe.step.iloc[-1]),value=last))
    axes.flat[0].legend();fig.suptitle('Fixed TRAINING probe; no held-out validation set');fig.tight_layout()
    fig.savefig(figdir/'training_curves.png',dpi=170);plt.close(fig)
    save_json(result/'training_integrity.json',summaries);pd.DataFrame(probes).to_csv(result/'logged_probe_comparison.csv',index=False)
    pred=pd.read_csv(out/'final_probe_predictions.csv');a=pred[pred.model.eq('v2')];b=pred[pred.model.eq('rope')]
    matched=a.merge(b,on=['sample_id','task','field'],suffixes=('_v2','_rope'),validate='one_to_one')
    assert np.array_equal(matched.target_v2,matched.target_rope)
    rows=[];rng=np.random.default_rng(42)
    for (task,field),g in matched.groupby(['task','field'],dropna=False):
        if task==0:
            for model in ('v2','rope'):
                err=g['prediction_'+model]-g.target_v2
                rows.append(dict(task='numeric_value',model=model,n=len(g),mae=abs(err).mean(),rmse=np.sqrt((err**2).mean()),
                                 abs_error_p95=np.quantile(abs(err),.95),abs_error_p99=np.quantile(abs(err),.99)))
        else:
            av=g.prediction_v2.eq(g.target_v2);bv=g.prediction_rope.eq(g.target_v2)
            rows.append(dict(task='element' if task==1 else field,model='paired',n=len(g),
                             both_correct=int((av&bv).sum()),rope_only_correct=int((~av&bv).sum()),
                             v2_only_correct=int((av&~bv).sum()),both_wrong=int((~av&~bv).sum())))
    pd.DataFrame(rows).to_csv(result/'final_probe_error_detail.csv',index=False)


@torch.inference_mode()
def controlled_background(args):
    """Hold the perturbed element at the SAME z=0 in every background."""
    cfg,vocab,tags=settings();out=args.output
    index=pd.read_csv(out/'stimulus_index.csv',low_memory=False);z=np.load(out/'stimulus_z.npy')
    bases=index[index.family.eq('derivative')&index.h.eq(0)]
    values=[];rows=[]
    for i,r in bases.iterrows():
        for h in (-.1,-.025,-.005,0.,.005,.025,.1):
            v=z[i].copy();v[int(r.target)]=h;values.append(v)
            rows.append(dict(case=r.case,sample=r['sample'],natural=r.natural,background=r.background,
                             element=r.element,target=int(r.target),h=h))
    values=np.stack(values);idx=pd.DataFrame(rows);idx.to_csv(out/'controlled_index.csv',index=False)
    np.save(out/'controlled_z.npy',values);metrics=[];relations=[]
    for name in MODELS:
        ck=torch.load(args.rope_checkpoint if 'rope' in name else args.v2_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        torch.manual_seed(42);model=build_model(ck['config'],vocab)
        if not name.startswith('random'):model.load_state_dict(ck['model_state_dict'])
        model.to(args.device).eval();del ck
        x=infer(model,values,vocab,args.device,128)['mean'];np.save(out/f'{name}_controlled.npy',x)
        vectors=[];records=[]
        for case,g in idx.groupby('case',sort=False):
            lut={r.h:i for i,r in g.iterrows()}
            ref=(x[lut[.005]].astype(float)-x[lut[-.005]])/.01
            r=g.iloc[0]
            for h in (.025,.1):
                jac=(x[lut[h]].astype(float)-x[lut[-h]])/(2*h)
                metrics.append(dict(model=name,case=case,h=h,natural=r.natural,element=r.element,
                                    fine_reference_cos=float(cosine(jac,ref)),
                                    fine_reference_relative_error=float(np.linalg.norm(jac-ref)/max(np.linalg.norm(ref),1e-12))))
                if h==.1:vectors.append(jac);records.append(r)
        for i,a in enumerate(records):
            for j in range(i+1,len(records)):
                b=records[j]
                if a.element!=b.element or a.natural!=b.natural:continue
                relations.append(dict(model=name,element=a.element,natural=a.natural,
                                      relation='within' if a.background==b.background else 'across',
                                      direction_cos=float(cosine(vectors[i],vectors[j]))))
        del model;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    pd.DataFrame(metrics).to_csv(out/'results/controlled_fine_steps.csv',index=False)
    pd.DataFrame(relations).to_csv(out/'results/controlled_backgrounds.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['extract','analyze','controlled'])
    p.add_argument('--output',type=Path,default=output_path("audit"))
    p.add_argument('--v2-checkpoint',type=Path,required=True)
    p.add_argument('--rope-checkpoint',type=Path,default=asset("rope_z_checkpoint"))
    p.add_argument('--device',default='mps');args=_guard_args(p.parse_args());torch.set_num_threads(8)
    _guard_args(args)
    if args.phase=='extract':extract_extra(args)
    elif args.phase=='controlled':controlled_background(args)
    else:
        with threadpool_limits(limits=8):cpu_analysis(args)


if __name__=='__main__':main()
