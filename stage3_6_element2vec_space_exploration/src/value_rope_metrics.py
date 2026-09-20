"""Geometry and paired quality metrics for value_rope_audit (no training)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .value_rope_audit import MODELS, STEPS, save_json, settings


def cosine(a, b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide(np.sum(a*b,axis=-1),den,out=np.full(np.shape(den),np.nan),where=den>1e-10)


def path_metrics(x, t):
    """Nonuniform grids use dx/dt. Zero motion is undefined, never 'perfect'."""
    x,t=np.asarray(x,dtype=float),np.asarray(t,dtype=float)
    dt=np.diff(t)
    if np.any(dt<=0):raise ValueError('Path parameters must be strictly increasing')
    dx=np.diff(x,axis=0); length=np.linalg.norm(dx,axis=1); speed=length/dt
    arc=length.sum()
    if arc<1e-8:
        return dict(arc=arc, speed_cv=np.nan, tangent_cos=np.nan, turn_rate=np.nan,
                    straightness=np.nan, speed_ratio=np.nan, parameter_arc_rho=np.nan,
                    uniformity_rmse=np.nan, zero_motion=True)
    cs=np.clip(cosine(dx[:-1],dx[1:]),-1,1)
    turn=np.arccos(cs)
    normalized_arc=np.r_[0,np.cumsum(length)]/arc
    normalized_t=(t-t[0])/(t[-1]-t[0])
    return dict(arc=arc, speed_cv=float(np.std(speed)/np.mean(speed)),
                tangent_cos=float(np.nanmean(cs)),
                turn_rate=float(np.nanmean(turn/((dt[:-1]+dt[1:])/2))),
                straightness=float(np.linalg.norm(x[-1]-x[0])/arc),
                speed_ratio=float(np.max(speed)/max(np.median(speed),1e-12)),
                parameter_arc_rho=float(spearmanr(t,np.linalg.norm(x-x[0],axis=1)).statistic),
                uniformity_rmse=float(np.sqrt(np.mean((normalized_arc-normalized_t)**2))), zero_motion=False)


def neighbors(x, k=15):
    x=np.asarray(x,dtype=np.float32)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    sim=x@x.T;np.fill_diagonal(sim,-np.inf)
    return np.argpartition(-sim,k-1,axis=1)[:,:k]


def purity(nn, labels):
    labels=np.asarray(labels).astype(str)
    valid=~np.isin(labels,['nan','None','','<NA>'])
    usable=valid[:,None]&valid[nn]
    values=np.divide(((labels[:,None]==labels[nn])&usable).sum(1),usable.sum(1),
                     out=np.full(len(labels),np.nan),where=usable.sum(1)>0)
    return values


def quality(x,meta):
    nn=neighbors(x)
    row={}
    for column in ('rock_type','canonical_ref_id','data_source'):
        values=purity(nn,meta[column])
        row[column+'_purity']=float(np.nanmean(values))
    n=meta.num_numeric_tokens.to_numpy()
    row['token_count_neighbor_mae']=float(np.mean(abs(n[:,None]-n[nn])))
    return row,nn


def projections(global_x,fit):
    pca=PCA(n_components=min(64,global_x.shape[1]),svd_solver='randomized',random_state=42)
    pca.fit(global_x[fit])
    spaces={'FULL':None}
    for d in (64,32,3,2):
        if d<=pca.n_components_:spaces[f'PCA{d}']=pca.components_[:d].T
    for seed in range(1,11):
        q,_=np.linalg.qr(np.random.default_rng(seed).normal(size=(global_x.shape[1],min(64,global_x.shape[1]))))
        for d in (64,32,3,2):
            if d<=q.shape[1]:spaces[f'RP{d}_{seed:02}']=q[:,:d].astype(np.float32)
    return pca,spaces


def transformed(x,basis,center):
    return x if basis is None else (x-center)@basis


def derivative_metrics(x,index):
    rows=[];directions={}
    for case,g in index[index.family.eq('derivative')].groupby('case',sort=False):
        values=x[g.index].astype(float); hs=g.h.to_numpy(); lookup={round(h,5):i for i,h in enumerate(hs)}
        base=values[lookup[0.]]
        ref=(values[lookup[.025]]-values[lookup[-.025]])/.05
        rec=g.iloc[0]
        for h in STEPS:
            plus,minus=values[lookup[h]],values[lookup[-h]]
            jac=(plus-minus)/(2*h); displacement=plus-base
            speed=np.linalg.norm(jac)
            if h==.1:directions[case]=(rec,jac)
            rows.append(dict(case=case,sample=rec['sample'],background=rec.background,natural=rec.natural,
                             element=rec.element,h=h, direction_cos=float(cosine(jac,ref)), speed=speed,
                             jac_relative_error=float(np.linalg.norm(jac-ref)/max(np.linalg.norm(ref),1e-12)),
                             predictor_error=float(np.linalg.norm(displacement-h*ref)/max(np.linalg.norm(displacement),1e-12)),
                             relative_curvature=float(np.linalg.norm(plus-2*base+minus)/(h*h*max(speed,1e-12)))))
    bgrows=[]
    for tag in index.element.dropna().unique():
        selected=[v for v in directions.values() if v[0].element==tag]
        for i,(a,va) in enumerate(selected):
            for b,vb in selected[i+1:]:
                if bool(a.natural)!=bool(b.natural):continue
                bgrows.append(dict(element=tag,natural=bool(a.natural),
                                   relation='within' if a.background==b.background else 'across',
                                   backgrounds=':'.join(sorted([a.background,b.background])),
                                   direction_cos=float(cosine(va,vb))))
    return pd.DataFrame(rows),pd.DataFrame(bgrows)


def path_table(x,index):
    rows=[]
    for case,g in index[index.family.isin(['sweep','process'])].groupby('case',sort=False):
        g=g.sort_values('t');r=g.iloc[0]
        rows.append(dict(case=case,family=r.family,scope=r.scope,process=r.process,
                         sample=r['sample'],natural=r.natural,element=r.element,background=r.background,
                         **path_metrics(x[g.index],g.t.to_numpy())))
    return pd.DataFrame(rows)


def joint_table(x,index):
    rows=[]
    for case,g in index[index.family.eq('joint')].groupby('case',sort=False):
        base=x[g.index[g.component.eq('base')][0]].astype(float)
        actual=x[g.index[g.component.eq('joint')][0]].astype(float)-base
        summed=(x[g.index[~g.component.isin(['base','joint'])]].astype(float)-base).sum(0)
        r=g.iloc[0]
        rows.append(dict(case=case,sample=r['sample'],natural=r.natural,group=r.group,h=r.h,
                         direction_cos=float(cosine(actual,summed)),
                         residual=float(np.linalg.norm(actual-summed)/max(np.linalg.norm(actual),1e-12)),
                         symmetric_residual=float(2*np.linalg.norm(actual-summed)/max(np.linalg.norm(actual)+np.linalg.norm(summed),1e-12))))
    return pd.DataFrame(rows)


def collision_table(x,index):
    rows=[]
    for case,g in index[index.family.eq('sweep') & index.scope.eq('wide')].groupby('case',sort=False):
        g=g.sort_values('t');v=x[g.index].astype(float);t=g.t.to_numpy()
        distance=np.sqrt(np.maximum(((v[:,None,:]-v[None,:,:])**2).sum(-1),0))
        gap=abs(t[:,None]-t[None,:]); allowed=(gap>=4)&np.triu(np.ones_like(gap,dtype=bool))
        local=np.median(np.linalg.norm(np.diff(v,axis=0),axis=1))
        if local<1e-8:continue
        d=np.where(allowed,distance,np.inf);a,b=np.unravel_index(np.argmin(d),d.shape)
        r=g.iloc[0]
        rows.append(dict(case=case,element=r.element,background=r.background,
                         closest_far_over_local_step=float(d[a,b]/local),
                         far_pairs_below_one_local_step=int(np.sum(allowed&(distance<local))),
                         min_z=t[a],max_z=t[b],phase_2pi_over_local_step=float(
                             np.median(distance[np.isclose(gap,6.25,atol=.001)])/local)))
    return pd.DataFrame(rows)


def context_scores(x,index):
    rows=[]
    for element,g in index[index.family.eq('context')].groupby('element',sort=False):
        features=x[g.index]; y=g.background.to_numpy();groups=g.source.fillna('unknown').astype(str).to_numpy()
        for mode in ('stratified','reference_grouped'):
            cv=(StratifiedKFold(5,shuffle=True,random_state=42) if mode=='stratified'
                else StratifiedGroupKFold(5,shuffle=True,random_state=42))
            for fold,(train,test) in enumerate(cv.split(features,y,groups if mode!='stratified' else None)):
                pipe=make_pipeline(StandardScaler(),PCA(min(16,features.shape[1]),random_state=42),
                                   LogisticRegression(C=1.,max_iter=1000,class_weight='balanced'))
                # Exact fixed input vectors have no background information.
                if np.max(np.std(features.astype(float),axis=0))<1e-6:
                    pred=np.repeat(pd.Series(y[train]).mode().iloc[0],len(test))
                else:
                    pipe.fit(features[train],y[train]);pred=pipe.predict(features[test])
                rows.append(dict(element=element,split=mode,fold=fold,
                                 balanced_accuracy=balanced_accuracy_score(y[test],pred),test_n=len(test)))
    return pd.DataFrame(rows)


def paired_ci(frame, metric, keys, group_keys, baseline='v2', treatment='rope'):
    """Paired cluster bootstrap: samples, not perturbation steps, are replicates."""
    a=frame[frame.model.eq(baseline)];b=frame[frame.model.eq(treatment)]
    merged=a.merge(b,on=keys,suffixes=('_base','_new'))
    if merged.empty:return []
    merged['delta']=merged[metric+'_new']-merged[metric+'_base']
    # Merge keys keep group/sample labels unsuffixed.
    rows=[];rng=np.random.default_rng(42)
    grouper=group_keys[0] if len(group_keys)==1 else group_keys
    for key,g in merged.groupby(grouper,dropna=False):
        cluster=g.groupby('sample',dropna=False).delta.mean().dropna().to_numpy()
        if len(cluster)<2:continue
        boot=np.mean(rng.choice(cluster,(1000,len(cluster)),replace=True),axis=1)
        labels=(key,) if len(group_keys)==1 else key
        rows.append(dict(zip(group_keys,labels),metric=metric,baseline=baseline,treatment=treatment,
                         delta_mean=cluster.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
                         sample_clusters=len(cluster)))
    return rows


def analyze(args):
    with threadpool_limits(limits=8):
        _analyze(args)


def _analyze(args):
    out=args.output;reports=out/'results';reports.mkdir(exist_ok=True)
    meta=pd.read_csv(out/'metadata.csv',low_memory=False)
    index=pd.read_csv(out/'stimulus_index.csv',low_memory=False)
    inputs=np.load(out/'global_inputs.npz');fit=inputs['fit'];ev=inputs['evaluation']
    z=inputs['z'];stimz=np.load(out/'stimulus_z.npy')
    all_quality=[];all_pca=[];derivatives=[];backgrounds=[];paths=[];joints=[];collisions=[];contexts=[];shifts=[];drops=[]
    global_neighbors={};radii={}
    for name in args.models+['raw_z','missing_mask']:
        print(f'Analyzing {name}',flush=True)
        ismodel=name in MODELS
        if ismodel:
            global_data=np.load(out/f'{name}_global.npz');global_x=global_data['mean']
            data=np.load(out/f'{name}_stimuli.npz');x=data['mean']
        else:
            global_x=(np.nan_to_num(z,nan=0.) if name=='raw_z' else np.isfinite(z).astype(float)).astype(np.float32)
            x=(np.nan_to_num(stimz,nan=0.) if name=='raw_z' else np.isfinite(stimz).astype(float)).astype(np.float32)
        center=global_x[fit].mean(0)
        radius=float(np.sqrt(np.mean(np.sum((global_x[fit]-center)**2,axis=1))));radii[name]=radius
        pca,spaces=projections(global_x,fit)
        np.savez(out/f'{name}_pca_basis.npz',components=pca.components_,center=pca.mean_,fit=fit)
        eigen=pca.explained_variance_ratio_
        all_pca.append(dict(model=name,pc1=eigen[0],pc2_cumulative=eigen[:2].sum(),pc3_cumulative=eigen[:3].sum(),
                            pc32_cumulative=eigen[:32].sum(),pc64_cumulative=eigen.sum(),radius=radius))
        full_paths=path_table(x,index);full_deriv,full_bg=derivative_metrics(x,index);full_joint=joint_table(x,index)
        for space,basis in spaces.items():
            gx=transformed(global_x[ev],basis,center)
            q,nn=quality(gx,meta.iloc[ev]);all_quality.append(dict(model=name,space=space,**q))
            if space=='FULL':global_neighbors[name]=nn
            # All 45 spaces are evaluated on paths and derivatives; no cherry-picked dimension.
            sx=x if basis is None else x@basis
            pt=full_paths.copy() if basis is None else path_table(sx,index)
            pt['model']=name;pt['space']=space;paths.append(pt)
            dt,bg=(full_deriv.copy(),full_bg.copy()) if basis is None else derivative_metrics(sx,index)
            dt['model']=name;dt['space']=space;derivatives.append(dt)
            bg['model']=name;bg['space']=space;backgrounds.append(bg)
            jt=full_joint.copy() if basis is None else joint_table(sx,index)
            jt['model']=name;jt['space']=space;joints.append(jt)
        if ismodel:
            for key in ['input','q1','k1','layer1','layer3','layer6','layer9','mean','target']:
                if key not in data:continue
                level=data[key]
                if key not in ['mean']:
                    pt=path_table(level,index);pt['model']=name;pt['space']=key;paths.append(pt)
                ct=collision_table(level,index);ct['model']=name;ct['level']=key;collisions.append(ct)
            ct=context_scores(data['target'],index);ct['model']=name;ct['representation']='target_contextual';contexts.append(ct)
            # Input at fixed queried value is constant for BOTH architectures.
            # The pooled input can still expose reporting masks; don't call it the target input.
            mt=np.load(out/f'{name}_masked.npz');matched=mt['indices']
            q,nn=quality(mt['mean'],meta.iloc[matched]);all_quality.append(dict(model=name,space='masked_FULL_2000',**q))
            q,_=quality(global_x[matched],meta.iloc[matched]);all_quality.append(dict(model=name,space='numeric_FULL_2000',**q))
            all_pca[-1]['numeric_masked_cos']=float(np.mean(cosine(global_x[matched],mt['mean'])))
            # Save only binned global plot coordinates locally; no public sample plot arrays.
            xy=pca.transform(global_x[ev])[:,:2]
            np.save(out/f'{name}_pca2_evaluation.npy',xy)
        elif name=='raw_z':
            ct=context_scores(x,index);ct['model']=name;ct['representation']='raw_background';contexts.append(ct)
        for case,g in index[index.family.eq('shift')].groupby('case',sort=False):
            g=g.sort_values('h');r=g.iloc[0];base=x[g.index[0]].astype(float)
            for i,rr in g.iloc[1:].iterrows():
                input_norm=rr.input_norm if np.isfinite(rr.input_norm) else rr.h
                shifts.append(dict(model=name,case=case,sample=rr['sample'],natural=rr.natural,kind=rr.kind,h=rr.h,
                                   clipped=rr.clipped,normalized_sensitivity=float(np.linalg.norm(x[i]-base)/max(input_norm*radius,1e-12))))
        for case,g in index[index.family.eq('dropout')].groupby('case',sort=False):
            base=x[g.index[g.rate.eq(0)][0]]
            for i,r in g.iterrows():
                drops.append(dict(model=name,sample=case,rate=r.rate,cosine=float(cosine(base,x[i])),
                                  relative_displacement=float(np.linalg.norm(x[i]-base)/max(radius,1e-12))))
        if ismodel:global_data.close();data.close()
        # Persist completed model tables so a late plotting failure cannot lose the audit.
        pd.DataFrame(all_quality).to_csv(reports/'quality.csv',index=False)
        print(f'  {name}: all projections and geometry completed',flush=True)
    tables={'quality':pd.DataFrame(all_quality),'pca':pd.DataFrame(all_pca),
            'derivatives':pd.concat(derivatives,ignore_index=True),'backgrounds':pd.concat(backgrounds,ignore_index=True),
            'paths':pd.concat(paths,ignore_index=True),'joints':pd.concat(joints,ignore_index=True),
            'collisions':pd.concat(collisions,ignore_index=True),'context':pd.concat(contexts,ignore_index=True),
            'shifts':pd.DataFrame(shifts),'dropout':pd.DataFrame(drops)}
    for key,frame in tables.items():frame.to_csv(reports/f'{key}.csv',index=False)
    # Explicit contiguous linear range; large h cannot rescue a smaller failed h.
    dt=tables['derivatives']; ranges=[]
    for (model,space,case),g in dt.groupby(['model','space','case'],sort=False):
        g=g.sort_values('h');bound=0.
        for _,r in g.iterrows():
            if r.predictor_error<=.1 and r.direction_cos>=.99:bound=r.h
            else:break
        ranges.append(dict(model=model,space=space,case=case,sample=g.iloc[0]['sample'],
                           natural=g.iloc[0].natural,element=g.iloc[0].element,linear_radius=bound))
    tables['linear_range']=pd.DataFrame(ranges);tables['linear_range'].to_csv(reports/'linear_range.csv',index=False)
    ci=[]
    for baseline in ('v2','random_rope'):
        for metric in ('direction_cos','predictor_error','relative_curvature'):
            ci+=paired_ci(dt[dt.space.eq('FULL') & dt.natural.eq(True)],metric,
                          ['case','sample','h'],['h'],baseline=baseline)
        pt=tables['paths'];pt=pt[pt.space.eq('FULL')&pt.family.eq('sweep')&pt.natural.eq(True)]
        for metric in ('speed_cv','tangent_cos','turn_rate','uniformity_rmse'):
            ci+=paired_ci(pt,metric,['case','sample','scope'],['scope'],baseline=baseline)
    pd.DataFrame(ci).to_csv(reports/'paired_bootstrap.csv',index=False)
    # Quality CIs pair individual evaluation samples, including source-purity changes.
    qr=[];rng=np.random.default_rng(42)
    for baseline in ('v2','random_rope','random_v2','raw_z','missing_mask'):
        for col in ('rock_type','canonical_ref_id','data_source'):
            a=purity(global_neighbors[baseline],meta.iloc[ev][col]);b=purity(global_neighbors['rope'],meta.iloc[ev][col])
            d=b-a;usable=np.isfinite(d);d=d[usable]
            # Reference-cluster bootstrap, avoiding treating one publication as many independent replications.
            cluster=pd.DataFrame({'d':d,'ref':meta.iloc[ev].canonical_ref_id.astype(str).to_numpy()[usable]}).groupby('ref').agg(['sum','count'])['d']
            ix=rng.integers(0,len(cluster),size=(1000,len(cluster)))
            boot=cluster['sum'].to_numpy()[ix].sum(1)/cluster['count'].to_numpy()[ix].sum(1)
            qr.append(dict(baseline=baseline,metric=col+'_purity',delta_mean=d.mean(),
                           ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),reference_clusters=len(cluster)))
    pd.DataFrame(qr).to_csv(reports/'quality_paired_bootstrap.csv',index=False)
    # Geometric phase code has many frequencies; a 2pi rotation of the fastest
    # pair is not a full-vector period. Test only the training-supported [-12,12].
    gap=np.linspace(.001,24,24000);freq=10000.**(-2*np.arange(32)/64)
    kernel=np.cos(gap[:,None]*freq).mean(1)
    mask=gap>=4
    save_json(reports/'phase_kernel.json',dict(similarity_at_2pi=float(np.cos(2*np.pi*freq).mean()),
              maximum_similarity_gap_ge4=float(kernel[mask].max()),gap_at_max=float(gap[mask][np.argmax(kernel[mask])]),
              clipping_collisions='z values beyond +12 or below -12 become identical at preprocessing; both models share this limitation'))
    plot_results(args,tables)
    save_json(reports/'verification.json',dict(models=args.models,global_samples=len(meta),fit_samples=len(fit),
              evaluation_samples=len(ev),stimuli=len(index),spaces_per_768_model=45,
              all_outputs_finite={m:all(np.isfinite(np.load(out/f'{m}_global.npz')[k]).all() for k in ('mean','cls','input')) for m in args.models},
              private_output_directory=str(out),training_set_diagnostic=True))
    print(f'Completed analysis: {reports}',flush=True)


def plot_results(args,tables):
    out=args.output;figs=out/'figures';figs.mkdir(exist_ok=True)
    colors=dict(v2='#5275a3',rope='#d56639',random_rope='#67a580',random_v2='#aa93b9')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    dt=tables['derivatives'];pt=tables['paths'];lr=tables['linear_range']
    for name in args.models:
        g=dt[dt.model.eq(name)&dt.space.eq('FULL')&dt.natural.eq(True)].groupby('h')
        a=g.direction_cos.median();axes[0,0].plot(a.index,a.values,'o-',label=name,color=colors[name])
        a=g.predictor_error.median();axes[0,1].plot(a.index,a.values,'o-',label=name,color=colors[name])
        g=pt[pt.model.eq(name)&pt.space.eq('FULL')&pt.family.eq('sweep')&pt.scope.eq('local')]
        axes[1,0].scatter([name]*len(g),g.speed_cv,alpha=.3,s=18,color=colors[name])
        g=lr[lr.model.eq(name)&lr.space.eq('FULL')&lr.natural.eq(True)]
        axes[1,1].plot(np.sort(g.linear_radius),np.linspace(0,1,len(g)),label=name,color=colors[name])
    axes[0,0].set(title='Central derivative direction vs h=0.025',xlabel='Step h (z)',ylabel='Median cosine',xscale='log')
    axes[0,1].set(title='Local tangent prediction error',xlabel='Step h (z)',ylabel='Median relative error',xscale='log')
    axes[1,0].set(title='Local sweep speed variability (±1 z)',ylabel='Speed CV')
    axes[1,1].set(title='Contiguous local linear range',xlabel='Largest tested h satisfying 10% error, cosine ≥0.99',ylabel='Empirical CDF')
    axes[0,0].legend();fig.tight_layout();fig.savefig(figs/'local_geometry.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in args.models:
        g=pt[pt.model.eq(name)&pt.family.eq('sweep')&pt.scope.eq('local')]
        levels=['input','q1','layer1','layer3','layer6','layer9','FULL']
        for ax,metric in zip(axes,['speed_cv','tangent_cos','turn_rate']):
            values=[g[g.space.eq(k)][metric].median() for k in levels]
            ax.plot(levels,values,'o-',label=name,color=colors[name]);ax.set_title(metric);ax.tick_params(axis='x',rotation=35)
    axes[0].legend();fig.suptitle('Where smoothness changes: input, rotated Q, Transformer depth');fig.tight_layout()
    fig.savefig(figs/'layer_geometry.png',dpi=170);plt.close(fig)
    q=tables['quality'];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,['rock_type_purity','canonical_ref_id_purity','data_source_purity']):
        for name in args.models:
            g=q[q.model.eq(name)].set_index('space')
            levels=['FULL','PCA64','PCA32','PCA3','PCA2']
            ax.plot(levels,g.loc[levels,metric],'o-',label=name,color=colors[name])
        ax.set_title(metric);ax.set_ylim(0,1)
    axes[0].legend();fig.tight_layout();fig.savefig(figs/'quality_guardrails.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,9));meta=pd.read_csv(out/'metadata.csv',low_memory=False)
    ev=np.load(out/'global_inputs.npz')['evaluation'];bg=meta.iloc[ev].background.to_numpy()
    for ax,name in zip(axes.flat,args.models):
        xy=np.load(out/f'{name}_pca2_evaluation.npy')
        for group,c in [('mafic','#4485b3'),('intermediate','#79a76a'),('felsic','#d48454')]:
            mask=bg==group;ax.scatter(xy[mask,0],xy[mask,1],s=4,alpha=.35,c=c,label=group,rasterized=True)
        ax.set_title(name+' (independent PCA; axes not aligned)');ax.set_xlabel('PC1');ax.set_ylabel('PC2')
    axes.flat[0].legend();fig.tight_layout();fig.savefig(figs/'global_pca.png',dpi=170);plt.close(fig)
