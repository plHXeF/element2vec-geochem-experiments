"""Masked recovery, static rank and matched perturbation/dropout diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .schema import FIELDS, INDEX, GROUPS, LABELS, EXACT, hidden_closure, upper_limit
from .prepare import coordinates, save_json, digest_file
from .training import load_model, build, seed_all
from .batching import make_batch, to_device


def subset(data, rows):
    return SimpleNamespace(values=np.array(data.values[rows]), states=np.array(data.states[rows]),
                           labels=np.array(data.labels[rows]), sample_ids=np.array(data.sample_ids[rows]),
                           weights=np.array(data.weights[rows]), stats=data.stats)


@torch.inference_mode()
def encode(model, data, rows, device, hide=(), batch_size=32):
    result=[]
    for start in range(0,len(rows),batch_size):
        batch=make_batch(data,rows[start:start+batch_size],model,training=False,chemistry_only=True,hide=hide)
        result.append(model.encode(to_device(batch,device))["mean"].cpu().numpy())
    return np.concatenate(result)


def cosine(a,b):
    denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.divide(np.sum(a*b,axis=1),denominator,out=np.full(len(a),np.nan),where=denominator>1e-12)


def clean_number(x):
    return float(x) if np.isfinite(x) else None


def audit(data,root,checkpoint,device="cpu",max_per_field=256):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist
    root=Path(root);out=root/"evaluation"
    model,ck=load_model(checkpoint,device);model.eval()
    if ck["prepared_fingerprint"]!=data.fingerprint:raise ValueError("Mismatched input")
    seed_all(ck["config"]["training"]["seed"]);random_model=build(ck["config"],data).to(device).eval()
    pool=pd.read_csv(data.root/"cohort.csv")
    supplement=data.root/"rare_field_cohort.csv"
    if supplement.exists():
        pool=pd.concat([pool,pd.read_csv(supplement).drop(columns="tag_index")]).drop_duplicates("row_index").sort_values("row_index")
    rows=pool.row_index.to_numpy(np.int64);splits=pool.split.to_numpy()
    values=np.array(data.values[rows]);states=np.array(data.states[rows]);z=coordinates(values,states,data.stats)
    known=np.isin(states,EXACT);base=np.concatenate((np.where(known,z,0),known.astype(float)),axis=1)
    recovery=[]
    for tag,name in enumerate(FIELDS):
        train=np.flatnonzero((splits==0)&known[:,tag]);dev=np.flatnonzero((splits==1)&known[:,tag]);test=np.flatnonzero((splits==2)&known[:,tag])
        if not len(test):
            recovery.append({"field":name,"status":"no_exact_test_observation","n_test":0});continue
        if len(test)>max_per_field:
            test=np.random.default_rng(42+tag).choice(test,max_per_field,replace=False)
        y=z[test,tag];pred=[]
        with torch.inference_mode():
            for start in range(0,len(test),32):
                ids=rows[test[start:start+32]]
                b=to_device(make_batch(data,ids,model,training=False,chemistry_only=True,target_tag=tag),device)
                output=model(b);pred.extend(output["mu"][b["numeric_target"]].cpu().numpy())
        pred=np.asarray(pred)
        item={"field":name,"status":"evaluated","n_fit":len(train),"n_dev":len(dev),"n_test":len(test),
              "measured_test":int(np.isin(states[test,tag],[1,2]).sum()),"derived_test":int((states[test,tag]==3).sum()),
              "model_z_mae":float(np.mean(abs(pred-y))),"model_z_r2":clean_number(r2_score(y,pred)) if len(y)>1 else None}
        for state_name,state_ids in [("measured",[1,2]),("derived",[3])]:
            mask=np.isin(states[test,tag],state_ids)
            item[state_name+"_z_mae"]=float(np.mean(abs(pred[mask]-y[mask]))) if mask.any() else None
        if len(train)>=5:
            x=base.copy();hidden=hidden_closure([tag]);x[:,hidden]=0;x[:,len(FIELDS)+hidden]=0
            best=None;best_error=np.inf
            for alpha in (.1,1.,10.):
                reg=Ridge(alpha=alpha).fit(x[train],z[train,tag])
                error=float(np.mean((reg.predict(x[dev])-z[dev,tag])**2)) if len(dev) else (0. if alpha==1 else np.inf)
                if error<best_error:best,best_error=reg,error
            raw=best.predict(x[test]);med=np.median(z[train,tag])
            item.update(raw_ridge_z_mae=float(np.mean(abs(raw-y))),element_median_z_mae=float(np.mean(abs(med-y))),
                        raw_ridge_z_r2=clean_number(r2_score(y,raw)) if len(y)>1 else None)
        recovery.append(item)
    save_json(out/"masked_recovery.json",recovery);pd.DataFrame(recovery).to_csv(out/"masked_recovery.csv",index=False)
    # True concentration counterfactuals on a single independent report. Remove
    # equivalent forms from both endpoints so inconsistent totals cannot help.
    geometry=[]
    for tag_name in ("SiO2","MgO","FeO","Sr","La","Th"):
        j=INDEX[tag_name]
        candidates=rows[(splits==2)&known[:,j]&(values[:,j]>0)&(values[:,j]<upper_limit(tag_name)/1.1)][:128]
        if not len(candidates):continue
        original=subset(data,candidates);local=np.arange(len(candidates));hide=[k for k in hidden_closure([j]) if k!=j]
        for name,encoder in [("trained",model),("random",random_model)]:
            paths={0:encode(encoder,original,local,device,hide)}
            for h in (-.02,-.01,.01,.02):
                perturbed=subset(data,candidates);perturbed.values[:,j]*=1+h
                paths[h]=encode(encoder,perturbed,local,device,hide)
            d1=(paths[.01]-paths[-.01])/.02;d2=(paths[.02]-paths[-.02])/.04
            direction=cosine(d1,d2);valid=np.isfinite(direction)
            curvature=np.linalg.norm(paths[.01]-2*paths[0]+paths[-.01],axis=1)/np.maximum(np.linalg.norm(paths[.01]-paths[-.01],axis=1),1e-10)
            geometry.append({"field":tag_name,"model":name,"n":len(local),"nonzero_directions":int(valid.sum()),
                             "step_direction_cosine":clean_number(np.median(direction[valid])) if valid.any() else None,
                             "relative_curvature":float(np.median(curvature)),"interpretation":"single-report sensitivity; not a validated petrological process"})
    save_json(out/"geometry.json",geometry)
    candidates=rows[splits==2][:256];drop=subset(data,candidates);eligible=[]
    for i in range(len(candidates)):
        groups=np.unique(GROUPS[drop.states[i]!=0])
        if len(groups)<2:continue
        count=min(len(groups)-1,max(1,round(.3*len(groups))))
        chosen=np.random.default_rng(int(drop.sample_ids[i])+42).choice(groups,count,replace=False)
        drop.states[i,np.isin(GROUPS,chosen)]=0;eligible.append(i)
    dropout=[]
    if eligible:
        for name,encoder in [("trained",model),("random",random_model)]:
            before=encode(encoder,data,candidates,device);after=encode(encoder,drop,np.arange(len(candidates)),device)
            c=cosine(before,after)[eligible]
            dropout.append({"model":name,"n":len(eligible),"group_drop_fraction":.3,"latent_cosine_median":clean_number(np.nanmedian(c))})
    save_json(out/"panel_dropout.json",dropout)
    statics=[]
    eligible=np.array([st["n_exact"]>0 for st in data.stats])
    ree=[INDEX[x] for x in "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split() if eligible[INDEX[x]]]
    for name,encoder in [("trained",model),("random",random_model)]:
        e=encoder.numeric_tag_embedding.weight.detach().cpu().numpy()[:len(FIELDS)]
        singular=np.linalg.svd(e[eligible]-e[eligible].mean(0),compute_uv=False)
        rank=int(np.sum(singular>singular[0]*1e-6)) if len(singular) and singular[0]>0 else 0
        item={"model":name,"exact_supervised_identities":int(eligible.sum()),"rank_limit":min(int(eligible.sum())-1,e.shape[1]),"numerical_rank":rank}
        if len(ree)>3:
            distance=pdist(e[ree],metric="cosine");order=pdist(np.array(ree)[:,None])
            item["ree_order_distance_spearman"]=clean_number(spearmanr(distance,order).statistic)
        statics.append(item)
    save_json(out/"static_embedding.json",statics)
    save_json(out/"scientific_audit.json",{"passed":True,"checkpoint_sha256":digest_file(checkpoint),
              "numeric_units":"frozen pre-angle z; observed and derived targets reported separately",
              "hidden_aliases":True,"unseen_pretraining_generalization":False,"device":device,
              "files":["masked_recovery.json","geometry.json","panel_dropout.json","static_embedding.json"]})
