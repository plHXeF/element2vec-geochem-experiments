"""Physical replay of frozen z stimuli, with explicit admissibility/closure flags."""
import gc
import json

import numpy as np
import pandas as pd
import torch

from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame
from stage3_5_element2vec_v2.src.utils.io import load_config
from .geo_mask15_audit import context, model_for, cache, MODELS
from .value_rope_audit import infer, sha, save_json, STEPS
from .value_rope_metrics import cosine, path_metrics, context_scores, paired_ci
from .value_rope_supplement import vote, classification_metrics


def physical_replay(z,index,meta,cfg,vocab):
    fields=cfg["data"]["numeric_fields"];cols=[f["db_column"] for f in fields]
    stats=vocab.transform_stats.set_index("db_column")
    raw=np.empty(z.shape,dtype=float)
    for j,f in enumerate(fields):
        s=stats.loc[f["db_column"]]
        value=z[:,j].astype(float)*s.iqr+s["median"]
        raw[:,j]=10.**value-s.eps if s.log_transform else value
    # Preserve the unperturbed raw concentrations (including values clipped in z).
    lookup={str(int(v)):i for i,v in enumerate(meta.sample_id)}
    baseline=meta[[f"z__{f['name']}" for f in fields]].to_numpy(float)
    for sample,group in index.groupby("sample",sort=False):
        key=str(sample).replace("natural_","")
        try:key=str(int(float(key)))
        except ValueError:continue
        if key not in lookup:continue
        i=lookup[key];positions=group.index.to_numpy()
        unchanged=np.isclose(z[positions],baseline[i],rtol=0,atol=1e-6)
        original=meta.iloc[i][cols].to_numpy(float)
        raw[positions]=np.where(unchanged,original,raw[positions])
    frame=pd.DataFrame(raw,columns=cols)
    transformed,qc=transform_numeric_frame(frame,cfg,vocab)
    coordinates=transformed[cols].to_numpy(np.float32)
    # Deleting a token must not trigger reclosure of the retained panel.
    if "family" in index:
        global_geo,_=transform_numeric_frame(meta,cfg,vocab)
        for i,row in index[index.family.eq("dropout")].iterrows():
            coordinates[i]=global_geo.iloc[int(row.global_pos)][cols].to_numpy(np.float32)
            coordinates[i,~np.isfinite(z[i])]=np.nan
    panel=frame[cols[:11]]
    eligible=panel.notna().all(axis=1)&panel.ge(0).all(axis=1)&panel.sum(axis=1).between(90,105)
    valid=~(panel.lt(0).any(axis=1)|frame["co2"].lt(0))
    for col in cols[13:]:valid &= ~(frame[col].notna() & frame[col].le(0))
    baseline_cfg=load_config("stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml")
    reconstructed,_=transform_numeric_frame(frame,baseline_cfg,vocab)
    rz=reconstructed[cols].to_numpy(float)
    matches=np.all(np.isclose(z,rz,atol=2e-5,rtol=1e-5,equal_nan=True),axis=1)
    flags=index.copy()
    flags["physical_valid"]=valid.to_numpy()
    flags["closed"]=eligible.to_numpy()
    flags["baseline_roundtrip"]=matches
    flags["boundary_or_clipped_coordinate"]=np.any(abs(z)>=11.99999,axis=1)
    # Clipped synthetic stimuli cannot be uniquely inverted; keep only as stress tests.
    flags["admissible"]=flags.physical_valid & flags.baseline_roundtrip & ~flags.boundary_or_clipped_coordinate
    return coordinates,flags,qc


def prepare_geometry(args):
    cfg,vocab,meta,fit,ev=context(args)
    manifest=args.output/"stimuli_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["baseline_stimuli_sha256"]==sha(args.baseline/"stimulus_z.npy")
        return
    specs=[("stimulus","stimulus_index.csv","stimulus_z.npy"),
           ("controlled","controlled_index.csv","controlled_z.npy"),
           ("ratio","ratio_inputs.csv","ratio_z.npy")]
    counts={}
    for name,idxfile,arrayfile in specs:
        index=pd.read_csv(args.baseline/idxfile,low_memory=False)
        if name=="ratio":index["case"]=index["sample"].astype(str)+":"+index.kind
        z=np.load(args.baseline/arrayfile)
        coords,flags,qc=physical_replay(z,index,meta,cfg,vocab)
        np.save(args.output/f"{name}_geo.npy",coords)
        flags.to_csv(args.output/f"{name}_index.csv",index=False)
        counts[name]=dict(rows=len(index),admissible=int(flags.admissible.sum()),
                          negative_or_invalid=int((~flags.physical_valid).sum()),
                          clipped=int(flags.boundary_or_clipped_coordinate.sum()),
                          failed_roundtrip=int((~flags.baseline_roundtrip).sum()))
    save_json(manifest,dict(baseline_stimuli_sha256=sha(args.baseline/"stimulus_z.npy"),counts=counts,
                           physical_parameterization="same concentrations reconstructed from baseline robust-z; unperturbed natural values restored",
                           noninvertible="clipped stimuli excluded from main geometry; retained only in explicit stress diagnostics"))


def extract_geometry(args):
    cfg,vocab,meta,fit,ev=context(args)
    prepare_geometry(args)
    index=pd.read_csv(args.output/"stimulus_index.csv",low_memory=False)
    coordinates=np.load(args.output/"stimulus_geo.npy")
    for name in ("geo","random_geo"):
        marker=args.output/f"{name}_geometry_manifest.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            assert saved["checkpoint_sha256"]==sha(args.geo_checkpoint)
            assert saved["stimulus_sha256"]==sha(args.output/"stimulus_geo.npy")
            print(name+": geometry cache verified",flush=True);continue
        print("Physical geometry "+name,flush=True)
        model=model_for(args,name,cfg,vocab)
        outputs=infer(model,coordinates,vocab,args.device,args.batch_size,
                      targets=index.target.fillna(-1).to_numpy(int),layers=True)
        np.savez(args.output/f"{name}_stimuli.npz",**outputs)
        subset=index.index[index.family.eq("derivative")][:26]
        model.to("cpu")
        cpu=infer(model,coordinates[subset],vocab,"cpu",8)["mean"]
        error=float(np.linalg.norm(cpu-outputs["mean"][subset])/np.linalg.norm(cpu))
        model.to(args.device)
        for panel in ("controlled","ratio"):
            result=infer(model,np.load(args.output/f"{panel}_geo.npy"),vocab,args.device,args.batch_size)
            if panel=="controlled":np.save(args.output/f"{name}_controlled.npy",result["mean"])
            else:np.savez(args.output/f"{name}_ratio.npz",**result)
        # Additional geo control: same coordinate at queried element across backgrounds.
        selected=index[index.family.eq("context")]
        fixed=coordinates[selected.index].copy()
        for tag,g in selected.groupby("element",sort=False):
            j=int(g.target.iloc[0]);positions=np.flatnonzero(selected.element.eq(tag))
            fixed[positions,j]=np.nanmedian(fixed[positions,j])
        extra=infer(model,fixed,vocab,args.device,args.batch_size,targets=selected.target.to_numpy(int))
        np.savez(args.output/f"{name}_fixed_coordinate_context.npz",**extra)
        save_json(marker,dict(checkpoint_sha256=sha(args.geo_checkpoint),
                              stimulus_sha256=sha(args.output/"stimulus_geo.npy"),cpu_mps_relative_error=error))
        del model,outputs,extra
        gc.collect()
        if args.device=="mps":torch.mps.empty_cache()


def derivatives(x,index,reference=.025,steps=STEPS):
    rows=[]
    for case,g in index.groupby("case",sort=False):
        lut={round(r.h,5):i for i,r in g.iterrows()}
        needed=[0.,reference,-reference]
        if not all(h in lut for h in needed):continue
        refrows=g.loc[[lut[h] for h in needed]]
        if not refrows.admissible.all() or refrows.closed.nunique()!=1:continue
        base=x[lut[0.]].astype(float)
        ref=(x[lut[reference]].astype(float)-x[lut[-reference]])/(2*reference)
        if np.linalg.norm(ref)<1e-6:continue
        r=g.iloc[0]
        for h in steps:
            if h not in lut or -h not in lut:continue
            use=g.loc[[lut[0.],lut[h],lut[-h]]]
            if not use.admissible.all() or use.closed.nunique()!=1:continue
            plus=x[lut[h]].astype(float);minus=x[lut[-h]].astype(float)
            jac=(plus-minus)/(2*h);d=plus-base
            rows.append(dict(case=case,sample=r["sample"],natural=r.natural,element=r.element,background=r.background,
                             h=h,speed=np.linalg.norm(jac),direction_cos=float(cosine(jac,ref)),
                             predictor_error=np.linalg.norm(d-h*ref)/max(np.linalg.norm(d),1e-12),
                             relative_curvature=np.linalg.norm(plus-2*base+minus)/(h*h*max(np.linalg.norm(jac),1e-12))))
    return pd.DataFrame(rows)


def geometry_metrics(args):
    cfg,vocab,meta,fit,ev=context(args)
    index=pd.read_csv(args.output/"stimulus_index.csv",low_memory=False)
    controlled=pd.read_csv(args.output/"controlled_index.csv")
    ratio_index=pd.read_csv(args.output/"ratio_index.csv")
    collected={k:[] for k in ("derivatives","fine_steps","paths","joints","context","dropout","ratio","collisions","backgrounds","linearity")}
    exclusions=[]
    for name in MODELS:
        print("Geometry metrics "+name,flush=True)
        data=np.load(cache(args,name,"stimuli.npz"));x=data["mean"]
        global_x=np.load(cache(args,name,"global.npz"))["mean"]
        radius=float(np.sqrt(np.mean(np.sum((global_x[fit]-global_x[fit].mean(0))**2,axis=1))))
        basis=np.load(args.output/f"{name}_pca_basis.npz")["components"]
        for space,matrix in [("FULL",None)]+[(f"PCA{d}",basis[:d].T) for d in (64,32,3,2)]:
            sx=x if matrix is None else x@matrix
            dt=derivatives(sx,index[index.family.eq("derivative")]);dt["model"]=name;dt["space"]=space
            collected["derivatives"].append(dt)
            for case,g in dt.groupby("case",sort=False):
                bound=0.
                for h in STEPS:
                    v=g[g.h.eq(h)]
                    if len(v)==1 and v.predictor_error.iloc[0]<=.1 and v.direction_cos.iloc[0]>=.99:bound=h
                    else:break
                collected["linearity"].append(dict(model=name,space=space,case=case,natural=g.natural.iloc[0],linear_radius=bound))
            for case,g in index[index.family.isin(["sweep","process"])].groupby("case",sort=False):
                if not g.admissible.all() or g.closed.nunique()!=1:
                    if space=="FULL":exclusions.append(dict(model=name,case=case,reason="invalid/clipped or closure boundary",family=g.family.iloc[0]))
                    continue
                g=g.sort_values("t");r=g.iloc[0]
                collected["paths"].append(dict(model=name,space=space,case=case,sample=r["sample"],natural=r.natural,
                                               family=r.family,scope=r.scope,process=r.process,**path_metrics(sx[g.index],g.t)))
            for case,g in index[index.family.eq("joint")].groupby("case",sort=False):
                if not g.admissible.all() or g.closed.nunique()!=1:continue
                base=sx[g.index[g.component.eq("base")][0]].astype(float)
                actual=sx[g.index[g.component.eq("joint")][0]].astype(float)-base
                summed=(sx[g.index[~g.component.isin(["base","joint"])]].astype(float)-base).sum(0)
                r=g.iloc[0]
                collected["joints"].append(dict(model=name,space=space,case=case,sample=r["sample"],natural=r.natural,group=r.group,h=r.h,
                                                residual=np.linalg.norm(actual-summed)/max(np.linalg.norm(actual),1e-12),
                                                direction_cos=float(cosine(actual,summed))))
        # Fine central differences use exactly the same physical perturbations at h=.005.
        fine=derivatives(np.load(cache(args,name,"controlled.npy")),controlled,reference=.005,steps=(.025,.1))
        fine["model"]=name;collected["fine_steps"].append(fine)
        for level in ("input","q1","k1","layer1","layer3","layer6","layer9"):
            if level not in data:continue
            for case,g in index[index.family.eq("sweep")&index.scope.eq("local")].groupby("case",sort=False):
                if not g.admissible.all() or g.closed.nunique()!=1:continue
                g=g.sort_values("t");r=g.iloc[0]
                collected["paths"].append(dict(model=name,space=level,case=case,sample=r["sample"],natural=r.natural,
                                               family="sweep",scope="local",process=None,**path_metrics(data[level][g.index],g.t)))
        cx=context_scores(data["target"],index[index.admissible])
        cx["model"]=name;cx["condition"]="fixed_raw_value";collected["context"].append(cx)
        if name in ("geo","random_geo"):
            cindex=index[index.family.eq("context")].reset_index(drop=True)
            cx=context_scores(np.load(args.output/f"{name}_fixed_coordinate_context.npz")["target"],cindex)
            cx["model"]=name;cx["condition"]="fixed_coordinate";collected["context"].append(cx)
        # Cross-background response is a within-model cosine; never align model axes.
        vectors=[]
        for case,g in index[index.family.eq("derivative")].groupby("case",sort=False):
            rows=g[g.h.isin([-.1,.1,0.])]
            if len(rows)!=3 or not rows.admissible.all() or rows.closed.nunique()!=1:continue
            plus=g.index[g.h.eq(.1)][0];minus=g.index[g.h.eq(-.1)][0]
            vectors.append((g.iloc[0],(x[plus].astype(float)-x[minus])/.2))
        for i,(a,va) in enumerate(vectors):
            for b,vb in vectors[i+1:]:
                if a.element!=b.element or a.natural!=b.natural:continue
                collected["backgrounds"].append(dict(model=name,element=a.element,natural=a.natural,
                                                      relation="within" if a.background==b.background else "across",
                                                      direction_cos=float(cosine(va,vb))))
        reference=global_x[ev]/np.maximum(np.linalg.norm(global_x[ev],axis=1,keepdims=True),1e-12)
        for rate,g in index[index.family.eq("dropout")].groupby("rate"):
            query=x[g.index];unit=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12)
            sim=unit@reference.T
            for row,pos in enumerate(g.global_pos.to_numpy(int)):sim[row,np.flatnonzero(ev==pos)]=-np.inf
            nn=np.argpartition(-sim,14,axis=1)[:,:15];pred=vote(nn,meta.iloc[ev].rock_type)
            bases=global_x[g.global_pos.to_numpy(int)]
            collected["dropout"].append(dict(model=name,rate=rate,cosine=float(np.nanmean(cosine(query,bases))),
                                             relative_displacement=float(np.mean(np.linalg.norm(query-bases,axis=1))/radius),
                                             **classification_metrics(meta.iloc[g.global_pos.to_numpy(int)].rock_type,pred)))
        ratio=np.load(cache(args,name,"ratio.npz"))["mean"]
        for case,g in ratio_index.groupby("case",sort=False):
            if not g.admissible.all() or g.closed.nunique()!=1:continue
            g=g.sort_values("h");base=ratio[g.index[0]]
            for i,r in g.iloc[1:].iterrows():
                collected["ratio"].append(dict(model=name,kind=r.kind,h=r.h,
                                                normalized_sensitivity=float(np.linalg.norm(ratio[i]-base)/(r.input_norm*radius))))
        for case,g in index[index.family.eq("sweep")&index.scope.eq("wide")&index.physical_valid&index.baseline_roundtrip].groupby("case",sort=False):
            g=g.sort_values("t");v=x[g.index].astype(float);t=g.t.to_numpy()
            if len(v)<3:continue
            dist=np.linalg.norm(v[:,None]-v[None,:],axis=-1)
            local=np.median(np.linalg.norm(np.diff(v,axis=0),axis=1))
            allowed=(abs(t[:,None]-t[None,:])>=4)&np.triu(np.ones((len(t),len(t)),bool))
            collected["collisions"].append(dict(model=name,case=case,element=g.element.iloc[0],pairs=int(allowed.sum()),
                                                 collisions=int((allowed&(dist<local)).sum()),
                                                 min_far_over_local=float(dist[allowed].min()/local) if allowed.any() and local>1e-8 else np.nan))
        for table,parts in collected.items():
            if parts:
                frame=pd.concat(parts,ignore_index=True) if isinstance(parts[0],pd.DataFrame) else pd.DataFrame(parts)
                frame.to_csv(args.output/f"results/{table}.csv",index=False)
        data.close()
    pd.DataFrame(exclusions).to_csv(args.output/"results/geometry_exclusions.csv",index=False)
    df=pd.concat(collected["derivatives"],ignore_index=True)
    ci=[]
    for baseline in ("v2","rope","random_geo"):
        for metric in ("direction_cos","predictor_error","relative_curvature"):
            ci+=paired_ci(df[df.space.eq("FULL")&df.natural.eq(True)],metric,["case","sample","h"],["h"],baseline=baseline,treatment="geo")
    pd.DataFrame(ci).to_csv(args.output/"results/geometry_paired_ci.csv",index=False)
    print("Paired geometry complete",flush=True)


def numerical_check(args):
    cfg,vocab,meta,fit,ev=context(args)
    index=pd.read_csv(args.output/"controlled_index.csv")
    chosen=[]
    for background,g in index[index.natural.eq(True)].groupby("background"):
        sample=g["sample"].iloc[0]
        chosen.extend(g.index[g["sample"].eq(sample)&g.element.isin(["SiO2","MgO","CaO","Rb_ppm","Sr_ppm","Ni_ppm"])])
    chosen=np.array(sorted(chosen))
    args_cpu=type("Args",(),dict(geo_checkpoint=args.geo_checkpoint,device="cpu"))()
    model=model_for(args_cpu,"geo",cfg,vocab)
    coordinates=np.load(args.output/"controlled_geo.npy")[chosen]
    with torch.inference_mode():
        cpu=infer(model,coordinates,vocab,"cpu",32)["mean"]
    mps=np.load(args.output/"geo_controlled.npy")[chosen]
    local=index.loc[chosen].reset_index(drop=True);rows=[]
    for case,g in local.groupby("case",sort=False):
        lut={r.h:i for i,r in g.iterrows()}
        for h in (.005,.025,.1):
            a=(cpu[lut[h]].astype(float)-cpu[lut[-h]])/(2*h)
            b=(mps[lut[h]].astype(float)-mps[lut[-h]])/(2*h)
            rows.append(dict(case=case,element=g.element.iloc[0],h=h,
                             cpu_mps_tangent_cos=float(cosine(a,b)),
                             relative_derivative_error=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))))
    pd.DataFrame(rows).to_csv(args.output/"results/numerical_floor.csv",index=False)
    del model;gc.collect()
