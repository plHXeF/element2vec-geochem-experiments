"""Support-aware phase checks and Jacobian spectra for the independent audit."""

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .geo_mask15_audit import context, cache, MODELS
from .value_rope_audit import save_json


def raw_processes(args):
    """Recover exact physical endpoints where the old z cache was clipped."""
    import gc
    import torch
    from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame
    from stage3_5_element2vec_v2.src.utils.io import load_config
    from .geo_mask15_audit import model_for
    from .value_rope_audit import infer, sha
    from .value_rope_metrics import path_metrics
    cfg,vocab,meta,fit,ev=context(args)
    old_cfg=load_config("stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml")
    index=pd.read_csv(args.baseline/"stimulus_index.csv",low_memory=False)
    old_z=np.load(args.baseline/"stimulus_z.npy")
    cols=[f["db_column"] for f in cfg["data"]["numeric_fields"]]
    raw=[];rows=[];hashes={}
    for process,file,grouping,parameter in [
        ("mixing","E1_two_endmember_mixing/synthetic_mixing_compositions.csv",["mix_pair","closure"],"alpha"),
        ("melting","E4_partial_melting/partial_melting_compositions.csv",["source","residual"],"melt_fraction")]:
        path=args.legacy/"results"/file
        frame=pd.read_csv(path);hashes[file]=sha(path)
        for _,row in frame.iterrows():
            case=process+":"+":".join(str(row[k]) for k in grouping)
            matches=index[index.case.eq(case)&np.isclose(index.t,float(row[parameter]))]
            assert len(matches)==1
            rows.append(dict(case=case,process=process,t=float(row[parameter]),old_row=int(matches.index[0])))
            raw.append(row[cols])
    frame=pd.DataFrame(raw).reset_index(drop=True);records=pd.DataFrame(rows)
    baseline,_=transform_numeric_frame(frame,old_cfg,vocab)
    np.testing.assert_allclose(baseline[cols],old_z[records.old_row],atol=2e-5,rtol=1e-5,equal_nan=True)
    geo,_=transform_numeric_frame(frame,cfg,vocab)
    panel=frame[cols[:11]]
    records["closed"]=panel.notna().all(axis=1)&panel.ge(0).all(axis=1)&panel.sum(axis=1).between(90,105)
    records.to_csv(args.output/"raw_process_index.csv",index=False)
    metrics=[]
    for name in MODELS:
        if name in ("geo","random_geo"):
            model=model_for(args,name,cfg,vocab)
            result=infer(model,geo[cols].to_numpy(np.float32),vocab,args.device,256)
            x=result["mean"]
            np.savez(args.output/f"{name}_raw_process.npz",**result)
            del model,result;gc.collect()
            if args.device=="mps":torch.mps.empty_cache()
        else:x=np.load(cache(args,name,"stimuli.npz"))["mean"][records.old_row]
        for case,g in records.groupby("case",sort=False):
            g=g.sort_values("t")
            metrics.append(dict(model=name,case=case,process=g.process.iloc[0],
                                closure_crossing=g.closed.nunique()>1,**path_metrics(x[g.index],g.t)))
    pd.DataFrame(metrics).to_csv(args.output/"results/raw_process_paths.csv",index=False)
    pd.DataFrame(metrics).groupby(["model","process","closure_crossing"]).median(numeric_only=True).reset_index().to_csv(
        args.output/"results/raw_process_summary.csv",index=False)
    save_json(args.output/"raw_process_manifest.json",dict(source_sha256=hashes,rows=len(records),
              roundtrip_matches_original_z=True,note="Exact raw endpoints, including originally clipped carbonate mixtures."))
    print("Exact raw mixing and melting replay complete",flush=True)


def run(args):
    cfg,vocab,meta,fit,ev=context(args)
    ix=pd.read_csv(args.output/"stimulus_index.csv",low_memory=False)
    support=pd.read_csv(args.baseline/"results/full_data_numeric_summary.csv").set_index("element")
    coords=np.load(args.output/"stimulus_geo.npy")
    frequency=10000.**(-2*np.arange(32)/64)
    collisions=[];phase=[];spectra=[];coverage=[]
    for name in MODELS:
        x=np.load(cache(args,name,"stimuli.npz"))["mean"]
        for case,g in ix[ix.family.eq("sweep")&ix.scope.eq("wide")&ix.physical_valid&ix.baseline_roundtrip].groupby("case",sort=False):
            s=support.loc[g.element.iloc[0]]
            for mode,lo,hi in [("full_observed",s.z_min,s.z_max),("p01_p99",s.z_p01,s.z_p99)]:
                gg=g[g.t.between(lo,hi)].sort_values("t")
                if len(gg)<3:continue
                v=x[gg.index].astype(float);t=gg.t.to_numpy()
                dist=np.linalg.norm(v[:,None]-v[None,:],axis=-1)
                local=np.median(np.linalg.norm(np.diff(v,axis=0),axis=1))
                allowed=(abs(t[:,None]-t[None,:])>=4)&np.triu(np.ones((len(t),len(t)),bool))
                collisions.append(dict(model=name,element=gg.element.iloc[0],support=mode,pairs=int(allowed.sum()),
                                        collisions=int((allowed&(dist<local)).sum()),
                                        min_far_over_local=float(dist[allowed].min()/local) if allowed.any() and local>1e-8 else np.nan))
                if name=="geo" and allowed.any():
                    position=coords[gg.index,int(gg.target.iloc[0])]
                    gap=abs(position[:,None]-position[None,:])
                    kernel=np.cos(gap[...,None]*frequency).mean(-1)
                    phase.append(dict(element=gg.element.iloc[0],support=mode,
                                      max_far_phase_similarity=float(kernel[allowed].max()),
                                      largest_angle_gap=float(gap[allowed].max()),
                                      smallest_angle_gap=float(gap[allowed].min())))
        for sample,g in ix[ix.family.eq("derivative")].groupby("sample",sort=False):
            jac=[]
            for element,gg in g.groupby("element",sort=False):
                rows=gg[gg.h.isin([-.1,.1,0])]
                if len(rows)!=3 or not rows.admissible.all() or rows.closed.nunique()!=1:continue
                plus=gg.index[gg.h.eq(.1)][0];minus=gg.index[gg.h.eq(-.1)][0]
                jac.append((x[plus].astype(float)-x[minus])/.2)
            if len(jac)<2:continue
            singular=np.linalg.svd(np.stack(jac),compute_uv=False);energy=singular**2;energy/=energy.sum()
            spectra.append(dict(model=name,background=g.background.iloc[0],natural=g.natural.iloc[0],elements=len(jac),
                                 participation_rank=float(1/(energy**2).sum()),rank90=int(np.searchsorted(np.cumsum(energy),.9)+1),
                                 top1_energy=float(energy[0])))
    pd.DataFrame(collisions).groupby(["model","element","support"]).agg(
        pairs=("pairs","sum"),collisions=("collisions","sum"),minimum_far_over_local=("min_far_over_local","min")
    ).reset_index().to_csv(args.output/"results/support_collisions_summary.csv",index=False)
    pd.DataFrame(phase).groupby(["element","support"]).agg(
        max_far_phase_similarity=("max_far_phase_similarity","max"),
        largest_angle_gap=("largest_angle_gap","max"),smallest_angle_gap=("smallest_angle_gap","min")
    ).reset_index().to_csv(args.output/"results/phase_summary.csv",index=False)
    pd.DataFrame(spectra).groupby(["model","background","natural"]).median().reset_index().to_csv(
        args.output/"results/jacobian_summary.csv",index=False)
    for family,g in ix.groupby("family"):
        coverage.append(dict(family=family,rows=len(g),admissible_rows=int(g.admissible.sum()),
                             cases=g.case.nunique(),complete_admissible_cases=sum(gg.admissible.all() for _,gg in g.groupby("case")),
                             closure_crossing_cases=sum(gg.closed.nunique()>1 for _,gg in g.groupby("case"))))
    pd.DataFrame(coverage).to_csv(args.output/"results/stimulus_coverage.csv",index=False)
    manifest=json.loads((args.output.parent/"logs/run_manifest.json").read_text())
    ranges=manifest["coordinate_qc"]["ranges"]
    phase_support=[]
    for tag,value in ranges.items():
        if value["min"] is None:continue
        span=value["max"]-value["min"]
        gaps=np.linspace(0,span,10001)
        similarity=np.cos(gaps[:,None]*frequency).mean(1)
        distant=gaps>=4
        phase_support.append(dict(element=tag,minimum=value["min"],maximum=value["max"],span=span,
                                  maximum_kernel_gap_ge4=float(similarity[distant].max()) if distant.any() else None))
    pd.DataFrame(phase_support).to_csv(args.output/"results/full_coordinate_phase_support.csv",index=False)
    save_json(args.output/"results/phase_interpretation.json",dict(
        kernel_at_2pi=float(np.cos(2*np.pi*frequency).mean()),
        distant_physical_definition="shared baseline-z parameter gap >=4; same physical endpoints per model",
        interpretation="Large phase similarity at a small geo angle gap is coordinate compression, not proof of wrapping. No finite sweep proves global injectivity.",
        closure_rule="Threshold and completeness switches are excluded from within-regime smoothness claims."))
    print("Support-aware phase and Jacobian checks complete",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=output_path("audit"))
    p.add_argument("--baseline",type=Path,default=asset("rope_z_run") / "audit")
    p.add_argument("--legacy",type=Path)
    p.add_argument("--geo-checkpoint",type=Path,default=asset("geo15_checkpoint"))
    p.add_argument("--device",default="mps")
    args=_guard_args(p.parse_args())
    _guard_args(args)
    with threadpool_limits(limits=8):
        if args.legacy:raw_processes(args)
        else:run(args)
