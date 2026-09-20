"""Publish aggregate-only artifacts for the independent geo15 experiment."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geo_mask15_audit import MODELS
from .value_rope_audit import save_json

COLORS=dict(v2="#3973ac",rope="#d48227",geo="#bc3e4a",
            random_v2="#7f9bbb",random_rope="#c3a67e",random_geo="#b9929a")


def publish(args):
    out=args.output;result=out/"results";dest=args.report
    dest.mkdir(parents=True,exist_ok=True);figs=dest/"figures";figs.mkdir(exist_ok=True)
    quality=pd.read_csv(result/"quality.csv");paths=pd.read_csv(result/"paths.csv")
    derivative=pd.read_csv(result/"derivatives.csv");fine=pd.read_csv(result/"fine_steps.csv")
    joint=pd.read_csv(result/"joints.csv")
    for table in ("quality","quality_paired_ci","pca","closure_strata","dropout","context","geometry_paired_ci"):
        shutil.copyfile(result/f"{table}.csv",dest/f"{table}.csv")
    for table in ("support_collisions_summary","phase_summary","jacobian_summary","stimulus_coverage",
                  "full_coordinate_phase_support","raw_process_summary","final_numeric_by_family","final_label_accuracy"):
        if (result/f"{table}.csv").exists():
            shutil.copyfile(result/f"{table}.csv",dest/f"{table}.csv")
    def aggregate(data,groups,metrics,name):
        grouped=data.groupby(groups,dropna=False)[metrics]
        frame=grouped.median().reset_index()
        frame["records"]=grouped.size().to_numpy()
        frame.to_csv(dest/f"{name}.csv",index=False)
        return frame
    aggregate(derivative[derivative.natural],["model","space","h"],
              ["direction_cos","predictor_error","relative_curvature","speed"],"derivatives_summary")
    aggregate(fine[fine.natural],["model","h"],["direction_cos","predictor_error","relative_curvature"],"fine_steps_summary")
    aggregate(joint[joint.natural],["model","space","h"],["residual","direction_cos"],"joints_summary")
    aggregate(paths,["model","space","family","scope","process"],
              ["speed_cv","tangent_cos","turn_rate","uniformity_rmse","straightness"],"paths_summary")
    for name,groups,metrics in [
        ("backgrounds",["model","natural","relation"],["direction_cos"]),
        ("linearity",["model","space","natural"],["linear_radius"]),
        ("ratio",["model","kind","h"],["normalized_sensitivity"]),
        ("numerical_floor",["element","h"],["cpu_mps_tangent_cos","relative_derivative_error"])]:
        aggregate(pd.read_csv(result/f"{name}.csv"),groups,metrics,name+"_summary")
    collisions=pd.read_csv(result/"collisions.csv")
    collisions.groupby(["model","element"]).agg(pairs=("pairs","sum"),collisions=("collisions","sum"),
                                               minimum_far_over_local=("min_far_over_local","min")).reset_index().to_csv(dest/"collisions_summary.csv",index=False)
    numerical=pd.read_csv(result/"numerical_floor.csv")
    save_json(dest/"analysis_verification.json",dict(
        models=list(MODELS),training_steps=13200,paired_samples=12000,fit_samples=8000,evaluation_samples=4000,
        chemistry_only=True,quality_random_projection_seeds=list(range(1,11)),
        main_geometry_spaces=["FULL","PCA64","PCA32","PCA3","PCA2"],
        cpu_mps_fine_tangent_min=float(numerical.cpu_mps_tangent_cos.min()),
        private_sample_artifacts_published=False,independent_validation=False))
    integrity=json.loads((result/"training_integrity.json").read_text())
    integrity.pop("checkpoint_sha256");integrity.pop("coordinate_fingerprint")
    save_json(dest/"training_summary.json",integrity)
    shutil.copyfile(result/"final_probe.json",dest/"final_probe.json")
    if args.tables_only:
        print("Aggregate tables updated",flush=True)
        return
    # No sample IDs, latent arrays, neighbor tables or projection bases are exported.
    q=quality[quality["mode"].eq("numeric")&quality.k.eq(15)]
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    levels=["FULL","PCA64","PCA32","PCA3","PCA2"]
    for ax,metric in zip(axs,["rock_type_purity","canonical_ref_id_purity","data_source_purity"]):
        for name in MODELS:
            g=q[q.model.eq(name)].set_index("space")
            ax.plot(levels,g.loc[levels,metric],"o--" if name.startswith("random") else "o-",
                    label=name,color=COLORS[name],alpha=.65 if name.startswith("random") else 1)
        ax.set_title(metric.replace("_"," "));ax.set_ylim(0,1);ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle("Paired chemistry-only neighborhoods (k=15; 4,000 evaluation rows)")
    fig.tight_layout();fig.savefig(figs/"global_quality.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for name in MODELS:
        g=fine[fine.natural&fine.model.eq(name)].groupby("h")
        v=g.predictor_error.median();axs[0,0].plot(v.index,v,"o-",label=name,color=COLORS[name])
        g=joint[joint.natural&joint.model.eq(name)&joint.space.eq("FULL")].groupby("h").residual.median()
        axs[0,1].plot(g.index,g,"o-",color=COLORS[name])
        g=paths[paths.model.eq(name)&paths.scope.eq("local")&paths.space.eq("FULL")]
        axs[1,0].bar(name,g.speed_cv.median(),color=COLORS[name])
        g=paths[paths.model.eq(name)&paths.family.eq("process")&paths.space.eq("FULL")].groupby("process").uniformity_rmse.median()
        axs[1,1].plot(g.index,g,"o-",color=COLORS[name])
    axs[0,0].set(title="Fine-reference tangent prediction error",xlabel="Shared physical input parameter h",ylabel="Median relative error")
    axs[0,1].set(title="Multi-element additivity error",xlabel="Shared input parameter h",ylabel="Median residual",xscale="log")
    axs[1,0].set(title="Local sweep speed CV",ylabel="Median CV")
    axs[1,1].set(title="Process parameter vs normalized path length",ylabel="Median uniformity RMSE")
    axs[0,0].legend(fontsize=8)
    for ax in axs[1]:ax.tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(figs/"local_geometry.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    levels=["input","q1","layer1","layer3","layer6","layer9","FULL"]
    for name in ("v2","rope","geo"):
        g=paths[paths.model.eq(name)&paths.scope.eq("local")]
        for ax,metric in zip(axs,["speed_cv","tangent_cos","turn_rate"]):
            values=[g[g.space.eq(level)][metric].median() for level in levels]
            ax.plot(levels,values,"o-",label=name,color=COLORS[name]);ax.set_title(metric);ax.tick_params(axis="x",rotation=30)
    axs[0].legend();fig.suptitle("Layer progression; undefined zero motion left blank")
    fig.tight_layout();fig.savefig(figs/"layer_geometry.png",dpi=170);plt.close(fig)
    meta=pd.read_csv(args.baseline/"metadata.csv",low_memory=False)
    ev=np.load(out/"paired_pools.npz")["evaluation"]
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axs,("v2","rope","geo")):
        xy=np.load(out/f"{name}_pca2.npy")
        for bg,c in [("mafic","#3589bd"),("intermediate","#5c9c68"),("felsic","#ce8751")]:
            mask=meta.iloc[ev].background.eq(bg)
            ax.scatter(xy[mask,0],xy[mask,1],s=3,alpha=.3,color=c,label=bg,rasterized=True)
        ax.set(title=name,xlabel="PC1",ylabel="PC2")
    axs[0].legend(markerscale=3);fig.suptitle("Independent PCA fits on the same fitting pool; axes are not aligned")
    fig.tight_layout();fig.savefig(figs/"global_pca.png",dpi=170);plt.close(fig)
    root=args.output.parent
    t=pd.read_csv(root/"logs/train_log.csv");p=pd.read_csv(root/"logs/probe_log.csv")
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axs.flat,["total_loss","numeric_value_loss","numeric_tag_loss","rock_group_loss","rock_origin_loss","rock_type_loss"]):
        ax.plot(t.step,t[key].rolling(500).mean(),label="Train rolling 500")
        ax.plot(p.step,p["probe_"+key],label="Fixed train probe")
        ax.set(title=key,xlabel="Optimizer step",yscale="log")
    axs.flat[0].legend(fontsize=8);fig.suptitle("RoPE_geo + 15% mask: full-data training, no held-out validation")
    fig.tight_layout();fig.savefig(figs/"training_curves.png",dpi=170);plt.close(fig)
    print("Aggregate tables and figures published to "+str(dest),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/audit"))
    p.add_argument("--baseline",type=Path,default=Path("stage3_5_element2vec_v2/outputs/value_rope_z/audit"))
    p.add_argument("--report",type=Path,default=Path("stage3_5_element2vec_v2/experiments/rope_geo_mask15"))
    p.add_argument("--tables-only",action="store_true")
    publish(p.parse_args())


if __name__=="__main__":main()
