"""Aggregate final training-probe errors against a tag-median numeric baseline."""

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from stage3_5_element2vec_v2.src.data.dataset import WholeRockTokenDataset
from stage3_5_element2vec_v2.src.training.train_v2 import build_fixed_probe_batches, move_batch_to_device
from .geo_mask15_audit import context, model_for


@torch.inference_mode()
def run(args):
    cfg,vocab,meta,fit,ev=context(args);cfg["data"]["sqlite_path"]=str(args.sqlite_path)
    dataset=WholeRockTokenDataset(cfg,vocab);batches=build_fixed_probe_batches(dataset,cfg)
    tags=vocab.numeric_vocab["numeric_tags"]
    columns={f["name"]:f["db_column"] for f in cfg["data"]["numeric_fields"]}
    medians={tag:float(dataset.df[col].median()) for tag,col in columns.items()}
    model=model_for(args,"geo",cfg,vocab)
    errors=[];labels=[]
    for raw in batches:
        batch=move_batch_to_device(raw,torch.device(args.device));out=model(batch)
        mask=raw["numeric_prediction_mask"].numpy()
        ids=raw["numeric_tag_ids"].numpy()[mask]
        target=raw["numeric_targets_all"].numpy()[mask]
        prediction=out["numeric_value_predictions"].cpu().numpy()[mask]
        for tag,y,p in zip(ids,target,prediction):
            name=tags[int(tag)]
            errors.append(dict(element=name,target=float(y),prediction=float(p),baseline=medians[name],
                               family="trace" if name.endswith("_ppm") else ("auxiliary" if name in ("LOI","CO2") else "major")))
        dec=out["decoder_outputs"]
        selected=raw["task_ids"].eq(1).numpy()
        for y,p in zip(raw["numeric_tag_targets"].numpy()[selected],dec["numeric_tag"].argmax(-1).cpu().numpy()[selected]):
            labels.append(dict(task="numeric_tag",correct=int(y==p)))
        for field in cfg["data"]["pure_label_fields"]:
            selected=np.array(raw["pure_fields"])==field
            for y,p in zip(raw["pure_class_targets"].numpy()[selected],dec["pure_labels"][field].argmax(-1).cpu().numpy()[selected]):
                labels.append(dict(task=field,correct=int(y==p)))
    frame=pd.DataFrame(errors)
    def huber(error):
        a=abs(error);return np.where(a<=1,.5*a*a,a-.5)
    frame["model_huber"]=huber(frame.prediction-frame.target)
    frame["median_baseline_huber"]=huber(frame.baseline-frame.target)
    rows=[]
    for group in ("element","family"):
        for key,g in frame.groupby(group):
            rows.append(dict(group=group,name=key,n=len(g),model_huber=g.model_huber.mean(),
                             tag_median_huber=g.median_baseline_huber.mean(),
                             improvement_fraction=1-g.model_huber.mean()/max(g.median_baseline_huber.mean(),1e-12),
                             rmse=np.sqrt(np.mean((g.prediction-g.target)**2))))
    pd.DataFrame(rows).to_csv(args.output/"results/final_numeric_by_family.csv",index=False)
    pd.DataFrame(labels).groupby("task").agg(n=("correct","size"),accuracy=("correct","mean")).reset_index().to_csv(
        args.output/"results/final_label_accuracy.csv",index=False)
    print("Final probe aggregate decoder checks complete",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=output_path("audit"))
    p.add_argument("--baseline",type=Path,default=asset("rope_z_run") / "audit")
    p.add_argument("--geo-checkpoint",type=Path,default=asset("geo15_checkpoint"))
    p.add_argument("--sqlite-path",type=Path,required=True)
    p.add_argument("--device",default="mps")
    with threadpool_limits(limits=8):run(_guard_args(p.parse_args()))
