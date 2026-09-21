"""Verify older cache records by frozen checkpoint hashes and direct subset replay."""

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from stage3_5_element2vec_v2.src.utils.io import load_config
from stage3_5_element2vec_v2.src.data.dataset import load_vocab_bundle
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from .value_rope_audit import sha,infer,save_json


def verify(args):
    cfg=load_config("stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml")
    vocab=load_vocab_bundle(cfg);out=args.output;old=args.baseline
    prior=json.loads((old/"protocol.json").read_text())
    paths=dict(v2=args.v2_checkpoint,rope=args.rope_checkpoint)
    hashes={key:sha(path) for key,path in paths.items()}
    assert hashes==prior["checkpoints"]
    assert sha(old/"metadata.csv")==prior["sample_hash"]
    meta=pd.read_csv(old/"metadata.csv",low_memory=False)
    global_z=np.load(old/"global_inputs.npz")["z"]
    stimulus=np.load(old/"stimulus_z.npy");index=pd.read_csv(old/"stimulus_index.csv",low_memory=False)
    chosen=np.unique(np.r_[np.linspace(0,len(meta)-1,24).astype(int),0,len(meta)-1])
    stimuli=np.array(sorted(index.groupby("family",sort=False).head(4).index))
    tests={};file_hashes={}
    for name in ("v2","rope","random_v2","random_rope"):
        key="rope" if "rope" in name else "v2"
        checkpoint=torch.load(paths[key],map_location="cpu",weights_only=False,mmap=True)
        assert checkpoint["numeric_vocab"]==vocab.numeric_vocab and checkpoint["pure_vocab"]==vocab.pure_vocab
        torch.manual_seed(42);model=build_model(checkpoint["config"],vocab)
        if not name.startswith("random"):model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        model.to(args.device).eval();errors={}
        jobs=[("global.npz",global_z[chosen],chosen,None,None),
              ("masked_full.npz",global_z[chosen],chosen,meta.iloc[chosen],None),
              ("stimuli.npz",stimulus[stimuli],stimuli,None,index.target.fillna(-1).to_numpy(int)[stimuli]),
              ("controlled.npy",np.load(old/"controlled_z.npy")[:21],np.arange(21),None,None),
              ("ratio.npz",np.load(old/"ratio_z.npy")[:12],np.arange(12),None,None)]
        for suffix,values,positions,labels,targets in jobs:
            actual=infer(model,values,vocab,args.device,32,labels=labels,targets=targets)
            path=old/f"{name}_{suffix}"
            cached=np.load(path)
            keys=["mean"] if suffix.endswith(".npy") else ["mean","cls"]
            for representation in keys:
                expected=cached[positions] if suffix.endswith(".npy") else cached[representation][positions]
                delta=float(np.linalg.norm(actual[representation]-expected)/max(np.linalg.norm(expected),1e-12))
                assert delta<2e-5,(name,suffix,representation,delta)
                errors[suffix+":"+representation]=delta
            if hasattr(cached,"close"):cached.close()
            file_hashes[name+"_"+suffix]=sha(path)
        tests[name]=dict(passed=True,relative_errors=errors,global_rows_checked=len(chosen),stimulus_rows_checked=len(stimuli))
        print(name+": checkpoint and cache subset replay passed",flush=True)
        del model;gc.collect()
        if args.device=="mps":torch.mps.empty_cache()
    save_json(out/"baseline_replay_verification.json",dict(protocol_sha256=sha(old/"protocol.json"),
              checkpoint_sha256=hashes,models=tests,cache_sha256=file_hashes))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=output_path("audit"))
    p.add_argument("--baseline",type=Path,default=asset("rope_z_run") / "audit")
    p.add_argument("--v2-checkpoint",type=Path,required=True)
    p.add_argument("--rope-checkpoint",type=Path,default=asset("rope_z_checkpoint"))
    p.add_argument("--device",default="mps")
    with threadpool_limits(limits=8):verify(_guard_args(p.parse_args()))
