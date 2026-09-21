"""Independent post-training geo15 audit. All caches are private and versioned."""

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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from stage3_5_element2vec_v2.src.data.dataset import WholeRockTokenDataset, load_vocab_bundle
from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame, coordinate_fingerprint
from stage3_5_element2vec_v2.src.evaluation.extract_embeddings import validate_checkpoint_compatibility
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from stage3_5_element2vec_v2.src.training.train_v2 import build_fixed_probe_batches, evaluate_probe
from stage3_5_element2vec_v2.src.utils.io import load_config
from .value_rope_audit import infer, sha, save_json
from .value_rope_metrics import projections, transformed, quality, neighbors, purity
from .value_rope_supplement import vote, classification_metrics

CONFIG = "stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml"
MODELS = ("v2", "rope", "geo", "random_v2", "random_rope", "random_geo")


def context(args):
    cfg = load_config(CONFIG)
    vocab = load_vocab_bundle(cfg)
    old = args.baseline
    out = args.output
    prior = json.loads((old/"protocol.json").read_text())
    frozen = json.loads((out/"protocol.json").read_text())
    assert sha(old/"metadata.csv") == prior["sample_hash"] == frozen["baseline_metadata_sha256"]
    assert sha(out/"paired_sample_ids.csv") == frozen["paired_ids_sha256"]
    assert sha(out/"paired_pools.npz") == frozen["paired_pools_sha256"]
    for name in ("v2","rope","random_v2","random_rope"):
        provenance=json.loads((old/f"{name}_extraction.json").read_text())
        key="rope" if "rope" in name else "v2"
        if "protocol_sha256" in provenance:
            assert provenance["protocol_sha256"]==sha(old/"protocol.json")
            assert provenance["checkpoint_sha256"]==prior["checkpoints"][key]
        else:
            replay=json.loads((out/"baseline_replay_verification.json").read_text())
            assert replay["protocol_sha256"]==sha(old/"protocol.json")
            assert replay["checkpoint_sha256"][key]==prior["checkpoints"][key]
            assert replay["models"][name]["passed"]
    meta = pd.read_csv(old/"metadata.csv", low_memory=False)
    ids = pd.read_csv(out/"paired_sample_ids.csv")
    assert np.array_equal(meta.sample_id.astype(str), ids.sample_id.astype(str))
    inputs = np.load(old/"global_inputs.npz")
    pools = np.load(out/"paired_pools.npz")
    assert np.array_equal(inputs["fit"], pools["fit"]) and np.array_equal(inputs["evaluation"], pools["evaluation"])
    (out/"results").mkdir(exist_ok=True)
    return cfg, vocab, meta, pools["fit"], pools["evaluation"]


def model_for(args, name, cfg, vocab):
    path = args.geo_checkpoint
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_compatibility(cfg, ck, vocab)
    assert ck["step"] == 13200
    torch.manual_seed(42)
    model = build_model(cfg, vocab)
    if name == "geo":
        model.load_state_dict(ck["model_state_dict"])
    model.to(args.device).eval()
    del ck
    return model


def global_extract(args):
    cfg, vocab, meta, fit, ev = context(args)
    root = args.geo_checkpoint.parent.parent
    train = pd.read_csv(root/"logs/train_log.csv")
    probe = pd.read_csv(root/"logs/probe_log.csv")
    assert np.array_equal(train.step, np.arange(1,13201))
    assert np.isfinite(train.filter(regex="loss").values).all()
    ck = torch.load(args.geo_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    validate_checkpoint_compatibility(cfg, ck, vocab)
    assert ck["step"] == 13200
    assert all(torch.isfinite(t).all().item() for t in ck["model_state_dict"].values())
    integrity = dict(steps=13200, final_epoch=float(train.epoch.iloc[-1]),
                     final_lr=float(train.learning_rate.iloc[-1]), weights_finite=True,
                     loss_finite=True, train_last500=train.filter(regex="loss$").tail(500).mean().to_dict(),
                     numeric_mask_fraction=float(train.numeric_masked_tokens.sum()/train.numeric_eligible_tokens.sum()),
                     checkpoint_sha256=sha(args.geo_checkpoint), coordinate_fingerprint=coordinate_fingerprint(cfg),
                     last_logged_probe_step=int(probe.step.iloc[-1]),
                     pretraining_validation="none; full-data training")
    del ck
    save_json(args.output/"results/training_integrity.json", integrity)
    values, qc = transform_numeric_frame(meta, cfg, vocab)
    cols = [f["db_column"] for f in cfg["data"]["numeric_fields"]]
    coordinates = values[cols].to_numpy(np.float32)
    np.save(args.output/"global_geo.npy", coordinates)
    for name in ("geo", "random_geo"):
        manifest = args.output/f"{name}_global_manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            assert saved["checkpoint_sha256"] == integrity["checkpoint_sha256"]
            assert saved["coordinate_fingerprint"] == integrity["coordinate_fingerprint"]
            assert saved["paired_ids_sha256"] == sha(args.output/"paired_sample_ids.csv")
            print(name+": global cache verified", flush=True)
            continue
        print("Extracting "+name+" paired global and masked-label latents", flush=True)
        model = model_for(args,name,cfg,vocab)
        result = infer(model, coordinates, vocab, args.device, args.batch_size)
        assert all(np.isfinite(v).all() for v in result.values())
        np.savez(args.output/f"{name}_global.npz", **result)
        masked = infer(model, coordinates, vocab, args.device, args.batch_size, labels=meta)
        np.savez(args.output/f"{name}_masked_full.npz", **masked)
        if name == "geo":
            cfg["data"]["sqlite_path"] = str(args.sqlite_path)
            dataset = WholeRockTokenDataset(cfg,vocab)
            batches = build_fixed_probe_batches(dataset,cfg)
            final = evaluate_probe(model,batches,torch.device(args.device),cfg["data"]["pure_label_fields"],cfg)
            save_json(args.output/"results/final_probe.json",dict(step=13200,**final))
            del dataset,batches
        save_json(manifest,dict(checkpoint_sha256=integrity["checkpoint_sha256"],
                               coordinate_fingerprint=integrity["coordinate_fingerprint"],
                               paired_ids_sha256=sha(args.output/"paired_sample_ids.csv"), coordinate_qc=qc))
        del model,result,masked
        gc.collect()
        if args.device=="mps":torch.mps.empty_cache()


def cache(args,name,suffix):
    return (args.output if name in ("geo","random_geo") else args.baseline)/f"{name}_{suffix}"


def cluster_ci(delta, refs):
    frame = pd.DataFrame({"delta":delta,"ref":np.asarray(refs).astype(str)}).dropna()
    grouped = frame.groupby("ref").delta.agg(["sum","count"])
    rng = np.random.default_rng(42)
    idx = rng.integers(0,len(grouped),(1000,len(grouped)))
    means = grouped["sum"].to_numpy()[idx].sum(1)/grouped["count"].to_numpy()[idx].sum(1)
    return dict(delta=float(frame.delta.mean()), ci_low=float(np.quantile(means,.025)),
                ci_high=float(np.quantile(means,.975)), reference_clusters=len(grouped))


def global_metrics(args):
    cfg,vocab,meta,fit,ev = context(args)
    rows=[];pcrows=[];details={};strata=[]
    panel=[f["db_column"] for f in cfg["data"]["numeric_fields"][:11]]
    raw=meta[panel]
    eligible=(raw.notna().all(axis=1)&raw.ge(0).all(axis=1)&raw.sum(axis=1).between(90,105)).to_numpy()
    for name in MODELS:
        print("Global quality "+name,flush=True)
        data=np.load(cache(args,name,"global.npz"))
        masked=np.load(cache(args,name,"masked_full.npz"))
        for mode,x in (("numeric",data["mean"]),("masked_labels",masked["mean"]),("cls_numeric",data["cls"])):
            pca,spaces=projections(x,fit)
            if mode!="numeric":spaces={k:v for k,v in spaces.items() if not k.startswith("RP")}
            pcrows.append(dict(model=name,mode=mode,pc1=pca.explained_variance_ratio_[0],
                               pc3=pca.explained_variance_ratio_[:3].sum(),pc32=pca.explained_variance_ratio_[:32].sum(),
                               pc64=pca.explained_variance_ratio_.sum()))
            if mode=="numeric":
                np.savez(args.output/f"{name}_pca_basis.npz",components=pca.components_,center=pca.mean_)
                np.save(args.output/f"{name}_pca2.npy",pca.transform(x[ev])[:,:2])
            for space,basis in spaces.items():
                projected=transformed(x[ev],basis,pca.mean_)
                for k in ((15,30) if not space.startswith("RP") else (15,)):
                    nn=neighbors(projected,k)
                    pred=vote(nn,meta.iloc[ev].rock_type)
                    q=classification_metrics(meta.iloc[ev].rock_type,pred)
                    for col in ("rock_type","canonical_ref_id","data_source"):
                        vector=purity(nn,meta.iloc[ev][col])
                        q[col+"_purity"]=float(np.nanmean(vector))
                        if mode=="numeric" and space in ("FULL","PCA64","PCA32"):
                            details[(name,space,k,col)]=vector
                    rows.append(dict(model=name,mode=mode,space=space,k=k,**q))
                    if mode=="numeric" and space=="FULL" and k==15:
                        for label,mask in (("closed",eligible[ev]),("fallback",~eligible[ev])):
                            strata.append(dict(model=name,stratum=label,n=int(mask.sum()),
                                               rock_purity=float(np.nanmean(purity(nn,meta.iloc[ev].rock_type)[mask])),
                                               ref_purity=float(np.nanmean(purity(nn,meta.iloc[ev].canonical_ref_id)[mask])),
                                               source_purity=float(np.nanmean(purity(nn,meta.iloc[ev].data_source)[mask]))))
            pd.DataFrame(rows).to_csv(args.output/"results/quality.csv",index=False)
        data.close();masked.close()
    ci=[]
    for baseline in ("v2","rope","random_geo","random_rope","random_v2"):
        for space in ("FULL","PCA64","PCA32"):
            for k in (15,30):
                for col in ("rock_type","canonical_ref_id","data_source"):
                    delta=details[("geo",space,k,col)]-details[(baseline,space,k,col)]
                    ci.append(dict(baseline=baseline,treatment="geo",space=space,k=k,metric=col,
                                   **cluster_ci(delta,meta.iloc[ev].canonical_ref_id)))
    pd.DataFrame(ci).to_csv(args.output/"results/quality_paired_ci.csv",index=False)
    pd.DataFrame(pcrows).to_csv(args.output/"results/pca.csv",index=False)
    pd.DataFrame(strata).to_csv(args.output/"results/closure_strata.csv",index=False)
    print("Global paired quality complete",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["global","quality","geometry_extract","geometry","numerical"])
    parser.add_argument("--output",type=Path,default=output_path("audit"))
    parser.add_argument("--baseline",type=Path,default=asset("rope_z_run") / "audit")
    parser.add_argument("--geo-checkpoint",type=Path,default=asset("geo15_checkpoint"))
    parser.add_argument("--sqlite-path",type=Path,default=asset("whole_rock"))
    parser.add_argument("--device",default="mps")
    parser.add_argument("--batch-size",type=int,default=256)
    args=_guard_args(parser.parse_args())
    _guard_args(args)
    from .geo_mask15_geometry import extract_geometry, geometry_metrics, numerical_check
    with threadpool_limits(limits=8):
        {"global":global_extract,"quality":global_metrics,
         "geometry_extract":extract_geometry,"geometry":geometry_metrics,"numerical":numerical_check}[args.command](args)


if __name__=="__main__":main()
