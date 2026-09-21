import json,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT,NAMES,save_json,sha,array_digest,digest
from src.data import task_inputs
from src.representations.baselines import ZVectorRepresentation,GeoNormRepresentation,PCAGeoRepresentation

def baselines(cfg,records):
 marker=ROOT/'cache/baselines.pkl'
 if marker.exists():return pickle.loads(marker.read_bytes())
 train=records[records.split.eq('train')];reps=[c(cfg).fit_reference(train) for c in [ZVectorRepresentation,GeoNormRepresentation,PCAGeoRepresentation]]
 marker.write_bytes(pickle.dumps(reps));stats={}
 for r in reps:
  attrs={k:v for k,v in vars(r).items() if k in ['mean','std','center','scale','n']}
  if r.name=='PCAGeo':attrs.update(components=r.pca.components_[:r.n],mean=r.pca.mean_,variance=r.pca.explained_variance_ratio_[:r.n])
  stats[r.name]=attrs
 save_json(ROOT/'configs/fitted_baseline_statistics.json',stats)
 return reps

def extract_real(cfg,records,models=None):
 from src.adapters.checkpoints import CheckpointRepresentation
 import torch
 torch.set_num_threads(8)
 selected=models or NAMES;refs=baselines(cfg,records);jobs=list(task_inputs(records,cfg));train=np.flatnonzero(records.split.eq('train'))
 for name in selected:
  directory=ROOT/'cache/real'/name;directory.mkdir(parents=True,exist_ok=True)
  if all((directory/(t+'.npy')).exists() for t,x in jobs):continue
  rep=next((r for r in refs if r.name==name),None)
  if rep is None:rep=CheckpointRepresentation(next(s for s in cfg['experiments'] if s['name']==name),cfg['features'],cfg['device'],cfg['batch_size'])
  proof={}
  for task,x in jobs:
   p=directory/(task+'.npy')
   if p.exists():continue
   print(f'Extract {name} {task} n={len(x)}',flush=True);start=time.time()
   # Exact record hashes + model hash make cache reuse auditable.
   result=np.empty((len(x),rep.encode(x.iloc[:1]).shape[1]),np.float32)
   for a in range(0,len(x),2048):
    result[a:a+2048]=rep.encode(x.iloc[a:a+2048]);print(f'  {name}/{task} {min(a+2048,len(x))}/{len(x)} {time.time()-start:.1f}s',flush=True)
   if task=='unmasked':
    twice=rep.encode(x.iloc[:8]);err=float(np.max(abs(twice-result[:8])));assert np.allclose(twice,result[:8],rtol=2e-5,atol=2e-5)
    scale=float(np.sqrt(np.mean(np.sum((result[train]-result[train].mean(0))**2,axis=1))))
    save_json(directory/'scale.json',{'global_rms_radius':scale,'repeated_encode_max_abs_error':err,'fit_ids_hash':digest(records.iloc[train].sample_id.tolist())})
   np.save(p,result);save_json(directory/(task+'.json'),{'input_hash':array_digest(x.to_numpy()),'output_sha256':sha(p),'fingerprint':getattr(rep,'fingerprint',sha(ROOT/'cache/baselines.pkl')),'seconds':time.time()-start})
  if hasattr(rep,'close'):rep.close()
