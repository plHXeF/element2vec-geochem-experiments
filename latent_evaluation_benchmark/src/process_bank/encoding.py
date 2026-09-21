import json,time,gc
import numpy as np
import pandas as pd
from src.common import ROOT,NAMES,save_json,digest,sha
from src.extraction import baselines

def extract_bank(cfg,records,models=None,limit=None):
 from src.adapters.checkpoints import CheckpointRepresentation
 import torch
 torch.set_num_threads(8)
 bank=ROOT/'cache/process_bank';idx=json.loads((bank/'index.json').read_text());fs=json.loads((bank/'schema.json').read_text())['fields'];refs=baselines(cfg,records)
 if limit:idx=idx[:limit]
 for name in models or NAMES:
  out=ROOT/'cache/process_latents'/name;out.mkdir(parents=True,exist_ok=True);signatures_path=out/'input_signatures.json';signatures=json.loads(signatures_path.read_text()) if signatures_path.exists() else {};expected={r['episode_id']:digest([r.get('states_hash'),r['process_parameters']]) for r in idx};missing=[r for r in idx if not (out/(r['episode_id']+'.npz')).exists() or signatures.get(r['episode_id'])!=expected[r['episode_id']]]
  if not missing:continue
  rep=next((r for r in refs if r.name==name),None)
  if rep is None:rep=CheckpointRepresentation(next(s for s in cfg['experiments'] if s['name']==name),cfg['features'],cfg['device'],cfg['batch_size'])
  start=time.time()
  for a in range(0,len(missing),32):
   batch=missing[a:a+32];arrays=[];lengths=[]
   for row in batch:
    data=np.load(bank/row['state_file']);s=data['states'];o=data['order'];arrays.append(np.vstack([s,o]));lengths.append((len(s),len(o)))
   x=np.vstack(arrays);z=rep.encode(pd.DataFrame(x,columns=fs));pos=0
   for row,(n,m) in zip(batch,lengths):
    p=out/(row['episode_id']+'.npz');tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,z=z[pos:pos+n],order=z[pos+n:pos+n+m]);tmp.replace(p);pos+=n+m;signatures[row['episode_id']]=expected[row['episode_id']]
   save_json(signatures_path,signatures)
   print(f'Process {name} {min(a+32,len(missing))}/{len(missing)} {time.time()-start:.0f}s',flush=True)
  save_json(out/'manifest.json',{'bank_ids_hash':digest([r['episode_id'] for r in idx]),'schema_hash':sha(bank/'schema.json'),'episode_content_hash':digest([(r['episode_id'],r.get('states_hash'),r['process_parameters']) for r in idx]),'checkpoint_fingerprint':getattr(rep,'fingerprint',sha(ROOT/'cache/baselines.pkl')),'representation':name,'latent':'native; no metric PCA','episodes':len(idx)})
  if hasattr(rep,'close'):rep.close()
