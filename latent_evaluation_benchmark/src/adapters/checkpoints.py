"""Read-only adapters; import the audited backward-compatible model implementation."""
import copy,json,sys,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common import sha,digest,save_json,ROOT,array_digest
class CheckpointRepresentation:
 def __new__(cls,spec,features,device='cpu',batch_size=256):
  if spec.get('format')=='e010':
   sys.path.insert(0,spec['implementation_root'])
   from stage3_5_element2vec_v2.src.e010.adapter import E010CheckpointRepresentation
   return E010CheckpointRepresentation(spec,features,device,batch_size)
  return super().__new__(cls)
 def __init__(self,spec,features,device='cpu',batch_size=256):
  self.spec=spec;self.name=spec['name'];self.features=features;self.device=device;self.batch_size=batch_size
  sys.path.insert(0,spec['implementation_root'])
  from stage3_5_element2vec_v2.src.data.dataset import VocabBundle
  from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
  from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame
  self.transform=transform_numeric_frame
  ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
  self.config=copy.deepcopy(ck['config']);vd=Path(spec['vocab'])
  self.vocab=VocabBundle(json.loads((vd/'numeric_tag_vocab.json').read_text()),json.loads((vd/'pure_label_vocab.json').read_text()),pd.read_csv(vd/'value_transform_stats.csv'))
  assert ck['numeric_vocab']==self.vocab.numeric_vocab and ck['pure_vocab']==self.vocab.pure_vocab
  if 'value_transform_stats' in ck:assert ck['value_transform_stats']==self.vocab.transform_stats.to_dict('records')
  assert [f['db_column'] for f in self.config['data']['numeric_fields']]==features
  self.model=build_model(self.config,self.vocab);self.model.load_state_dict(ck['model_state_dict'],strict=True)
  self.model.to(device).eval();self.model.requires_grad_(False);self.step=ck['step'];del ck
  self.fingerprint=digest({'checkpoint':sha(spec['checkpoint']),'config':self.config,'stats':sha(vd/'value_transform_stats.csv'),'pooling':'numeric_mean','mode':'chemistry_only'})
 def fit_reference(self,train_records):return self
 @torch.inference_mode()
 def encode(self,records):
  frame=records.reindex(columns=self.features)
  coords,_=self.transform(frame,self.config,self.vocab)
  z=coords[self.features].to_numpy(np.float32);out=[]
  ids=torch.arange(len(self.features),device=self.device)
  for start in range(0,len(z),self.batch_size):
   a=z[start:start+self.batch_size];b,n=a.shape
   values=torch.tensor(np.nan_to_num(a,nan=0.),device=self.device);valid=torch.tensor(np.isfinite(a),device=self.device)
   batch={'kind':torch.zeros(b,n,dtype=torch.long,device=self.device),'numeric_tag_ids':ids.expand(b,-1),'pure_label_ids':torch.zeros(b,n,dtype=torch.long,device=self.device),'values':values,'mask_tag_flags':torch.zeros_like(values),'mask_value_flags':torch.zeros_like(values),'attention_mask':valid,'numeric_token_mask':valid}
   h=self.model.encode(batch)['encoded'][:,1:]
   pooled=(h*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
   out.append(pooled.cpu().numpy())
  result=np.concatenate(out);assert np.isfinite(result).all();return result
 def close(self):
  del self.model;gc.collect()
  if self.device=='mps':torch.mps.empty_cache()
class Experiment1Representation(CheckpointRepresentation):pass
class Experiment2Representation(CheckpointRepresentation):pass
class Experiment3Representation(CheckpointRepresentation):pass
