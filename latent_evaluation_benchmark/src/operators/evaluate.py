import json,pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.common import ROOT,NAMES,table,save_json
from src.operators.metrics import trajectory_metrics,boundary_detection
from src.operators.linear import fit_operators,rollout,rollout_batch

def evaluate(cfg):
 bank=ROOT/'cache/process_bank';index=json.loads((bank/'index.json').read_text());geometry=[];composites=[];orders=[]
 for name in NAMES:
  scale=json.loads((ROOT/'cache/real'/name/'scale.json').read_text())['global_rms_radius'];print('Trajectory metrics',name,flush=True)
  cache_path=ROOT/'cache'/f'trajectory_metrics_{name}.parquet'
  cached=pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame();done=set(cached.episode_id) if len(cached) else set();local=cached.to_dict('records')
  for row in index:
   if row['episode_id'] in done:continue
   z=np.load(ROOT/'cache/process_latents'/name/(row['episode_id']+'.npz'))['z'].astype(float)/scale
   meta={k:row.get(k) for k in ['episode_id','kind','source_type','split','initial_state_id','composite_type','ood_fraction']};local.append({'representation':name,**meta,**trajectory_metrics(z,cfg)})
  pd.DataFrame(local).to_parquet(cache_path,index=False);geometry.extend(local)
  table(pd.DataFrame(geometry),ROOT/'outputs/process/trajectory_metrics')
 operators=fit_operators(cfg,index)
 for name in NAMES:
  scale=json.loads((ROOT/'cache/real'/name/'scale.json').read_text())['global_rms_radius']
  # Tune one detection threshold on validation composites only; freeze for test.
  validation=[r for r in index if r['kind']=='Composite' and r['split']=='val'];candidates=[.01,.03,.1,.2,.4,.8,1.2];values=[]
  for threshold in candidates:
   vals=[]
   for r in validation:
    z=np.load(ROOT/'cache/process_latents'/name/(r['episode_id']+'.npz'))['z'];vals.append(boundary_detection(z,r['stage_boundaries'],cfg['process']['boundary_tolerance'],threshold)['boundary_f1'])
   values.append(np.mean(vals))
  threshold_path=ROOT/'outputs/composite'/f'{name}_boundary_threshold.json'
  if threshold_path.exists():threshold=json.loads(threshold_path.read_text())['threshold']
  else:
   threshold=candidates[int(np.argmax(values))];save_json(threshold_path,{'threshold':threshold,'candidates':candidates,'validation_F1':values,'fit_split':'val','frozen_after_initial_bank':True})
  models={}
  composite_rows=[r for r in index if r['kind']=='Composite'];predictions={}
  for start in range(0,len(composite_rows),256):
   batch=composite_rows[start:start+256];pred=np.stack([np.load(ROOT/'cache/process_latents'/name/(r['episode_id']+'.npz'))['z'][0] for r in batch]).astype(float);controls=[np.load(bank/r['state_file'])['theta'] for r in batch]
   for stage in range(3):
    for process in ['Mixing','Fractionation','Alteration']:
     positions=[i for i,r in enumerate(batch) if len(r['process_sequence'])>stage and r['process_sequence'][stage]==process]
     if not positions:continue
     if process not in models:models[process]=pickle.loads((ROOT/'cache/operators'/f'{name}_{process}.pkl').read_bytes())
     grids=[batch[i]['stage_grids'][stage] for i in positions];theta_batch=[controls[i][stage*(cfg['process']['states']-1):(stage+1)*(cfg['process']['states']-1)] for i in positions]
     pred[positions]=rollout_batch(models[process],pred[positions],theta_batch,grids)
   predictions.update({r['episode_id']:pred[i] for i,r in enumerate(batch)})
  for r in index:
   if r['kind']!='Composite':continue
   lat=np.load(ROOT/'cache/process_latents'/name/(r['episode_id']+'.npz'));z=lat['z'];raw=np.load(bank/r['state_file']);theta=raw['theta'];pred=predictions[r['episode_id']]
   composition_error=float(np.linalg.norm(pred-z[-1])/scale) if np.isfinite(pred).all() else 1e15
   result={'representation':name,'episode_id':r['episode_id'],'source_type':r['source_type'],'split':r['split'],'initial_state_id':r['initial_state_id'],'composite_type':r['composite_type'],'composition_error':composition_error,'rollout_diverged':not np.isfinite(pred).all(),**boundary_detection(z,r['stage_boundaries'],cfg['process']['boundary_tolerance'],threshold)};composites.append(result)
   if len(lat['order'])==3:
    raworder=raw['order'];valid=np.isfinite(raworder).all(0);rawdiff=np.linalg.norm(raworder[1,valid]-raworder[2,valid]);ldiff=np.linalg.norm(lat['order'][1]-lat['order'][2])/scale;orders.append({'representation':name,'episode_id':r['episode_id'],'source_type':r['source_type'],'split':r['split'],'raw_order_effect':float(rawdiff),'latent_order_effect':float(ldiff),'process_pair':' -> '.join(r['process_sequence'][:2])})
  print('Composite',name,flush=True)
 table(pd.DataFrame(composites),ROOT/'outputs/composite/metrics');table(pd.DataFrame(orders),ROOT/'outputs/composite/order_effects')
 correlations=[]
 for (name,source),g in pd.DataFrame(orders).query("split == 'test'").groupby(['representation','source_type']):
  correlations.append({'representation':name,'source_type':source,'n':len(g),'Spearman':float(spearmanr(g.raw_order_effect,g.latent_order_effect).statistic) if len(g)>3 else np.nan})
 table(pd.DataFrame(correlations),ROOT/'outputs/composite/noncommutativity')
 # Exact trace composition law at same initial state and independent stage parameters.
 fs=json.loads((bank/'schema.json').read_text())['fields'];js=np.array([i for i,f in enumerate(fs) if f.endswith('_ppm')]);law=[]
 for r in index:
  if r['composite_type']!='same-process' or r['process_sequence'][0] not in ['Fractionation','Alteration']:continue
  x=np.array(r['initial_state'],dtype=float);final=np.load(bank/r['state_file'])['states'][-1];kind=r['process_sequence'][0];expected=x.copy()
  if kind=='Alteration':
   exponent=sum(np.array(p['a'])*(g[-1]-g[0]) for p,g in zip(r['process_parameters'],r['stage_grids']));expected=x*np.exp(exponent)
  else:
   exponent=sum((np.array(p['D'])-1)*np.log(g[-1]/g[0]) for p,g in zip(r['process_parameters'],r['stage_grids']));expected[js]=x[js]*np.exp(exponent[js])
  valid=np.isfinite(expected[js])&np.isfinite(final[js]);error=np.linalg.norm(expected[js][valid]-final[js][valid])/max(np.linalg.norm(final[js][valid]),1e-12) if valid.any() else np.nan;law.append({'episode_id':r['episode_id'],'process':kind,'source_type':r['source_type'],'split':r['split'],'trace_semigroup_error':float(error),'observed_trace_fields':int(valid.sum()),'status':'ok' if valid.any() else 'unavailable_no_observed_trace','scope':'physical trace law; learned full-latent rollout is reported separately'})
 table(pd.DataFrame(law),ROOT/'outputs/composite/semigroup')
