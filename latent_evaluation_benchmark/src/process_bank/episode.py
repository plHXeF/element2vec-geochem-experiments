from src.common import config_digest
"""One persisted physical EpisodeBank shared by every representation."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT,save_json,digest,split_for,table,array_digest
from src.process_bank.rock_generator import RockGenerator
from src.process_bank.trace_generator import TracePrior
from src.process_bank.operators import mixing,fractionation,partial_melting,alteration
from src.process_bank.composite import concatenate,trace_semigroup
PROCESSES=['Mixing','Fractionation','PartialMelting','Alteration']

def fields(cfg):return list(dict.fromkeys(cfg['features']+list(cfg['baseline']['ree_reference_ppm'])))
def parameters(kind,x,rng,cfg,gen,composite=False):
 fs=gen.fields;n=len(fs);trace=[j for j,f in enumerate(fs) if f.endswith('_ppm')];major=list(range(len(cfg['major'])));grid=np.linspace(0,1,cfg['process']['states']);p={'trace_indices':trace,'major_indices':major,'initial':x.tolist()}
 if kind=='Mixing':
  b,_=gen.sample(rng);b[~np.isfinite(x)]=np.nan;p['endmember']=b.tolist()
 elif kind=='Alteration':
  a=np.zeros(n);js=rng.choice(n,max(1,round(.3*n)),replace=False);a[js]=rng.normal(0,.45,len(js));common=rng.normal(0,.3)
  ree=[i for i in js if fs[i] in cfg['baseline']['ree_reference_ppm']];a[ree]+=common;p['a']=a.tolist()
 elif kind in ['Fractionation','PartialMelting']:
  # Synthetic group priors, not calibrated mineral partition coefficients.
  D=np.ones(n);comp={'ni_ppm','cr_ppm','co_ppm','sc_ppm','v_ppm'};neutral={'cu_ppm','zn_ppm','ga_ppm'}
  groups={k:rng.normal(0,.25) for k in ['compatible','neutral','incompatible']}
  for j in trace:
   group='compatible' if fs[j] in comp else 'neutral' if fs[j] in neutral else 'incompatible';base={'compatible':3.,'neutral':1.,'incompatible':.08}[group];D[j]=base*np.exp(groups[group]+rng.normal(0,.15))
  p['D']=D.tolist()
  if kind=='PartialMelting':grid=np.linspace(*cfg['process']['melting_F'],cfg['process']['states'])
  else:
   endpoint=rng.uniform(.75,.95) if composite else cfg['process']['fractionation_F'][1];grid=np.linspace(1,endpoint,cfg['process']['states']);removed,controls=fractionation.plan(x,grid,cfg,rng);p['removed_masses']=removed.tolist();p['cumulate_controls']=controls.tolist()
 return p,grid

def apply(kind,x,p,grid):return {'Mixing':mixing,'Fractionation':fractionation,'PartialMelting':partial_melting,'Alteration':alteration}[kind].apply(x,p,grid)
def theta(kind,p,grid,n):
 # Controls use fixed physical units and training-only standardization in Ridge.
 out=np.zeros((len(grid)-1,3*n+12),np.float32)
 if kind=='Mixing':
  out[:,:n]=np.nan_to_num(np.asarray(p['endmember'],float));out[:,n:2*n]=np.nan_to_num(np.asarray(p['initial'],float))
 if kind in ['Fractionation','PartialMelting']:out[:,n:2*n]=p['D']
 if kind=='Alteration':out[:,2*n:3*n]=p['a']
 if kind=='Fractionation':out[:,3*n:3*n+11]=p['cumulate_controls']
 out[:,-1]=grid[:-1];return out

def legal(x,fs,cfg):
 m=x[:len(cfg['major'])];nonloi=np.array([j for j,f in enumerate(fs) if f!='loi']);observed=x[nonloi][np.isfinite(x[nonloi])];return np.isfinite(m).all() and (m>=-1e-7).all() and (m<=100+1e-7).all() and m.sum()>1 and (observed>=-1e-7).all() and not np.isinf(x).any() and np.nanmax(abs(x))<1e15

def prepare_bank(cfg,records,target=None):
 directory=ROOT/'cache/process_bank';directory.mkdir(exist_ok=True);fs=fields(cfg);train=records[records.split.eq('train')];prior=TracePrior(cfg,train);gen=RockGenerator(cfg,prior,fs)
 with open(directory/'trace_prior.pkl','wb') as f:pickle.dump(prior,f)
 save_json(directory/'trace_prior.json',{'fit_sample_hash':digest(train.sample_id.tolist()),'fields':prior.fields,'log_mean':prior.pca.mean_,'components':prior.pca.components_,'variances':prior.pca.explained_variance_})
 # Semi-synthetic chemistry is complete on major panel, otherwise no imputation.
 major=records[cfg['major']];eligible=major.notna().all(1)&major.ge(0).all(1)&major.sum(1).between(90,105);pool=records[eligible].copy();assert len(pool)>20
 low=train.reindex(columns=fs).quantile(.01).to_numpy();high=train.reindex(columns=fs).quantile(.99).to_numpy()
 previous=json.loads((directory/'schema.json').read_text()) if (directory/'schema.json').exists() else {}
 targets=target if isinstance(target,dict) else ({k:target for k in PROCESSES+['Composite']} if target else {})
 counts={k:max(cfg['process']['per_source']*2,previous.get('per_process_counts',{}).get(k,previous.get('per_process',0)),targets.get(k,0)) for k in PROCESSES};composite_n=max(cfg['process']['composites'],previous.get('composites',0),targets.get('Composite',0));index=[];attempts={}
 def initial(source,i,rng):
  if source=='synthetic':
   x,info=gen.sample(rng);identity=f'synthetic:{i}'
  else:
   row=pool.iloc[int(rng.integers(len(pool)))];x=row.reindex(fs).to_numpy(float);x[:11]*=100/np.sum(x[:11]);identity='natural:'+str(row.sample_id);info={'sample_id':str(row.sample_id),'major_closed_to_100':True,'record_split':row.split}
  return x,identity,info
 def write_episode(eid,source,i,kind,composite=False):
  path=directory/(eid+'.npz');meta=directory/(eid+'.json')
  if path.exists() and meta.exists():return json.loads(meta.read_text())
  seed=int(digest([cfg['seed'],eid])[:8],16);rng=np.random.default_rng(seed)
  for attempt in range(300):
   try:
    x,identity,initial_info=initial(source,i,rng);initial_x=x.copy();stages=[];params=[];grids=[];controls=[]
    if composite:
     hom=(i//2)%2==0;length=int(rng.choice([2,3]));sequence=[str(rng.choice(['Mixing','Fractionation','Alteration']))]*length if hom else rng.choice(['Mixing','Fractionation','Alteration'],length,replace=False).tolist()
    else:sequence=[kind];hom=False
    for process in sequence:
     p,g=parameters(process,x,rng,cfg,gen,composite);states=apply(process,x,p,g)
     if not all(legal(v,fs,cfg) for v in states):raise ValueError('illegal chemistry')
     stages.append(states);params.append(p);grids.append(g);controls.append(theta(process,p,g,len(fs)));x=states[-1]
    states,bounds=concatenate(stages) if composite else (stages[0],[])
    progress=np.arange(len(states),dtype=float)/32 if composite else grids[0]
    # Raw order diagnostic uses exactly the same A/B parameters, not resampled alternatives.
    order=None
    if composite and not hom:
     pa,pb=params[:2];ga,gb=grids[:2];a,b=sequence[:2]
     try:
      bfirst=apply(b,initial_x,pb,gb)[-1];ba=apply(a,bfirst,pa,ga)[-1]
      if legal(ba,fs,cfg):order=np.stack([initial_x,stages[1][-1],ba])
     except ValueError:pass
    np.savez_compressed(path,states=states.astype(np.float64),theta=np.vstack(controls),progress=progress,stage_id=np.concatenate([np.repeat(j,len(g)-1) for j,g in enumerate(grids)]),order=np.empty((0,len(fs))) if order is None else order)
    ood=(np.isfinite(states)&np.isfinite(low)&((states<low)|(states>high)));den=np.isfinite(states)&np.isfinite(low)
    item={'episode_id':eid,'seed':seed,'source_type':source,'initial_state_id':identity,'split':split_for(identity,cfg['seed']),'initial_state':initial_x.tolist(),'initial_metadata':initial_info,'process_sequence':sequence,'process_parameters':params,'progress_parameter':progress.tolist(),'stage_boundaries':bounds,'state_file':path.name,'states_hash':array_digest(states),'states':{'storage':'npz','key':'states','shape':list(states.shape)},'source_initial_not_first_melt':kind=='PartialMelting','stage_grids':[g.tolist() for g in grids],'kind':kind,'composite_type':'same-process' if hom else 'heterogeneous' if composite else 'single','ood_fraction':float(ood.sum()/max(den.sum(),1)),'attempts':attempt+1,'order_available':order is not None}
    save_json(meta,item);return item
   except (ValueError,FloatingPointError):continue
  raise RuntimeError(f'Cannot generate {eid} after 300 legal-state attempts')
 for kind in PROCESSES:
  per_source=counts[kind]//2
  for source in ['synthetic','semi-synthetic']:
   for i in range(per_source):
    eid=f'{kind}_{source}_{i:05d}';index.append(write_episode(eid,source,i,kind))
    if i%100==0:print(f'Bank {kind} {source} {i}/{per_source}',flush=True)
 for i in range(composite_n):
  source='synthetic' if i%2==0 else 'semi-synthetic';eid=f'Composite_{i:05d}';index.append(write_episode(eid,source,i,'Composite',True))
  if i%100==0:print(f'Bank Composite {i}/{composite_n}',flush=True)
 # Ideal solid-solution calibration trajectories, formula-derived endpoints.
 from src.process_bank.minerals import matrix
 names,m=matrix(cfg)
 for a,b in [('Ab','An'),('Fa','Fo'),('En','Fs'),('Di','Hd')]:
  eid=f'Calibration_{a}_{b}';x=np.full((33,len(fs)),np.nan);t=np.linspace(0,1,33);x[:,:11]=100*((1-t[:,None])*m[names.index(a)]+t[:,None]*m[names.index(b)])
  np.savez_compressed(directory/(eid+'.npz'),states=x,theta=np.zeros((32,3*len(fs)+12)),progress=t,stage_id=np.zeros(32),order=np.empty((0,len(fs))))
  item={'episode_id':eid,'seed':cfg['seed'],'source_type':'synthetic','initial_state_id':eid,'split':'test','initial_state':x[0].tolist(),'process_sequence':['Calibration'],'process_parameters':[{'from':a,'to':b}],'progress_parameter':t.tolist(),'stage_boundaries':[],'stage_grids':[t.tolist()],'state_file':eid+'.npz','states':{'storage':'npz','key':'states'},'kind':'Calibration','composite_type':'single','ood_fraction':None,'order_available':False};save_json(directory/(eid+'.json'),item);index.append(item)
 save_json(directory/'index.json',index);save_json(directory/'schema.json',{'fields':fs,'units':{f:'ppm' if f.endswith('_ppm') else 'wt_pct' for f in fs},'seed':cfg['seed'],'config_hash':config_digest(cfg),'episode_ids_hash':digest([r['episode_id'] for r in index]),'episode_content_hash':digest([(r['episode_id'],r.get('states_hash'),r['process_parameters']) for r in index]),'episodes':len(index),'per_process':min(counts.values()),'per_process_counts':counts,'composites':composite_n,'semi_pool':len(pool),'ood_reference':'train-only p1/p99; no model-dependent clipping','fractionation':'random constrained mineral removal; trace Rayleigh uses independent correlated synthetic D','composite_processes':['Mixing','Fractionation','Alteration'],'melting_composite_exclusion':'batch melt first state F=.01 is a product, not identical to source; no fabricated continuous stage join'})
 print(f'Bank saved: {len(index)} episodes',flush=True)
