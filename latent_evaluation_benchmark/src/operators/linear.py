"""Native-dimensional Ridge vector fields. Whole initial-state groups are held out."""
import json,pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from src.common import ROOT,save_json,table,NAMES
from src.process_bank.episode import PROCESSES

def fit_operators(cfg,index):
 bank=ROOT/'cache/process_bank';out=ROOT/'outputs/process';out.mkdir(exist_ok=True);cache=ROOT/'cache/operators';cache.mkdir(exist_ok=True);rows=[]
 for name in NAMES:
  scale=json.loads((ROOT/'cache/real'/name/'scale.json').read_text())['global_rms_radius']
  for process in PROCESSES:
   path=cache/f'{name}_{process}.pkl'
   if path.exists():continue
   arrays={s:[] for s in ['train','val','test']};xs={s:[] for s in arrays};ys={s:[] for s in arrays}
   for row in index:
    if row['kind']!=process:continue
    raw=np.load(bank/row['state_file']);z=np.load(ROOT/'cache/process_latents'/name/(row['episode_id']+'.npz'))['z'].astype(float)/scale;t=raw['progress'];dt=np.diff(t);controls=raw['theta'].copy()
    if process=='Mixing':
     n=(controls.shape[1]-12)//3;controls[:,n:2*n]=0
     features=np.c_[z[:-1],controls,np.repeat(z[:1],len(dt),axis=0)]
    else:features=np.c_[z[:-1],controls]
    deriv=np.diff(z,axis=0)/dt[:,None]
    if row['split'] in ['train','val']:
     # Fixed evenly spaced fit rows per episode; evaluation always uses every step.
     take=np.arange(0,len(dt),4);xs[row['split']].append(features[take]);ys[row['split']].append(deriv[take])
   X=np.vstack(xs['train']);Y=np.vstack(ys['train']);V=np.vstack(xs['val']);W=np.vstack(ys['val']);d=z.shape[1]
   # A single global latent scale; control columns have training-only per-parameter scales.
   center=X.mean(0);ss=np.ones(X.shape[1]);end=X.shape[1]-d if process=='Mixing' else X.shape[1];ss[d:end]=np.maximum(X[:,d:end].std(0),1e-8);X=(X-center)/ss;V=(V-center)/ss
   best=None
   for alpha in cfg['process']['ridge_alphas']:
    m=Ridge(alpha=alpha,solver='cholesky').fit(X,Y);err=float(np.mean((m.predict(V)-W)**2))
    if best is None or err<best[0]:best=(err,m,alpha)
   art={'model':best[1],'center':center,'control_scale':ss,'latent_scale':scale,'latent_dim':d,'process':process,'mixing_initial_latent_condition':process=='Mixing','alpha':best[2],'train_rows':len(X),'val_rows':len(V),'fit_stride':4,'parameter_count':int(best[1].coef_.size+best[1].intercept_.size),'train_initial_ids':sorted({r['initial_state_id'] for r in index if r['kind']==process and r['split']=='train'}),'test_initial_ids':sorted({r['initial_state_id'] for r in index if r['kind']==process and r['split']=='test'})}
   assert not set(art['train_initial_ids'])&set(art['test_initial_ids']);path.write_bytes(pickle.dumps(art));print('Operator fit',name,process,flush=True)
 metric_signature=operator_cache_signature(index);metric_meta=out/'operator_metrics_signature.json'
 if metric_meta.exists() and (out/'operator_metrics.parquet').exists() and json.loads(metric_meta.read_text())['signature']==metric_signature:
  return pd.read_parquet(out/'operator_metrics.parquet').to_dict('records')
 for name in NAMES:
  for process in PROCESSES:
   art=pickle.loads((cache/f'{name}_{process}.pkl').read_bytes());scale=art['latent_scale']
   for row in index:
    if row['kind']!=process:continue
    raw=np.load(bank/row['state_file']);z=np.load(ROOT/'cache/process_latents'/name/(row['episode_id']+'.npz'))['z'].astype(float)/scale;dt=np.diff(raw['progress']);controls=raw['theta'].copy()
    if process=='Mixing':
     n=(controls.shape[1]-12)//3;controls[:,n:2*n]=0;inp=np.c_[z[:-1],controls,np.repeat(z[:1],len(dt),axis=0)]
    else:inp=np.c_[z[:-1],controls]
    pred=z[:-1]+dt[:,None]*art['model'].predict((inp-art['center'])/art['control_scale']);delta=z[1:]-z[:-1];err=np.linalg.norm(pred-z[1:],axis=1)
    rows.append({'representation':name,'process':process,'episode_id':row['episode_id'],'source_type':row['source_type'],'split':row['split'],'normalized_next_state_error':float(np.mean(err)),'relative_step_error':float(np.sqrt(np.mean(err**2))/max(np.sqrt(np.mean(np.sum(delta**2,axis=1))),1e-12)),'R2':float(r2_score(z[1:],pred,multioutput='variance_weighted')) if np.sum(np.var(z[1:],axis=0))>1e-12 else np.nan,'alpha':art['alpha'],'parameter_count':art['parameter_count']})
 table(pd.DataFrame(rows),out/'operator_metrics');save_json(metric_meta,{'signature':metric_signature});return rows

def rollout(art,z0,theta,t):
 z=np.array(z0,float)/art['latent_scale'];initial=z.copy()
 for control,dt in zip(theta,np.diff(t)):
  control=control.copy()
  if art.get('mixing_initial_latent_condition'):
   n=(len(control)-12)//3;control[n:2*n]=0;v=np.r_[z,control,initial]
  else:v=np.r_[z,control]
  z=z+dt*art['model'].predict(((v-art['center'])/art['control_scale'])[None,:])[0]
  if not np.isfinite(z).all() or np.linalg.norm(z)>1e15:return np.full_like(z,np.nan)
 return z*art['latent_scale']

def rollout_batch(art,initial_raw,controls,grids):
 """Same float64 Euler recurrence as rollout, batched without true intermediate states."""
 initial=np.asarray(initial_raw,float)/art['latent_scale'];z=initial.copy();controls=np.asarray(controls,float).copy();d=art['latent_dim'];nstep=controls.shape[1]
 if art.get('mixing_initial_latent_condition'):
  n=(controls.shape[2]-12)//3;controls[:,:,n:2*n]=0
 coefficients=art['model'].coef_/art['control_scale'][None,:];bias=art['model'].intercept_-coefficients@art['center'];A=coefficients[:,:d];B=coefficients[:,d:d+controls.shape[2]]
 forces=(controls.reshape(-1,controls.shape[2])@B.T).reshape(len(z),nstep,d)+bias
 if art.get('mixing_initial_latent_condition'):forces+= (initial@coefficients[:,-d:].T)[:,None,:]
 dt=np.diff(np.asarray(grids,float),axis=1)
 for step in range(nstep):
  z+=dt[:,step,None]*(z@A.T+forces[:,step]);bad=~np.isfinite(z).all(1)|(np.linalg.norm(z,axis=1)>1e15);z[bad]=np.nan
 return z*art['latent_scale']

def operator_cache_signature(index):
 from src.common import digest,sha
 paths=sorted((ROOT/'cache/operators').glob('*.pkl'))
 return digest({'operators':{p.name:sha(p) for p in paths},'single_processes':[(r['episode_id'],r.get('states_hash'),r['process_parameters']) for r in index if r['kind'] in PROCESSES],'metric_version':'native-euler-v2-r2-temporal-variance'})
