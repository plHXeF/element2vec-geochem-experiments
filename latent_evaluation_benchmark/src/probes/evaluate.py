import json,pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from src.common import ROOT,NAMES,table,save_json,digest
from src.probes.train import probe,scores

def decodability(cfg,records):
 out=ROOT/'outputs/decodability';out.mkdir(exist_ok=True);y=pd.read_parquet(ROOT/'cache/targets.parquet');split=records.split.to_numpy();rows=[]
 for target in ['rock_group','rock_origin','SiO2','Mg_number','SigmaREE10']:
  task='unmasked' if target.startswith('rock_') else target
  for name in NAMES:
   x=np.load(ROOT/'cache/real'/name/(task+'.npy'))
   for j,width in enumerate(cfg['probe']['widths']):
    key=f'{target}_{name}_{j}';p=out/(key+'.json')
    if p.exists():rows.append(json.loads(p.read_text()));continue
    print('Probe',key,flush=True);result,art=probe(x,y[target].to_numpy(),split,width,cfg['probe'],target.startswith('rock_'),cfg['seed'])
    row={'target':target,'representation':name,'probe':['Linear','MLP32','MLP128_64','MLP512_256_128'][j],'status':'ok' if result else 'insufficient_labels',**(result or {})};save_json(p,row);rows.append(row)
    if art:
     (ROOT/'cache/probes').mkdir(exist_ok=True);(ROOT/'cache/probes'/(key+'.pkl')).write_bytes(pickle.dumps(art))
   table(pd.DataFrame(rows),out/'metrics')
 frame=pd.DataFrame(rows);auc=[]
 for (target,name),g in frame[frame.status.eq('ok')].groupby(['target','representation']):
  metric='balanced_accuracy' if target.startswith('rock_') else 'R2';g=g.sort_values('parameter_count');xx=np.log10(g.parameter_count);area=np.trapezoid(g[metric],xx);auc.append({'target':target,'representation':name,'metric':metric,'auc':area,'normalized_auc':area/(xx.iloc[-1]-xx.iloc[0]),'log_parameters_min':xx.iloc[0],'log_parameters_max':xx.iloc[-1],'warning':'different parameter-count support; curves remain primary'})
 table(pd.DataFrame(auc),out/'curve_auc');return frame

def sufficiency(cfg,records):
 out=ROOT/'outputs/sufficiency';out.mkdir(exist_ok=True);mask=np.load(ROOT/'cache/recovery_mask.npy');cols=json.loads((ROOT/'cache/recovery_columns.json').read_text());split=records.split.to_numpy();rows=[]
 for name in NAMES:
  x=np.load(ROOT/'cache/real'/name/'recovery.npy').astype(float)
  for j,c in enumerate(cols):
   y=records[c].to_numpy(float);valid=(mask[:,j] if c in cfg['features'] else np.ones(len(y),bool))&np.isfinite(y);ix={s:np.flatnonzero(valid&(split==s)) for s in ['train','val','test']};group='REE' if c in cfg['ree_input'] else 'major' if c in cfg['major'] else 'trace' if c in cfg['trace'] else 'isotope' if c in cfg['isotope'] else 'age/derived'
   row={'representation':name,'field':c,'group':group,'encoder_input':c in cfg['features'],'protocol':'masked20_recovery' if c in cfg['features'] else 'external_target_on_masked_chemistry_all_labels','status':'ok' if min(map(len,ix.values()))>=10 else 'insufficient_masked_labels','n_train':len(ix['train']),'n_test':len(ix['test'])}
   if row['status']=='ok':
    center=x[ix['train']].mean(0);scale=max(np.sqrt(np.mean((x[ix['train']]-center)**2)),1e-8);xx=(x-center)/scale;yc=y[ix['train']].mean();ys=max(y[ix['train']].std(),1e-8);best=None
    for alpha in cfg['process']['ridge_alphas']:
     model=Ridge(alpha=alpha).fit(xx[ix['train']],(y[ix['train']]-yc)/ys);err=np.mean((model.predict(xx[ix['val']])*ys+yc-y[ix['val']])**2)
     if best is None or err<best[0]:best=(err,model,alpha)
    pred=best[1].predict(xx[ix['test']])*ys+yc;row.update(scores(y[ix['test']],pred,False),alpha=best[2],parameter_count=x.shape[1]+1)
   rows.append(row)
  print('Recovery',name,flush=True);table(pd.DataFrame(rows),out/'masked_recovery')
 # Reporting clusters are fitted to train availability only, then assigned to val/test.
 avail=records[cfg['features']].notna().to_numpy(float);train=split=='train';km=KMeans(n_clusters=12,random_state=cfg['seed'],n_init=10).fit(avail[train]);clusters=km.predict(avail).astype(str)
 save_json(ROOT/'cache/missingness_clusters.json',{'centers':km.cluster_centers_,'train_ids_hash':digest(records.loc[train,'sample_id'].tolist()),'n_clusters':12})
 nuisance=[]
 for name in NAMES:
  x=np.load(ROOT/'cache/real'/name/'unmasked.npy')
  for label in ['data_source','canonical_ref_id','missingness_cluster']:
   y=clusters if label=='missingness_cluster' else records[label].astype(str).to_numpy();keep=None
   if label=='canonical_ref_id':
    counts=pd.Series(y[train]).value_counts();keep=counts[counts>=10].head(64).index.tolist();y=np.array([v if v in keep else None for v in y],object)
   result,art=probe(x,y,split,[],cfg['probe'],True,cfg['seed']);row={'representation':name,'target':label,'status':'ok' if result else 'insufficient_labels','reference_scope':'top 64 train references with >=10 train samples' if keep else 'all','coverage':float(np.mean(pd.notna(y))),**(result or {})};nuisance.append(row)
   if art:(ROOT/'cache/probes'/f'nuisance_{name}_{label}.pkl').write_bytes(pickle.dumps(art))
  ix=np.flatnonzero(split=='test');z=x[ix];nn=NearestNeighbors(n_neighbors=21,metric='euclidean').fit(z).kneighbors(z,return_distance=False)[:,1:];s=records.data_source.iloc[ix].fillna('unknown').to_numpy();purity=np.mean(s[nn]==s[:,None]);counts=pd.Series(s).value_counts().to_numpy();chance=np.sum(counts*(counts-1))/(len(s)*(len(s)-1));nuisance.append({'representation':name,'target':'same_source_knn_enrichment','status':'ok','purity':purity,'chance':chance,'enrichment':purity/max(chance,1e-12)})
  print('Nuisance',name,flush=True);table(pd.DataFrame(nuisance),out/'nuisance')
