"""Native-space Euclidean semantic geometry; no PCA/UMAP in metrics."""
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.common import ROOT,NAMES,save_json,table

def generate_triplets(y,group,avail,cfg,hierarchy=False):
 rng=np.random.default_rng(cfg['seed']);triples=[];fallback=0;valid=pd.notna(y)
 if not hierarchy:valid&=np.isfinite(np.asarray(y,float))
 ids=np.flatnonzero(valid)
 if len(ids)<30:return np.empty((0,3),int),0
 for a in rng.choice(ids,min(len(ids),cfg['semantic']['triplets']),replace=len(ids)<cfg['semantic']['triplets']):
  pool=ids[(ids!=a)&(np.abs(avail[ids]-avail[a])<=cfg['semantic']['availability_tolerance'])]
  if hierarchy:
   positive=pool[y[pool]==y[a]];negative=pool[y[pool]!=y[a]]
  else:
   same=pool[group[pool]==group[a]]
   if len(same)>=20:pool=same
   else:fallback+=1
   if len(pool)<10:continue
   d=abs(y[pool]-y[a]);low,high=np.quantile(d,[cfg['semantic']['low_quantile'],cfg['semantic']['high_quantile']]);positive=pool[d<=low];negative=pool[d>=high]
   if high<=low:continue
  if len(positive) and len(negative):triples.append([a,rng.choice(positive),rng.choice(negative)])
 return np.asarray(triples,int).reshape(-1,3),fallback

def evaluate(cfg,records):
 out=ROOT/'outputs/semantic_geometry';out.mkdir(exist_ok=True);ev=records.split.eq('test').to_numpy();meta=records[ev].reset_index(drop=True);ys=pd.read_parquet(ROOT/'cache/targets.parquet')[ev].reset_index(drop=True);tasks=['SiO2','Mg_number','total_alkali','SigmaREE10','La_Yb','Eu_anomaly','Sr_Y','rock_group','rock_origin'];availability=meta[cfg['features']].notna().sum(1).to_numpy();group=meta.rock_group.fillna('unknown').to_numpy();triplets={};rows=[]
 for task in tasks:
  y=ys[task].to_numpy();tri,fb=generate_triplets(y,group,availability,cfg,task.startswith('rock_'));triplets[task]=tri;save_json(out/(task+'_triplet_protocol.json'),{'triplets':len(tri),'group_match_fallback':fb,'availability_tolerance':cfg['semantic']['availability_tolerance']})
 np.savez(ROOT/'cache/semantic_triplets.npz',**triplets)
 for name in NAMES:
  x=np.load(ROOT/'cache/real'/name/'unmasked.npy')[ev].astype(float)
  scale=json.loads((ROOT/'cache/real'/name/'scale.json').read_text())['global_rms_radius'];x=x/max(scale,1e-12)
  nn=NearestNeighbors(n_neighbors=cfg['semantic']['knn']+1,metric='euclidean').fit(x).kneighbors(x,return_distance=False)[:,1:]
  for task in tasks:
   tri=triplets[task];row={'representation':name,'target':task,'triplets':len(tri),'triplet_accuracy':np.nan}
   if len(tri):
    a,p,n=tri.T;dp=np.linalg.norm(x[a]-x[p],axis=1);dn=np.linalg.norm(x[a]-x[n],axis=1);row['triplet_accuracy']=float(np.mean(dp<dn));row['tie_rate']=float(np.mean(np.isclose(dp,dn,atol=1e-12)))
   if not task.startswith('rock_'):
    y=ys[task].to_numpy(float);valid=np.isfinite(y);anchors=np.repeat(np.arange(len(y)),nn.shape[1]);targets=nn.ravel();ok=valid[anchors]&valid[targets];a=anchors[ok];b=targets[ok]
    if len(a)>20 and np.var(y[valid])>1e-20:
     vals=y[valid];variance=np.var(vals);observed=np.mean((y[a]-y[b])**2)/max(2*variance,1e-12);rng=np.random.default_rng(cfg['seed']);null=[]
     for _ in range(cfg['semantic']['permutations']):
      shuffled=y.copy();shuffled[valid]=rng.permutation(vals);null.append(np.mean((shuffled[a]-shuffled[b])**2)/max(2*variance,1e-12))
     row.update(graph_energy=observed,permutation_mean=float(np.mean(null)),smoothness=1-observed/max(np.mean(null),1e-12),permutation_p=(1+np.sum(np.asarray(null)<=observed))/(len(null)+1),valid_edges=len(a))
   rows.append(row)
  print('Semantic',name,flush=True)
 table(pd.DataFrame(rows),out/'metrics')
