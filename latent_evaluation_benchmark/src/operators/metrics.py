import numpy as np
from scipy.spatial.distance import cdist

def trajectory_metrics(z,cfg):
 z=np.asarray(z,float);dz=np.diff(z,axis=0);norm=np.linalg.norm(dz,axis=1);valid=norm>1e-10;cos=np.sum(dz[:-1]*dz[1:],axis=1)/np.maximum(norm[:-1]*norm[1:],1e-20);ok=valid[:-1]&valid[1:];length=norm.sum();endpoint=np.linalg.norm(z[-1]-z[0]);ranks=[];window=cfg['process']['window']
 for i in range(len(z)-window+1):
  a=z[i:i+window]-z[i:i+window].mean(0);e=np.linalg.eigvalsh(a@a.T);ranks.append(e[-1]/e.sum() if e.sum()>1e-20 else np.nan)
 d=cdist(z,z);ii=np.arange(len(z));d[abs(ii[:,None]-ii[None,:])<=cfg['process']['fold_gap']]=np.inf;typical=np.median(norm[valid]) if valid.any() else np.nan
 collision=float(np.mean(d.min(1)<typical)) if np.isfinite(typical) else np.nan
 return {'tangent_coherence':float(np.mean(cos[ok])) if ok.any() else np.nan,'simplicity':float(endpoint/length) if length>1e-10 else np.nan,'local_rank1':float(np.nanmedian(ranks)) if np.isfinite(ranks).any() else np.nan,'no_fold':1-collision,'collision_rate':collision,'path_length':length,'endpoint_distance':endpoint,'zero_step_fraction':1-float(valid.mean())}

def boundary_detection(z,boundaries,tolerance=2,threshold=.2):
 dz=np.diff(z.astype(float),axis=0);norm=np.linalg.norm(dz,axis=1);cos=np.sum(dz[:-1]*dz[1:],axis=1)/np.maximum(norm[:-1]*norm[1:],1e-20);score=1-cos;candidate=np.flatnonzero((score>=threshold)&(score>=np.r_[0,score[:-1]])&(score>=np.r_[score[1:],0]))+1
 selected=[]
 for i in candidate[np.argsort(score[candidate-1])[::-1]]:
  if all(abs(i-j)>tolerance for j in selected):selected.append(int(i))
 unmatched=set(boundaries);hits=0
 for p in selected:
  possible=[b for b in unmatched if abs(b-p)<=tolerance]
  if possible:b=min(possible,key=lambda b:abs(b-p));unmatched.remove(b);hits+=1
 precision=hits/len(selected) if selected else 0.;recall=hits/len(boundaries) if boundaries else np.nan;f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0.
 return {'boundary_precision':precision,'boundary_recall':recall,'boundary_f1':f1,'predicted_boundaries':selected}
