import numpy as np
from sklearn.decomposition import PCA
class TracePrior:
 def __init__(self,cfg,train):
  self.cfg=cfg;self.fields=cfg['trace'];x=train[self.fields].to_numpy(float);x=np.where(x>0,x,np.nan);log=np.log(np.maximum(x,cfg['baseline']['trace_pseudocount_ppm']));self.mean=np.nanmedian(log,axis=0);log=np.where(np.isfinite(log),log,self.mean);self.pca=PCA(n_components=min(cfg['process']['trace_prior_dims'],len(self.fields))).fit(log);self.sd=np.sqrt(self.pca.explained_variance_)
 def sample(self,rng):return dict(zip(self.fields,np.exp(self.pca.mean_+(rng.normal(size=len(self.sd))*self.sd)@self.pca.components_)))
 def ree(self,rng):
  c=self.cfg['process'];ref=self.cfg['baseline']['ree_reference_ppm'];u=np.linspace(-1,1,len(ref));a=[rng.uniform(*c[k]) for k in ['ree_a0','ree_a1','ree_a2','ree_eu']]
  return {e:float(r*np.exp(a[0]+a[1]*t+a[2]*t*t+a[3]*(e=='eu_ppm'))) for (e,r),t in zip(ref.items(),u)}
