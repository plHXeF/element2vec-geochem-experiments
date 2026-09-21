"""Training-only baselines; no target values are consulted during encoding."""
import numpy as np
from sklearn.decomposition import PCA
class Representation:
 def fit_reference(self,train_records):raise NotImplementedError
 def encode(self,records):raise NotImplementedError
class ZVectorRepresentation(Representation):
 name='ZVector'
 def __init__(self,cfg):self.cfg=cfg;self.features=cfg['features']
 def fit_reference(self,train_records):
  x=train_records[self.features].to_numpy(float);self.mean=np.nanmean(x,0);self.std=np.nanstd(x,0);self.mean=np.nan_to_num(self.mean);self.std=np.where(np.isfinite(self.std)&(self.std>self.cfg['baseline']['min_scale']),self.std,1.);return self
 def encode(self,records):
  x=records.reindex(columns=self.features).to_numpy(float);mask=~np.isfinite(x);z=np.where(mask,0,(x-self.mean)/self.std);return np.c_[z,mask.astype(float)].astype(np.float32)
class GeoNormRepresentation(Representation):
 name='GeoNorm'
 def __init__(self,cfg):self.cfg=cfg;self.features=cfg['features'];self.spec=cfg['baseline'];self.major=[self.features.index(v) for v in cfg['major']];self.trace=[self.features.index(v) for v in cfg['trace']];self.ree=[self.features.index(v) for v in cfg['ree_input']]
 def unscaled(self,records):
  x=records.reindex(columns=self.features).to_numpy(float);missing=~np.isfinite(x);r=x.copy();m=x[:,self.major]
  # Observed subcomposition only. Hidden targets do not enter closure or CLR.
  valid=np.isfinite(m)&(m>=0);p=np.where(valid,np.maximum(m,self.spec['major_pseudocount_wt']),np.nan)
  closed=p/np.nansum(p,axis=1,keepdims=True);log=np.log(closed);r[:,self.major]=log-np.nanmean(log,axis=1,keepdims=True)
  t=x[:,self.trace];r[:,self.trace]=np.log10(np.maximum(t,self.spec['trace_pseudocount_ppm']))
  aux=[self.features.index(v) for v in self.cfg['auxiliary']];r[:,aux]=np.sign(x[:,aux])*np.log1p(abs(x[:,aux]))
  refs=np.array([self.spec['ree_reference_ppm'][self.features[j]] for j in self.ree]);ree=np.log10(np.maximum(x[:,self.ree],self.spec['trace_pseudocount_ppm'])/refs)
  return r,ree,missing
 def fit_reference(self,train_records):
  r,ree,_=self.unscaled(train_records);self.center=np.nanmedian(r,0);self.scale=np.nanquantile(r,.75,axis=0)-np.nanquantile(r,.25,axis=0)
  # Major CLR is kept as CLR, trace and auxiliary robust-scaled.
  self.center[self.major]=0;self.scale[self.major]=1
  self.center=np.nan_to_num(self.center);self.scale=np.where(np.isfinite(self.scale)&(self.scale>self.spec['min_scale']),self.scale,1.);return self
 def encode(self,records):
  r,ree,mask=self.unscaled(records);r=(r-self.center)/self.scale
  return np.c_[np.nan_to_num(r),np.nan_to_num(ree),mask.astype(float)].astype(np.float32)
class PCAGeoRepresentation(Representation):
 name='PCAGeo'
 def __init__(self,cfg):self.cfg=cfg;self.geo=GeoNormRepresentation(cfg)
 def fit_reference(self,train_records):
  self.geo.fit_reference(train_records);x=self.geo.encode(train_records);self.pca=PCA(svd_solver='full').fit(x)
  self.n=min(self.cfg['baseline']['pca_max_dims'],int(np.searchsorted(np.cumsum(self.pca.explained_variance_ratio_),self.cfg['baseline']['pca_variance'])+1));return self
 def encode(self,records):return self.pca.transform(self.geo.encode(records))[:,:self.n].astype(np.float32)
