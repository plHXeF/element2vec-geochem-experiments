"""Fixed-budget GELU/AdamW probes. Early stopping and hyperparameters use val only."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score,f1_score,r2_score
from scipy.stats import spearmanr

def scores(y,p,classification):
 if classification:return {'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_F1':float(f1_score(y,p,average='macro',zero_division=0))}
 return {'R2':float(r2_score(y,p)) if len(y)>1 and np.std(y)>0 else np.nan,'Spearman':float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else np.nan}

def probe(x,y,split,widths,cfg,classification=False,seed=0):
 torch.set_num_threads(2);rng=np.random.default_rng(seed);torch.manual_seed(seed)
 valid=np.asarray([v is not None and str(v) not in ['nan','None','<NA>',''] for v in y]) if classification else np.isfinite(np.asarray(y,float));ix={s:np.flatnonzero(valid&(split==s)) for s in ['train','val','test']}
 if min(map(len,ix.values()))<10 or len(ix['train'])<cfg['min_samples']:return None,None
 classes=None;yy=np.zeros(len(y),dtype=int if classification else float)
 if classification:
  classes=sorted(set(np.asarray(y)[ix['train']].astype(str)));mapping={v:i for i,v in enumerate(classes)}
  if len(classes)<2:return None,None
  yy=np.array([mapping.get(str(v),-1) for v in y]);coverage={s:float(np.mean(yy[ix[s]]>=0)) for s in ix}
  # Unknown test classes remain errors, not silently excluded from scores.
  ix['val']=ix['val'][yy[ix['val']]>=0]
  if len(ix['val'])<10:return None,None
  output=len(classes);yc=0;ys=1
 else:
  yy=np.asarray(y,float);yc=float(yy[ix['train']].mean());ys=max(float(yy[ix['train']].std()),1e-8);yy=(yy-yc)/ys;output=1;coverage={s:1. for s in ix}
 center=x[ix['train']].mean(0);scale=max(float(np.sqrt(np.mean((x[ix['train']]-center)**2))),1e-8)
 xx=torch.from_numpy(np.asarray((x-center)/scale,np.float32));yt=torch.tensor(yy,dtype=torch.long if classification else torch.float32)
 best=None;traces=[]
 for lr in cfg['learning_rates']:
  torch.manual_seed(seed);layers=[];dims=[x.shape[1]]+widths+[output]
  for j in range(len(dims)-1):
   layers.append(nn.Linear(dims[j],dims[j+1]));
   if j<len(dims)-2:layers.append(nn.GELU())
  model=nn.Sequential(*layers);params=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay']);criterion=nn.CrossEntropyLoss() if classification else nn.MSELoss();bestloss=np.inf;state=None;wait=0;history=[]
  for epoch in range(cfg['epochs']):
   model.train();order=rng.permutation(ix['train']);losses=[]
   for start in range(0,len(order),cfg['batch_size']):
    batch=order[start:start+cfg['batch_size']];pred=model(xx[batch]);loss=criterion(pred if classification else pred[:,0],yt[batch]);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.no_grad():
    pred=model(xx[ix['val']]);vl=float(criterion(pred if classification else pred[:,0],yt[ix['val']]))
   history.append([epoch,float(np.mean(losses)),vl])
   if vl<bestloss-1e-6:bestloss=vl;state=copy.deepcopy(model.state_dict());wait=0
   else:wait+=1
   if wait>=cfg['patience']:break
  traces.append({'learning_rate':lr,'history':history,'best_val_loss':bestloss})
  if best is None or bestloss<best[0]:best=(bestloss,state,model,lr,epoch+1,params)
 _,state,model,lr,epochs,params=best;model.load_state_dict(state);model.eval()
 with torch.no_grad():pred=model(xx[ix['test']]).numpy()
 if classification:yp=pred.argmax(1);true=yy[ix['test']]
 else:yp=pred[:,0]*ys+yc;true=np.asarray(y,float)[ix['test']]
 metrics=scores(true,yp,classification);metrics.update(parameter_count=params,epochs=epochs,learning_rate=lr,n_train=len(ix['train']),n_val=len(ix['val']),n_test=len(ix['test']),known_test_class_fraction=coverage['test'],search_budget=len(cfg['learning_rates']))
 artifact={'indices':ix['test'],'truth':true,'prediction':yp,'state_dict':state,'center':center,'scale':scale,'target_center':yc,'target_scale':ys,'classes':classes,'history':traces,'widths':widths}
 return metrics,artifact
