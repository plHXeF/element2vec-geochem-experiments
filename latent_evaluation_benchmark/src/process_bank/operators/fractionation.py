"""Rayleigh trace law and mass-balanced incremental mineral removal."""
import numpy as np
from scipy.optimize import linprog
from src.process_bank.minerals import matrix

def apply(x,params,grid):
 out=np.repeat(x[None,:],len(grid),axis=0);j=params['trace_indices'];D=np.array(params['D']);out[:,j]=x[j]*grid[:,None]**(D[j]-1)
 major=params['major_indices'];mass=x[major]/100;total0=mass.sum();cum=np.array(params['cumulate_controls'])*(-np.diff(grid))[:,None]*total0
 for i in range(1,len(grid)):
  mass=mass-cum[i-1];out[i,major]=mass/grid[i]*100
 if np.nanmin(out[:,major]) < -1e-7:raise ValueError('negative residual')
 out[:,major]=np.maximum(out[:,major],0);return out

def plan(x,grid,cfg,rng):
 _,minerals=matrix(cfg);mass=x[:len(cfg['major'])]/100;total0=mass.sum();removals=[];controls=[]
 for i in range(1,len(grid)):
  dm=(grid[i-1]-grid[i])*total0;objective=-rng.lognormal(0,.7,len(minerals))
  sol=linprog(objective,A_ub=minerals.T,b_ub=mass+1e-12,A_eq=np.ones((1,len(minerals))),b_eq=[dm],bounds=(0,None),method='highs')
  if not sol.success:raise ValueError('infeasible mineral removal to requested F')
  removed=sol.x@minerals;mass-=removed;removals.append(removed);controls.append(removed/dm)
 return np.asarray(removals),np.asarray(controls)
