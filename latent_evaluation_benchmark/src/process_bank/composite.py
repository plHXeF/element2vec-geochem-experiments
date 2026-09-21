"""Strict stage chaining and physical trace semigroup controls."""
import numpy as np
def concatenate(stage_states):
 for a,b in zip(stage_states[:-1],stage_states[1:]):
  if not np.allclose(a[-1],b[0],equal_nan=True):raise ValueError('discontinuous stage input')
 states=stage_states[0];bounds=[]
 for stage in stage_states[1:]:bounds.append(len(states)-1);states=np.vstack([states,stage[1:]])
 return states,bounds

def trace_semigroup(x,D,a,F1,F2,t1,t2):
 f12=x*(F1*F2)**(D-1);fchain=(x*F1**(D-1))*F2**(D-1)
 alt12=x*np.exp((t1+t2)*a);altchain=(x*np.exp(t1*a))*np.exp(t2*a)
 return np.nanmax(abs(f12-fchain))/max(np.nanmax(abs(f12)),1e-12),np.nanmax(abs(alt12-altchain))/max(np.nanmax(abs(alt12)),1e-12)
