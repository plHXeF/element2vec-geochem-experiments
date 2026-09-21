import numpy as np
def apply(x,params,grid):
 out=np.repeat(x[None,:],len(grid),axis=0);j=params['trace_indices'];d=np.array(params['D'])[j];F=grid[:,None];out[:,j]=x[j]/(d+F*(1-d));return out
