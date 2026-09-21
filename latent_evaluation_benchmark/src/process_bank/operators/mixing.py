import numpy as np
def apply(x,params,grid):
 b=np.array(params['endmember'],dtype=float);return (1-grid[:,None])*x+grid[:,None]*b
