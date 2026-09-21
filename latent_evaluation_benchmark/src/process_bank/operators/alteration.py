import numpy as np
def apply(x,params,grid):return x[None,:]*np.exp(grid[:,None]*np.array(params['a'],dtype=float)[None,:])
