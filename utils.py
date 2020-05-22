import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# from clx to class(y)
def cly(clx):
    y=np.array([])
    for i in range(len(clx)):
        y=np.append(y,np.repeat(list(clx.keys())[i],len(clx[list(clx.keys())[i]])))
    return y

#from clx to dataframe
def df(clx):
    x=pd.DataFrame() 
    for i in clx.keys():
        x=x.append(clx[i])
    return x


#dictionary construction
def dict_cons(clx, gamma, epsilon):
    rbf_clx=rbf_kernel(clx,gamma=gamma)
    if len(clx)>1:
        init_idx=np.array([],dtype='i')
        init_idx=np.append(init_idx,np.unravel_index(rbf_clx.argmin(),rbf_clx.shape))
        cand_sim=np.vstack([rbf_clx[init_idx[0]],rbf_clx[init_idx[1]]])
        
        if rbf_clx.min()<epsilon:
            while len(clx)-len(init_idx)>0:
                if np.min(cand_sim.max(axis=0))<epsilon:
                    init_idx=np.append(init_idx,np.where(cand_sim.max(axis=0)==np.min(cand_sim.max(axis=0)))[0][0])
                    cand_sim=np.vstack([cand_sim,rbf_clx[init_idx[len(init_idx)-1]]])
                else:
                    break
    
    init_clx=pd.DataFrame()
    for k in init_idx:
        init_clx=init_clx.append(clx.iloc[k:k+1])
    clx=init_clx.copy()
    
    return clx