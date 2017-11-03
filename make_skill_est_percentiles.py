import numpy as np
import os
from scipy.stats import rankdata

params = '0.8-0.7-0.8-0.94-5'
save_path = 'percentiles/'+params+'/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

dirname = 'ranks/ranks-'+params+'/'
fns = os.listdir(dirname)
if '.DS_Store' in fns:
    fns.remove('.DS_Store')

def rank(a,mask):
    res = np.ones_like(a)
    res[mask] = rankdata(a[mask])/mask.sum()
    res[~mask] = np.nan
    return res

for u in range(0,len(fns),2):
    ratings = np.load(dirname+fns[u])[:,-1]
    reps = np.load(dirname+fns[u+1])[:,-1]
    ratings_ = np.copy(ratings)
    reps_ =  np.copy(reps)
    mask = np.logical_or(ratings_==0,reps_==0)
    ratings_[mask] = np.nan
    reps_[mask] = np.nan
    np.save(save_path+fns[u].split('.')[0]+'_percentile.npy',
            np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,ratings_))
    np.save(save_path+fns[u+1].split('.')[0]+'_percentile.npy',
            np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,reps_))
    
    