import numpy as np
from scipy.stats import rankdata
import os

dirname = 'ranks/ranks-0.8-0.7-0.8-0.94-5/'
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
	mask = np.logical_or(ratings==0,reps==0)
	ratings[mask] = np.nan
	reps[mask] = np.nan
	ratings_percentile = rank(ratings,~np.isnan(ratings))
	reps_percentile = rank(reps,~np.isnan(reps))
	np.save('latest_percentiles/'+fns[u],ratings_percentile)
	np.save('latest_percentiles/'+fns[u+1],reps_percentile)