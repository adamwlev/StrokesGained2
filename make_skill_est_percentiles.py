import numpy as np
import os
from scipy.stats import rankdata

beta_map = {'green5':'16',
            'green0':'16',
            'tee45':'4',
            'rough0':'16',
            'fairway540':'16',
            'rough375':'16',
            'green10':'16',
            'other':'10',
            'green20':'16',
            'fairway0':'16',
            'bunker':'16',
            'rough90':'16',
            'tee3':'8'}

params = '0.8-0.7-0.8-0.95'
save_path = 'percentiles/'+params+'/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

dirname = 'ranks/ranks-'+params+'-%s'+'/'

def rank(a,mask):
    res = np.ones_like(a)
    res[mask] = rankdata(a[mask])/mask.sum()
    res[~mask] = np.nan
    return res

for cat in beta_map:
    ratings = np.load(dirname % (beta_map[cat],) + '%s_ranks.npy' % (cat,))
    reps = np.load(dirname % (beta_map[cat],) + '%s_reps.npy' % (cat,))
    ratings_ = np.copy(ratings)
    reps_ =  np.copy(reps)
    mask = np.logical_or(ratings_==0,reps_==0)
    ratings_[mask] = np.nan
    reps_[mask] = np.nan
    np.save(save_path+'%s_ranks' % (cat,)+'_percentile.npy',
            np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,ratings_))
    np.save(save_path+'%s_reps' % (cat,)+'_percentile.npy',
            np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,reps_))
    
    