import pandas as pd
import numpy as np
from scipy.stats import rankdata
import sys, pickle

if __name__=="__main__":
    cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
            'rough375','fairway0','fairway300','fairway540','bunker','other']

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
                'fairway300':'16',
                'tee3':'8'}

    _,e_d,e_t,w_d,a = sys.argv
    ratings, reps = {},{}
    for cat in cats:
        beta = beta_map[cat]
        ratings[cat] = np.load('ranks/ranks-%s-%s-%s-%s-%s/%s_ranks.npy' % (e_d,e_t,w_d,a,beta,cat))
        reps[cat] = np.load('ranks/ranks-%s-%s-%s-%s-%s/%s_reps.npy' % (e_d,e_t,w_d,a,beta,cat))
    
    def rank(a,mask):
        res = np.ones_like(a)
        res[mask] = rankdata(a[mask])/mask.sum()
        res[~mask] = np.nan
        return res

    ratings_percentile, reps_percentile = {}, {}
    for cat in cats:
        ratings_ = np.copy(ratings[cat])
        reps_ =  np.copy(reps[cat])
        mask = np.logical_or(ratings_==0,reps_==0)
        ratings_[mask] = np.nan
        reps_[mask] = np.nan
        ratings_percentile[cat] = np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,ratings_)
        reps_percentile[cat] = np.apply_along_axis(lambda x: rank(x,~np.isnan(x)),0,reps_)
        
    with open('PickleFiles/num_to_ind_shot.pkl','rb') as pickle_file:
        num_to_ind = pickle.load(pickle_file,encoding='latin1')

    for year in range(2003,2019):
        print(year)
        data = pd.read_csv('../GolfData/Shot/%d.csv.gz' % (year,))
        data['Player_Index'] = [num_to_ind[num] for num in data['Player_#']]
        #data = data.drop(['skill_estimate','observation_count','not_seen'],axis=1)
        for cat in cats:
            data['skill_estimate_%s' % (cat,)] = [ratings[cat][player_ind,tourn_num-1]
                                                  if tourn_num>0 else np.nan
                                                  for player_ind,tourn_num in zip(data['Player_Index'],
                                                                                  data['tourn_num'])]
            data['observation_count_%s' % (cat,)] = [reps[cat][player_ind,tourn_num-1]
                                                     if tourn_num>0 else np.nan
                                                     for player_ind,tourn_num in zip(data['Player_Index'],
                                                                                     data['tourn_num'])]
            data['not_seen_%s' % (cat,)] = (data['skill_estimate_%s' % (cat,)]==0) | (data['observation_count_%s' % (cat,)]==0)
            data['skill_estimate_percentile_%s' % (cat,)] = [ratings_percentile[cat][player_ind,tourn_num-1]
                                                             if tourn_num>0 else np.nan
                                                             for player_ind,tourn_num in zip(data['Player_Index'],
                                                                                             data['tourn_num'])]
            data['observation_count_percentile_%s' % (cat,)] = [reps_percentile[cat][player_ind,tourn_num-1]
                                                                if tourn_num>0 else np.nan
                                                                for player_ind,tourn_num in zip(data['Player_Index'],
                                                                                                data['tourn_num'])]
        data.sort_values('tourn_num').to_csv('../GolfData/Shot/%d.csv.gz' % (year,), compression='gzip', index=False)

