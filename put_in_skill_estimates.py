import pandas as pd
import numpy as np
from scipy.stats import rankdata
import sys, pickle

if __name__=="__main__":
    cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
            'rough375','fairway0','fairway300','fairway540','bunker','other']

    queries = {}
    queries['green0'] = '(Cat=="Green" | Cat=="Fringe") & Distance_from_hole<5'
    queries['green5'] = '(Cat=="Green" | Cat=="Fringe") & Distance_from_hole>=5 & Distance_from_hole<10'
    queries['green10'] = '(Cat=="Green" | Cat=="Fringe") & Distance_from_hole>=10 & Distance_from_hole<20'
    queries['green20'] = '(Cat=="Green" | Cat=="Fringe") & Distance_from_hole>=20'
    queries['rough0'] = '(Cat=="Primary Rough" | Cat=="Intermediate Rough") & Distance_from_hole<90'
    queries['rough90'] = '(Cat=="Primary Rough" | Cat=="Intermediate Rough") & Distance_from_hole>=90 & Distance_from_hole<375'
    queries['rough375'] = '(Cat=="Primary Rough" | Cat=="Intermediate Rough") & Distance_from_hole>=375'
    queries['fairway0'] = 'Cat=="Fairway" & Distance_from_hole<300'
    queries['fairway300'] = 'Cat=="Fairway" & Distance_from_hole>=300 & Distance_from_hole<540'
    queries['fairway540'] = 'Cat=="Fairway" & Distance_from_hole>=540'
    queries['bunker'] = 'Cat=="Bunker"'
    queries['tee3'] = 'Cat=="Tee Box" & Par_Value==3'
    queries['tee45'] = 'Cat=="Tee Box" & (Par_Value==4 | Par_Value==5)'
    queries['other'] = 'Cat=="Other"'

    _,e_d,e_t,w_d,a,beta = sys.argv
    ratings, reps = {},{}
    for cat in cats:
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
        num_to_ind = pickle.load(pickle_file)

    for year in range(2003,2018):
        print year
        data = pd.read_csv('data/%d.csv' % (year,))
        cols_with_hashtags = [col for col in data.columns if '#' in col]
        data = data.rename(columns={col:col.replace('#','') for col in cols_with_hashtags})
        to_c_stack = []
        for cat in cats:
            mask = np.zeros(len(data))
            mask[data.query(queries[cat]).index.values] = 1
            to_c_stack.append(mask)
        cat_dummies = np.column_stack(to_c_stack)
        assert (cat_dummies.sum(1)==1).mean()==1
        cat_map = {u:cat for u,cat in enumerate(cats)}
        data['baby_cat'] = [cat_map[i] for i in cat_dummies.argmax(1)]
        #data = data.rename(columns={col.replace('#',''):col for col in cols_with_hashtags})
        data = data.drop([col for col in data.columns if col.startswith('Unnamed')],axis=1)
        data['Player_Index'] = [num_to_ind[num] for num in data['Player_#']]
        data['skill_estimate'] = [ratings[baby_cat][player_ind,tourn_num-1]
                                  if tourn_num>0 else np.nan
                                  for baby_cat,player_ind,tourn_num in zip(data['baby_cat'],
                                                                           data['Player_Index'],
                                                                           data['tourn_num'])]
        data['observation_count'] = [reps[baby_cat][player_ind,tourn_num-1]
                                     if tourn_num>0 else np.nan
                                     for baby_cat,player_ind,tourn_num in zip(data['baby_cat'],
                                                                              data['Player_Index'],
                                                                              data['tourn_num'])]
        data['not_seen'] = (data['skill_estimate']==0) | (data['observation_count']==0)
        data['skill_estimate_percentile'] = [ratings_percentile[baby_cat][player_ind,tourn_num-1]
                                             if tourn_num>0 else np.nan
                                             for baby_cat,player_ind,tourn_num in zip(data['baby_cat'],
                                                                                      data['Player_Index'],
                                                                                      data['tourn_num'])]
        data['observation_count_percentile'] = [reps_percentile[baby_cat][player_ind,tourn_num-1]
                                                if tourn_num>0 else np.nan
                                                for baby_cat,player_ind,tourn_num in zip(data['baby_cat'],
                                                                                      data['Player_Index'],
                                                                                      data['tourn_num'])]
        data.sort_values('tourn_num').to_csv('data/%d.csv' % (year,), index=False)

