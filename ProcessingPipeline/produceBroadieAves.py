import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from scipy.stats import norm
import sys
import multiprocessing

if __name__=='__main__':

    _,BETA = sys.argv
    BETA = float(BETA)

    def convert_adam_cats(cat,dist,par):
        if cat=="Fringe" or cat=="Green":
            if dist<5:
                return 'green0'
            elif dist<10:
                return 'green5'
            elif dist<20:
                return 'green10'
            else:
                return 'green20'
        if cat=="Intermediate Rough" or cat=="Primary Rough":
            if dist<90:
                return 'rough0'
            elif dist<375:
                return 'rough90'
            else:
                return 'rough375'
        if cat=="Fairway":
            if dist<300:
                return 'fairway0'
            elif dist<540:
                return 'fairway300'
            else:
                return 'fairway540'
        if cat=="Tee Box":
            if par==3:
                return 'tee3'
            else:
                return 'tee45'
        if cat=="Bunker":
            return 'bunker'
        if cat=="Other":
            return 'other'    

    data = pd.concat([pd.read_csv('./../data/2003.csv')[['Year','Permanent_Tournament_#','Distance_from_hole',
                                                         'Course_#','Round','Hole','Player_#','Cat','Par_Value']]] +
                     [pd.read_csv('./../data/%d.csv' % year)[['Strokes_Gained/Baseline','Year','Permanent_Tournament_#','Distance_from_hole',
                                                              'Course_#','Round','Hole','Player_#','Cat','Par_Value']] for year in range(2004,2017)])
    
    with open('./../PickleFiles/tourn_order.pkl','r') as pickleFile:
        tourn_order = pickle.load(pickleFile)

    data = pd.concat([data[(data.Year==year) & (data['Permanent_Tournament_#']==tourn)] for year,tourn in tourn_order])
    tups = data.drop_duplicates(['Year','Permanent_Tournament_#'])[['Year','Permanent_Tournament_#']].values.tolist()
    tournament_groups = {tuple(tup):u/4 for u,tup in enumerate(tups)}
    data.insert(len(data.columns),'Tournament_Group',[tournament_groups[tuple(tup)] for tup in data[['Year','Permanent_Tournament_#']].values.tolist()])
    n_tournament_groups = len(pd.unique(data.Tournament_Group))

    data = data[data['Strokes_Gained/Baseline'].notnull()]
    data.insert(len(data.columns),'Adam_cat',[convert_adam_cats(cat,dist,par) for cat,dist,par in zip(data.Cat,data.Distance_from_hole,data.Par_Value)])
    field_for_cat = data.groupby(['Year','Course_#','Round','Adam_cat'])
    d = field_for_cat['Strokes_Gained/Baseline'].mean().to_dict()
    data.insert(len(data.columns),'SG_of_Field',[d[tup] for tup in zip(data.Year,data['Course_#'],data.Round,data.Adam_cat)])
    data.insert(len(data.columns),'Strokes_Gained_Broadie',data['Strokes_Gained/Baseline']-data.SG_of_Field)

    with open('./../PickleFiles/num_to_ind_shot.pkl','r') as pickleFile:
        num_to_ind = pickle.load(pickleFile)

    data.insert(5,'Player_Index',[num_to_ind[num] for num in data['Player_#']])
    n_players = len(num_to_ind)
    players = range(len(num_to_ind))

    player_perfs = defaultdict(dict)
    for cat,df in data.groupby('Adam_cat'):
        d = df.groupby(['Tournament_Group','Player_Index'],as_index=False).agg({'Round':'count','Strokes_Gained_Broadie':'sum'})
        player_perfs[cat].update({tuple(tup[0:2]):tup[2:] for tup in d.values.tolist()})

    def my_norm(x,BETA):
        return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

    def take_weighted_ave(a,beta,window_size=28):
        if not a:
            return np.nan
        counts,sums = zip(*a[max(0,len(a)-window_size):])
        if np.sum(counts)==0:
            return np.nan
        weights = np.array([my_norm(len(counts)-j-1,beta) for j in range(len(counts))])
        return np.sum(np.dot(weights,sums)/np.sum(np.dot(weights,counts)))

    def partition (lst, n):
        return [lst[i::n] for i in xrange(n)]

    def run_a_cat(cat):
        cat = cat[0]
        A = np.array([[take_weighted_ave([player_perfs[cat][(tournament_group,player_ind)] 
                                                  if (tournament_group,player_ind) in player_perfs[cat] else (0.,0.)
                                                  for tournament_group in range(i)],BETA)
                               for i in range(n_tournament_groups)]
                              for player_ind in players])
        np.save('./../Broadie_Aves/%s_%g.npy' % (cat,BETA),A)

    cats = pd.unique(data.Adam_cat).tolist()
    num_cores = len(cats)
    slices = partition(cats,num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_cat, slices)
    pool.close()

        
