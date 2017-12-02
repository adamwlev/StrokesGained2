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

    cols = ['Strokes_Gained/Baseline','Year','Permanent_Tournament_#','Course_#',
            'Round','Player_#','Strokes_Gained_Category']
    data = pd.concat([pd.read_csv('data/rawdata/shot/%d.txt' % (year,),sep=';',
                      usecols=lambda x: x.strip().replace(' ','_') in cols) for year in range(2004,2019)])
    data.columns = [col.strip().replace(' ','_') for col in data.columns]
    print data.info(null_counts=True)
    data = data.dropna(subset=('Strokes_Gained/Baseline','Strokes_Gained_Category'))
    print pd.unique(data['Strokes_Gained_Category'])
    data.loc[data['Strokes_Gained_Category']=='Approach to the Green','Strokes_Gained_Category'] = 'Approach the Green'
    print pd.unique(data['Strokes_Gained_Category'])
    tourn_map = {}
    for year,ptn in data[['Year','Permanent_Tournament_#']].values:
        if (year,ptn) not in tourn_map:
            tourn_map[(year,ptn)] = len(tourn_map)
    data['tourn_num'] = [tourn_map[(y,ptn)] for y,ptn in data[['Year','Permanent_Tournament_#']].values]
    print len(tourn_map)
    field_for_cat = data.groupby(['Year','Course_#','Round','Strokes_Gained_Category'])
    d = field_for_cat['Strokes_Gained/Baseline'].mean().to_dict()
    data['SG_of_Field'] = [d[tup] for tup in zip(data.Year,data['Course_#'],data.Round,data.Strokes_Gained_Category)]
    data['Strokes_Gained_Broadie'] = data['Strokes_Gained/Baseline'] - data.SG_of_Field

    with open('PickleFiles/num_to_ind_shot.pkl','r') as pickleFile:
        num_to_ind = pickle.load(pickleFile)

    data['Player_Index'] = [num_to_ind[num] if num in num_to_ind else np.nan for num in data['Player_#']]
    print data['Player_Index'].isnull().sum(),data['Player_Index'].isnull().mean()
    data = data.dropna(subset=['Player_Index'])
    n_players = len(num_to_ind)
    players = range(len(num_to_ind))
    n_tourns = data.tourn_num.max()+1

    player_perfs = defaultdict(dict)
    for cat,df in data.groupby('Strokes_Gained_Category'):
        print type(df)
        d = df.groupby(['tourn_num','Player_Index'],as_index=False).agg({'Round':'count',
                                                                         'Strokes_Gained_Broadie':'sum'})
        player_perfs[cat].update({tuple(tup[0:2]):tup[2:] for tup in d.values})

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

    def for_counts(a,beta,window_size=28):
        if not a:
            return np.nan
        counts,sums = zip(*a[max(0,len(a)-window_size):])
        if np.sum(counts)==0:
            return np.nan
        weights = np.array([my_norm(len(counts)-j-1,beta) for j in range(len(counts))])
        return np.sum(np.dot(weights,counts)/np.sum(weights))

    def partition (lst, n):
        return [lst[i::n] for i in xrange(n)]

    def run_a_cat(cats):
        for cat in cats:
            A = np.array([[take_weighted_ave([player_perfs[cat][(tournament,player_ind)] 
                                             if (tournament,player_ind) in player_perfs[cat] else (0.,0.)
                                             for tournament in range(i)],BETA)
                                   for i in range(n_tourns)]
                                  for player_ind in players])
            
            G = np.array([[for_counts([player_perfs[cat][(tournament,player_ind)] 
                                      if (tournament,player_ind) in player_perfs[cat] else (0.,0.)
                                      for tournament in range(i)],BETA)
                                   for i in range(n_tourns)]
                                  for player_ind in players])
            

            np.save('Broadie_Aves/%s_%g.npy' % (cat,BETA),A)
            np.save('Broadie_Aves/%s_%gG.npy' % (cat,BETA),G)

    cats = pd.unique(data.Strokes_Gained_Category)
    num_cores = 4
    slices = partition(cats,num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_cat, slices)
    pool.close()
