import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist,squareform
import math,os,sys,multiprocessing,gc,pickle,itertools
from collections import defaultdict
    
def doit(data,e_d,e_t,w_d,m,r):
    cats = {}
    cats['green0'] = 'Cat=="Green" & Distance_from_hole<5'
    cats['fringe0'] = 'Cat=="Fringe" & Distance_from_hole<5'
    cats['green5'] = 'Cat=="Green" & Distance_from_hole>=5 & Distance_from_hole<10'
    cats['fringe5'] = 'Cat=="Fringe" & Distance_from_hole>=5 & Distance_from_hole<10'
    cats['green10'] = 'Cat=="Green" & Distance_from_hole>=10 & Distance_from_hole<20'
    cats['fringe10'] = 'Cat=="Fringe" & Distance_from_hole>=10 & Distance_from_hole<20'
    cats['green20'] = 'Cat=="Green" & Distance_from_hole>=20'
    cats['fringe20'] = 'Cat=="Fringe" & Distance_from_hole>=20'
    cats['prough0'] = 'Cat=="Primary Rough" & Distance_from_hole<90'
    cats['irough0'] = 'Cat=="Intermediate Rough" & Distance_from_hole<90'
    cats['prough90'] = 'Cat=="Primary Rough" & Distance_from_hole>=90 & Distance_from_hole<375'
    cats['irough90'] = 'Cat=="Intermediate Rough" & Distance_from_hole>=90 & Distance_from_hole<375'
    cats['prough375'] = 'Cat=="Primary Rough" & Distance_from_hole>=375'
    cats['irough375'] = 'Cat=="Intermediate Rough" & Distance_from_hole>=375'
    cats['fairway0'] = 'Cat=="Fairway" & Distance_from_hole<300'
    cats['fairway300'] = 'Cat=="Fairway" & Distance_from_hole>=300 & Distance_from_hole<540'
    cats['fairway540'] = 'Cat=="Fairway" & Distance_from_hole>=540'
    cats['bunker'] = 'Cat=="Bunker"'
    cats['tee3'] = 'Cat=="Tee Box" & Par_Value==3'
    cats['tee45'] = 'Cat=="Tee Box" & (Par_Value==4 | Par_Value==5)'
    cats['other'] = 'Cat=="Other"'

    meta_cats = {}
    meta_cats['tee3'] = ['tee3']
    meta_cats['tee45'] = ['tee45']
    meta_cats['green0'] = ['green0','fringe0']
    meta_cats['green5'] = ['green5','fringe5']
    meta_cats['green10'] = ['green10','fringe10']
    meta_cats['green20'] = ['green20','fringe20']
    meta_cats['rough0'] = ['prough0','irough0']
    meta_cats['rough90'] = ['prough90','irough90']
    meta_cats['rough375'] = ['prough375','irough375']
    meta_cats['fairway0'] = ['fairway0']
    meta_cats['fairway300'] = ['fairway300']
    meta_cats['fairway540'] = ['fairway540']
    meta_cats['bunker'] = ['bunker']
    meta_cats['other'] = ['other']

    def partition (lst, n):
        return [lst[i::n] for i in xrange(n)]

    def run_a_slice(slice):
        def sigmoid(x):
            return (1./(1. + np.exp(m)**(-x)) + (np.tanh(r*x) + 1.)/2.)/2.

        def get_matrix(tournament,conditon):
            arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
            for (round,course,hole),df in data[data.tourn_num==tournament].groupby(['Round','Course_','Hole']):
                subset = df.query(condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index']].values
                num_shots = subset.shape[0]
                dists = squareform(pdist(subset[:,0:2]))
                w_1 = w_1 = 1/(dists/(np.add.outer(subset[:,2],subset[:,2])/2) + .01)**e_d
                w_2 = 1/((np.abs(np.subtract.outer(subset[:,4]-subset[:,4]))+5)/100.0)**e_t
                w = w_1*w_d + w_2*(1-w_d)
                inds_ = np.arange(num_shots**2)
                inds = np.arange(num_shots**2)[inds_%num_shots!=0]
                vals = np.squeeze(sigmoid(np.subtract.outer(subset[:,3],subset[:,3])).reshape(-1,1))
                vals = vals[inds_%num_shots!=0]
                np.add.at(arr,inds,w*vals)
                np.add.at(arr1,inds,w*.5)
            mat,mat1 = csc_matrix(arr),csc_matrix(arr1)
            return (mat,mat1)

        def save_sparse_csc(filename,array):
            np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
            return

        for tournament in slice:
            tournament += run_a_slice.base_number_tournaments ## for incremental
            for big_cat in meta_cats:
                if os.path.exists('./../cats/cats_w%g-%g-%g/%s_%d.npz' % (e_d,e_t,w_d,big_cat,tournament)):
                    continue
                mat,mat1 = None,None
                for small_cat in meta_cats[big_cat]:
                    condition = cats[small_cat] 
                    try:
                        mat.data
                    except:
                        mat,mat1 = get_matrix(tournament,condition)
                        gc.collect()
                    else:
                        res = get_matrix(tournament,condition)
                        gc.collect()
                        mat += res[0]
                        mat1 += res[1]
                save_sparse_csc('./../cats/cats_w%g-%g-%g-%g/%s_%d' % (epsilon*100,e_d,e_t,w_d,big_cat,tournament_group),mat)
                save_sparse_csc('./../cats/cats_w%g-%g-%g-%g/%s_%d_g' % (epsilon*100,e_d,e_t,w_d,big_cat,tournament_group),mat1)
                #cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/cats/cats_w%g-%g-%g-%g ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/cats/" % (epsilon*100,e_d,e_t,w_d)
                #os.system(cmd)
        return

    if not os.path.exists('../cats/cats_w%s-%s-%s' % (e_d,e_t,w_d)):
        os.makedirs('../cats/cats_w%s-%s-%s' % (e_d,e_t,w_d))
    e_d,e_t,w_d = tuple(map(float,[e_d,e_t,w_d]))

    cols = ('Year','Permanent_Tournament_#')
    data = pd.concat([pd.read_csv('data/rawdata/%d.txt' % year, sep=';', 
                                  usecols=lambda x: x.strip().replace(' ','_') in cols) for year in range(2003,2018)])
    tourn_order = data.drop_duplicates().values.tolist()

    data.columns = [col.replace('#','') for col in data.columns]
    
    with open('./../PickleFiles/num_to_ind_shot.pkl','r') as pickleFile:
        num_to_ind = pickle.load(pickleFile)

    for player_num in data['Player_'].drop_duplicates():
        if player_num not in num_to_ind:
            num_to_ind[player_num] = len(num_to_ind)
    data.insert(5,'Player_Index',[num_to_ind[num] for num in data.Player_])
    n_players = len(num_to_ind)
    data.Time = data.Time.values/100 * 60 + data.Time.values%100
    
    tourns_in_data = data[['Year','Permanent_Tournament_']].drop_duplicates.values.tolist()
    tourns_in_data = set(tuple(tup) for tup in tourns_in_data)
    tourn_order = [tup for tup in tourn_order if tuple(tup) in tourns_in_data]
    tourn_seq = {tuple(tup):u for u,tup in enumerate(tourn_order)}
    data['tourn_num'] = [tourn_seq[tuple(tup)] for tup in data[['Year','Permanent_Tournament_']].values]
    n_tournaments = len(tourn_seq)

    num_cores = multiprocessing.cpu_count()-3
    slices = partition(range(n_tournaments),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()

