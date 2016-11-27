import pandas as pd
import numpy as np
import itertools
import math
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist,squareform
import os,sys
import multiprocessing
import gc
import pickle
from collections import defaultdict
    
if __name__=='__main__':
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
        def get_matrix(tournament_group,conditon):
            arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
            for (round,course,hole),df in data[data.Tournament_Group==tournament_group].groupby(['Round','Course_','Hole']):
                subset = df.query(condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index']].values
                dists = squareform(pdist(subset[:,0:2]))
                inds = [(i,j) for i,j in itertools.product(xrange(len(dists)),xrange(len(dists))) if i!=j and dists[i,j]<epsilon*subset[i,2] and dists[i,j]<epsilon*subset[j,2]]
                for i,j in inds:
                    w_1 = 1/(dists[i,j]/((subset[i,2]+subset[j,2])/2) + .05)**e_d
                    w_2 = 1/((np.abs(subset[i,4]-subset[j,4])+5)/100.0)**e_t
                    w = w_1*w_d + w_2*(1-w_d)
                    arr[int(subset[i,5]),int(subset[j,5])] += w/(1.0 + math.exp(subset[j,3]-subset[i,3]))
                    arr1[int(subset[i,5]),int(subset[j,5])] += w
            mat,mat1 = csc_matrix(arr),csc_matrix(arr1)
            return (mat,mat1)

        def save_sparse_csc(filename,array):
            np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
            return

        for tournament_group in slice:
            for big_cat in meta_cats:
                if os.path.exists('./../cats/cats_w%g-%g-%g-%g/%s_%d.npz' % (epsilon*100,e_d,e_t,w_d,big_cat,tournament_group)):
                    continue
                mat,mat1 = None,None
                for small_cat in meta_cats[big_cat]:
                    condition = cats[small_cat] 
                    try:
                        mat.data
                    except:
                        mat,mat1 = get_matrix(tournament_group,condition)
                        gc.collect()
                    else:
                        res = get_matrix(tournament_group,condition)
                        gc.collect()
                        mat += res[0]
                        mat1 += res[1]
                save_sparse_csc('./../cats/cats_w%g-%g-%g-%g/%s_%d' % (epsilon*100,e_d,e_t,w_d,big_cat,tournament_group),mat)
                save_sparse_csc('./../cats/cats_w%g-%g-%g-%g/%s_%d_g' % (epsilon*100,e_d,e_t,w_d,big_cat,tournament_group),mat1)
                cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/cats/cats_w%g-%g-%g-%g ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/cats/" % (epsilon*100,e_d,e_t,w_d)
                os.system(cmd)
        return

    _,epsilon,e_d,e_t,w_d = sys.argv
    if not os.path.exists('./../cats/cats_w%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d)):
        os.makedirs('./../cats/cats_w%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d))
    epsilon = float(epsilon)/100
    e_d,e_t,w_d = tuple(map(float,[e_d,e_t,w_d]))
    data = pd.concat([pd.read_csv('./../data/%d.csv' % (year)) for year in range(2003,2017)])
    data.columns = [col.replace('#','') for col in data.columns]
    
    with open('./../PickleFiles/num_to_ind_shot.pkl','r') as pickleFile:
        num_to_ind = pickle.load(pickleFile)

    with open('./../PickleFiles/tourn_order.pkl','r') as pickleFile:
        tourn_order = pickle.load(pickleFile)

    data.insert(5,'Player_Index',[num_to_ind[num] for num in data.Player_])
    n_players = len(num_to_ind)
    data.Time = data.Time.values/100 * 60 + data.Time.values%100
    data = data[['Cat','Year','Round','Permanent_Tournament_','Course_','Hole','Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index','Par_Value']]
    gc.collect()
    
    data = pd.concat([data[(data.Year==year) & (data.Permanent_Tournament_==tourn)] for year,tourn in tourn_order])
    tups = data.drop_duplicates(['Year','Permanent_Tournament_'])[['Year','Permanent_Tournament_']].values.tolist()
    tournament_groups = {tuple(tup):u/4 for u,tup in enumerate(tups)}
    data.insert(len(data.columns),'Tournament_Group',[tournament_groups[tuple(tup)] for tup in data[['Year','Permanent_Tournament_']].values.tolist()])
    n_tournament_groups = len(pd.unique(data.Tournament_Group))

    num_cores = 8
    slices = partition(range(n_tournament_groups),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()

