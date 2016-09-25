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
        def get_matrix(tups,conditon):
            arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
            for tup in tups:
                year,tournament,round,course,hole = tup
                subset = data.query(condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index']].values
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

        for group,tups in slice:
            for big_cat in meta_cats:
                mat,mat1 = None,None
                for small_cat in meta_cats[big_cat]:
                    condition = 'Year==@year & Permanent_Tournament_==@tournament & Round==@round & Course_==@course & Hole==@hole & ' + cats[small_cat] 
                    try:
                        mat.data
                    except:
                        mat,mat1 = get_matrix(tups,condition)
                        gc.collect()
                    else:
                        res = get_matrix(tups,condition)
                        gc.collect()
                        mat += res[0]
                        mat1 += res[1]
                save_sparse_csc('./../cats_w%g-%g-%g-%g/%s_%d' % (epsilon*100,e_d,e_t,w_d,big_cat,group),mat)
                save_sparse_csc('./../cats_w%g-%g-%g-%g/%s_%d_g' % (epsilon*100,e_d,e_t,w_d,big_cat,group),mat1)
        return

    _,epsilon,e_d,e_t,w_d = sys.argv
    if os.path.isfile('./../cats_w%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d)):
        sys.exit('File already exists.')
    else:
        os.makedirs('./../cats_w%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d))
    epsilon = float(epsilon)/100
    e_d,e_t,w_d = tuple(map(float,[e_d,e_t,w_d]))
    data = pd.concat([pd.read_csv('./../data/%d.csv' % (year)) for year in range(2003,2017)])
    data.columns = [col.replace('#','') for col in data.columns]
    inds = {num:ind for ind,num in enumerate(pd.unique(data.Player_))}
    data.insert(5,'Player_Index',[inds[num] for num in data.Player_])
    data.Time = data.Time.values/100 * 60 + data.Time.values%100
    data = data[['Cat','Year','Round','Permanent_Tournament_','Course_','Hole','Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index','Par_Value']]
    gc.collect()

    with open('./../hole_tups.pkl','r') as pickleFile:
        hole_tups = pickle.load(pickleFile)
    n_players = len(inds)
    n_holes = len(hole_tups)
    n_tournaments = len(pd.DataFrame(np.array(hole_tups))[[0,1]].drop_duplicates())

    bin_size = 4
    window_size = 28
    n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))
    current_group = 0
    tournament_groups = defaultdict(set)
    tournaments = set()
    group_to_tups = {}
    holes_to_inflate = []
    for tup in hole_tups:
       tournaments.add(tuple(tup[0:2]))
       tournament_group = (len(tournaments)-1)/bin_size
       if tournament_group>current_group:
           current_group = tournament_group
           group_to_tups[current_group] = holes_to_inflate
           holes_to_inflate = []
       tournament_groups[current_group].add(tuple(tup[0:2]))
       holes_to_inflate.append(tuple(tup))

    num_cores = multiprocessing.cpu_count()-1
    slices = partition(group_to_tups.items(),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()

