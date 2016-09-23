import pandas as pd
import numpy as np
import itertools
import math
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist,squareform
import os,sys
import multiprocessing
import gc
    
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

    def partition (lst, n):
        return [lst[i::n] for i in xrange(n)]

    def run_a_slice(slice):
        def get_matrix(tup,conditon):
            year,tournament,round,course,hole = tup
            subset = data.query(condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index']].values
            arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
            dists = squareform(pdist(subset[:,0:2]))
            inds = [(i,j) for i,j in itertools.product(xrange(len(dists)),xrange(len(dists))) if i!=j and dists[i,j]<epsilon*subset[i,2] and dists[i,j]<epsilon*subset[j,2]]
            for i,j in inds:
                if arr[i,j]!=0:
                    continue
                w = 1/((dists[i,j]/((subset[i,2]+subset[j,2])/2))**e_d + (np.abs(subset[i,4]-subset[j,4])/1000.0)**e_t)
                arr[int(subset[i,5]),int(subset[j,5])] += w/(1.0 + math.exp(subset[j,3]-subset[i,3]))
                arr1[int(subset[i,5]),int(subset[j,5])] += w
            if (arr!=0).sum()==0:
                return
            else:
                mat,mat1 = csc_matrix(arr),csc_matrix(arr1)
                return (mat,mat1)

        def save_sparse_csc(filename,array):
            np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
            return

        for ind,tup in slice:
            for cat in cats:
                condition = 'Year==@year & Permanent_Tournament_==@tournament & Round==@round & Course_==@course & Hole==@hole & ' + cats[cat] 
                mat,mat1 = get_matrix(tuple(tup),condition)
                gc.collect()
                try:
                    mat.data
                except:
                    continue
                else:
                    save_sparse_csc('./../cats_w%g-%g-%g/%s_%d' % (epsilon*100,e_d,e_t,cat,ind),mat)
                    save_sparse_csc('./../cats_w%g-%g-%g/%s_%d_g' % (epsilon*100,e_d,e_t,cat,ind),mat1)
        return

    _,epsilon,e_d,e_t = sys.argv
    if os.path.isfile('./../cats_w%s-%s-%s' % (epsilon,e_d,e_t)):
        sys.exit('File already exists.')
    else:
        os.makedirs('./../cats_w%s-%s-%s' % (epsilon,e_d,e_t))
    epsilon = float(epsilon)/100
    e_d,e_t = float(e_d),float(e_t)
    data = pd.concat([pd.read_csv('./../data/%d.csv' % (year)) for year in range(2003,2017)])
    data.columns = [col.replace('#','') for col in data.columns]
    inds = {num:ind for ind,num in enumerate(pd.unique(data.Player_))}
    data.insert(5,'Player_Index',[inds[num] for num in data.Player_])
    data.Time = data.Time.values/100 * 60 + data.Time.values%100
    n_players = len(inds)
    hole_tups = data[['Year','Permanent_Tournament_','Round','Course_','Hole']].drop_duplicates().reset_index().drop('index',axis=1).T.to_dict('list').items()
    hole_tups = sorted(hole_tups)
    print len(hole_tups)
    with open('./../cats%g/key_file.csv' % (epsilon*100,),'w') as keyFile:
        for tup in hole_tups:
            keyFile.write(','.join(map(str,[tup[0]] + tup[1])) + '\n')

    num_cores = multiprocessing.cpu_count()-1
    slices = partition(hole_tups,num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()

