import pandas as pd
import numpy as np
import itertools
import math
from scipy.sparse import csc_matrix,eye,bmat
from scipy.sparse.linalg import eigs,inv,gmres
from scipy.spatial.distance import pdist,squareform
from scipy.stats import norm
import os,sys
import multiprocessing
import gc
import pickle
from collections import defaultdict

if __name__=='__main__':
    def convert_cats(cat,dist,shot):
        if cat in ['Green Side Bunker','Fairway Bunker']:
            return 'Bunker'
        elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
            return 'Other'
        elif cat=='Fringe' and dist>120:
            return 'Intermediate Rough'
        elif cat=='Tee Box' and shot!=1:
            return 'Fairway'
        else:
            return cat

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
        little_d = defaultdict(dict)
        def get_matrix(tups,conditon):
            arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
            for year,tournament,round,course,hole in tups:
                subset = data.query(condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Time','Player_Index']].values
                if len(subset)<2:
                    continue
                dists = squareform(pdist(subset[:,0:2]))
                inds = [(i,j) for i,j in itertools.product(xrange(len(dists)),xrange(len(dists))) if i!=j and dists[i,j]<epsilon*subset[i,2] and dists[i,j]<epsilon*subset[j,2]]
                for i,j in inds:
                    if arr[i,j]!=0:
                        continue
                    w = 1/((dists[i,j]/((subset[i,2]+subset[j,2])/2))**e_d + (np.abs(subset[i,4]-subset[j,4])/1000.0)**e_t)
                    arr[int(subset[i,5]),int(subset[j,5])] += w/(1.0 + math.exp(subset[j,3]-subset[i,3]))
                    arr1[int(subset[i,5]),int(subset[j,5])] += w
            return (csc_matrix(arr),csc_matrix(arr1))

        for group,tups in slice:
            for cat in cats:
                condition = 'Year==@year & Permanent_Tournament_==@tournament & Round==@round & Course_==@course & Hole==@hole & ' + cats[cat] 
                little_d[group][cat] = get_matrix(tups,condition)
        return little_d

    _,epsilon,e_d,e_t,a,beta = sys.argv

    epsilon = float(epsilon)/100
    e_d,e_t,a,beta = tuple(map(float,[e_d,e_t,a,beta]))
    data = pd.concat([pd.read_csv('./../data/%d.csv' % (year)) for year in range(2003,2004)])
    data.columns = [col.replace('#','') for col in data.columns]
    inds = {num:ind for ind,num in enumerate(pd.unique(data.Player_))}
    data.insert(5,'Player_Index',[inds[num] for num in data.Player_])
    data.insert(len(data.columns),'Cat',[convert_cats(c,d,s) for c,d,s in zip(data['From_Location(Scorer)'],data['Distance_from_hole'],data.Shot)])
    data.Time = data.Time.values/100 * 60 + data.Time.values%100
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

    results = {key:value for little_dict in results for key,value in little_dict.iteritems()}

    cats = {}
    cats['tee3'] = ['tee3']
    cats['tee45'] = ['tee45']
    cats['green0'] = ['green0','fringe0']
    cats['green5'] = ['green5','fringe5']
    cats['green10'] = ['green10','fringe10']
    cats['green20'] = ['green20','fringe20']
    cats['rough0'] = ['prough0','irough0']
    cats['rough90'] = ['prough90','irough90']
    cats['rough375'] = ['prough375','irough375']
    cats['fairway0'] = ['fairway0']
    cats['fairway300'] = ['fairway300']
    cats['fairway540'] = ['fairway540']
    cats['bunker'] = ['bunker']
    cats['other'] = ['other']

    As,Gs = {},{}
    for cat in cats:
        for sub_cat in cat:
            if cat not in A:
                A[cat] = bmat([[results[i][sub_cat][0] for i in range(1,n_tournament_groups)]],format='csc')
                G[cat] = bmat([[results[i][sub_cat][1] for i in range(1,n_tournament_groups)]],format='csc')
            else:
                A[cat] += bmat([[results[i][sub_cat][0] for i in range(1,n_tournament_groups)]],format='csc')
                G[cat] += bmat([[results[i][sub_cat][1] for i in range(1,n_tournament_groups)]],format='csc')
    results = None
    gc.collect()

    def run_a_slice2(slice):
        def my_norm(x,BETA):
            return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

        def alpha(A,a):
            A.data[A.data<1e-6] = 0
            A.data[np.isnan(A.data)]=0
            w,v = eigs(A,k=1,which='LM')
            return a/w[0].real

        def solve(mat,mat_1,a,min_reps,x_guess=None,x_guess1=None):
            mat.data[mat_1.data<1e-6] = 0
            mat_1.data[mat_1.data<1e-6] = 0
            mat.data[np.isnan(mat.data)] = 0
            mat_1.data[np.isnan(mat_1.data)] = 0
            
            S = eye(mat.shape[0],format='csc')-alpha(mat,a)*mat
            w_a = gmres(S,mat.sum(1),x0=x_guess)[0]
            
            S = eye(mat_1.shape[0],format='csc')-alpha(mat_1,a)*mat_1 
            w_g = gmres(S,mat_1.sum(1),x0=x_guess1)[0]
            
            solve.w_a = w_a
            solve.w_g = w_g
            w_a[w_g<min_reps] = 0
            return ((w_a/w_g)[-n_players:],w_g[-n_players:])

        ranks,reps = [],[]
        cat = slice[0]
        if len(cat)==0:
            return
        for group in range(1,n_tournament_groups):
            min_ = max(0,group-window_size)*n_players
            max_ = group*n_players
            A,G = As[cat][min_:max_,min_:max_],Gs[cat][min_:max_,min_:max_]
            res = solve(A,G,a,1)
            ranks.append(res[0])
            reps.append(res[1])
        return (ranks,reps)
    

    num_cores = multiprocessing.cpu_count()-1
    slices = partition(cats.keys(),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice2, slices)
    pool.close()

    if not os.path.exists('./../ranks/ranks-%g-%g-%g-%g-%g' % (epsilon,e_d,e_t,a,beta)):
        os.makedirs('./../ranks/ranks-%g-%g-%g-%g-%g' % (epsilon,e_d,e_t,a,beta))
        
    for u,cat in enumerate(cats.keys()):
        np.save('./../ranks/ranks-%g-%g-%g-%g-%g/%s_ranks.npy' % (epsilon,e_d,e_t,a,beta,cat), np.array(results[u][0]).T)
        np.save('./../ranks/ranks-%g-%g-%g-%g-%g/%s_reps.npy' % (epsilon,e_d,e_t,a,beta,cat), np.array(results[u][1]).T)
    






