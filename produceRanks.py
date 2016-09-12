import pandas as pd
import numpy as np
import math
from scipy.sparse import csc_matrix,csr_matrix,eye,bmat
from scipy.sparse.linalg import eigs,inv,gmres
from scipy.stats import norm
import pickle
from collections import defaultdict
import multiprocessing
import gc
import os,sys

if __name__=='__main__':
	with open('num_to_ind.pkl','r') as pickleFile:
	    num_to_ind = pickle.load(pickleFile)

	with open('hole_tups.pkl','r') as pickleFile:
	    hole_tups = pickle.load(pickleFile)

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

	_,cat,epsilon,a,beta = sys.argv
	epsilon,a,beta = tuple(map(float,[epsilon,a,beta]))

	key = pd.read_csv('cats%g/key_file.csv' % (epsilon,),header=None,index_col=0)
	key_dict = {tuple(value):key for key,value in key.T.to_dict('list').iteritems()}

	n_players = len(inds_to_name)
	n_holes = len(hole_tups)
	n_tournaments = len(pd.DataFrame(np.array(hole_tups))[[0,1]].drop_duplicates())

	def load_sparse_csc(filename):
	    loader = np.load(filename)
	    return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

	def run_a_slice(slice):
	    def inflate(cat,tournament_group,holes_to_inflate,n_tournament_groups,beta,window_size):
	        mat = csc_matrix((n_players*n_tournament_groups,n_players),dtype=float)
	        mat_1 = csc_matrix((n_players*n_tournament_groups,n_players),dtype=float)
	        for j in holes_to_inflate:
	            ind = key_dict[j]
	            for c in cats[cat]:
	                fname = 'cats%g/%s_%d.npz' % (epsilon,c,ind)
	                if not os.path.isfile(fname):
	                	continue
	                mat += bmat([[load_sparse_csc(fname)*my_norm(tournament_group-k,beta)] for k in range(1,n_tournament_groups+1)],format='csc')
	                mat_1 += bmat([[(load_sparse_csc(fname)!=0).astype(float)*my_norm(tournament_group-k,beta)] for k in range(1,n_tournament_groups+1)],format='csc')
	        return {tournament_group:(mat,mat_1)}
	    d = {}
	    for group,tups in slice:
	        d.update(inflate(cat,group,tups,n_tournament_groups,beta,window_size))
	    return d

	def partition (lst, n):
	    return [lst[i::n] for i in xrange(n)]

	bin_size = 4
	window_size = 28
	n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))
	current_group = 0
	tournament_groups=[set()]
	tournaments = set()
	group_to_tups = {}
	holes_to_inflate = []
	for tup in hole_tups:
	    tournaments.add(tuple(tup[0:2]))
	    tournament_group = (len(tournaments)-1)/bin_size
	    if tournament_group>current_group:
	        current_group = tournament_group
	        tournament_groups.append(set())
	        group_to_tups[current_group] = holes_to_inflate
	        holes_to_inflate = []
	    tournament_groups[current_group].add(tuple(tup[0:2]))
	    holes_to_inflate.append(tuple(tup))

	num_cores = multiprocessing.cpu_count()-1
    slices = partition(group_to_tups.items(),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()
    
    results = {key:value for res in results for key,value in res.iteritems()}
    
    A = bmat([[results[i][0] for i in range(1,n_tournament_groups)]],format='csc')
    for i in results:
        results[i] = (None,results[i][1])
    gc.collect()
    G = bmat([[results[i][1] for i in range(1,n_tournament_groups)]],format='csc')
    results = None
    gc.collect()

    def run_a_slice(slice):
    	def my_norm(x,beta):
    	    return norm.pdf(x,0,beta)/norm.pdf(0,0,beta)
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

    	d = defaultdict(list)
    	for group in slice:
    		min_ = max(0,group-window_size)*n_players
    		max_ = group*n_players
    		A_,G_ = A[min_:max_,min_:max_],G[min_:max_,min_:max_]
    		if group==1:
    		    res = solve(A_,G_,a,1)
    		    d[group].append((res[0],res[1]))
    		else:
    		    w_a_approx = np.append(solve.w_a[0 if group<=window_size else n_players:],solve.w_a[-n_players:])
    		    w_g_approx = np.append(solve.w_g[0 if group<=window_size else n_players:],solve.w_g[-n_players:])
    		    res = solve(A_,G_,a,1,w_a_approx,w_g_approx)
    		    d[group].append((res[0],res[1]))
    	return d

	num_cores = multiprocessing.cpu_count()-1
    slices = partition(range(1,n_tournament_groups),num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)
    pool.close()

    results = {key:value for res in results for key,value in res.iteritems()}

    os.makedirs('ranks-%g-%g-%g' % (epsilon,a,beta))
    np.save('ranks-%g-%g-%g/ranks.npy' % (epsilon,a,beta), np.array([res[i][0] for i in range(1,n_tournament_groups)]))
    np.save('ranks-%g-%g-%g/reps.npy' % (epsilon,a,beta), np.array([res[i][1] for i in range(1,n_tournament_groups)]))
