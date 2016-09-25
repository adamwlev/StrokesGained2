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

if __name__=="__main__":
	cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
			'rough375','fairway0','fairway300','fairway540','bunker','other']

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

	def load_sparse_csc(filename):
	    loader = np.load(filename)
	    return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

	with open('./../hole_tups.pkl','r') as pickleFile:
	    hole_tups = pickle.load(pickleFile)

	with open('./../num_to_ind.pkl','r') as pickleFile:
	    num_to_ind = pickle.load(pickleFile)
	
	n_players = len(num_to_ind)
	n_tournaments = len(pd.DataFrame(np.array(hole_tups))[[0,1]].drop_duplicates())

	bin_size = 4
	window_size = 28
	n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))

	_,epsilon,e_d,e_t,w_d,a,beta = sys.argv
	a,beta = tuple(map(float,[a,beta]))
	for cat in cats:
		print cat
		ranks,reps = [],[]
		A = bmat([[bmat([[load_sparse_csc('./../cats/cats_w%s-%s-%s-%s/%s_%d.npz' % (epsilon,e_d,e_t,w_d,cat,group)) * my_norm(abs(i-group),beta)] for i in range(1,n_tournament_groups)]) for group in range(1,n_tournament_groups)]],format='csc')
		G = bmat([[bmat([[load_sparse_csc('./../cats/cats_w%s-%s-%s-%s/%s_%d_g.npz' % (epsilon,e_d,e_t,w_d,cat,group)) * my_norm(abs(i-group),beta)] for i in range(1,n_tournament_groups)]) for group in range(1,n_tournament_groups)]],format='csc')
		for group in range(1,n_tournament_groups):
		    min_ = max(0,group-window_size)*n_players
		    max_ = group*n_players
		    A_,G_ = A[min_:max_,min_:max_],G[min_:max_,min_:max_]
		    if group==1:
		        res = solve(A_,G_,a,1)
		        ranks.append(res[0])
		        reps.append(res[1])
		    else:
		        w_a_approx = np.append(solve.w_a[0 if group<=window_size else n_players:],solve.w_a[-n_players:])
		        w_g_approx = np.append(solve.w_g[0 if group<=window_size else n_players:],solve.w_g[-n_players:])
		        res = solve(A_,G_,a,1,w_a_approx,w_g_approx)
		        ranks.append(res[0])
		        reps.append(res[1])
		np.save('./../ranks/ranks-%s-%s-%s-%s-%g-%g/%s_ranks.npy' % (epsilon,e_d,e_t,w_d,a,beta,cat), np.array(ranks).T)
		np.save('./../ranks/ranks-%s-%s-%s-%s-%g-%g/%s_reps.npy' % (epsilon,e_d,e_t,w_d,a,beta,cat), np.array(reps).T)


	