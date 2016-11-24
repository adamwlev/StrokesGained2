from scipy.sparse import bmat
from scipy.stats import norm

def my_norm(x,BETA):
    return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

def return_mats(beta):
	n_tournament_groups = 156

	A = bmat([[bmat([[load_sparse_csc('./../rounds/%dA.npz' % i)*my_norm(i-k,BETA)] 
                 for k in range(0,n_tournament_groups)],format='csc') 
              for i in range(0,n_tournament_groups)]],format='csc')

	G = bmat([[bmat([[load_sparse_csc('./../rounds/%dG.npz' % i)*my_norm(i-k,BETA)] 
                 for k in range(0,n_tournament_groups)],format='csc') 
              for i in range(0,n_tournament_groups)]],format='csc')

	return (A,G)