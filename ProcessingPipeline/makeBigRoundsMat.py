def return_mats(BETA):
	from scipy.sparse import bmat,csc_matrix
	from scipy.stats import norm
	import numpy as np
	import pandas as pd

	def load_sparse_csc(filename):
	    loader = np.load(filename)
	    return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

	def my_norm(x,BETA):
	    return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

	data = pd.read_csv('./../data/round.csv')
	data = data.loc[data['Permanent_Tournament_#']!=470] ## this is the match play championship, no round scores available
	tups = data.drop_duplicates(['Tournament_Year','Permanent_Tournament_#'])[['Tournament_Year','Permanent_Tournament_#']].values.tolist()
	tournament_groups = {tuple(tup):u/4 for u,tup in enumerate(tups)}
	data.insert(len(data.columns),'Tournament_Group',[tournament_groups[tuple(tup)] for tup in data[['Tournament_Year','Permanent_Tournament_#']].values.tolist()])
	n_tournament_groups = len(pd.unique(data.Tournament_Group))

	A = bmat([[bmat([[load_sparse_csc('./../rounds/%dA.npz' % i)*my_norm(i-k,BETA)] 
                 for k in range(n_tournament_groups)],format='csc') 
              for i in range(n_tournament_groups)]],format='csc')

	G = bmat([[bmat([[load_sparse_csc('./../rounds/%dG.npz' % i)*my_norm(i-k,BETA)] 
                 for k in range(n_tournament_groups)],format='csc') 
              for i in range(n_tournament_groups)]],format='csc')

	return (A,G)