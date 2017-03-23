import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import math
import pickle

data = pd.read_csv('./../data/round.csv')
data = data.loc[data['Permanent_Tournament_#']!=470] ## this is the match play championship, no round scores available

with open('./../PickleFiles/num_to_ind_round.pkl','r') as pickleFile:
    num_to_inds = pickle.load(pickleFile)

data.insert(5,'Player_Index',[num_to_inds[num] for num in data.Player_Number])
tups = data.drop_duplicates(['Tournament_Year','Permanent_Tournament_#'])[['Tournament_Year','Permanent_Tournament_#']].values.tolist()
tournament_groups = {tuple(tup):u/4 for u,tup in enumerate(tups)}
data.insert(len(data.columns),'Tournament_Group',[tournament_groups[tuple(tup)] 
                                                  for tup in data[['Tournament_Year','Permanent_Tournament_#']].values.tolist()])
n_tournament_groups = len(pd.unique(data.Tournament_Group))
n_players = len(pd.unique(data.Player_Index))

def save_sparse_csc(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def sigmoid(x,m,r):
    return (1/(1 + np.exp(m)**(-x)) + (np.tanh(r*x) + 1)/2)/2

def save_mats(m,r):
	for (year,tourn_group),df in data.groupby(['Tournament_Year','Tournament_Group'],sort=False):
	    A,G = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
	    
	    for (tourn,round),df_ in df.groupby(['Course_#','Round_Number']):
	        A[np.ix_(df_.Player_Index.values,df_.Player_Index.values)] += \
	                sigmoid(np.subtract.outer(df_.Round_Score.values,df_.Round_Score.values),m,r)
	        G[np.ix_(df_.Player_Index.values,df_.Player_Index.values)] += .5

	    np.fill_diagonal(A,0)
	    np.fill_diagonal(G,0)
	    A = csc_matrix(A)
	    G = csc_matrix(G)
	    save_sparse_csc('./../rounds/%dA' % (tourn_group),A)
	    save_sparse_csc('./../rounds/%dG' % (tourn_group),G)