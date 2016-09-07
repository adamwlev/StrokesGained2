import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import os
from scipy.spatial.distance import pdist

cats = {}
cats['green0'] = ['Cat=="Green" & Distance_from_hole<5','Cat=="Fringe" & Distance_from_hole<5']
cats['green5'] = ['Cat=="Green" & Distance_from_hole>=5 & Distance_from_hole<10','Cat=="Fringe" & Distance_from_hole>=5 & Distance_from_hole<10']
cats['green10'] = ['Cat=="Green" & Distance_from_hole>=10 & Distance_from_hole<20','Cat=="Fringe" & Distance_from_hole>=10 & Distance_from_hole<20']
cats['green20'] = ['Cat=="Green" & Distance_from_hole>=20','Cat=="Fringe" & Distance_from_hole>=20']
cats['rough0'] = ['Cat=="Primary Rough" & Distance_from_hole<90','Cat=="Intermediate Rough" & Distance_from_hole<90']
cats['rough90'] = ['Cat=="Primary Rough" & Distance_from_hole>=90 & Distance_from_hole<375','Cat=="Intermediate Rough" & Distance_from_hole>=90 & Distance_from_hole<375']
cats['rough375'] = ['Cat=="Primary Rough" & Distance_from_hole>=375','Cat=="Intermediate Rough" & Distance_from_hole>=375']
cats['fairway0'] = ['Cat=="Fairway" & Distance_from_hole<300']
cats['fairway300'] = ['Cat=="Fairway" & Distance_from_hole>=300 & Distance_from_hole<540']
cats['fairway540'] = ['Cat=="Fairway" & Distance_from_hole>=540']
cats['bunker'] = ['Cat=="Bunker"']
cats['tee3'] = ['Cat=="Tee Box" & Par_Value==3']
cats['tee45'] = ['Cat=="Tee Box" & (Par_Value==4 | Par_Value==5)']

class SaveTheCats(object):
	def __init__(self,epsilon):
		self.epsilon = epsilon
		if os.path.isfile('cats%g' % (epsilon,)):
			pass
		else:
			os.makedirs('cats%g' % (epsilon,))
			self.save_files()

	def save_files(self):
		self.data = pd.concat([pd.read_csv('data/%d.csv' % (year)) for year in range(2003,2017)])
		self.data.columns = [col.replace('#','') for col in self.data.columns]
		inds = {num:ind for ind,num in enumerate(pd.unique(data.Player_Number))}
		data.insert(5,'Player_Index',[inds[num] for num in data.Player_Number])
		self.n_players = len(inds)
		rounds = data.groupby(['Tournament_Year','Permanent_Tournament_','Round_Number','Course_'])
		# n_rounds = len(rounds)
		for round_ind,df in enumerate(rounds):
		    tup,df = df
		    year,tournament,round,course = tup
		    for cat in cats:
		    	A,G = csc_matrix((n_players,n_players)),csc_matrix((n_players,n_players))
		    	for condition in cats[cat]:
		    		self.condition = 'Tournament_Year==@year & Permanent_Tournament_==@tournament & Round_Number==@round & Course_==@course & ' + condition 
		    		A_,G_ = get_matrices()
		    		A += A_
		    		G += G_
			    self.filename,self.array = '%s_%dA' % (cat,round_ind),A
			    self.save_sparse_csc()
			    self.filename,self.array = '%s_%dG' % (cat,round_ind),G
			    self.save_sparse_csc()
		    # if round_ind%300==0:
		    #     print round_ind

	def get_matrices(self):
		subset = self.data.query(self.condition)[['Started_at_X','Started_at_Y','Distance_from_hole','Strokes_Gained','Player_Index']].values
		arr,arr1 = np.zeros((self.n_players,self.n_players)),np.zeros((self.n_players,self.n_players))
		dists = pdist(subset[:,0:2])
		inds = [(i,j) for i,j in zip(xrange(len(subset)),xrange(len(subset))) if  
											i!=j and dists[i,j]<self.epsilon*subset[i,2] and dists[i,j]<self.epsilon*subset[j,2]]
		for i,j in inds:
			arr[subset[i,4],subset[j,4]] += 1.0/(1.0 + math.exp(subset[i,3]-subset[j,3])) + .5
			arr1[subset[i,4],subset[j,4]] += 1.0
		A,G = csc_matrix(arr),csc_matrix(arr1)
		return A,G

	def save_sparse_csc(self):
    	np.savez(self.filename,data=self.array.data,indices=self.array.indices,indptr=self.array.indptr,shape=self.array.shape)



		
		
