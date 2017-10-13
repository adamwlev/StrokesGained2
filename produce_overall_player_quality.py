import pandas as pd
import numpy as np
from scipy.stats import norm
import sys, json

def main(beta):
	cols = ['Player_#','Hole_Score','Par_Value','Course_#','Permanent_Tournament_#','Year','Hole','Round']
	data = pd.concat([pd.read_csv('data/rawdata/hole/%d.txt' % year,sep=';',
								  usecols=lambda x: x.strip().replace(' ','_') in cols)
	                  for year in range(2003,2019)])
	data.columns = [col.strip().replace(' ','_') for col in data.columns]
	data = data.drop_duplicates(['Player_#','Course_#','Permanent_Tournament_#','Year','Hole','Round'])
	data.Hole_Score = pd.to_numeric(data.Hole_Score,errors='coerce')
	data = data.dropna(subset=['Hole_Score'])
	id_cols = ['Course_#','Permanent_Tournament_#','Year','Hole','Round']
	stroke_ave = data.groupby(id_cols)['Hole_Score'].mean().to_dict()
	data['SG'] = [stroke_ave[tuple(tup)]-hole_score
	              for hole_score,tup in zip(data.Hole_Score,data[id_cols].values)]
	tourn_num = []
	tourn_num_dict = {}
	for tup in data[['Permanent_Tournament_#','Year']].values: 
	    if tuple(tup) not in tourn_num_dict:
	        tourn_num_dict[tuple(tup)] = len(tourn_num_dict) + 1
	    tourn_num.append(tourn_num_dict[tuple(tup)])
	data['tourn_num'] = tourn_num
	def my_norm(x,BETA):
	    return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)
	def normalized_performace(current_tourn_num,df):
	    when = current_tourn_num-df.tourn_num.values
	    weights = my_norm(when,normalized_performace.BETA)
	    return np.dot(df.SG.values,weights)/np.sum(weights)
	def number_observations(current_tourn_num,df):
	    when = current_tourn_num-df.tourn_num.values
	    weights = my_norm(when,normalized_performace.BETA)
	    return np.sum(weights)
	normalized_performace.BETA = beta
	results = {}
	for player,df in data.groupby('Player_#'):
	    for perm_tourn_num,year in sorted(tourn_num_dict,key=tourn_num_dict.get):
	        tourn_num = tourn_num_dict[(perm_tourn_num,year)]
	        for par_value in [3,4,5]:
	            sub = df.loc[(df.tourn_num<tourn_num) & (df.Par_Value==par_value),['tourn_num','SG']]
	            results[(player,perm_tourn_num,year,par_value)] = (normalized_performace(tourn_num,sub),
	                                                               number_observations(tourn_num,sub))
	results = {'%d-%d-%d-%d' % tuple(map(int,key)):tuple(map(float,value)) for key,value in results.iteritems()}
	with open('PickleFiles/overall_player_quality_%d.json' % (int(beta),),'w') as json_file:
		json_file.write(json.dumps(results))

	# for player,df in data.groupby('Player_#'):
 #        tourn_num = len(tourn_num_dict)+1
 #        for par_value in [3,4,5]:
 #            sub = df.loc[(df.Par_Value==par_value),['tourn_num','SG']]
 #            results[(player,par_value)] = (normalized_performace(tourn_num,sub),
 #                                           number_observations(tourn_num,sub))
	# with open('PickleFiles/overall_player_quality.pkl','wb') as pickle_file:
	# 	pickle.dump(results,pickle_file)

if __name__=='__main__':
	_,beta = sys.argv
	beta = float(beta)
	main(beta)