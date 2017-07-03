import pandas as pd
import numpy as np
from cluster import cluster

def doit(year):
	data = pd.read_csv('data/%d.csv' % year)
	results = {}
	for (course,hole),df in data.groupby(['Course_#','Hole']):
		if len(df)==1:
			continue
		closest_to_pin = df.groupby(['Year','Round']).apply(lambda x: x[x.last_shot_mask].sort_values('Distance_from_hole').iloc[0]).reset_index(drop=True)
		assignments, cluster_centers = cluster(closest_to_pin)
		clusters = dict(zip(np.sort(pd.unique(assignments)),cluster_centers))
		results.update({(course,hole) + tuple(tup):(a,clusters[a])
		                for tup,a in zip(closest_to_pin[['Year','Round']].values,assignments)})
	data['Cluster'] = [0]*len(data)
	data['Cluster'] = [results[tuple(tup)][0]
	                   if tuple(tup) in results else 0
	                   for tup in data[['Course_#','Hole','Year','Round']].values]
	data['Cluster_Green_X'] = [results[tuple(tup)][1][0]
	                           if tuple(tup) in results else 0
	                           for tup in data[['Course_#','Hole','Year','Round']].values]
	data['Cluster_Green_Y'] = [results[tuple(tup)][1][1]
	                           if tuple(tup) in results else 0
	                           for tup in data[['Course_#','Hole','Year','Round']].values]
	cluster_tee_box_dict = {index:tuple(row) for index,row in data.groupby(['Course_#','Hole','Cluster']).apply(lambda x: x.loc[x.from_the_tee_box_mask,
																	  						 									['Start_X_Coordinate',
	                                                                   						  									 'Start_Y_Coordinate']].mean()).iterrows()}
	data['Cluster_Tee_X'] = [cluster_tee_box_dict[tuple(tup)][0] for tup in data[['Course_#','Hole','Cluster']].values]
	data['Cluster_Tee_Y'] = [cluster_tee_box_dict[tuple(tup)][1] for tup in data[['Course_#','Hole','Cluster']].values]
	data = data[data.Cluster_Green_X!=0]
	data.to_csv('data/%d.csv' % year, index=False)