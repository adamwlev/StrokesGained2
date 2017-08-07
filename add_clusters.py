import pandas as pd
import numpy as np
from cluster import cluster
import gc,sys

def doit():
	cols = ['Course_#','Round','Hole','last_shot_mask','Distance_from_hole',
	        'Year','End_X_Coordinate','End_Y_Coordinate']
	data = pd.concat([pd.read_csv('data/%d.csv' % year, usecols=cols)
	                  for year in range(2003,2018)])
	results = {}
	grouped = data.groupby(['Course_#','Hole'])
	print len(grouped)
	for u,((course,hole),df) in enumerate(grouped):
		if len(df)==1: print 'Skipping %d' % (u,)
		if u%300==0: print u
		closest_to_pin = df.groupby(['Year','Round']).apply(lambda x: x[x.last_shot_mask].sort_values('Distance_from_hole').iloc[0]).reset_index(drop=True)
		assignments, cluster_centers = cluster(closest_to_pin)
		clusters = dict(zip(np.sort(pd.unique(assignments)),cluster_centers))
		results.update({(course,hole,year,round):(a,clusters[a])
		                for (year,round),a in zip(closest_to_pin[['Year','Round']].values,assignments)})
	
	data = None
	gc.collect()
	for year in range(2003,2018):
		data = pd.read_csv('data/%d.csv' % year)
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
		len_before = len(data)
		data = data[data.Cluster_Green_X!=0]
		print 'dropping %d singleton shots' % (len_before-len(data),)
		data.to_csv('data/%d.csv' % year, index=False)
		data = None
		gc.collect()

	cols = ['Course_#','Hole','from_the_tee_box_mask','Cluster',
	        'Start_X_Coordinate','Start_Y_Coordinate']
	data = pd.concat([pd.read_csv('data/%d.csv' % year, usecols=cols)
	                  for year in range(2003,2018)])
	cluster_tee_box_dict = {index:tuple(row) for index,row in data.groupby(['Course_#','Hole','Cluster'])\
																  .apply(lambda x: x.loc[x.from_the_tee_box_mask,
							  						 									 ['Start_X_Coordinate',
                               						  									  'Start_Y_Coordinate']].mean()).iterrows()}
	data = None
	gc.collect()
	for year in range(2003,2018):
		data = pd.read_csv('data/%d.csv' % year)
		data['Cluster_Tee_X'] = [cluster_tee_box_dict[tuple(tup)][0] for tup in data[['Course_#','Hole','Cluster']].values]
		data['Cluster_Tee_Y'] = [cluster_tee_box_dict[tuple(tup)][1] for tup in data[['Course_#','Hole','Cluster']].values]
		data.to_csv('data/%d.csv' % year, index=False)
		data = None
		gc.collect()
