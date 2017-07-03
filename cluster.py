import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def cluster(df):
	"""
	Accepts 'closest_to_pin' dataframe which is assumed to have
	columns 'End_X_Coordinate', 'End_Y_Coordinate', and 'Year' and returns an 
	array of cluster assignments for each row (round).
	"""
	ks = []
	X = df.groupby('Year')[['End_X_Coordinate','End_Y_Coordinate']].median().values
	for _ in range(15):
		n_points = len(df)
		random_points = np.column_stack([np.random.uniform(low=np.amin(X[:,i]),
														   high=np.amax(X[:,i]),
														   size=n_points)
										 for i in range(2)])
		random_inertias,real_inertias = [],[]
		for k in range(1,len(pd.unique(df.Year))+1):
		    kmeans = KMeans(n_clusters=k,n_init=12)
		    kmeans.fit(random_points)
		    random_inertias.append(kmeans.inertia_)
		    kmeans.fit(X)
		    real_inertias.append(kmeans.inertia_)
		diff = np.array(random_inertias) - np.array(real_inertias)
		k = np.arange(1,len(pd.unique(df.Year))+1)[np.where(diff==np.amax(diff))][0]
		ks.append(k)
	k = pd.Series(ks).value_counts().index[0]
	kmeans = KMeans(n_clusters=k,n_init=13)
	kmeans.fit(X)
	labels = dict(zip(pd.unique(df.Year),kmeans.labels_))
	return [labels[yr] for yr in df.Year],kmeans.cluster_centers_

