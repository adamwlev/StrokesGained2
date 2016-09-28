import pandas as pd
import numpy as np
import gc
import multiprocessing

broadie_baseline = {}

for year in range(2004,2017):
	data = pd.read_csv('./../data/%d.csv' % year)
	broadie_baseline.update({tuple(tup[:-1]):tup[-1] for tup in data[['Year','Course_#','Player_#','Hole','Round','Shot','Strokes_Gained/Baseline']].values.tolist()})
	data = None
	gc.collect()

shots_taken_from_location = []
model_prediction = []
broadie_prediction = []
for year in range(2004,2017):
	data = pd.read_csv('./../data_old/%d.csv' % year)
	if year!=2016:
		print year, sum(tuple(tup) not in broadie_baseline for tup in data[['Year','Course_#','Player_#','Hole','Round','Shot']].values.tolist())/float(len(data))
		data.insert(len(data.columns),'Strokes_Gained/Baseline',[broadie_baseline[tuple(tup)] if tuple(tup) in broadie_baseline else np.nan for tup in data[['Year','Course_#','Player_#','Hole','Round','Shot']].values.tolist()])
	data = data[data['Strokes_Gained/Baseline'].notnull()]
	data.insert(len(data.columns),'Difficulty_Start_broadie',[0]*len(data))
	data.loc[data.Shot==data.Hole_Score,'Difficulty_Start_broadie'] = data[data.Shot==data.Hole_Score]['Strokes_Gained/Baseline'] + 1
	data=data.sort_values(['Player_#','Course_#','Round','Hole'])
	for i in range(1,int(data.Hole_Score.max())+1): ##i is hole_score
	    for j in range(i-1,0,-1): ##j is shot
	        data.loc[(data.Hole_Score==i) & (data.Shot==j),'Difficulty_Start_broadie'] = \
	                  data[(data.Hole_Score==i) & (data.Shot==j+1)].Difficulty_Start_broadie.values + \
	                  data[(data.Hole_Score==i) & (data.Shot==j)]['Strokes_Gained/Baseline'].values + 1
	shots_taken_from_location.extend(data.Shots_taken_from_location.values.tolist())
	model_prediction.extend(data.Difficulty_Start.values.tolist())
	broadie_prediction.extend(data.Difficulty_Start_broadie.values.tolist())
	data = None
	gc.collect()

def confidence_interval_mean(its):
    means = []
    for _ in xrange(its):
    	print _
        mean = np.mean(np.random.choice(sample,len(sample)))
        means.append(mean)
    return means

sample = (np.array(shots_taken_from_location) - np.array(model_prediction))**2
print np.mean(sample)

num_cores = multiprocessing.cpu_count()
slices = [50 for _ in range(num_cores)]
pool = multiprocessing.Pool(num_cores)
results = pool.map(confidence_interval_mean, slices)
pool.close()
results = [item for little_list in results for item in little_list]
print np.percentile(results,[(100.0-95)/2,(95+100.0)/2])

sample = (np.array(shots_taken_from_location) - np.array(broadie_prediction))**2
print np.mean(sample)

num_cores = multiprocessing.cpu_count()
slices = [50 for _ in range(num_cores)]
pool = multiprocessing.Pool(num_cores)
pool.close()
results = pool.map(confidence_interval_mean, slices)
results = [item for little_list in results for item in little_list]
print np.percentile(results,[(100.0-95)/2,(95+100.0)/2])

