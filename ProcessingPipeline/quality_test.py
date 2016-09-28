import pandas as pd
import numpy as np
from collections import defaultdict


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    #added = d1_keys - d2_keys
    #removed = d2_keys - d1_keys
    #modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    different = set(o for o in intersect_keys if d1[o] != d2[o])
    return len(same), len(different), different


for year in range(2015,2016):
	print year

	data = pd.read_csv('./../data/%d.csv' % (year,)) 

	id_cols = ['Course_#','Player_#','Hole','Round','Shot']

	unique_shots = set(tuple(tup) for tup in data[id_cols].values.astype(int).tolist())

	try:
		assert len(data) == len(unique_shots)
	except:
		print 'Discrepency'
		print len(data), len(unique_shots)
	else:
		print 'Passed Unique C-H-R-S combinations test.'


	cols = ['Course_#','Player_#','Round','Hole']

	hole_scores = {tuple(i[0:4]): range(1,i[4]+1) for i in data[cols+['Hole_Score']].values.astype(int).tolist()}
	shots_in_data = defaultdict(list)
	for i in data[cols+['Shot']].values.astype(int).tolist():
		shots_in_data[tuple(i[0:4])].append(i[4])
		shots_in_data[tuple(i[0:4])].sort()


	try:
		assert hole_scores==dict(shots_in_data)
	except:
		res = dict_compare(hole_scores,dict(shots_in_data))
		print res
		print 'Dropping these %d Player-Holes.' % res[1]
		to_drop = np.array([tuple(tup) in res[2] for tup in data[['Course_#','Player_#','Round','Hole']].values.tolist()])
		data = data[~to_drop]
		data.to_csv('./../data/%d.csv' % year,index=False)
	else:
		print 'Passed completeness and non duplicate shots test.'