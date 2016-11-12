import numpy as np
import pandas as pd
import pickle

d = {}
for year in range(2003,2017):
	data = pd.read_csv('../data/rawdata/%d.txt' % year, sep = ';')
	data.columns = np.array([str(i).strip().replace(' ','_') for i in list(data.columns.values)])
	data.Hole_Score = np.array([str(i).strip() for i in data.Hole_Score.values])
	data.Hole_Score = pd.to_numeric(data.Hole_Score)
	data.Shot = pd.to_numeric(data.Shot)
	df = data[data.Shot==1]
	for tup,df_ in df.groupby(['Year','Course_#','Hole','Round']):
		d[tup] = df_.Hole_Score.mean()

with open('TeeDifficulty.pkl','w') as pickleFile:
	pickle.dump(d,pickleFile)

