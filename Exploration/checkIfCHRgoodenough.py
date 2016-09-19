import pandas as pd
import numpy as np

for year in range(2003,2017):
	data = pd.read_csv('data/%d.csv' % (year,))
	if len(data.groupby(['Course_#','Round','Hole']))!=len(data.groupby(['Permanent_Tournament_#','Course_#','Round','Hole'])):
		print len(data.groupby(['Course_#','Round','Hole'])),len(data.groupby(['Permanent_Tournament_#','Course_#','Round','Hole']))