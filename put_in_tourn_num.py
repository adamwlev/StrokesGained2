import pandas as pd

if __name__=='__main__':
	cols = ('Year','Permanent_Tournament_#')
	data = pd.concat([pd.read_csv('data/%d.csv' % year,usecols=cols) for year in range(2003,2018)])

	cols = ('Year','Permanent_Tournament_#')
	rawdata = pd.concat([pd.read_csv('data/rawdata/shot/%d.txt' % year, sep=';', 
	                                 usecols=lambda x: x.strip().replace(' ','_') in cols)
	                     for year in range(2003,2018)])
	tourn_order = rawdata.drop_duplicates().values.tolist()

	data.columns = [col.replace('#','') for col in data.columns]
	tourns_in_data = data[['Year','Permanent_Tournament_']].drop_duplicates().values.tolist()
	tourns_in_data = set(tuple(tup) for tup in tourns_in_data)
	tourn_order = [tup for tup in tourn_order if tuple(tup) in tourns_in_data]
	tourn_seq = {tuple(tup):u for u,tup in enumerate(tourn_order)}

	for year in range(2017,2018):
		data = pd.read_csv('data/%d.csv' % (year,))
		data['tourn_num'] = [tourn_seq[tuple(tup)] for tup in data[['Year','Permanent_Tournament_#']].values]
		data.to_csv('data/%d.csv' % (year,), index=False)
		data.to_csv('data/%d.csv.gz' % (year,), compression='gzip', index=False)