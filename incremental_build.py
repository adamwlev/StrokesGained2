import pandas as pd
from intitial_processing import doit as p1
from put_in_green_to_work_with import doit as p2
from produce_difficulty import doit as p3

def doit(years,year,path_to_new_file,path_to_old_file):
	old_data = pd.read_csv(path_to_old_file,sep=';')
	new_data = pd.read_csv(path_to_new_file,sep=';')

	records_old = set([tuple(row) for row in old_data.values])
	in_old_mask = np.array([tuple(row) in records_old for row in new_data.values])
	new_data = new_data[~in_old_mask]

	print '%d new rows' % (len(new_data),)

	data = p1(new_data)
	data = p2(data)
	data = p3(data)
	
	current_data = pd.read_csv('data/%d.csv' % year)
	data = pd.concat([current_data,data])
	data.to_csv('data/%d.csv' % year,index=False)


