import pandas as pd
from initial_processing import doit as p1
from put_in_green_to_work_with import doit as p2
from produce_difficulty import doit as p3
from add_skill_estimates import doit as p4
import gc

def doit(years,fit_model=False):
	for year in years:
		data = pd.read_csv('data/rawdata/%d.txt' % year,sep=';')
		data = p1(data)
		data = p2(data)
		data.to_csv('data/%d.csv' % year, index=False)
	data = pd.concat([pd.read_csv('data/%d.csv' % year, 
	                              usecols=['Year','Course_#','Permanent_Tournament_#','Round','Hole','Player_#',
	                                       'Start_X_Coordinate','End_X_Coordinate',
	                                       'Start_Y_Coordinate','End_Y_Coordinate',
	                                       'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
	                                       'Strokes_from_starting_location','Cat','Distance_from_hole',
	                                       'Green_to_work_with','Real_Shots'])
	                  for year in years])
	data = p3(data,full=fit_model)
	shot_id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole','Player_#','Real_Shots']
	data = data.drop(['Start_X_Coordinate','End_X_Coordinate',
	                  'Start_Y_Coordinate','End_Y_Coordinate',
	                  'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
	                  'Strokes_from_starting_location','Cat','Distance_from_hole','Green_to_work_with'],axis=1)
	gc.collect()
	difficulty_start = {tuple(tup[:-1]):tup[-1] for tup in data[shot_id_cols+['Difficulty_Start']].values}
	difficulty_end = {tuple(tup[:-1]):tup[-1] for tup in data[shot_id_cols+['Difficulty_End']].values}
	strokes_gained = {tuple(tup[:-1]):tup[-1] for tup in data[shot_id_cols+['Strokes_Gained']].values}
	data = None
	gc.collect()

	for year in years:
		data = pd.read_csv('data/%d.csv' % (year,))
		data['Difficulty_Start'] = [difficulty_start[tuple(tup)] for tup in data[shot_id_cols].values]
		data['Difficulty_End'] = [difficulty_end[tuple(tup)] for tup in data[shot_id_cols].values]
		data['Strokes_Gained'] = [strokes_gained[tuple(tup)] for tup in data[shot_id_cols].values]
		cols = ('Cat','Year','Round','Permanent_Tournament_#','Course_#','Hole','Start_X_Coordinate',
		        'Start_Y_Coordinate','Distance_from_hole','Strokes_Gained','Time','Par_Value')
		data = p4(data[cols])
		data.to_csv('data/%d.csv' % (year,))