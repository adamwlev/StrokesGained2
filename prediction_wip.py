import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb

data = pd.concat([pd.read_csv('data/%d.csv' % year, 
                              usecols=['Year','Course_#','Hole','Player_#','Permanent_Tournament_#','Round',
                                       'Start_X_Coordinate','End_X_Coordinate',
                                       'Start_Y_Coordinate','End_Y_Coordinate',
                                       'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
                                       'Strokes_from_starting_location','Cat','Distance_from_hole',
                                       'Green_to_work_with','loc_string','loc_string_hole'])
                  for year in range(2003,2018)])

cats = ['Tee Box','Green','Fairway','Rough','Bunker','Other']
id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
z_of_hole = data[data.last_shot_mask].groupby(id_cols)['End_Z_Coordinate'].max().to_dict()
data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'] - np.array([z_of_hole[tuple(tup)] for tup in data[id_cols].values])
data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'].fillna(0)

data['dist_using_coords'] = ((data.Start_X_Coordinate-data.End_X_Coordinate)**2
                             + (data.Start_Y_Coordinate-data.End_Y_Coordinate)**2)**.5
data['dist_error'] = (data.Distance/12.0 - data.dist_using_coords)/(data.Distance/12.0).replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=['dist_error'])
data = data[~((data.Cat=='Green') & (data.Distance_from_hole>130))]
data = data[data.dist_error.abs()<.1]
data = data.drop(['End_Z_Coordinate','last_shot_mask','dist_using_coords','dist_error',
				  'Permanent_Tournament_#','Round','Distance','Start_X_Coordinate',
				  'End_X_Coordinate','Start_Y_Coordinate','End_Y_Coordinate'],axis=1)
data.loc[data.Cat.isin(('Primary Rough','Intermediate Rough')),'Cat'] = 'Rough'
data.loc[data.Cat=='Fringe','Cat'] = 'Fairway'

