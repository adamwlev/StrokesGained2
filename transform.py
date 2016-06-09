import pandas as pd
import numpy as np
import eliminate_holes_with_issues as e

year=2003
data = e.make_df(year)

hole_locations = {}
tee_box_locations = {}
z_of_closest = {}

f = open('hole_locations.csv','r')
for line in f:
	hole_locations[tuple(map(int,line.split('-')[0].strip().split(',')))] = tuple(map(float,line.split('-')[1].strip().split(',')))
f.close()

f = open('tee_box_locations.csv','r')
for line in f:
	tee_box_locations[tuple(map(int,line.split('-')[0].strip().split(',')))] = tuple(map(float,line.split('-')[1].strip().split(',')))
f.close()

f = open('z_of_closest.csv','r')
for line in f:
	z_of_closest[tuple(map(int,line.split('-')[0].strip().split(',')))] = float(line.split('-')[1].strip())
f.close()

print z_of_closest

data.insert(len(data.columns),'Shots_taken_after',data.Hole_Score-data.Shot)
data.sort_values('Shots_taken_after')
cols = ['Year','Course_#','Round','Hole']
print data[data.Shots_taken_after==0][cols].as_matrix().tolist()
data.insert(len(data.columns),'Went_to_X',[hole_locations[tuple(map(int,tup))][0] for tup in data[data.Shots_taken_after==0][cols].as_matrix().tolist()]
                +data[data.Shots_taken_after>0].X_Coordinate.tolist())
data.insert(len(data.columns),'Went_to_Y',[hole_locations[tuple(map(int,tup))][1] for tup in data[data.Shots_taken_after==0][cols].as_matrix().tolist()]
                +data[data.Shots_taken_after>0].Y_Coordinate.tolist())
data.insert(len(data.columns),'Went_to_Z',[z_of_closest[tuple(map(int,tup))] for tup in data[data.Shots_taken_after==0][cols].as_matrix().tolist()]
                +data[data.Shots_taken_after>0].Z_Coordinate.tolist())
data.sort_values('Shot')
cols2 = ['Course_#','Player_#','Hole','Round','Shot']
data.insert(len(data.columns),'Came_from_X',[tee_box_locations[tuple(tup)][0] for tup in data[data.Shot==1][cols].as_matrix().tolist()]
                +[data[(data['Course_#']==tup[0]) & (data['Player_#']==tup[1]) & (data['Hole']==tup[2])
                   & (data['Round']==tup[3]) & (data['Shot']==tup[4]-1)].X_Coordinate for tup in data[data.Shot!=1][cols2].as_matrix().tolist()])
data.insert(len(data.columns),'Came_from_Y',[tee_box_locations[tuple(tup)][1] for tup in data[data.Shot==1][cols].as_matrix().tolist()]
                +[data[(data['Course_#']==tup[0]) & (data['Player_#']==tup[1]) & (data['Hole']==tup[2])
                   & (data['Round']==tup[3]) & (data['Shot']==tup[4]-1)].Y_Coordinate for tup in data[data.Shot!=1][cols2].as_matrix().tolist()])
data.insert(len(data.columns),'Came_from_Z',[np.nan for _ in xrange(len(data[data.Shot==1]))]
                +[data[(data['Course_#']==tup[0]) & (data['Player_#']==tup[1]) & (data['Hole']==tup[2])
                   & (data['Round']==tup[3]) & (data['Shot']==tup[4]-1)].Z_Coordinate for tup in data[data.Shot!=1][cols2].as_matrix().tolist()])


