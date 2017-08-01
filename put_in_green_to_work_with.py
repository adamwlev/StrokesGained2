import pandas as pd
import numpy as np
from math import atan2, radians, pi
import gc

def convert_cats(cat,dist,shot):
    if cat in ['Green Side Bunker','Fairway Bunker']:
        return 'Bunker'
    elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
        return 'Other'
    elif cat=='Fringe' and dist>120:
        return 'Intermediate Rough'
    elif cat=='Tee Box' and shot!=1:
        return 'Fairway'
    else:
        return cat

def wrap(angle):
    if angle > pi:
        return angle - 2*pi
    elif angle < -pi:
        return angle + 2*pi
    else:
        return angle

def get_sub(df,angle,slack):
    bottom, top = angle-slack, angle+slack
    _or = (abs(bottom)>pi) or (abs(top)>pi)
    bottom, top = wrap(bottom), wrap(top)
    return df[(df.angle>bottom) | (df.angle<top)] if _or else df[(df.angle>bottom) & (df.angle<top)]

def doit(data):
    data = data.reset_index(drop=True)
    hole_locs = {}
    for tup,df in data.groupby(['Permanent_Tournament_#','Round','Hole']):
        x,y = df.sort_values('Strokes_from_starting_location').iloc[0][['End_X_Coordinate','End_Y_Coordinate']]
        hole_locs[tuple(tup)] = (x,y)
    cols = ['Permanent_Tournament_#','Round','Hole','Start_X_Coordinate','Start_Y_Coordinate']
    data['Distance_from_hole'] = [((hole_locs[tuple(tup[:-2])][0] - tup[-2])**2 + 
                                   (hole_locs[tuple(tup[:-2])][1] - tup[-1])**2)**.5
                                  for tup in data[cols].values]
    data.insert(len(data.columns),'Cat',[convert_cats(c,d,s) for c,d,s in zip(data['From_Location(Scorer)'],data['Distance_from_hole'],data.Shot)])
    data.insert(len(data.columns),'Green_to_work_with',[np.nan]*len(data))
    grouped = data.groupby(['Course_#','Round','Hole'])
    print len(grouped)
    for i,((course,round,hole),df) in enumerate(grouped):
        if i%300==0:
            print i
        df = df.copy()
        hole_x, hole_y = df[df.Strokes_from_starting_location==1].iloc[0][['End_X_Coordinate','End_Y_Coordinate']]
        df.Start_X_Coordinate = df.Start_X_Coordinate - hole_x
        df.Start_Y_Coordinate = df.Start_Y_Coordinate - hole_y
        df.End_X_Coordinate = df.End_X_Coordinate - hole_x
        df.End_Y_Coordinate = df.End_Y_Coordinate - hole_y
        non_green = df[df.Cat!='Green']
        green = df[df.Cat=='Green']
        green.insert(len(green.columns),'angle',[atan2(y,x) for x,y in zip(green.Start_X_Coordinate,green.Start_Y_Coordinate)])
        work_with = []
        for u,(x,y) in enumerate(zip(non_green.Start_X_Coordinate,non_green.Start_Y_Coordinate)):
            angle = atan2(y,x)
            slack = 23
            sub = get_sub(green,angle,radians(slack))
            c = 0
            # if len(sub)==0:
            #     print green.angle.sort_values().values
            while len(sub)==0:
                #print u,c,len(green),x,y,angle,slack,radians(slack)
                c += 1
                slack += .5
                if c==134:
                    break
                sub = get_sub(green,angle,radians(slack))
            if c==134:
                work_with.append(np.nan)
            else:
                work_with.append(sub.Distance_from_hole.max())
        #assert np.all(data[(data['Course_#']==course) & (data.Round==round) & (data.Hole==hole) & (data.Cat!='Green')].index==non_green.index)
        data.loc[non_green.index,'Green_to_work_with'] = work_with
        #print work_with
    
    print 'Replacing %d nulls with mean.' % len(data[(data.Cat!='Green') & (data.Green_to_work_with.isnull())])
    data.loc[(data.Cat!='Green') & data.Green_to_work_with.isnull(),'Green_to_work_with'] = 36.5

    return data

