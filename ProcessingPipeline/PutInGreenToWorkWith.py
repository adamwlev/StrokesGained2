import pandas as pd
import numpy as np
from math import atan2,radians

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

def get_sub(df,angle,slack):
    return df[(df.angle>angle-slack) & (df.angle<angle+slack)]

for year in range(2003,2017):
    print year
    data = pd.read_csv('./../data/%d.csv' % year)
    data = data.reset_index(drop=True)
    data.insert(len(data.columns),'Cat',[convert_cats(c,d,s) for c,d,s in zip(data['From_Location(Scorer)'],data['Distance_from_hole'],data.Shot)])
    data.insert(len(data.columns),'Green_to_work_with',[np.nan]*len(data))
    grouped = data.groupby(['Course_#','Round','Hole'])
    print len(grouped)
    for i,((course,round,hole),df) in enumerate(grouped):
        if i%300==0:
            print i
        non_green = df[df.Cat!='Green']
        green = df[df.Cat=='Green']
        green.insert(len(green.columns),'angle',[atan2(y,x) for x,y in zip(green.Started_at_X,green.Started_at_Y)])
        work_with = []
        for x,y in zip(non_green.Started_at_X,non_green.Started_at_Y):
            angle = atan2(y,x)
            slack = 20
            sub = get_sub(green,angle,radians(slack))
            c = 0
            while len(sub)==0:
                c += 1
                slack += 2
                if c==15:
                    break
                sub = get_sub(green,angle,radians(slack))
            if c==15:
                work_with.append(np.nan)
            else:
                work_with.append(sub.Distance_from_hole.max())
        assert np.all(data[(data['Course_#']==course) & (data.Round==round) & (data.Hole==hole) & (data.Cat!='Green')].index==non_green.index)
        data.loc[non_green.index,'Green_to_work_with'] = work_with
    
    data.loc[data.Green_to_work_with.isnull(),'Green_to_work_with'] = data[data.Green_to_work_with.notnull()].Green_to_work_with.mean()
    data.to_csv('./../data/%d.csv' % year,index=False)

