import numpy as np
import pandas as pd
import json, gc

def doit():
    with open('difficulty.json','r') as json_file:
        results = json.loads(json_file.read())
    results = {tuple(map(int,key[1:-1].split(','))):value for key,value in results.iteritems()}

    id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
    shot_id_cols = id_cols + ['Player_#','Real_Shots']
    for year in range(2003,2018):
        gc.collect()
        data = pd.read_csv('data/%d.csv' % year)
        data['Difficulty_Start'] = np.nan
        data['Difficulty_End'] = np.nan
        tee_difficulty = data[data.from_the_tee_box_mask].groupby(id_cols).Strokes_from_starting_location.mean().to_dict()
        data.loc[data.from_the_tee_box_mask,'Difficulty_Start'] = [tee_difficulty[tuple(tup)]
                                                                   if tuple(tup) in tee_difficulty else np.nan
                                                                   for tup in data[data.from_the_tee_box_mask][id_cols].values]
        data.loc[~data.from_the_tee_box_mask,'Difficulty_Start'] = [results[tuple(tup)]
                                                                    if tuple(tup) in results else np.nan
                                                                    for tup in data[~data.from_the_tee_box_mask][shot_id_cols].values]
        data.loc[data.last_shot_mask,'Difficulty_End'] = 0
        data.loc[~data.last_shot_mask,'Difficulty_End'] = [results[tuple(tup[:-1])+(tup[-1]+1,)]
                                                           if tuple(tup) in results else np.nan
                                                           for tup in data[~data.last_shot_mask][shot_id_cols].values]

        data['Strokes_Gained'] = data.Difficulty_Start - data.Difficulty_End - 1

        print year,len(data)
        data = data.dropna(subset=['Strokes_Gained'])
        print len(data)

        data.to_csv('data/%d.csv' % year,index=False)