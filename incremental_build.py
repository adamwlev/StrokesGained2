import pandas as pd
import numpy as np
from initial_processing import doit as p1
from put_in_green_to_work_with import doit as p2
from produce_difficulty import doit as p3
import gc, subprocess

def doit(years,fit_model=False):
    for year in years:
        data = pd.read_csv('../GolfData/Shot-Raw/%d.txt' % year,sep=';',encoding='latin1')
        data = p1(data)
        data = p2(data,16)
        data.to_csv('../GolfData/Shot/%d.csv.gz' % year, index=False, compression='gzip')
    data = None
    gc.collect()
    data = pd.concat([pd.read_csv('../GolfData/Shot/%d.csv.gz' % year, 
                                  usecols=['Year','Course_#','Permanent_Tournament_#','Round','Hole','Player_#',
                                           'Start_X_Coordinate','End_X_Coordinate',
                                           'Start_Y_Coordinate','End_Y_Coordinate',
                                           'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
                                           'Strokes_from_starting_location','Cat','Distance_from_hole',
                                           'Green_to_work_with','Real_Shots','from_the_tee_box_mask'])
                      for year in years])

    cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']
    id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
    shot_id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole','Player_#','Real_Shots']
    
    results = {}
    for cat in cats:
        print(cat)
        results_ = p3(data[data.Cat==cat].copy(),cat,full=fit_model)
        results = {key:value for d in [results,results_] for key,value in d.items()}
        gc.collect()
        print(len(results))

    for year in years:
        data = pd.read_csv('../GolfData/Shot/%d.csv.gz' % (year,))
        data['Difficulty_Start'] = np.nan
        data['Difficulty_End'] = np.nan
        tee_difficulty = data[data.from_the_tee_box_mask].groupby(id_cols).Strokes_from_starting_location.mean()
        tee_difficulty = tee_difficulty.to_dict()
        data.loc[data.from_the_tee_box_mask,'Difficulty_Start'] = [tee_difficulty[tuple(tup)]
                                                                   if tuple(tup) in tee_difficulty else np.nan
                                                                   for tup in data[data.from_the_tee_box_mask][id_cols].values]
        data.loc[~data.from_the_tee_box_mask,'Difficulty_Start'] = [results[tuple(tup)]
                                                                    if tuple(tup) in results else np.nan
                                                                    for tup in data[~data.from_the_tee_box_mask][shot_id_cols].values]
        data.loc[data.last_shot_mask,'Difficulty_End'] = 0
        data.loc[~data.last_shot_mask,'Difficulty_End'] = [results[tuple(tup[:-1])+(tup[-1]+1,)]
                                                           if tuple(tup[:-1])+(tup[-1]+1,) in results else np.nan
                                                           for tup in data[~data.last_shot_mask][shot_id_cols].values]
        data['Strokes_Gained'] = data.Difficulty_Start - data.Difficulty_End - 1
        data.to_csv('../GolfData/Shot/%d.csv.gz' % (year,),index=False, compression='gzip')

    subprocess.call(['python','fill_empty_dates.py'])
    subprocess.call(['python','put_in_tourn_num.py'])
    subprocess.call(['python','save_blocks.py'])
    subprocess.call(['python','crawler.py'])
    subprocess.call(['python','make_skill_est_percentiles.py'])
