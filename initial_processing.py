import numpy as np
import pandas as pd
from impute_locations import Imputer

def find_all(s,substr):
    lens = np.array([len(sub) for sub in s.split(substr)])
    lens[1:] = lens[1:] + len(substr)
    inds = np.cumsum(lens)
    return inds[inds!=len(s)]

def doit(year):
    data = pd.read_csv('data/rawdata/%d.txt' % (year,), sep=';')
    data.columns = [col.strip().replace(' ','_') for col in data.columns]

    data['X_Coordinate'] = [str(i).replace(' ','') for i in data['X_Coordinate']]
    data['Y_Coordinate'] = [str(i).replace(' ','') for i in data['Y_Coordinate']]
    data['Z_Coordinate'] = [str(i).replace(' ','') for i in data['Z_Coordinate']]
    data['X_Coordinate'] = [str(i).replace(',','') for i in data['X_Coordinate']] 
    data['Y_Coordinate'] = [str(i).replace(',','') for i in data['Y_Coordinate']]
    data['Z_Coordinate'] = [str(i).replace(',','') for i in data['Z_Coordinate']]
    data['X_Coordinate'] = ['-' + str(i)[:-1] if str(i)[-1]=='-' else i for i in data['X_Coordinate']] 
    data['Y_Coordinate'] = ['-' + str(i)[:-1] if str(i)[-1]=='-' else i for i in data['Y_Coordinate']]
    data['Z_Coordinate'] = ['-' + str(i)[:-1] if str(i)[-1]=='-' else i for i in data['Z_Coordinate']]
    data['X_Coordinate'] = pd.to_numeric(data['X_Coordinate'])
    data['Y_Coordinate'] = pd.to_numeric(data['Y_Coordinate'])
    data['Z_Coordinate'] = pd.to_numeric(data['Z_Coordinate'])

    id_cols = ['Permanent_Tournament_#','Player_#','Round','Hole']

    ## filter out match play
    before = len(data)
    data = data[data['Permanent_Tournament_#']!=470]
    print 'Dropped %d shots from matchplay.' % (before-len(data),)

    ## filter out holes with conceded strokes
    before = len(data)
    holes_w_conceded_strokes = set(tuple(tup) for tup in data.loc[data['Shot_Type(S/P/D)']=='C',id_cols].values.tolist())
    holes_w_conceded_strokes = np.array([tuple(tup) in holes_w_conceded_strokes for tup in data[id_cols].values])
    data = data.iloc[~holes_w_conceded_strokes]
    print 'Dropped %d shots from dropping all player-holes with conceded strokes.' % (before-len(data),)

    ## filter out holes with any Shot of Type = 'S', #_of_Strokes = 0
    before = len(data)
    holes_w_S_w_0strokes = set(tuple(tup) for tup in data.loc[(data['Shot_Type(S/P/D)']=='S') & 
                                                              (data['#_of_Strokes']==0),id_cols].values.tolist())
    holes_w_S_w_0strokes = np.array([tuple(tup) in holes_w_S_w_0strokes for tup in data[id_cols].values])
    data = data.iloc[~holes_w_S_w_0strokes]
    print 'Dropped %d shots from dropping all player-holes with Shots of type S but #_of_Strokes=0.' % (before-len(data),)

    data = data.rename(columns={'X_Coordinate':'End_X_Coordinate',
                                'Y_Coordinate':'End_Y_Coordinate',
                                'Z_Coordinate':'End_Z_Coordinate'})

    data['Start_loc'] = 0
    data['End_loc'] = [(x,y,z) for x,y,z in zip(data.End_X_Coordinate,data.End_Y_Coordinate,data.End_Z_Coordinate)]
    data['Penalty_Shots'] = 0

    data = data[data['Shot_Type(S/P/D)']!='Pr']

    start_loc_equal_to_prev_start_loc, penalty_shots = set(), {}
    for hole_id_tup,df in data.groupby(id_cols):
        seq = df['Shot_Type(S/P/D)'].str.cat(sep='')
        SPUs = find_all(seq,'SPU')
        for ind in SPUs:
            shot_id_tup_1, shot_id_tup_2, shot_id_tup_3 = [hole_id_tup + (df.iloc[i].Shot,) for i in range(ind,ind+3)]
            penalty_shots[shot_id_tup_1] = df.iloc[ind+1]['#_of_Strokes']
            start_loc_equal_to_prev_start_loc.add(shot_id_tup_3)
        SPSs = [i for i in find_all(seq,'SPS') if df.iloc[i+2]['From_Location(Scorer)'].strip()=='Tee Box']
        for ind in SPSs:
            shot_id_tup_1, shot_id_tup_2, shot_id_tup_3 = [hole_id_tup + (df.iloc[i].Shot,) for i in range(ind,ind+3)]
            penalty_shots[shot_id_tup_1] = df.iloc[ind+1]['#_of_Strokes']
            start_loc_equal_to_prev_start_loc.add(shot_id_tup_3)
        SPDs = find_all(seq,'SPD')
        for ind in SPDs:
            shot_id_tup_1, shot_id_tup_2, shot_id_tup_3 = [hole_id_tup + (df.iloc[i].Shot,) for i in range(ind,ind+3)]
            penalty_shots[shot_id_tup_1] = df.iloc[ind+1]['#_of_Strokes']

    id_col_with_shot = id_cols + ['Shot']
    data.loc[data['Shot_Type(S/P/D)']=='U','Shot_Type(S/P/D)'] = 'S'
    data['start_loc_equal_to_prev_start_loc'] = [tuple(tup) in start_loc_equal_to_prev_start_loc 
                                                 for tup in data[id_col_with_shot].values]
    data['Penalty_Shots'] = [penalty_shots[tuple(tup)]
                             if tuple(tup) in penalty_shots else 0
                             for tup in data[id_col_with_shot].values]

    data = data[data['Shot_Type(S/P/D)']!='P']

    data = data.sort_values(id_cols + ['Shot'])

    loc_seq, counter = [], {}
    for tup in data[id_cols].values:
        tup = tuple(tup)
        if tup not in counter:
            loc_seq.append(1)
            counter[tup] = 1
        else:
            loc_seq.append(counter[tup] + 1)
            counter[tup] += 1
    data['loc_seq'] = loc_seq

    locs = {tuple(tup[:-1]):tup[-1] for tup in data[id_cols + ['loc_seq','End_loc']].values}
    start_loc = []
    for tup in data[id_cols+['loc_seq']].values:
        tup = tuple(tup)
        if tup[:-1] + (tup[-1]-1,) in locs:
            start_loc.append(locs[tup[:-1] + (tup[-1]-1,)])
        else:
            start_loc.append(0)
    data['Start_loc'] = start_loc

    data = data[data['Shot_Type(S/P/D)']!='D']

    data['Start_X_Coordinate'] = [x[0] if not isinstance(x,int) else 0 for x in data.Start_loc]
    data['Start_Y_Coordinate'] = [x[1] if not isinstance(x,int) else 0 for x in data.Start_loc]
    data['Start_Z_Coordinate'] = [x[2] if not isinstance(x,int) else 0 for x in data.Start_loc]

    ## this is impossible to impute so setting to nan
    data.loc[data.Shot==1,'Start_Z_Coordinate'] = np.nan

    data = data.drop(['loc_seq','Start_loc','End_loc'],axis=1)
    real_shots, counter = [], {}
    for tup in data[id_cols].values:
        tup = tuple(tup)
        if tup not in counter:
            real_shots.append(1)
            counter[tup] = 1
        else:
            real_shots.append(counter[tup] + 1)
            counter[tup] += 1
    data['Real_Shots'] = real_shots

    last_shot_mask = []
    for tup in data[id_cols + ['Real_Shots']].values:
        if tup[-1]==counter[tuple(tup[:-1])]:
            last_shot_mask.append(True)
        else:
            last_shot_mask.append(False)
    data['last_shot_mask'] = last_shot_mask

    hole_locs, tee_locs = {}, {}
    hole_id_cols = ['Permanent_Tournament_#','Round','Hole']
    for tup,df in data.groupby(hole_id_cols):
        tup = tuple(tup)
        imputer = Imputer(1,df)
        if len(imputer.df)==0:
            tee_locs[tup] = (0.,0.)
        else:
            result = imputer.find_location(7)
            if result:
                tee_locs[tup] = (imputer.loc_x,imputer.loc_y)
            else:
                tee_locs[tup] = (0.,0.)

        imputer = Imputer(0,df)
        if len(imputer.df)==0:
            hole_locs[tup] = (0.,0.,0.)
        else:
            result = imputer.find_location(1)
            if result:
                hole_locs[tup] = (imputer.loc_x,imputer.loc_y,imputer.loc_z)
            else:
                hole_locs[tup] = (0.,0.,0.)

    data.loc[data.Shot==1,'Start_X_Coordinate'] = [tee_locs[tuple(tup)][0] 
                                                   for tup in data.loc[data.Shot==1,hole_id_cols].values]
    data.loc[data.Shot==1,'Start_Y_Coordinate'] = [tee_locs[tuple(tup)][1] 
                                                   for tup in data.loc[data.Shot==1,hole_id_cols].values]
    
    data.loc[data.last_shot_mask,'End_X_Coordinate'] = [hole_locs[tuple(tup)][0] 
                                                        for tup in data.loc[data.last_shot_mask,hole_id_cols].values]
    data.loc[data.last_shot_mask,'End_Y_Coordinate'] = [hole_locs[tuple(tup)][1] 
                                                        for tup in data.loc[data.last_shot_mask,hole_id_cols].values]
    data.loc[data.last_shot_mask,'End_Z_Coordinate'] = [hole_locs[tuple(tup)][2] 
                                                        for tup in data.loc[data.last_shot_mask,hole_id_cols].values]

    for _ in range(3): ## dealing with edge case of multiple shots in a row OB
        start_locs = {tuple(tup[:-3]):tup[-3:] for tup in data[id_cols + ['Real_Shots','Start_X_Coordinate',
                                                                          'Start_Y_Coordinate','Start_Z_Coordinate']].values}    
        data.loc[data.start_loc_equal_to_prev_start_loc,'Start_X_Coordinate'] = [start_locs[tuple(tup[:-1]) + (tup[-1]-1,)][0] 
                                                                                 for tup in data.loc[data.start_loc_equal_to_prev_start_loc,id_cols+['Real_Shots']].values]
        data.loc[data.start_loc_equal_to_prev_start_loc,'Start_Y_Coordinate'] = [start_locs[tuple(tup[:-1]) + (tup[-1]-1,)][1] 
                                                                                 for tup in data.loc[data.start_loc_equal_to_prev_start_loc,id_cols+['Real_Shots']].values]
        data.loc[data.start_loc_equal_to_prev_start_loc,'Start_Z_Coordinate'] = [start_locs[tuple(tup[:-1]) + (tup[-1]-1,)][2] 
                                                                                 for tup in data.loc[data.start_loc_equal_to_prev_start_loc,id_cols+['Real_Shots']].values]

    non_zero_mask = (data.Start_X_Coordinate!=0) & (data.Start_Y_Coordinate!=0) & (data.Start_Z_Coordinate!=0) & \
                    (data.End_X_Coordinate!=0) & (data.End_Y_Coordinate!=0) & (data.End_Z_Coordinate!=0)
    before = len(data)
    data = data[non_zero_mask]
    print 'Dropped %d shots from dropping shots with missing coordinates.' % (before-len(data),)

    data['Stroke'] = data.groupby(id_cols)['Penalty_Shots'].cumsum() + data['Real_Shots']
    scores = data.groupby(id_cols)['Stroke'].max().to_dict()
    data['Strokes_from_starting_location'] = [scores[tuple(tup[:-1])] - tup[-1] + 1 for tup in data[id_cols + ['Stroke']].values]

    data['from_the_tee_box_mask'] = [tee_locs[tuple(tup[:-2])]==tuple(tup[-2:])
                                     for tup in data[hole_id_cols+['Start_X_Coordinate','Start_Y_Coordinate']].values]

    data.to_csv('data/%d.csv' % year, index=False)

