data = None
gc.collect()
for year in range(2003,2018):
    data = pd.read_csv('../data/%d.csv' % year)
    if year==2017:
        data = data[data['Permanent_Tournament_#']!=18]
        data['Hole_Score'] = pd.to_numeric(data['Hole_Score'])
    if 'Difficulty_Start' in data.columns:
        data = data.drop('Difficulty_Start',axis=1)
    tee_difficulty_dict = {}
    for tup,df in data.groupby(id_cols):
        tee_difficulty_dict[tup] = df.groupby('Player_#').Stroke.max().mean()
    data.insert(len(data.columns),'Difficulty_Start',[0]*len(data))
    data.loc[data.Shot==1,'Difficulty_Start'] = [tee_difficulty_dict[tuple(tup)]
                                                 if tuple(tup) in tee_difficulty_dict else np.nan
                                                 for tup in data[data.Shot==1][id_cols].values]
    data.loc[data.Shot!=1,'Difficulty_Start'] = [results[tuple(tup)]
                                                 if tuple(tup) in results else np.nan
                                                 for tup in data[data.Shot!=1][shot_id_cols].values]
    data = data.dropna(subset=['Difficulty_Start'])
    z_of_hole = data[data.last_shot_mask].groupby(id_cols)['End_Z_Coordinate'].max().to_dict()
    data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'] - np.array([z_of_hole[tuple(tup)] for tup in data[id_cols].values])
    data.to_csv('%d.csv' % year,index=False)