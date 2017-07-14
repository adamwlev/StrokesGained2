import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import GradientBoostingRegressor

def make_monitor(running_mean_len):
    def monitor(i,self,args):
        if np.mean(self.oob_improvement_[max(0,i-running_mean_len+1):i+1])<0:
            return True
        else:
            return False
    return monitor

def doit(learn_rate,num_folds,with_slope,with_gtww,with_cryh_dum,
         with_exp_dum,with_shot_locs,with_hole_locs):
    data = pd.concat([pd.read_csv('data/%d.csv' % year, 
                                  usecols=['Year','Course_#','Permanent_Tournament_#','Round','Hole','Player_#',
                                           'Start_X_Coordinate','End_X_Coordinate',
                                           'Start_Y_Coordinate','End_Y_Coordinate',
                                           'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
                                           'Strokes_from_starting_location','Cat','Distance_from_hole',
                                           'Green_to_work_with','Shot','loc_string','loc_string_hole'])
                      for year in range(2003,2018)])

    id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
    cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']

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
                      'Start_X_Coordinate','End_X_Coordinate','Start_Y_Coordinate','End_Y_Coordinate'],axis=1)

    dum_cols = []
    if with_cryh_dum:
        dum_cols = ['Course_#','Year','Hole','Round']
    if with_hole_locs:
        dum_cols.append('loc_string_hole')
    if with_shot_locs:
        dum_cols.append('loc_string')
    
    results = {}

    for cat in cats:
        print cat
        data_ = data[data.Cat==cat]
        groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
        le = LabelEncoder()
        groups = le.fit_transform(groups)
        groups_dict = {group:u for u,group in enumerate(set(groups))}
        perm = np.random.permutation(len(groups_dict))
        groups = [perm[groups_dict[group]] for group in groups]

        cols = ['Distance_from_hole']
        if with_slope:
            cols.append('Start_Z_Coordinate')
        if cat!='Green' and with_gtww:
            cols.append('Green_to_work_with')
        X = data_[cols].values.astype(float)
        X = csc_matrix(X)
        if with_exp_dum:
            lb = LabelBinarizer(sparse_output=True)
            cols = ['Course_#',['Course_#','Year'],['Course_#','Year','Round'],['Course_#','Hole']]
            to_encode = []
            for col in cols:
                if isinstance(col,str):
                    to_encode.append([str(s) for s in data_[col].values])
                else:
                    to_encode.append(['-'.join(map(str,tup)) for tup in data_[col].values])
            X_ = bmat([[lb.fit_transform(c) for c in to_encode]],format='csc')
            X = bmat([[X,X_]],format='csc')
        if dum_cols:
            lb = LabelBinarizer(sparse_output=True)
            X_ = bmat([[lb.fit_transform(data_[col].values) for col in dum_cols]],format='csc')
            X = bmat([[X,X_]],format='csc')

        X.data[np.isnan(X.data)] = 0
        X.data[np.isinf(X.data)] = 0
        y = data_.Strokes_from_starting_location.values

        if learn_rate>=.5:
            mon_size = 5
        elif learn_rate>=.1:
            mon_size = 8
        else:
            mon_size = 12

        cv = GroupKFold(n_splits=num_folds)
        for u,(train,test) in enumerate(cv.split(X,y,groups)):
            gbr = GradientBoostingRegressor(loss='huber',learning_rate=learn_rate,n_estimators=40000,
                                            subsample=.75,verbose=1,alpha=.93,max_features=.8)
            monitor = make_monitor(mon_size)
            gbr.fit(X[train],y[train],monitor=monitor)
            predictions = gbr.predict(X[test])
            print np.mean(predictions<1.0)
            predictions[predictions<1.0] = 1.0
            train_error = np.mean(np.abs(gbr.predict(X[train])-y[train]))
            test_error = np.mean(np.abs(predictions-y[test]))
            if cat not in results:
                results[cat] = []
            results[cat].append((train_error,test_error))
            print '*** FOLD %d *** TRAIN_ERROR %g *** TEST_ERROR %g  ***' % (u,train_error,test_error)
            #results.update({tuple(tup):pred for tup,pred in zip(data_.iloc[test][shot_id_cols].values,predictions)})
    return results


# for year in range(2003,2018):
#     data = pd.read_csv('../data/%d.csv' % year)
#     if year==2017:
#         data = data[data['Permanent_Tournament_#']!=18]
#         data['Hole_Score'] = pd.to_numeric(data['Hole_Score'])
#     if 'Difficulty_Start' in data.columns:
#         data = data.drop('Difficulty_Start',axis=1)
#     tee_difficulty_dict = {}
#     for tup,df in data.groupby(id_cols):
#         tee_difficulty_dict[tup] = df.groupby('Player_#').Stroke.max().mean()
#     data.insert(len(data.columns),'Difficulty_Start',[0]*len(data))
#     data.loc[data.Shot==1,'Difficulty_Start'] = [tee_difficulty_dict[tuple(tup)]
#                                                  if tuple(tup) in tee_difficulty_dict else np.nan
#                                                  for tup in data[data.Shot==1][id_cols].values]
#     data.loc[data.Shot!=1,'Difficulty_Start'] = [results[tuple(tup)]
#                                                  if tuple(tup) in results else np.nan
#                                                  for tup in data[data.Shot!=1][shot_id_cols].values]
#     data = data.dropna(subset=['Difficulty_Start'])
#     z_of_hole = data[data.last_shot_mask].groupby(id_cols)['End_Z_Coordinate'].max().to_dict()
#     data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'] - np.array([z_of_hole[tuple(tup)] for tup in data[id_cols].values])
#     data.to_csv('%d.csv' % year,index=False)


