import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat

from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupKFold

#import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.concat([pd.read_csv('../data/%d.csv' % year, 
                              usecols=['Year','Course_#','Permanent_Tournament_#','Round','Hole','Player_#',
                                       'Start_X_Coordinate','End_X_Coordinate',
                                       'Start_Y_Coordinate','End_Y_Coordinate',
                                       'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
                                       'Strokes_from_starting_location','Cat','Distance_from_hole',
                                       'Green_to_work_with','Shot'])
                  for year in range(2003,2018)])

id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']

data['dist_using_coords'] = ((data.Start_X_Coordinate-data.End_X_Coordinate)**2
                             + (data.Start_Y_Coordinate-data.End_Y_Coordinate)**2)**.5
data['dist_error'] = (data.Distance/12.0 - data.dist_using_coords)/(data.Distance/12.0).replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=['dist_error'])
data = data[data.dist_error.abs()<.05]

z_of_hole = data[data.last_shot_mask].groupby(id_cols)['End_Z_Coordinate'].max().to_dict()
data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'] - np.array([z_of_hole[tuple(tup)] for tup in data[id_cols].values])
data = data.drop(['End_Z_Coordinate','last_shot_mask','dist_using_coords','dist_error',
                  'Start_X_Coordinate','End_X_Coordinate','Start_Y_Coordinate','End_Y_Coordinate'],axis=1)

n_folds = 6

# def find_num_trees(X,y,groups,n_folds):
#     cv = GroupKFold(n_splits=n_folds)
#     params = {'objective':'reg:linear','eta':.75,'silent':1,'gamma':1.,
#               'tree_method':'approx','max_depth':6,'lamb':3.,'subsample':.75}
#     early_stopping_rounds = 50
#     num_round = 100000
#     n_trees = []
#     for u,(train,test) in enumerate(cv.split(X,y,groups)):
#         dtrain = xgb.DMatrix(X[train],label=y[train])
#         dtest = xgb.DMatrix(X[test],label=y[test])
#         watchlist  = [(dtrain,'train'),(dtest,'eval')]
#         bst = xgb.train(params,dtrain,num_round,watchlist,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
#         predictions = bst.predict(dtest,ntree_limit=bst.best_iteration)
#         print np.mean(predictions<1.0)

#         error = np.mean((predictions-y[test])**2)
#         print '*** FOLD %d *** ERROR %g *** %d TREES ***' % (u,error,bst.best_iteration)
#         n_trees.append(bst.best_iteration)
#     return int(round(np.mean(n_trees)))

cols = ['Course_#','Year','Hole','Round']
results = {}

for cat in cats:
    data_ = data[data.Cat==cat]
    groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
    le = LabelEncoder()
    groups = le.fit_transform(groups)
    groups_dict = {group:u for u,group in enumerate(set(groups))}
    perm = np.random.permutation(len(groups_dict))
    groups = [perm[groups_dict[group]] for group in groups]

    if cat=='Green':
        X = data_[['Start_Z_Coordinate','Distance_from_hole']].values.astype(float)
    else:
        X = data_[['Start_Z_Coordinate','Distance_from_hole','Green_to_work_with']].values.astype(float)

    lb = LabelBinarizer(sparse_output=True)
    X = csc_matrix(X)
    X_ = bmat([[lb.fit_transform(data_[col].values.astype(str)) for col in cols]],format='csc')
    X = bmat([[X,X_]],format='csc')
    y = data_.Strokes_from_starting_location.values
    #n_trees = find_num_trees(X,y,groups,n_folds)

    groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
    le = LabelEncoder()
    groups = le.fit_transform(groups)
    groups_dict = {group:u for u,group in enumerate(set(groups))}
    perm = np.random.permutation(len(groups_dict))
    groups = [perm[groups_dict[group]] for group in groups]
    cv = GroupKFold(n_splits=n_folds)
    params = {'objective':'reg:linear','eta':.75,'silent':1,'gamma':1.,
              'tree_method':'approx','max_depth':6,'lamb':3.,'subsample':.75}
    
    #num_round = n_trees
    rf = RandomForestRegressor(n_estimators=800, criterion='mse', max_depth=6, min_samples_split=20, 
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                               max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, 
                               oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False)

    shot_id_cols = id_cols+['Player_#','Shot'] 
    for u,(train,test) in enumerate(cv.split(X,y,groups)):
        #dtrain = xgb.DMatrix(X[train],label=y[train])
        #dtest = xgb.DMatrix(X[test],label=y[test])
        #bst = xgb.train(params,dtrain,num_round,verbose_eval=False)
        rf.fit(X[train],y[train])
        #predictions = bst.predict(dtest)
        predictions = rf.predict(X[test])
        print np.mean(predictions<1.0)
        predictions[predictions<1.0] = 1.0

        error = np.mean((predictions-y[test])**2)
        #print '*** FOLD %d *** ERROR %g *** %d TREES ***' % (u,error,bst.best_iteration)
        print '*** FOLD %d *** ERROR %g ***' % (u,error)
        results.update({tuple(tup):pred for tup,pred in zip(data_.iloc[test][shot_id_cols].values,predictions)})

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


