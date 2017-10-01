import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
import xgboost as xgb
import itertools, gc, dill
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelBinarizer

cols = ['tourn_num','skill_estimate_percentile','not_seen','observation_count_percentile',
        'Player_First_Name','Player_Last_Name','Cat','Distance_from_hole','Green_to_work_with',
        'from_the_tee_box_mask','Strokes_from_starting_location','Course_#','Hole','loc_string',
        'loc_string_hole']#,'Start_Z_Coordinate','windSpeed','temperature'
data = pd.concat([pd.read_csv('../data/%d.csv.gz' % year,usecols=cols) for year in range(2003,2018)])

data.loc[data.from_the_tee_box_mask,'Cat'] = 'Tee Box'
data = data.drop('from_the_tee_box_mask',axis=1)

data.not_seen = data.not_seen.astype(float)
data.Strokes_from_starting_location = data.Strokes_from_starting_location.astype(float)
data.skill_estimate_percentile = data.skill_estimate_percentile.fillna(.5)
data.observation_count_percentile = data.observation_count_percentile.fillna(.5)
# data.Start_Z_Coordinate = data.Start_Z_Coordinate/data.Distance_from_hole
# data.loc[data.Start_Z_Coordinate.abs()>1,'Start_Z_Coordinate'] = data.loc[data.Start_Z_Coordinate.abs()>1,
#                                                                           'Start_Z_Coordinate']\
#                                                                           .apply(lambda x: 1 if x>0 else -1)
#data.windSpeed = data.windSpeed.fillna(data.windSpeed.mean())
#data.temperature = data.temperature.fillna(data.temperature.mean())

def psuedo_huber(preds, dtrain):
    labels = dtrain.get_label()
    delta = psuedo_huber.delta
    resids = preds - labels
    grad = resids * (1 + (resids/delta)**2)**(-.5)
    hess = (1 + (resids/delta)**2)**(-1.5)
    return grad, hess

def find_num_trees(X,y,params,eval_pct,course_strings,course_hole_strings,loc_strings,loc_strings_hole,lbs):
    early_stopping_rounds = 25
    num_round = 10000
    num_train = int(X.shape[0]*(1-eval_pct))
    X_train, y_train, X_test, y_test = (csc_matrix(X[:num_train]), y[:num_train]
                                        csc_matrix(X[num_train:]), y[num_train:])
    X_train = bmat([[X_train,lbs['course'].fit_transform(course_strings[:num_train]),
                     lbs['course_hole'].fit_transform(course_hole_strings[:num_train]),
                     lbs['loc_string'].fit_transform(loc_strings[:num_train]),
                     lbs['loc_string_hole'].fit_transform(loc_strings_hole[:num_train])]],format='csc')
    X_test = bmat([[X_test,lbs['course'].transform(course_strings[num_train:]),
                    lbs['course_hole'].transform(course_hole_strings[num_train:]),
                    lbs['loc_string'].transform(loc_strings[num_train:]),
                    lbs['loc_string_hole'].transform(loc_strings_hole[num_train:])]],format='csc')
    dtrain = xgb.DMatrix(X_train,label=y_train)
    deval = xgb.DMatrix(X_test,label=y_test)
    watchlist  = [(dtrain,'train'),(deval,'eval')]
    params['base_score'] = y_train.mean()
    bst = xgb.train(params,dtrain,num_round,watchlist,obj=psuedo_huber,
                    early_stopping_rounds=early_stopping_rounds,verbose_eval=False) 
    return bst.best_iteration

cats = ['Green','Fairway','Rough','Other','Bunker','Tee']
cat_map = {'Green':set(['Green']),'Fairway':set(['Fringe','Fairway']),'Bunker':set(['Bunker']),
           'Rough':set(['Primary Rough','Intermediate Rough']),'Other':set(['Other']),'Tee':set(['Tee'])}
for cat in cats:
    cols = ['Distance_from_hole','skill_estimate_percentile','not_seen',
            'observation_count_percentile']#,'Start_Z_Coordinate','windSpeed','temperature'
    if cat!='Green':
        cols.append('Green_to_work_with')
    sub = data[data.Cat.isin(cat_map[cat])].reset_index(drop=True)
    X = sub[cols].values
    course_strings = np.array(['%d' % (num,) for num in sub['Course_#']])
    course_hole_strings = np.array(['%d-%d' % (tup[0],tup[1])
                                    for tup in sub[['Course_#','Hole']].values])
    loc_strings = sub.loc_string.values
    loc_strings_hole = sub.loc_string_hole.values
    lbs = {}
    lbs['course'],lbs['course_hole'] = LabelBinarizer(sparse_output=True),LabelBinarizer(sparse_output=True)
    lbs['loc_string'],lbs['loc_string_hole'] = LabelBinarizer(sparse_output=True),LabelBinarizer(sparse_output=True)
    y = sub.Strokes_from_starting_location.values
    params = {'objective':'reg:linear','min_child_weight':4,'eval_metric':'mae',
              'subsample':.75,'tree_method':'approx','silent':0,
              'eta':.007,'lambda':20,'max_depth':12}
    num_trees = find_num_trees(X,y,params,.22)
    print cat,num_trees
    X = csc_matrix(X)
    X = bmat([[X,lbs['course'].fit_transform(course_strings),
               lbs['course_hole'].fit_transform(course_hole_strings),
               lbs['loc_string'].fit_transform(loc_strings),
               lbs['loc_string_hole'].fit_transform(loc_strings_hole)]],format='csc')
    with open('lbs/F-lbs-%s.pkl' % (cat,), 'wb') as pickle_file:
        dill.dump(lbs, pickle_file)
    dmat = xgb.DMatrix(X,label=y)
    bst = xgb.train(params,dtrain,num_trees,obj=psuedo_huber)
    bst.save_model('difficulty_prediction_models/F-%s.model' % (u,cat))