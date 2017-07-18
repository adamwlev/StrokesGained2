import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import gc, pickle

def doit():
    # data = pd.concat([pd.read_csv('data/%d.csv' % year, 
    #                               usecols=['Year','Course_#','Permanent_Tournament_#','Round','Hole','Player_#',
    #                                        'Start_X_Coordinate','End_X_Coordinate',
    #                                        'Start_Y_Coordinate','End_Y_Coordinate',
    #                                        'Start_Z_Coordinate','End_Z_Coordinate','last_shot_mask','Distance',
    #                                        'Strokes_from_starting_location','Cat','Distance_from_hole',
    #                                        'Green_to_work_with','Shot','loc_string','loc_string_hole'])
    #                   for year in range(2003,2018)])
    data = pd.read_csv('to_send.csv.gz')

    id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
    shot_id_cols = id_cols + ['Player_#','Shot']
    cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']
    num_folds = 15

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

    # delta_map = {'Green':.4,'Fairway':.65,'Intermediate Rough':.7,'Primary Rough':.75,
    #              'Fringe':.5,'Bunker':.8,'Other':1.}
    le = LabelEncoder()
    lb = LabelBinarizer(sparse_output=True)

    results = {}
    for cat in cats:
        print cat
        data_ = data[data.Cat==cat]
        groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
        groups = le.fit_transform(groups)
        groups_dict = {group:u for u,group in enumerate(set(groups))}
        perm = np.random.permutation(len(groups_dict))
        groups = [perm[groups_dict[group]] for group in groups]

        cols = ['Distance_from_hole','Start_Z_Coordinate']
        if cat!='Green':
            cols.append('Green_to_work_with')
        X = data_[cols].values.astype(float)
        X = csc_matrix(X)
        cols = ['Course_#',['Course_#','Year'],['Course_#','Year','Round'],['Course_#','Hole']]
        to_encode = []
        for col in cols:
            if isinstance(col,str):
                to_encode.append([str(s) for s in data_[col].values])
            else:
                to_encode.append(['-'.join(map(str,tup)) for tup in data_[col].values])
        X_ = bmat([[lb.fit_transform(c) for c in to_encode]],format='csc')
        X = bmat([[X,X_]],format='csc')
        y = data_.Strokes_from_starting_location.values

        def psuedo_huber(preds, dtrain):
            labels = dtrain.get_label()
            delta = psuedo_huber.delta
            resids = preds - labels
            grad = resids * (1 + (resids/delta)**2)**(-.5)
            hess = (1 + (resids/delta)**2)**(-1.5)
            return grad, hess

        params = {'objective':'reg:linear','eval_metric':'mae','min_child_weight':20,
                  'subsample':.9,'tree_method':'approx',
                  'eta':.01,'lambda':5,'max_depth':8,'base_score':y.mean()}
        psuedo_huber.delta = .4#delta_map[cat]

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            resids = np.abs(preds - labels)
            mean_error = np.mean(resids)
            error_80 = np.percentile(resids,80)
            error_99 = np.percentile(resids,99)
            overall_error = np.mean(np.array([mean_error,error_80,error_99]))
            return [('error_80',error_80),('error_99',error_99),('error',overall_error)]

        def find_num_trees():
            early_stopping_rounds = 8
            num_round = 10000
            trees = []
            for train,test in cv.split(X,y,groups):
                dtrain = xgb.DMatrix(X[train],label=y[train])
                dtest = xgb.DMatrix(X[test],label=y[test])
                watchlist  = [(dtrain,'train'),(dtest,'eval')]
                bst = xgb.train(params,dtrain,num_round,watchlist,psuedo_huber,evalerror,
                                early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
                trees.append(bst.best_iteration)
            return int(round(np.mean(trees)))

        cv = GroupKFold(n_splits=num_folds)
        num_trees = find_num_trees()
        groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
        groups = le.fit_transform(groups)
        groups_dict = {group:u for u,group in enumerate(set(groups))}
        perm = np.random.permutation(len(groups_dict))
        groups = [perm[groups_dict[group]] for group in groups]

        for u,(train,test) in enumerate(cv.split(X,y,groups)):
            dtrain = xgb.DMatrix(X[train],label=y[train])
            dtest = xgb.DMatrix(X[test])
            bst = xgb.train(params,dtrain,num_trees,obj=psuedo_huber)
            predictions = bst.predict(dtest)
            print np.mean(predictions<1.0)
            predictions[predictions<1.0] = 1.0
            train_error = np.mean(np.abs(bst.predict(dtrain)-y[train]))
            test_error = np.mean(np.abs(predictions-y[test]))
            print '*** FOLD %d *** TRAIN_ERROR %g *** TEST_ERROR %g  ***' % (u,train_error,test_error)
            results.update({tuple(tup):pred for tup,pred in zip(data_.iloc[test][shot_id_cols].values,predictions)})

    with open('difficulty.pkl','wb') as pickle_file:
        pickle.dump(results,pickle_file)
    
