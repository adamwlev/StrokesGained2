import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import gc, dill

def doit(data,cat,full=False):
    gc.collect()
    id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole']
    shot_id_cols = id_cols + ['Player_#','Real_Shots']
    data['Start_Z_Coordinate'] = data['Start_Z_Coordinate'].fillna(0)
    data['dist_using_coords'] = ((data.Start_X_Coordinate-data.End_X_Coordinate)**2
                                 + (data.Start_Y_Coordinate-data.End_Y_Coordinate)**2)**.5
    data['dist_error'] = data.Distance/12.0 - data.dist_using_coords
    before = len(data)
    data = data.dropna(subset=['dist_error'])
    print 'dropped %d for dropping nulls in dist_error' % (before-len(data),)
    before = len(data)
    data = data[~((data.Cat=='Green') & (data.Distance_from_hole>130))]
    print 'dropped %d for dropping green long shots' % (before-len(data),)
    before = len(data)
    data = data[data.dist_error.abs()<30]
    print 'dropped %d for more than 10 percent shots' % (before-len(data),)
    print len(data)
    data = data.drop(['End_Z_Coordinate','dist_using_coords','dist_error',
                      'Start_X_Coordinate','End_X_Coordinate','Start_Y_Coordinate','End_Y_Coordinate'],axis=1)

    if full:
        num_folds = 12
        delta_map = {'Green':.5,'Fairway':.75,'Intermediate Rough':.8,'Primary Rough':.85,
                     'Fringe':1.0,'Bunker':.9,'Other':1.3}
        le = LabelEncoder()
        results = {}
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
        lbs = {}
        for col in cols:
            lb = LabelBinarizer(sparse_output=True)
            if isinstance(col,str):
                vec = [str(s) for s in data_[col].values]
                lbs[col] = lb.fit(vec)
            else:
                vec = ['-'.join(map(str,tup)) for tup in data_[col].values]
                lbs[tuple(col)] = lb.fit(vec)
            to_encode.append(vec)
        X_ = bmat([[lbs[col if isinstance(col,str) else tuple(col)].transform(vec)
                    for col,vec in zip(cols,to_encode)]],format='csc')
        X = bmat([[X,X_]],format='csc')
        y = data_.Strokes_from_starting_location.values

        with open('lbs_evaluation/lbs-%s.pkl' % (cat,), 'wb') as pickle_file:
            dill.dump(lbs, pickle_file)

        def psuedo_huber(preds, dtrain):
            labels = dtrain.get_label()
            delta = psuedo_huber.delta
            resids = preds - labels
            grad = resids * (1 + (resids/delta)**2)**(-.5)
            hess = (1 + (resids/delta)**2)**(-1.5)
            return grad, hess

        params = {'objective':'reg:linear','eval_metric':'mae','min_child_weight':20,
                  'subsample':.9,'tree_method':'approx','silent':1,
                  'eta':.07,'lambda':5,'max_depth':8,'base_score':y.mean()}
        psuedo_huber.delta = delta_map[cat]

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            resids = np.abs(preds - labels)
            mean_error = np.mean(resids)
            error_80, error_99 = np.percentile(resids,[80,99])
            overall_error = np.mean(np.array([mean_error,error_80,error_99]))
            return [('error_80',error_80),('error_99',error_99),('error',overall_error)]

        def find_num_trees():
            early_stopping_rounds = 25
            num_round = 10000
            trees = []
            for train,test in cv.split(X,y,groups):
                dtrain = xgb.DMatrix(X[train],label=y[train])
                dtest = xgb.DMatrix(X[test],label=y[test])
                watchlist  = [(dtrain,'train'),(dtest,'eval')]
                bst = xgb.train(params,dtrain,num_round,watchlist,psuedo_huber,evalerror,
                                early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
                trees.append(bst.best_iteration)
                print bst.best_iteration
            return int(round(np.mean(trees)))

        cv = GroupKFold(n_splits=num_folds)
        num_trees = find_num_trees()
        groups = ['-'.join(map(str,tup)) for tup in data_[id_cols].values.tolist()]
        groups = le.fit_transform(groups)
        groups_dict = {group:u for u,group in enumerate(set(groups))}
        perm = np.random.permutation(len(groups_dict))
        groups = [perm[groups_dict[group]] for group in groups]
        print X.shape

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
            bst.save_model('difficulty_evalutation_models/bst%d-%s.model' % (u,cat))

    else:
        results = {}
        with open('lbs_evaluation/lbs-%s.pkl' % (cat,), 'rb') as pickle_file:
            lbs = dill.load(pickle_file)
        data_ = data[data.Cat==cat]
        cols = ['Distance_from_hole','Start_Z_Coordinate']
        if cat!='Green':
            cols.append('Green_to_work_with')
        X = data_[cols].values.astype(float)
        X = csc_matrix(X)
        cols = ['Course_#',['Course_#','Year'],['Course_#','Year','Round'],['Course_#','Hole']]
        to_encode = []
        for col in cols:
            lb = LabelBinarizer(sparse_output=True)
            if isinstance(col,str):
                vec = [str(s) for s in data_[col].values]
            else:
                vec = ['-'.join(map(str,tup)) for tup in data_[col].values]
            to_encode.append(vec)
        X_ = bmat([[lbs[col if isinstance(col,str) else tuple(col)].transform(vec)
                    for col,vec in zip(cols,to_encode)]],format='csc')
        X = bmat([[X,X_]],format='csc')
        for fold in range(12):
            bst = xgb.Booster(model_file='difficulty_evalutation_models/bst%d-%s.model' % (fold,cat))
            dmat = xgb.DMatrix(X)
            predictions = bst.predict(dmat)
            for tup,pred in zip(data_[shot_id_cols].values,predictions):
                if tuple(tup) not in results:
                    results[tuple(tup)] = []
                results[tuple(tup)].append(pred)
        results = {key:np.mean(value) for key,value in results.iteritems()}
    return results
