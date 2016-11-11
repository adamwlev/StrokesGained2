import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,bmat

from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupKFold

import xgboost as xgb
import pickle

data = pd.concat([pd.read_csv('./../data/%d.csv' % year)[['Cat','Shots_taken_from_location','Started_at_Z',
       'Distance_from_hole','Hole','Round','Course_#','Year','Green_to_work_with']] for year in range(2003,2017)])

data.insert(len(data.columns),'Year-Course',data.Year.astype(str).str.cat(data['Course_#'].astype(str),sep='-'))
data.insert(len(data.columns),'Hole-Course',data.Hole.astype(str).str.cat(data['Course_#'].astype(str),sep='-'))
data.insert(len(data.columns),'Round-Year-Course',data.Round.astype(str).str.cat([data.Year.astype(str),data['Course_#'].astype(str)],sep='-'))

cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']

n_folds = 15

with open('./../Modeling/xgboost_results.pkl','r') as pickleFile:
	hyperparams = pickle.load(pickleFile)

complexity_choices = ['with_course','with_year-course','with_hole-course','with_round-year-course']

complexity_choice = {'Green':1,'Fairway':3,'Intermediate Rough':3,'Primary Rough':3,'Fringe':0,'Bunker':2,'Other':0}

cols = ['Course_#','Year-Course','Hole-Course','Round-Year-Course']

for cat in cats[4:]:
	results = {}
	data_ = data[data.Cat==cat]
	groups = ['-'.join(map(str,tup)) for tup in data_[['Hole','Round','Course_#','Year']].values.tolist()]
	le = LabelEncoder()
	groups = le.fit_transform(groups)

	if cat=='Green':
		X = data_[['Started_at_Z','Distance_from_hole']].values.astype(float)
	else:
		X = data_[['Started_at_Z','Distance_from_hole','Green_to_work_with']].values.astype(float)

	lb = LabelBinarizer(sparse_output=True)
	X = csr_matrix(X)
	X_ = bmat([[lb.fit_transform(data_[col].values.astype(str)) for col in cols[:complexity_choice[cat]+1]]],format='csr')
	X = bmat([[X,X_]],format='csr')
	y = data_.Shots_taken_from_location.values

	cv = GroupKFold(n_splits=n_folds)
	params = hyperparams[cat][complexity_choices[complexity_choice[cat]]]['max_params']
	params.update({'objective':'reg:linear','eta':.05,'silent':1,'tree_method':'approx','max_depth':int(params['max_depth'])})
	early_stopping_rounds = 50
	num_round = 100000
	for u,(train,test) in enumerate(cv.split(X,y,groups)):
	    dtrain = xgb.DMatrix(X[train],label=y[train])
	    dtest = xgb.DMatrix(X[test],label=y[test])
	    watchlist  = [(dtrain,'train'),(dtest,'eval')]
	    bst = xgb.train(params,dtrain,num_round,watchlist,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
	    predictions = bst.predict(dtest,ntree_limit=bst.best_iteration)
	    error = np.mean((predictions-y[test])**2)
	    print '***FOLD %d *** ERROR %g ***' % (u,error)

	    assert np.all(y[test]==data.loc[data_.index[test]]['Shots_taken_from_location'].values)
	    results.update({ind:pred for ind,pred in zip(data_.index[test],predictions)})

	print len(results),len(data_)

