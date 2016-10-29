import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,bmat

from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

import xgboost as xgb
from bayes_opt import BayesianOptimization

import warnings
import pickle
import os


data = pd.concat([pd.read_csv('./../new_data/%d.csv' % year)[['Cat','Shots_taken_from_location','Started_at_Z',
            'Distance_from_hole','Hole','Round','Course_#','Year','Green_to_work_with']] for year in range(2003,2017)])

data.insert(len(data.columns),'Year-Course',data.Year.astype(str).str.cat(data['Course_#'].astype(str),sep='-'))
data.insert(len(data.columns),'Hole-Course',data.Hole.astype(str).str.cat(data['Course_#'].astype(str),sep='-'))
data.insert(len(data.columns),'Round-Year-Course',data.Round.astype(str).str.cat(
                                    [data.Year.astype(str),data['Course_#'].astype(str)],sep='-'))
data.insert(len(data.columns),'Hole-Year-Course',data.Hole.astype(str).str.cat(
                                    [data.Year.astype(str),data['Course_#'].astype(str)],sep='-'))

cats = ['Green','Fairway','Intermediate Rough','Primary Rough','Fringe','Bunker','Other']

def xgbcv(gamma,max_depth,alpha,lamb,min_child_weight,subsample):
    params = {'objective':'reg:linear','eta':.1,'gamma':gamma,'max_depth':int(max_depth),'alpha':alpha,
              'lambda':lamb,'min_child_weight':int(min_child_weight),'subsample':subsample}
    cv_folds = 5
    early_stopping_rounds = 50
    cv = GroupShuffleSplit(n_splits=cv_folds, test_size=0.2)
    errors = []
    for u,(train,test) in enumerate(cv.split(X,y,groups)):
        dtrain = xgb.DMatrix(X[train],label=y[train])
        dtest = xgb.DMatrix(X[test],label=y[test])
        watchlist  = [(dtrain,'train'),(dtest,'eval')]
        num_round = 100000
        bst = xgb.train(params,dtrain,num_round,watchlist,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
        error = np.mean((bst.predict(dtest,ntree_limit=bst.best_iteration) - y[test])**2)
        errors.append(error)
    return 1/np.array(errors).mean()

result = {}

for cat in cats:
	print '******************* DOING %s *******************' % cat
	result[cat] = {}
	data_ = data[data.Cat==cat]
	groups = ['-'.join(map(str,tup)) for tup in data[['Hole','Round','Course_#','Year']].values.tolist()]
	le = LabelEncoder()
	groups = le.fit_transform(groups)
	y = data.Shots_taken_from_location.values
	lb = LabelBinarizer(sparse_output=True)

	if cat=='Green':
		X = data_[['Started_at_Z','Distance_from_hole']].values.astype(float)
	else:
		X = data_[['Started_at_Z','Distance_from_hole','Green_to_work_with']].values.astype(float)

	print '******************* DOING SIMPLEST *******************'

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})
	
	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['simplest'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

	print '******************* ADDING COURSE DUMMIES *******************'

	X = bmat([[csr_matrix(X),lb.fit_transform(data_['Course_#'].values.astype(str))]],format='csr')

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['with_course'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

	print '******************* ADDING YEAR-COURSE DUMMIES *******************'

	X = bmat([[X,lb.fit_transform(data_['Year-Course'].values.astype(str))]],format='csr')

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['with_year-course'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

	print '******************* ADDING HOLE-COURSE DUMMIES *******************'

	X = bmat([[X,lb.fit_transform(data_['Hole-Course'].values.astype(str))]],format='csr')

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['with_hole-course'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

	print '******************* ADDING ROUND-YEAR-COURSE DUMMIES *******************'

	X = bmat([[X,lb.fit_transform(data_['Round-Year-Course'].values.astype(str))]],format='csr')

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['with_round-year-course'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

	print '******************* ADDING HOLE-YEAR-COURSE DUMMIES *******************'

	X = bmat([[X,lb.fit_transform(data_['Hole-Year-Course'].values.astype(str))]],format='csr')

	xgbBO = BayesianOptimization(xgbcv, {'gamma':(0,4),'max_depth':(2,10),'alpha':(0,.5),
	                                     'lamb':(0,4),'min_child_weight':(1,10),'subsample':(.5,1)})

	with warnings.catch_warnings():
	    warnings.simplefilter("ignore")
	    xgbBO.maximize(init_points=5,n_iter=35)

	result[cat]['with_hole-year-course'] = xgbBO.res['max']

	with open('result.pkl','w') as pickleFile:
		pickle.dump(result,pickleFile)

	cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/Modeling/result.pkl ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/Modeling/"
	os.system(cmd)

print 'DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!'

