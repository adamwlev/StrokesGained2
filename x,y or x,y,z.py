

# coding: utf-8

import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc


data = pd.read_csv('data/rawdata/2015.txt', sep = ';')


data.columns = np.array([str(i).strip() for i in list(data.columns.values)]) #remove space in col names


data['X Coordinate'] = [str(i).replace(' ','') for i in data['X Coordinate']] #remove space in coordinates cols
data['Y Coordinate'] = [str(i).replace(' ','') for i in data['Y Coordinate']]
data['Z Coordinate'] = [str(i).replace(' ','') for i in data['Z Coordinate']]


data['X Coordinate'] = [str(i).replace(',','') for i in data['X Coordinate']] #remove commas in coordinates cols
data['Y Coordinate'] = [str(i).replace(',','') for i in data['Y Coordinate']]
data['Z Coordinate'] = [str(i).replace(',','') for i in data['Z Coordinate']]


#putting negative in front
data['X Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['X Coordinate']] 
data['Y Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Y Coordinate']]
data['Z Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Z Coordinate']]


data['X Coordinate'] = pd.to_numeric(data['X Coordinate'])
data['Y Coordinate'] = pd.to_numeric(data['Y Coordinate'])
data['Z Coordinate'] = pd.to_numeric(data['Z Coordinate'])


#unique Course-Round-Hole Tuples
uCRHtps = list(itertools.product(np.unique(data['Course Name']),np.unique(data['Round']),np.unique(data['Hole'])))


# coordinates of hole are not given. must be imputed.
# does the distance use the x,y, and z coordinates or just the x and y coordinates?
# test: first find the hole using the x,y, and z coordinates and record the average difference between calculated hole
# location and ball and recorded distance. then do the same using just the x and y coodinates. see which is better.


def f (a):
    x0,y0,z0 = a[0],a[1],a[2]
    return sum((((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5-d)**2)/len(x)

def find_the_hole ():
    xopt = fmin_tnc(f,[x0,y0,z0],approx_grad=1)[0].tolist()
    return xopt


#finding the coordinates of the hole
aveds=[]
for u,i in enumerate(uCRHtps):
#     if (u+1)%100==0:
#         print u
#         print u+1,sum(aveds)/len(aveds)
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2]))                   & (data['Distance to Hole after the Shot']!=0) & (data['X Coordinate']!=0) & (data['Y Coordinate']!=0)                   & (data['Z Coordinate']!=0)]
    if subset.shape[0] == 0:
        continue
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    z = subset['Z Coordinate'].values
    sorted_subset = subset.sort_values('Distance to Hole after the Shot')
    x0 = sorted_subset['X Coordinate'].values[0] ##assume that closest ball recorded to hole does not have an error
    y0 = sorted_subset['Y Coordinate'].values[0]
    z0 = sorted_subset['Z Coordinate'].values[0]
    subset.insert(len(subset.columns),'approx_dist',((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5)
    # dist_diff is difference between recorded distance and distance approximated from location of closest recorded shot
    subset.insert(len(subset.columns),'dist_diff',                   np.absolute(subset['approx_dist'].values - subset['Distance to Hole after the Shot'].values/12.0))
    before = subset.shape[0]
    # remove very inconsistant shots from record which are likely mistakes
    subset = subset[subset['dist_diff']<sorted_subset['Distance to Hole after the Shot'].values[0]]
    after = subset.shape[0]
    subset.drop('approx_dist',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    if before-after>0:
        print u, before-after
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    z = subset['Z Coordinate'].values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',                   np.absolute(subset['dist_w_impute'].values - subset['Distance to Hole after the Shot'].values/12.0))
    mean_err = subset['dist_diff'].mean()
    if mean_err>.27:
        print u, mean_err
    max_err = subset['dist_diff'].max()
    if max_err>.5:
        print u,max_err
    
#print sum(aveds)/len(aveds)


# now dist with just x and y coordinates
def f (a):
    x0,y0 = a[0],a[1]
    return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

def find_the_hole ():
    xopt = fmin_tnc(f,[x0,y0],approx_grad=1)[0].tolist()
    return xopt


#finding the coordinates of the hole
aveds=[]
for u,i in enumerate(uCRHtps):
#     if (u+1)%100==0:
#         print u
#         print u+1,sum(aveds)/len(aveds)
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2]))                   & (data['Distance to Hole after the Shot']!=0) & (data['X Coordinate']!=0) & (data['Y Coordinate']!=0)                   & (data['Z Coordinate']!=0)]
    if subset.shape[0] == 0:
        continue
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    sorted_subset = subset.sort_values('Distance to Hole after the Shot')
    x0 = sorted_subset['X Coordinate'].values[0] ##assume that closest ball recorded to hole does not have an error
    y0 = sorted_subset['Y Coordinate'].values[0]
    subset.insert(len(subset.columns),'approx_dist',((x-x0)**2 + (y-y0)**2)**.5)
    # dist_diff is difference between recorded distance and distance approximated from location of closest recorded shot
    subset.insert(len(subset.columns),'dist_diff',                   np.absolute(subset['approx_dist'].values - subset['Distance to Hole after the Shot'].values/12.0))
    before = subset.shape[0]
    # remove very inconsistant shots from record which are likely mistakes
    subset = subset[subset['dist_diff']<sorted_subset['Distance to Hole after the Shot'].values[0]]
    after = subset.shape[0]
    subset.drop('approx_dist',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    if before-after>0:
        print u, before-after
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    z = subset['Z Coordinate'].values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',                   np.absolute(subset['dist_w_impute'].values - subset['Distance to Hole after the Shot'].values/12.0))
    mean_err = subset['dist_diff'].mean()
    if mean_err>.27:
        print u, mean_err
    max_err = subset['dist_diff'].max()
    if max_err>.5:
        print u,max_err
    
#print sum(aveds)/len(aveds)


# based on results of number of points that have implausible coordinates, and average error between the recorded dist.
# and this dist. calculated with the imputed coordinates of the hole, it is clear that the recorded distances use 
# the x,y, and z coordinates.
# now recreate dataframe with concatonation of all subsets - without bunk records and with the x,y, and z coordinates
# of the hole.

