import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc

import initial_process
data = initial_process.create_df(2015)

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
    xopt = fmin_tnc(f,[x0,y0,z0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
    return xopt


mean_errs,median_errs,max_errs = [],[],[]
#finding the coordinates of the hole
for u,i in enumerate(uCRHtps[0:2000]):
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2])) &  \
             (data['Distance to Hole after the Shot']!=0) & (data['X Coordinate']!=0) & (data['Y Coordinate']!=0) & (data['Z Coordinate']!=0)]
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
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset['approx_dist'].values - subset['Distance to Hole after the Shot'].values/12.0))
    before = subset.shape[0]
    # remove very inconsistant shots from record which are likely mistakes
    subset = subset[subset['dist_diff']<sorted_subset['Distance to Hole after the Shot'].values[0]]
    after = subset.shape[0]
    subset.drop('approx_dist',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    if before-after>3:
        print u, before-after
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    z = subset['Z Coordinate'].values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset['dist_w_impute'].values - subset['Distance to Hole after the Shot'].values/12.0))
    mean_err = subset['dist_diff'].mean()
    max_err = subset['dist_diff'].max()
    median_err = subset['dist_diff'].median()
    mean_errs.append(mean_err)
    max_errs.append(max_err)
    median_errs.append(median_err)

print 'mean mean err = ', sum(mean_errs)/len(mean_errs)
print 'mean max err = ', sum(max_errs)/len(max_errs)
print 'mean median_err = ', sum(median_errs)/len(median_errs)

# now distance with just x and y coordinates
def f (a):
    x0,y0 = a[0],a[1]
    return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

def find_the_hole ():
    xopt = fmin_tnc(f,[x0,y0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
    return xopt


#finding the coordinates of the hole using just x and y coordinates
for u,i in enumerate(uCRHtps[0:2000]):
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2])) & \
             (data['Distance to Hole after the Shot']!=0) & (data['X Coordinate']!=0) & (data['Y Coordinate']!=0) & (data['Z Coordinate']!=0)]
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
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset['approx_dist'].values - subset['Distance to Hole after the Shot'].values/12.0))
    before = subset.shape[0]
    # remove very inconsistant shots from record which are likely mistakes
    subset = subset[subset['dist_diff']<sorted_subset['Distance to Hole after the Shot'].values[0]]
    after = subset.shape[0]
    subset.drop('approx_dist',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    if before-after>3:
        print u, before-after
    d = subset['Distance to Hole after the Shot'].values/12.0
    x = subset['X Coordinate'].values
    y = subset['Y Coordinate'].values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset['dist_w_impute'].values - subset['Distance to Hole after the Shot'].values/12.0))
    mean_err = subset['dist_diff'].mean()
    max_err = subset['dist_diff'].max()
    median_err = subset['dist_diff'].median()
    mean_errs.append(mean_err)
    max_errs.append(max_err)
    median_errs.append(median_err)

print 'mean mean err = ', sum(mean_errs)/len(mean_errs)
print 'mean max err = ', sum(max_errs)/len(max_errs)
print 'mean median_err = ', sum(median_errs)/len(median_errs)

## Based on the results - numer of points that don't fit, it is clear the distance is calculated using the x,y, and z coordinates.
## Will use the x,y, and z coordinates.