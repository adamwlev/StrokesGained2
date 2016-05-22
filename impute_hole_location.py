import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc

year = 2014
import initial_process
data = initial_process.create_df(year)

#unique Course-Round-Hole Tuples
uCRHtps = list(itertools.product(np.unique(data['Course Name']),np.unique(data['Round']),np.unique(data['Hole'])))

# coordinates of hole are not given. must be imputed.
# based on previous test, x,y, and z coordinates are all used for distance

def f (a):
    x0,y0,z0 = a[0],a[1],a[2]
    return sum((((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5-d)**2)/len(x)

def find_the_hole ():
    xopt = fmin_tnc(f,[x0,y0,z0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
    return xopt

# initializing new data frame with two rows which will be deleted after
newdata = pd.DataFrame(data.loc[1:2,:])

# finding the coordinates of the hole and recording the results
# must treat shots that were holed out differently since the coordinates are recorded as 0 for those shots
for u,i in enumerate(uCRHtps):
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2])) \
    & (~((data['Distance to Hole after the Shot']!=0) & ((data['X Coordinate']==0) | (data['Y Coordinate']==0) | (data['Z Coordinate']==0))))]
    if subset[subset['Distance to Hole after the Shot']!=0].shape[0] == 0:
        continue
    start = subset.shape[0]
    subset = subset.sort_values('Distance to Hole after the Shot')
    d = subset[subset['Distance to Hole after the Shot']!=0]['Distance to Hole after the Shot'].values/12.0
    x = subset[subset['Distance to Hole after the Shot']!=0]['X Coordinate'].values
    y = subset[subset['Distance to Hole after the Shot']!=0]['Y Coordinate'].values
    z = subset[subset['Distance to Hole after the Shot']!=0]['Z Coordinate'].values
    x0 = subset[subset['Distance to Hole after the Shot']!=0]['X Coordinate'].values[0] ##assume that closest ball recorded to hole does not have an error
    y0 = subset[subset['Distance to Hole after the Shot']!=0]['Y Coordinate'].values[0]
    z0 = subset[subset['Distance to Hole after the Shot']!=0]['Z Coordinate'].values[0]
    subset.insert(len(subset.columns),'approx_dist',np.array([0]*subset[subset['Distance to Hole after the Shot']==0].shape[0] + (((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5).tolist()))
    # dist_diff is difference between recorded distance and distance approximated from location of closest recorded shot
    subset.insert(len(subset.columns),'dist_diff', np.absolute(subset['approx_dist'].values - subset['Distance to Hole after the Shot'].values/12.0))
    # remove very inconsistant shots from record which are likely mistakes
    subset = subset[subset['dist_diff']<subset[subset['Distance to Hole after the Shot']!=0]['Distance to Hole after the Shot'].values[0]]
    finish = subset.shape[0]
    if start-finish>5:
        print u, start-finish
    d = subset[subset['Distance to Hole after the Shot']!=0]['Distance to Hole after the Shot'].values/12.0
    x = subset[subset['Distance to Hole after the Shot']!=0]['X Coordinate'].values
    y = subset[subset['Distance to Hole after the Shot']!=0]['Y Coordinate'].values
    z = subset[subset['Distance to Hole after the Shot']!=0]['Z Coordinate'].values
    a = find_the_hole()
    subset.drop('approx_dist',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    subset.insert(len(subset.columns),'Hole X Coordinate',np.array([a[0]]*subset.shape[0]))
    subset.insert(len(subset.columns),'Hole Y Coordinate',np.array([a[1]]*subset.shape[0]))
    subset.insert(len(subset.columns),'Hole Z Coordinate',np.array([a[2]]*subset.shape[0]))
    subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Distance to Hole after the Shot']==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
    subset.insert(len(subset.columns),'dist_diff', np.absolute(subset['dist_w_impute'].values - subset['Distance to Hole after the Shot'].values/12.0))
    mean_err = subset[subset['Distance to Hole after the Shot']!=0]['dist_diff'].mean()
    if mean_err>.27:
        print u, 'mean_err = ', mean_err
    if mean_err>1:
        print u, subset[subset['Distance to Hole after the Shot']!=0].shape[0]
        continue
    max_err = subset['dist_diff'].max()
    if max_err>.5:
        print u, 'max_err = ', max_err
    subset.drop('dist_w_impute',axis=1,inplace=True)
    subset.drop('dist_diff',axis=1,inplace=True)
    newdata = newdata.append(subset)

newdata.drop(newdata.head(2).index, inplace=True)

print data.shape
print newdata.shape
print (data.shape[0]-newdata.shape[0])/float(data.shape[0])

# data has been shrunk by about 10 % as a result of a couple of screening processes:
# I removed all shots with a x,y, or z coordinate equaling 0 but a nonzero recorded distance from hole
# I removed all shots from course-hole-round subsets without any shots that had nonzero recorded distances
# I removed all shots that had recored distances that were not consistant with the coordinates recorded
#
# *details* I did the last one by comparing the absolute value of the difference between distance calculated
# with coordinates of the closest recorded shot and distance recorded and the distance of the closest shot
# recorded. If the distance recorded and the coordinates recorded are plausible, the former must be less than
# the latter. If this was not true, it indicates some sort of error in the recording of the coordinates so these
# shots were discarded.

newdata.to_csv('data/'+str(year)+'_with_hole_coordinates.csv')