import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc

import eliminate_holes_with_issues as e
data = e.make_df(2004)

#unique Course-Round-Hole Tuples
uCRHtps = list(itertools.product(np.unique(data.Course_Name),np.unique(data.Round),np.unique(data.Hole)))


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
shots_removed = 0
#finding the coordinates of the hole
for u,i in enumerate(uCRHtps):
    subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2])) &  \
            (data.Distance_to_Hole_after_the_Shot!=0) & (data.X_Coordinate!=0) & (data.Y_Coordinate!=0) & (data.Z_Coordinate!=0)]
    if subset.shape[0] == 0:
        continue
    d = subset.Distance_to_Hole_after_the_Shot.values/12.0
    x = subset.X_Coordinate.values
    y = subset.Y_Coordinate.values
    z = subset.Z_Coordinate.values
    sorted_subset = subset.sort_values('Distance_to_Hole_after_the_Shot')
    x0 = sorted_subset.X_Coordinate.values[0] ##assume that closest ball recorded to hole does not have an error
    y0 = sorted_subset.Y_Coordinate.values[0]
    z0 = sorted_subset.Z_Coordinate.values[0]
    d0 = sorted_subset.Distance_to_Hole_after_the_Shot.values[0]
    subset.insert(len(subset.columns),'dist_to_shot_nearest_to_hole',((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5)
    before = subset.shape[0]
    # remove shots for which the distance to the closest shot to the hole is greater than the sum of the distance to the hole
    # after the shot and the distance to the hole from the closest recorded shot
    subset = subset[subset.dist_to_shot_nearest_to_hole <= subset.Distance_to_Hole_after_the_Shot.values + d0]
    after = subset.shape[0]
    shots_removed += before-after
    d = subset.Distance_to_Hole_after_the_Shot.values/12.0
    x = subset.X_Coordinate.values
    y = subset.Y_Coordinate.values
    z = subset.Z_Coordinate.values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
    mean_err = subset.dist_diff.mean()
    max_err = subset.dist_diff.max()
    median_err = subset.dist_diff.median()
    mean_errs.append(mean_err)
    max_errs.append(max_err)
    median_errs.append(median_err)

print 'mean mean err = ', sum(mean_errs)/len(mean_errs)
print 'mean max err = ', sum(max_errs)/len(max_errs)
print 'mean median_err = ', sum(median_errs)/len(median_errs)
print 'shots_removed = ', shots_removed

# now distance with just x and y coordinates
def f (a):
    x0,y0 = a[0],a[1]
    return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

def find_the_hole ():
    xopt = fmin_tnc(f,[x0,y0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
    return xopt

mean_errs,median_errs,max_errs = [],[],[]
shots_removed = 0
#finding the coordinates of the hole using just x and y coordinates
for u,i in enumerate(uCRHtps[0:2000]):
    subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2])) & \
             (data.Distance_to_Hole_after_the_Shot!=0) & (data.X_Coordinate!=0) & (data.Y_Coordinate!=0) & (data.Z_Coordinate!=0)]
    if subset.shape[0] == 0:
        continue
    d = subset.Distance_to_Hole_after_the_Shot.values/12.0
    x = subset.X_Coordinate.values
    y = subset.Y_Coordinate.values
    sorted_subset = subset.sort_values('Distance_to_Hole_after_the_Shot')
    x0 = sorted_subset.X_Coordinate.values[0] ##assume that closest ball recorded to hole does not have an error
    y0 = sorted_subset.Y_Coordinate.values[0]
    d0 = sorted_subset.Distance_to_Hole_after_the_Shot.values[0]
    subset.insert(len(subset.columns),'dist_to_shot_nearest_to_hole',((x-x0)**2 + (y-y0)**2)**.5)
    before = subset.shape[0]
    # remove shots for which the distance to the closest shot to the hole is greater than the sum of the distance to the hole
    # after the shot and the distance to the hole from the closest recorded shot
    subset = subset[subset.dist_to_shot_nearest_to_hole <= subset.Distance_to_Hole_after_the_Shot.values + d0]
    after = subset.shape[0]
    shots_removed += before-after
    d = subset.Distance_to_Hole_after_the_Shot.values/12.0
    x = subset.X_Coordinate.values
    y = subset.Y_Coordinate.values
    a = find_the_hole()
    subset.insert(len(subset.columns),'dist_w_impute',np.array(((x-a[0])**2 + (y-a[1])**2)**.5).tolist())
    subset.insert(len(subset.columns),'dist_diff',np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
    mean_err = subset.dist_diff.mean()
    max_err = subset.dist_diff.max()
    median_err = subset.dist_diff.median()
    mean_errs.append(mean_err)
    max_errs.append(max_err)
    median_errs.append(median_err)

print 'mean mean err = ', sum(mean_errs)/len(mean_errs)
print 'mean max err = ', sum(max_errs)/len(max_errs)
print 'mean median_err = ', sum(median_errs)/len(median_errs)
print 'shots_removed = ', shots_removed
