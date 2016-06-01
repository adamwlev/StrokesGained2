import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc
import eliminate_holes_with_issues as e



def add_hole_locs (y):
    def f (a):
        x0,y0,z0 = a[0],a[1],a[2]
        return sum((((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5-d)**2)/len(x)

    def find_the_hole ():
        xopt = fmin_tnc(f,[x0,y0,z0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
        return xopt

    data = e.make_df(y)

    #unique Course-Round-Hole Tuples
    uCRHtps = list(itertools.product(np.unique(data.Course_Name),np.unique(data.Round),np.unique(data.Hole)))

    # coordinates of hole are not given. must be imputed.
    # based on previous test, x,y, and z coordinates are all used for distance

    # finding the coordinates of the hole and recording the results
    # must treat shots that were holed out differently since the coordinates are recorded as 0 for those shots
    inds_to_remove = set()
    hole_locs = []
    shots_looked_at = 0
    for u,i in enumerate(uCRHtps):
        subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2]))]
        shots_looked_at += subset.shape[0]
        inds_to_remove.update(subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))].index.values.tolist())
        if len(subset)==0:
            continue
        if subset[subset.Distance_to_Hole_after_the_Shot!=0].shape[0] == 0:
            print u, ' No non 0 dists'
            inds_to_remove.update(subset.index.values.tolist())
            continue
        start = subset.shape[0]
        subset = subset.sort_values('Distance_to_Hole_after_the_Shot')
        d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
        x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
        z = subset[subset.Distance_to_Hole_after_the_Shot!=0].Z_Coordinate.values
        x0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values[0] ##assume that closest ball recorded to hole does not have an error
        y0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values[0]
        z0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].Z_Coordinate.values[0]
        d0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values[0]/12.0
        subset.insert(len(subset.columns),'dist_to_shot_nearest_to_hole',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5).tolist()))
        # remove very inconsistant shots from record which are likely mistakes
        bad_subset = subset[subset.dist_to_shot_nearest_to_hole>subset.Distance_to_Hole_after_the_Shot.values+d0]
        inds_to_remove.update(bad_subset.index.values.tolist())
        subset = subset[subset.dist_to_shot_nearest_to_hole<=subset.Distance_to_Hole_after_the_Shot.values+d0]
        finish = subset.shape[0]
        if start-finish>5:
            print u, start-finish
        d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
        x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
        z = subset[subset.Distance_to_Hole_after_the_Shot!=0].Z_Coordinate.values
        a = find_the_hole()
        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
        mean_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.mean()
        if mean_err>.27:
            print u, 'mean_err = ', mean_err
        if mean_err>1.5:
            inds_to_remove.update(subset.index.values.tolist())
            print u, 'skipping and losing ',subset.shape[0]
            continue
        max_err = subset.dist_diff.max()
        if max_err>2:
            print u, 'max_err = ', max_err
            inds_to_remove.update(subset[subset.dist_diff>1].index.values.tolist())
        subset.drop('dist_w_impute',axis=1,inplace=True)
        subset.drop('dist_diff',axis=1,inplace=True)
        hole_locs.append((a[0],a[1],a[2]))
        

    #print data.shape
    print len(inds_to_remove)
    print shots_looked_at
    print len(hole_locs)


add_hole_locs(2004)