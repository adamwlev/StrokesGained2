import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc
import eliminate_holes_with_issues as e



def get_hole_and_tee_box_locs (y):
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
    tups_to_remove = set()
    id_cols = ['Year','Course_#','Player_#','Round','Hole'] 
    shots_looked_at = 0
    uCRHs = 0
    hole_locs, tee_box_locs = [], []
    for u,i in enumerate(uCRHtps):
        subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2]))]
        shots_looked_at += subset.shape[0]
        tups_to_remove.update([tuple(j) for j in subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))][id_cols].as_matrix().tolist()])
        subset = subset.drop(subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))].index,axis=0)
        if len(subset)==0:
            continue
        if subset[subset.Distance_to_Hole_after_the_Shot!=0].shape[0] == 0:
            print u, ' No non 0 dists'
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
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
        # keep track of hole's with shots that have distances that are inconsistant with the coordinates
        bad_subset = subset[subset.dist_to_shot_nearest_to_hole>subset.Distance_to_Hole_after_the_Shot.values+d0]
        tups_to_remove.update([tuple(j) for j in bad_subset[id_cols].as_matrix().tolist()])
        subset = subset[subset.dist_to_shot_nearest_to_hole<=subset.Distance_to_Hole_after_the_Shot.values+d0]
        
        d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
        x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
        z = subset[subset.Distance_to_Hole_after_the_Shot!=0].Z_Coordinate.values
        a = find_the_hole()
        hole_locs.append(a)

        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
        mean_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.mean()
        std_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.std()
        c = 0
        while mean_err>1.5:
            c+=1
            if c>=25:
                break
            print u,'hole',mean_err
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
            subset = subset.drop('dist_w_impute',axis=1)
            subset = subset.drop('dist_diff',axis=1)
            d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance.values/12.0
            x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
            y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
            z = subset[subset.Distance_to_Hole_after_the_Shot!=0].Z_Coordinate.values
            a = find_the_hole()
            subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
            subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
            mean_err = subset[subset.dist_diff>0].dist_diff.mean()
            std_err = subset[subset.dist_diff>0].dist_diff.std()
        if c==25:
            print 'hole Skipping ', u, len(subset)
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
        max_err = subset.dist_diff.max()
        if max_err>2:
            print u, 'hole max_err = ', max_err
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff>1][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff>1].index,axis=0)
        subset.drop('dist_w_impute',axis=1,inplace=True)
        subset.drop('dist_diff',axis=1,inplace=True)

        subset = subset.sort_values('Shot',ascending=False)
        d = subset[subset.Shot==1].Distance.values/12.0
        x = subset[subset.Shot==1].X_Coordinate.values
        y = subset[subset.Shot==1].Y_Coordinate.values
        z = subset[subset.Shot==1].Z_Coordinate.values
        rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
        rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate','Z_Coordinate']].values[rand_ind,:].tolist()[0]
        x0,y0,z0 = rand_shot[0],rand_shot[1],rand_shot[2]
        b = find_the_hole()
        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Shot']!=1].shape[0] + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12.0) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
        mean_err = subset[subset.dist_diff>0].dist_diff.mean()
        std_err = subset[subset.dist_diff>0].dist_diff.std()
        c=0
        while mean_err>252:
            c+=1
            if c>=25:
                break
            print u,'tee_box',mean_err
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
            subset = subset.drop('dist_w_impute',axis=1)
            subset = subset.drop('dist_diff',axis=1)
            d = subset[subset.Shot==1].Distance.values/12.0
            x = subset[subset.Shot==1].X_Coordinate.values
            y = subset[subset.Shot==1].Y_Coordinate.values
            z = subset[subset.Shot==1].Z_Coordinate.values
            rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
            rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate','Z_Coordinate']].values[rand_ind,:].tolist()[0]
            x0,y0,z0 = rand_shot[0],rand_shot[1],rand_shot[2]
            b = find_the_hole()
            subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Shot!=1].shape[0] + (((x-b[0])**2 + (y-b[1])**2 + (z-b[2])**2)**.5).tolist()))
            subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
            mean_err = subset[subset.dist_diff>0].dist_diff.mean()
            std_err = subset[subset.dist_diff>0].dist_diff.std()
        print 'tee box mean_err = ', mean_err
        print 'tee box max_err = ', subset[subset.dist_diff>0].dist_diff.max()
        if c==25:
            print 'tee box Skipping ', u, len(subset)
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
        
        subset = subset.drop('dist_w_impute',axis=1)
        subset = subset.drop('dist_diff',axis=1)
        uCRHs+=1

        tee_box_locs.append(b)

    print uCRHs
    print len(tups_to_remove)
    print shots_looked_at
    return (a,b)

a,b = get_hole_and_tee_box_locs(2004)