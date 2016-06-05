import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc
import eliminate_holes_with_issues as e


def get_hole_and_tee_box_locs (y):
    def f (a):
        x0,y0 = a[0],a[1]
        return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

    def find_the_hole ():
        xopt = fmin_tnc(f,[x0,y0],approx_grad=1,maxfun=1000,disp=0)[0].tolist()
        return xopt

    data = e.make_df(y)

    #unique Course-Round-Hole Tuples
    uCRHtps = list(itertools.product(pd.unique(data.Course_Name),pd.unique(data.Round),pd.unique(data.Hole)))

    # coordinates of hole are not given. must be imputed.
    # based on previous test, x,y, and z coordinates are all used for distance

    # finding the coordinates of the hole and recording the results
    # must treat shots that were holed out differently since the coordinates are recorded as 0 for those shots
    tups_to_remove = set()
    id_cols = ['Year','Course_#','Player_#','Round','Hole'] 
    shots_looked_at = 0
    uCRHs = 0
    hole_locs, tee_box_locs = {}, {}
    zerocoordsnonzerodist, lackofdists, outofsynchwithclosestshot, over2stdshole, wholeholeout, maxholetoobig, over2stdteebox, wholeteeboxout, maxteeboxtoobig = 0,0,0,0,0,0,0,0,0
    for u,i in enumerate(uCRHtps):
        subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2]))]
        shots_looked_at += subset.shape[0]
        zerocoordsnonzerodist += sum([1 for j in subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))][id_cols].as_matrix().tolist()])
        tups_to_remove.update([tuple(j) for j in subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))][id_cols].as_matrix().tolist()])
        subset = subset.drop(subset[(subset.Distance_to_Hole_after_the_Shot!=0) & ((subset.X_Coordinate==0) | (subset.Y_Coordinate==0) | (subset.Z_Coordinate==0))].index,axis=0)
        if len(subset)==0:
            continue
        if subset[subset.Distance_to_Hole_after_the_Shot!=0].shape[0] == 0:
            print u, ' No non 0 dists'
            lackofdists += sum([1 for j in subset[id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
        subset = subset.sort_values('Distance_to_Hole_after_the_Shot')
        d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
        x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
        x0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values[0] ##assume that closest ball recorded to hole does not have an error
        y0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values[0]
        d0 = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values[0]/12.0
        subset.insert(len(subset.columns),'dist_to_shot_nearest_to_hole',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-x0)**2 + (y-y0)**2)**.5).tolist()))
        # keep track of hole's with shots that have distances that are inconsistant with the coordinates
        bad_subset = subset[subset.dist_to_shot_nearest_to_hole>subset.Distance_to_Hole_after_the_Shot.values+d0]
        outofsynchwithclosestshot += sum([1 for j in bad_subset[id_cols].as_matrix().tolist()])
        tups_to_remove.update([tuple(j) for j in bad_subset[id_cols].as_matrix().tolist()])
        subset = subset[subset.dist_to_shot_nearest_to_hole<=subset.Distance_to_Hole_after_the_Shot.values+d0]
        
        d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
        x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
        a = find_the_hole()

        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
        mean_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.mean()
        std_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.std()
        c = 0
        while mean_err>1.5:
            c+=1
            if c>=25:
                break
            print u,'hole',mean_err
            over2stdshole += sum([1 for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
            subset = subset.drop('dist_w_impute',axis=1)
            subset = subset.drop('dist_diff',axis=1)
            d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance.values/12.0
            x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
            y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
            a = find_the_hole()
            subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2))**.5.tolist()))
            subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
            mean_err = subset[subset.dist_diff>0].dist_diff.mean()
            std_err = subset[subset.dist_diff>0].dist_diff.std()
        if c==25:
            print 'hole Skipping ', u, len(subset)
            wholeholeout += sum([1 for j in subset[id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
        max_err = subset.dist_diff.max()
        if max_err>10:
            print u, 'hole max_err = ', max_err
            maxholetoobig += sum([1 for j in subset[subset.dist_diff>10][id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff>10][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff>1].index,axis=0)
        subset = subset.drop('dist_w_impute',axis=1)
        subset = subset.drop('dist_diff',axis=1)
        hole_locs[i] = a

        subset = subset.sort_values('Shot',ascending=False)
        d = subset[subset.Shot==1].Distance.values/12.0
        x = subset[subset.Shot==1].X_Coordinate.values
        y = subset[subset.Shot==1].Y_Coordinate.values
        rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
        rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate']].values[rand_ind,:].tolist()[0]
        x0,y0 = rand_shot[0],rand_shot[1]
        b = find_the_hole()
        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Shot']!=1].shape[0] + (((x-a[0])**2 + (y-a[1])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12.0) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
        mean_err = subset[subset.dist_diff>0].dist_diff.mean()
        std_err = subset[subset.dist_diff>0].dist_diff.std()
        c=0
        while mean_err>252:
            c+=1
            if c>=25:
                break
            print u,'tee_box',mean_err
            over2stdteebox += sum([1 for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
            subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
            subset = subset.drop('dist_w_impute',axis=1)
            subset = subset.drop('dist_diff',axis=1)
            d = subset[subset.Shot==1].Distance.values/12.0
            x = subset[subset.Shot==1].X_Coordinate.values
            y = subset[subset.Shot==1].Y_Coordinate.values
            rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
            rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate']].values[rand_ind,:].tolist()[0]
            x0,y0 = rand_shot[0],rand_shot[1]
            b = find_the_hole()
            subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Shot!=1].shape[0] + (((x-b[0])**2 + (y-b[1])**2)**.5).tolist()))
            subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
            mean_err = subset[subset.dist_diff>0].dist_diff.mean()
            std_err = subset[subset.dist_diff>0].dist_diff.std()
        if c==25:
            print 'tee box Skipping ', u, len(subset)
            wholeteeboxout += sum([1 for j in subset[id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
            continue
        if max_err>720:
            print u, 'tee box max_err = ', max_err
            maxteeboxtoobig += sum([1 for j in subset[subset.dist_diff>720][id_cols].as_matrix().tolist()])
            tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff>720][id_cols].as_matrix().tolist()])

        uCRHs+=1

        tee_box_locs[i] = b

    print uCRHs, len(hole_locs), len(tee_box_locs)
    print 'zerocoordsnonzerodist = ', zerocoordsnonzerodist
    print  'lackofdists = ', lackofdists
    print 'over2stdshole = ', over2stdshole
    print 'wholeholeout = ', wholeholeout
    print 'maxholetoobig = ', maxholetoobig
    print 'over2stdteebox = ', over2stdteebox
    print 'wholeteeboxout = ', wholeteeboxout
    print 'maxteeboxtoobig = ', maxteeboxtoobig
    print 'total = ', len(tups_to_remove)
    print shots_looked_at
    return (hole_locs,tee_box_locs,tups_to_remove)

a,b,c = get_hole_and_tee_box_locs(2007)