import pandas as pd
import itertools
import eliminate_holes_with_issues as e


def dist_to_shot_nearest_to_hole_in (subset,n_coor):
    """ Takes in a sorted subset of data and returns a numpy array with the
    distance from each shot to the closest recorded shot in inches

    subset - pandas DataFrame
    n_coor - int (either 2 for x and y coordinates or 3 for x,y, and z coordinates)
    """
    x_coors = subset.X_Coordinate.values
    y_coors = subset.Y_Coordinate.values
    cl_sh_x = subset.X_Coordinate.values[0]
    cl_sh_y = subset.Y_Coordinate.values[0]
    if n_coor==2:
        return ((x_coors-cl_sh_x)**2 + (y_coors-cl_sh_y)**2)**.5 * 12.0
    else:
        z_coors = subset.Z_Coordinate.values
        cl_sh_z = subset.Z_Coordinate.values[0]
        return ((x_coors-cl_sh_x)**2 + (y_coors-cl_sh_y)**2 + (z_coors-cl_sh_z)**2)**.5 * 12.0


two_vesus_three = [[0,0,0] for _ in range(14)]

for year in range(2003,2017):    
    data = e.make_df(year,verbose=False)
    print year
    #unique Course-Round-Hole Tuples
    uCRHtps = list(itertools.product(pd.unique(data.Course_Name),pd.unique(data.Round),pd.unique(data.Hole)))

    for u,i in enumerate(uCRHtps):
        if u%1000==0:
            print u
        subset = data[(data.Course_Name==i[0]) & (data.Round==int(i[1])) & (data.Hole==int(i[2])) &  \
                (data.Distance_to_Hole_after_the_Shot!=0) & (data.X_Coordinate!=0) & (data.Y_Coordinate!=0) & (data.Z_Coordinate!=0)]
        if subset.shape[0] == 0:
            continue
        subset = subset.sort_values('Distance_to_Hole_after_the_Shot')
        
        ## compare which distance is most compatible with the data based on recorded distance to hole
        d0 = subset.Distance_to_Hole_after_the_Shot.values[0]
        dist_to_shot_nearest_to_hole = dist_to_shot_nearest_to_hole_in(subset,2)
        n_badshots2 = subset[dist_to_shot_nearest_to_hole > subset.Distance_to_Hole_after_the_Shot.values + d0 + 1.0].shape[0]
        dist_to_shot_nearest_to_hole = dist_to_shot_nearest_to_hole_in(subset,3)
        n_badshots3 = subset[dist_to_shot_nearest_to_hole > subset.Distance_to_Hole_after_the_Shot.values + d0 + 1.0].shape[0]

        if n_badshots2>n_badshots3:
            two_vesus_three[year-2003][0] += 1
        elif n_badshots2==n_badshots3:
            two_vesus_three[year-2003][1] += 1
        else:
            two_vesus_three[year-2003][2] += 1
    print 'Number of Holes with more inconsistant shots using x,y coordinates: ', two_vesus_three[year-2003][0]
    print 'Number of Holes with more inconsistant shote using x,y, and z coordinates: ', two_vesus_three[year-2003][2]
    print 'Number of Hole with the same number of inconsistant shots either way: ', two_vesus_three[year-2003][1]
