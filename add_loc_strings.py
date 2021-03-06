import pandas as pd
import numpy as np
import gc

def standardize(df,digitize,dist_bins=None,angle_bins=None,dist_bins_hole=None,angle_bins_hole=None):
    def rotate_theta_radians(points,theta):
        points = np.array(points).T
        r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return np.dot(r,points).T
    
    def convert_to_degrees(angle):
        if isinstance(angle,float):
            angle = angle + (2*np.pi if angle<0 else 0)
        else:
            angle[angle<0] = angle[angle<0] + 2*np.pi
        return angle*180./np.pi
    
    points = np.column_stack([df.Start_X_Coordinate.values - df.Cluster_Green_X.values[0],
                              df.Start_Y_Coordinate.values - df.Cluster_Green_Y.values[0]])
    angle_of_tee_box = np.arctan2(df.Cluster_Tee_Y.values[0] - df.Cluster_Green_Y.values[0],
                                  df.Cluster_Tee_X.values[0] - df.Cluster_Green_X.values[0])
    rotated = rotate_theta_radians(points,3*np.pi/2-angle_of_tee_box)
    angles = convert_to_degrees(np.arctan2(rotated[:,1],rotated[:,0]))
    distances_from_origin = (rotated**2).sum(1)**.5
    if digitize:
        angles = np.digitize(angles,angle_bins)
        distances_from_origin = np.digitize(distances_from_origin,dist_bins)
    
    hole_locs = {}
    for (year,tourn,round),df_ in df.groupby(['Year','Permanent_Tournament_#','Round']):
        hole_loc = df_.loc[df_.last_shot_mask,['End_X_Coordinate','End_Y_Coordinate']].iloc[0].values
        hole_loc = hole_loc - np.array([df.Cluster_Green_X.values[0],df.Cluster_Green_Y.values[0]])
        rotated = rotate_theta_radians(hole_loc,3*np.pi/2-angle_of_tee_box)
        angle = convert_to_degrees(np.arctan2(rotated[1],rotated[0]))
        distance_from_origin = (rotated**2).sum()**.5
        if digitize:
            angle = np.digitize(angle,angle_bins_hole)
            distance_from_origin = np.digitize(distance_from_origin,dist_bins_hole)
        hole_locs[(year,tourn,round)] = (angle,distance_from_origin)
    return angles, distances_from_origin, hole_locs

def get_bins(data,n_dist_bins,n_angle_bins,n_dist_bins_hole,n_angle_bins_hole):
  angles, distances = [], []
  for (course,hole,cluster),df in data.groupby(['Course_#','Hole','Cluster']):
      if df.shape[0]>1:
          angles_, dists_, _ = standardize(df,False)
          angles = np.concatenate([angles,angles_[(~df.from_the_tee_box_mask).values]])
          distances = np.concatenate([distances,dists_[(~df.last_shot_mask).values]])
  _, angle_bins = pd.qcut(angles,np.linspace(0,1,n_angle_bins),retbins=True)
  angle_bins[0], angle_bins[-1] = 0, 360
  _, dist_bins = pd.qcut(distances,np.linspace(0,1,n_dist_bins),retbins=True)
  dist_bins[0], dist_bins[-1] = 0, 1e10

  angles, distances = [], []
  for (course,hole,cluster),df in data.groupby(['Course_#','Hole','Cluster']):
      if df.shape[0]>1:
          _, _, hole_locs = standardize(df,False)
          arr = np.array(list(hole_locs.values()))
          angles_, distances_ = arr[:,0],arr[:,1]
          angles = np.concatenate([angles,angles_])
          distances = np.concatenate([distances,distances_])

  _, angle_bins_hole = pd.qcut(angles,np.linspace(0,1,n_angle_bins_hole),retbins=True)
  angle_bins_hole[0], angle_bins_hole[-1] = 0, 360
  _, dist_bins_hole = pd.qcut(distances,np.linspace(0,1,n_dist_bins_hole),retbins=True)
  dist_bins_hole[0], dist_bins_hole[-1] = 0, 1e10

  return dist_bins,angle_bins,dist_bins_hole,angle_bins_hole

def addlocStrings(data,n_dist_bins,n_angle_bins,n_dist_bins_hole,n_angle_bins_hole):
    dist_bins,angle_bins,dist_bins_hole,angle_bins_hole = get_bins(data,n_dist_bins,
                                                                   n_angle_bins,n_dist_bins_hole,
                                                                   n_angle_bins_hole)
    shot_id_cols = ['Permanent_Tournament_#','Course_#','Round','Hole','Player_#','Shot','Year']
    hole_id_cols = ['Permanent_Tournament_#','Course_#','Round','Hole','Year']
    locStrings_shots,locStrings_holes = {},{}
    for (course,hole,cluster),df in data.groupby(['Course_#','Hole','Cluster']):
        angles, dists, hole_locs = standardize(df,True,dist_bins,angle_bins,
                                               dist_bins_hole,angle_bins_hole)
        locStrings_shots.update({tuple(tup):'%d-%d-%d-%d' % (tup[1],tup[3],angle,dist) 
                                 for tup,angle,dist in zip(df[shot_id_cols].values,angles,dists)})
        locStrings_holes.update({tuple(tup):'%d-%d-%d-%d' % (tup[1],tup[3],
                                                             hole_locs[(tup[-1],tup[0],tup[2])][0],
                                                             hole_locs[(tup[-1],tup[0],tup[2])][1])
                                 for tup in df[hole_id_cols].values})
    data['loc_string'] = [locStrings_shots[tuple(tup)] for tup in data[shot_id_cols].values]
    data['loc_string_hole'] = [locStrings_holes[tuple(tup)] for tup in data[hole_id_cols].values]
    return data

def doit(n_dist_bins,n_angle_bins,n_dist_bins_hole,n_angle_bins_hole):
    cols = ['Course_#','Hole','Cluster','Cluster_Green_X','Cluster_Green_Y','from_the_tee_box_mask',
            'Cluster_Tee_Y','Cluster_Tee_X','Start_X_Coordinate','Start_Y_Coordinate','last_shot_mask',
            'End_Y_Coordinate','End_X_Coordinate','Permanent_Tournament_#','Round','Year']
    data = pd.concat([pd.read_csv('../GolfData/Shot/%d.csv.gz' % year, usecols=cols)
                      for year in range(2003,2019)])
    dist_bins,angle_bins,dist_bins_hole,angle_bins_hole = get_bins(data,n_dist_bins,
                                                                   n_angle_bins,n_dist_bins_hole,
                                                                   n_angle_bins_hole)
    np.save('latest_bins/dist_bins.npy',dist_bins)
    np.save('latest_bins/angle_bins.npy',angle_bins)
    np.save('latest_bins/dist_bins_hole.npy',dist_bins_hole)
    np.save('latest_bins/angle_bins_hole.npy',angle_bins_hole)
    data = None
    gc.collect()
    for year in range(2003,2019):
        data = pd.read_csv('../GolfData/Shot/%d.csv.gz' % year)
        shot_id_cols = ['Permanent_Tournament_#','Course_#','Round','Hole','Player_#','Shot','Year']
        hole_id_cols = ['Permanent_Tournament_#','Course_#','Round','Hole','Year']
        locStrings_shots,locStrings_holes = {},{}
        for (course,hole,cluster),df in data.groupby(['Course_#','Hole','Cluster']):
            angles, dists, hole_locs = standardize(df,True,dist_bins,angle_bins,
                                                   dist_bins_hole,angle_bins_hole)
            locStrings_shots.update({tuple(tup):'%d-%d-%d-%d' % (tup[1],tup[3],angle,dist) 
                                     for tup,angle,dist in zip(df[shot_id_cols].values,angles,dists)})
            locStrings_holes.update({tuple(tup):'%d-%d-%d-%d' % (tup[1],tup[3],
                                                                 hole_locs[(tup[-1],tup[0],tup[2])][0],
                                                                 hole_locs[(tup[-1],tup[0],tup[2])][1])
                                     for tup in df[hole_id_cols].values})
        data['loc_string'] = [locStrings_shots[tuple(tup)] for tup in data[shot_id_cols].values]
        data['loc_string_hole'] = [locStrings_holes[tuple(tup)] for tup in data[hole_id_cols].values]
        data.to_csv('../GolfData/Shot/%d.csv.gz' % year, compression='gzip', index=False)
