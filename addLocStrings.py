import pandas as pd
import numpy as np

angle_bins = np.array([0.,30.51178516,61.67090908,94.04625862,
                       121.43171077,149.18913484,178.92034735,206.49623801,
                       230.27777578,245.70639433,255.40606863,262.38586754,
                       268.21019596,273.7904887 ,279.73740368,286.76391868,
                       296.08925169,310.25414107,331.99685933,360.        ])
dist_bins = np.array([0.00000000e+00,2.86803800e+01,5.04239539e+01,
                      1.86821980e+02,4.36973779e+02,5.62549775e+02,
                      8.56281230e+02,1.24960457e+03,1.44277181e+03,1.00000000e+10])

def standardize(df):
    def rotate_theta_radians(points,theta):
        points = np.array(points).T
        r = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return np.dot(r,points).T
    
    def convert_to_degrees(angle):
        angle[angle<0] = angle[angle<0] + 2*np.pi
        return angle*180./np.pi
    
    points = np.column_stack([df.Start_X_Coordinate.values - df.Cluster_Green_X.values[0],
                              df.Start_Y_Coordinate.values - df.Cluster_Green_Y.values[0]])
    angle_of_tee_box = np.arctan2(df.Cluster_Tee_Y.values[0] - df.Cluster_Green_Y.values[0],
                                  df.Cluster_Tee_X.values[0] - df.Cluster_Green_X.values[0])
    rotated = rotate_theta_radians(points,3*np.pi/2-angle_of_tee_box)
    angles = convert_to_degrees(np.arctan2(rotated[:,1],rotated[:,0]))
    distance_from_origin = (rotated**2).sum(1)**.5
    angles = np.digitize(angles,angle_bins)
    distance_from_origin = np.digitize(distance_from_origin,dist_bins)
    return angles, distance_from_origin

def doit(year):
    data = pd.read_csv('data/%d.csv' % year)
    shot_id_cols = ['Year','Permanent_Tournament_#','Course_#','Round','Hole','Player_#','Shot']
    locStrings = {}
    for (course,hole,cluster),df in data.groupby(['Course_#','Hole','Cluster']):
        angles, dists = standardize(df)
        locStrings.update({tuple(tup):'%d-%d-%d-%d' % (tup[2],tup[3],angle,dist) 
                           for tup,angle,dist in zip(df[shot_id_cols].values,angles,dists)})
    data['loc_string'] = [locStrings[tuple(tup)] for tup in data[shot_id_cols].values]
    data.to_csv('%d.csv' % year, index=False)
