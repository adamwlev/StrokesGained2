import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc
import eliminate_holes_with_issues as e


class Impute_a_Hole (object):
    """Represents process of imputing missing hole and tee box locations for a one single hole"""
    def __init__(self,course,round,hole,data):
        self.df = data[(data['Course_#']==course) & (data.Round==round) & (data.Hole==hole)]
        self.tuples_to_remove = set()
    def __str__(self):
        return self.df[['Course_#','Round','Hole','Hole_Score','Shot','Distance_to_Hole_after_the_Shot']].__str__()

    @staticmethod
    def _f(a,x,y,d):
        """Returns mean squared error from location to guess for hole or tee box location.
        Function to be minimized for imputing locations."""
        x0,y0 = a[0],a[1]
        return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

    def _find_the_hole(self):
        """Finds best estimate for hole or tee box

        Precondition: The following variables are defined for the instance
        x - x coordinates of the shots
        y - y coordinates of the shots
        d - distance recorded of shots from hole (or tee box)
        x0 - initial guess for location x coordinate
        y0 - initial guess for location y coordinate
        """
        xopt = fmin_tnc(Impute_a_Hole._f,[self.x0,self.y0],args=(self.x,self.y,self.d),approx_grad=1,maxfun=1000,disp=0)[0].tolist()
        return xopt

    id_cols = ['Year','Course_#','Player_#','Round','Hole']

    def filter_out_shots_with_zero_coordinates_but_nonzero_distances(self):
        """Removes said shots. Records the Player-Hole-Round tuples that have said shots."""
        self.tuples_to_remove.update(tuple(j) for j in self.df[(self.df.Distance_to_Hole_after_the_Shot!=0) & 
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))][Impute_a_Hole.id_cols].as_matrix().tolist())
        size = len(self.df[(self.df.Distance_to_Hole_after_the_Shot!=0) & 
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))])
        self.df = self.df.drop(self.df[(self.df.Distance_to_Hole_after_the_Shot!=0) & 
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))].index,axis=0)
        return size

    def filter_out_shots_with_zero_distances_from_hole_but_not_last_shot(self):
        """Removes said shots. Records the Player-Hole-Round tuples that have said shots."""
        self.tuples_to_remove.update(tuple(j) for j in self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & 
            (self.df.Shot!=self.df.Hole_Score)][Impute_a_Hole.id_cols].as_matrix().tolist())
        size = len(self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & (self.df.Shot!=self.df.Hole_Score)])
        self.df = self.df.drop(self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & (self.df.Shot!=self.df.Hole_Score)].index,axis=0)
        return size

    def are_there_any_non_zero_distances_from_the_hole_after_the_shot(self):
        """Checking to make sure that there are shots in the data that have nonzero distance from the hole.
        If this is not true, this hole is worthless and will be skipped over."""
        return bool(len(self.df[self.df.Distance_to_Hole_after_the_Shot!=0]))

    def set_attributes_before_finding_the_hole(self):
        """Sets attributes required for instance before calling find_the_hole"""
        self.d = self.df.Distance_to_Hole_after_the_Shot.values / 12.0
        self.x = self.df.X_Coordinate.values
        self.y = self.df.Y_Coordinate.values

        self.sorted_df = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].sort_values('Distance_to_Hole_after_the_Shot')
        self.x0 = self.sorted_df.X_Coordinate.values[0]
        self.y0 = self.sorted_df.Y_Coordinate.values[0]

    def find_the_hole(self,max_allowable_mean_error,max_allowable_max_error,iter=25):
        """This attempts to impute the hole location. If the mean error of the difference between the
        imputed distances and the recorded distance is larger than max_allowable_mean_error, Player-Hole-Round
        tuples are removed from the data according the amount of error present for a shot. Then hole location
        is reimputed. 25 attempts to reimpute are made on default.
        All shots that have differences between the imputed distance and the recorded distance that are greater 
        than the max_allowable_max_error are removed and the Player-Hole-Round tuples that they correspond to 
        are recorded.
        """
        self.set_attributes_before_finding_the_hole()
        hole_x, hole_y = self._find_the_hole()
        error = np.abs(((self.x - hole_x)**2 + (self.y - hole_y)**2)**.5 - self.d)
        
        c = 0
        while np.mean(error) > max_allowable_mean_error:
            c += 1
            if c>=25: return False
            mean_error = np.mean(error); std_error = np.std(error)
            self.tuples_to_remove.update(tuple(j) for j in self.df[error > mean_error + 2.5*std_error][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.df[error > mean_error + 2.5*std_error].index,axis=0)
            self.set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()
            error = np.abs(((self.x - hole_x)**2 + (self.y - hole_y)**2)**.5 - self.d)

        if np.amax(error) > max_allowable_max_error:
            self.tuples_to_remove.update(tuple(j) for j in self.df[error > max_allowable_max_error][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.df[error > max_allowable_max_error].index,axis=0)
            self.set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()

        if len(self.df)==0: return False
        else: self.hole_x = hole_x; self.hole_y = hole_y; return True

data = e.make_df(2003)
uCRHtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))
hole = Impute_a_Hole(uCRHtps[0][0],uCRHtps[0][1],uCRHtps[0][2],data)
print hole.filter_out_shots_with_zero_coordinates_but_nonzero_distances()
print hole.filter_out_shots_with_zero_distances_from_hole_but_not_last_shot()
print hole.are_there_any_non_zero_distances_from_the_hole_after_the_shot()
print hole.find_the_hole(1.5,10)



#     hole_locs, tee_box_locs = {}, {}
#     zerocoordsnonzerodist, lackofdists, outofsynchwithclosestshot, over2stdshole, wholeholeout, maxholetoobig, over2stdteebox, wholeteeboxout, maxteeboxtoobig = 0,0,0,0,0,0,0,0,0
#         if len(subset)==0:
#             continue
#         if subset[subset.Distance_to_Hole_after_the_Shot!=0].shape[0] == 0:
#             print u, ' No non 0 dists'
#             lackofdists += sum([1 for j in subset[id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
#             continue
#         
        
#         d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values/12.0
#         x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
#         y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
#         a = find_the_hole()

#         subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2)**.5).tolist()))
#         subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
#         mean_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.mean()
#         std_err = subset[subset.Distance_to_Hole_after_the_Shot!=0].dist_diff.std()
#         c = 0
#         while mean_err>1.5:
#             c+=1
#             if c>=25:
#                 break
#             print u,'hole',mean_err
#             over2stdshole += sum([1 for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
#             subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
#             subset = subset.drop('dist_w_impute',axis=1)
#             subset = subset.drop('dist_diff',axis=1)
#             d = subset[subset.Distance_to_Hole_after_the_Shot!=0].Distance.values/12.0
#             x = subset[subset.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
#             y = subset[subset.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values
#             a = find_the_hole()
#             subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Distance_to_Hole_after_the_Shot==0].shape[0] + (((x-a[0])**2 + (y-a[1])**2))**.5.tolist()))
#             subset.insert(len(subset.columns),'dist_diff', np.absolute(subset.dist_w_impute.values - subset.Distance_to_Hole_after_the_Shot.values/12.0))
#             mean_err = subset[subset.dist_diff>0].dist_diff.mean()
#             std_err = subset[subset.dist_diff>0].dist_diff.std()
#         if c==25:
#             print 'hole Skipping ', u, len(subset)
#             wholeholeout += sum([1 for j in subset[id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
#             continue
#         max_err = subset.dist_diff.max()
#         if max_err>10:
#             print u, 'hole max_err = ', max_err
#             maxholetoobig += sum([1 for j in subset[subset.dist_diff>10][id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff>10][id_cols].as_matrix().tolist()])
#             subset = subset.drop(subset[subset.dist_diff>1].index,axis=0)
#         subset = subset.drop('dist_w_impute',axis=1)
#         subset = subset.drop('dist_diff',axis=1)
#         hole_locs[i] = a

#         subset = subset.sort_values('Shot',ascending=False)
#         d = subset[subset.Shot==1].Distance.values/12.0
#         x = subset[subset.Shot==1].X_Coordinate.values
#         y = subset[subset.Shot==1].Y_Coordinate.values
#         rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
#         rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate']].values[rand_ind,:].tolist()[0]
#         x0,y0 = rand_shot[0],rand_shot[1]
#         b = find_the_hole()
#         subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Shot']!=1].shape[0] + (((x-a[0])**2 + (y-a[1])**2)**.5).tolist()))
#         subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12.0) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
#         mean_err = subset[subset.dist_diff>0].dist_diff.mean()
#         std_err = subset[subset.dist_diff>0].dist_diff.std()
#         c=0
#         while mean_err>252:
#             c+=1
#             if c>=25:
#                 break
#             print u,'tee_box',mean_err
#             over2stdteebox += sum([1 for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff > mean_err + 2.5*std_err][id_cols].as_matrix().tolist()])
#             subset = subset.drop(subset[subset.dist_diff > mean_err + 2.5*std_err].index,axis=0)
#             subset = subset.drop('dist_w_impute',axis=1)
#             subset = subset.drop('dist_diff',axis=1)
#             d = subset[subset.Shot==1].Distance.values/12.0
#             x = subset[subset.Shot==1].X_Coordinate.values
#             y = subset[subset.Shot==1].Y_Coordinate.values
#             rand_ind = np.random.choice(range(subset[subset.Shot==1].shape[0]),size=1)
#             rand_shot = subset[subset.Shot==1][['X_Coordinate','Y_Coordinate']].values[rand_ind,:].tolist()[0]
#             x0,y0 = rand_shot[0],rand_shot[1]
#             b = find_the_hole()
#             subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset.Shot!=1].shape[0] + (((x-b[0])**2 + (y-b[1])**2)**.5).tolist()))
#             subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset.dist_w_impute.values[j] - subset.Distance.values[j]/12) if subset.Shot.values[j]==1 else 0 for j in range(subset.shape[0])]))
#             mean_err = subset[subset.dist_diff>0].dist_diff.mean()
#             std_err = subset[subset.dist_diff>0].dist_diff.std()
#         if c==25:
#             print 'tee box Skipping ', u, len(subset)
#             wholeteeboxout += sum([1 for j in subset[id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[id_cols].as_matrix().tolist()])
#             continue
#         if max_err>720:
#             print u, 'tee box max_err = ', max_err
#             maxteeboxtoobig += sum([1 for j in subset[subset.dist_diff>720][id_cols].as_matrix().tolist()])
#             tups_to_remove.update([tuple(j) for j in subset[subset.dist_diff>720][id_cols].as_matrix().tolist()])

#         uCRHs+=1

#         tee_box_locs[i] = b

#     print uCRHs, len(hole_locs), len(tee_box_locs)
#     print 'zerocoordsnonzerodist = ', zerocoordsnonzerodist
#     print  'lackofdists = ', lackofdists
#     print 'over2stdshole = ', over2stdshole
#     print 'wholeholeout = ', wholeholeout
#     print 'maxholetoobig = ', maxholetoobig
#     print 'over2stdteebox = ', over2stdteebox
#     print 'wholeteeboxout = ', wholeteeboxout
#     print 'maxteeboxtoobig = ', maxteeboxtoobig
#     print 'total = ', len(tups_to_remove)
#     print shots_looked_at
#     return (hole_locs,tee_box_locs,tups_to_remove)

# a,b,c = get_hole_and_tee_box_locs(2007)