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
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))][Impute_a_Hole.id_cols].values.astype(int).tolist())
        size = len(self.df[(self.df.Distance_to_Hole_after_the_Shot!=0) & 
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))])
        self.df = self.df.drop(self.df[(self.df.Distance_to_Hole_after_the_Shot!=0) & 
            ((self.df.X_Coordinate==0) | (self.df.Y_Coordinate==0) | (self.df.Z_Coordinate==0))].index,axis=0)
        return size

    def filter_out_shots_with_zero_distances_from_hole_but_not_last_shot(self):
        """Removes said shots. Records the Player-Hole-Round tuples that have said shots."""
        self.tuples_to_remove.update(tuple(j) for j in self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & 
            (self.df.Shot!=self.df.Hole_Score)][Impute_a_Hole.id_cols].values.astype(int).tolist())
        size = len(self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & (self.df.Shot!=self.df.Hole_Score)])
        self.df = self.df.drop(self.df[(self.df.Distance_to_Hole_after_the_Shot==0) & (self.df.Shot!=self.df.Hole_Score)].index,axis=0)

    def are_there_any_non_zero_distances_from_the_hole_after_the_shot(self):
        """Checking to make sure that there are shots in the data that have nonzero distance from the hole.
        If this is not true, this hole is worthless and will be skipped over."""
        if bool(len(self.df[self.df.Distance_to_Hole_after_the_Shot!=0])):
            return True
        else:
            self.tuples_to_remove.update(tuple(j) for j in self.df[Impute_a_Hole.id_cols].values.astype(int).tolist())
            return False

    def _set_attributes_before_finding_the_hole(self):
        """Sets attributes required for instance before calling _find_the_hole.
        Since coordinates of shots that went into the hole are 0, we must exclude
        those that have a 0 distance to hole after the shot (the shots that wen in the hole)"""
        self.d = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].Distance_to_Hole_after_the_Shot.values / 12.0
        self.x = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].X_Coordinate.values
        self.y = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].Y_Coordinate.values

        self.sorted_df = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].sort_values('Distance_to_Hole_after_the_Shot')
        self.x0 = self.sorted_df.X_Coordinate.values[0]
        self.y0 = self.sorted_df.Y_Coordinate.values[0]

    def find_the_hole(self,max_allowable_mean_abs_deviation,max_allowable_max_abs_deviation,iter=25):
        """This attempts to impute the hole location. If the mean error of the difference between the
        imputed distances and the recorded distance is larger than max_allowable_mean_error, Player-Hole-Round
        tuples are removed from the data according the amount of error present for a shot. Then hole location
        is reimputed. 25 attempts to re-impute are made on default.
        All shots that have differences between the imputed distance and the recorded distance that are greater 
        than the max_allowable_max_error are removed and the Player-Hole-Round tuples that they correspond to 
        are recorded.
        """
        before = len(self.tuples_to_remove)
        self._set_attributes_before_finding_the_hole()
        self.without_holed_shots = self.df[self.df.Distance_to_Hole_after_the_Shot!=0]
        hole_x, hole_y = self._find_the_hole()
        abs_deviation = np.abs(((self.x - hole_x)**2 + (self.y - hole_y)**2)**.5 - self.d)
        
        c = 0
        while np.mean(abs_deviation) > max_allowable_mean_abs_deviation:
            #print len(self.df), np.mean(abs_deviation)
            c += 1
            if c>=25:
                #print '\033[91m','25 attempts were made to remove outliers. Imputation failed. Removed whole hole.','\033[0m'
                self.tuples_to_remove.update(tuple(j) for j in self.df[Impute_a_Hole.id_cols].values.astype(int).tolist())
                return False
            mean_abs_deviation = np.mean(abs_deviation); std_abs_deviation = np.std(abs_deviation)
            self.tuples_to_remove.update(tuple(j) for j in self.without_holed_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation][Impute_a_Hole.id_cols].values.astype(int).tolist())
            self.df = self.df.drop(self.without_holed_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self.without_holed_shots = self.without_holed_shots.drop(self.without_holed_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()
            abs_deviation = np.abs(((self.x - hole_x)**2 + (self.y - hole_y)**2)**.5 - self.d)

        if np.amax(abs_deviation) > max_allowable_max_abs_deviation:
            self.tuples_to_remove.update(tuple(j) for j in self.without_holed_shots[abs_deviation > max_allowable_max_abs_deviation][Impute_a_Hole.id_cols].values.astype(int).tolist())
            self.df = self.df.drop(self.without_holed_shots[abs_deviation > max_allowable_max_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()

        after = len(self.tuples_to_remove)
        if (after-before)>0:
            #print '\033[91m','There were %d player-holes removed during imputation of hole location.' % (after-before,),'\033[0m'
            pass
        else:
            #print '\033[92m', 'Imputation Successful, no shots removed.','\033[0m'
            pass
        if len(self.df)==0: 
            return False
        else: 
            self.hole_x = hole_x
            self.hole_y = hole_y
            return True

    def _set_attributes_before_finding_the_tee_box(self):
        """Sets attributes required for instance before calling _find_the_teebox.
        Since only information for distance from teebox is on first shot we will
        only count the first shots for the hole. Initial guess is set as the 
        coordinates that the shortest tee shot ended up at.
        """
        self.d = self.df[self.df.Shot==1].Distance.values / 12.0
        self.x = self.df[self.df.Shot==1].X_Coordinate.values
        self.y = self.df[self.df.Shot==1].Y_Coordinate.values

        self.sorted_df = self.df[self.df.Shot==1].sort_values('Distance')
        self.x0 = self.sorted_df.X_Coordinate.values[0]
        self.y0 = self.sorted_df.Y_Coordinate.values[0]

    def find_the_tee_box(self,max_allowable_mean_abs_deviation,max_allowable_max_abs_deviation,iter=25):
        """Same behavior as find_the_hole except now we will find the tee box.
        The distance measurements seem to be far less exact with these long shots so
        the max_allowable_mean_abs_deviation and max_allowable_max_abs_deviation
        will be set much higher.
        """

        before = len(self.tuples_to_remove)
        self._set_attributes_before_finding_the_tee_box()
        self.only_tee_shots = self.df[self.df.Shot==1]
        tee_box_x, tee_box_y = self._find_the_hole()
        abs_deviation = np.abs(((self.x - tee_box_x)**2 + (self.y - tee_box_y)**2)**.5 - self.d)
        
        c = 0
        while np.mean(abs_deviation) > max_allowable_mean_abs_deviation:
            #print len(self.df), np.mean(abs_deviation)
            c += 1
            if c>=25:
                #print '\033[91m','25 attempts were made to remove outliers. Imputation failed. Removed whole hole.','\033[0m'
                self.tuples_to_remove.update(tuple(j) for j in self.df[Impute_a_Hole.id_cols].values.astype(int).tolist())
                return False
            mean_abs_deviation = np.mean(abs_deviation); std_abs_deviation = np.std(abs_deviation)
            self.tuples_to_remove.update(tuple(j) for j in self.only_tee_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation][Impute_a_Hole.id_cols].values.astype(int).tolist())
            self.df = self.df.drop(self.only_tee_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self.only_tee_shots = self.only_tee_shots.drop(self.only_tee_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_tee_box()
            tee_box_x, tee_box_y = self._find_the_hole()
            abs_deviation = np.abs(((self.x - tee_box_x)**2 + (self.y - tee_box_y)**2)**.5 - self.d)

        if np.amax(abs_deviation) > max_allowable_max_abs_deviation:
            self.tuples_to_remove.update(tuple(j) for j in self.only_tee_shots[abs_deviation > max_allowable_max_abs_deviation][Impute_a_Hole.id_cols].values.astype(int).tolist())
            self.df = self.df.drop(self.only_tee_shots[abs_deviation > max_allowable_max_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_tee_box()
            tee_box_x, tee_box_y = self._find_the_hole()

        after = len(self.tuples_to_remove)
        if (after-before)>0:
            pass
            #print '\033[91m','There were %d player-holes removed during imputation of hole location.' % (after-before,),'\033[0m'
        else:
            pass
            #print '\033[92m','Imputation Successful, no shots removed.','\033[0m'
        if len(self.df)==0: 
            return False
        else: 
            self.tee_box_x = tee_box_x
            self.tee_box_y = tee_box_y
            return True

    def set_z_of_closest(self):
        """This sets an attribute for the z coordinate of the closest shot recorded to the hole. Since imputation
        of the z coordinate of the hole is impossible, I'll use this as an approximation for the z coordinate of the hole. """
        self.sorted_df = self.df[self.df.Distance_to_Hole_after_the_Shot!=0].sort_values('Distance_to_Hole_after_the_Shot')
        self.z_of_closest = self.sorted_df.Z_Coordinate.values[0]

f = open('data/data.csv','w')
f.close()

for year in range(2003,2017):
    print year
    data = e.make_df(year,verbose=True)
    uCRHtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))
    hole_locations = {}
    tee_box_locations = {}
    z_of_closest = {}
    tuples_to_remove = set()
    for tup in uCRHtps:
        hole = Impute_a_Hole(tup[0],tup[1],tup[2],data)
        #print tup
        #print 'Number of shots with zero coordinates but nonzero distances: %d' % (hole.filter_out_shots_with_zero_coordinates_but_nonzero_distances(),)
        hole.filter_out_shots_with_zero_coordinates_but_nonzero_distances()
        #print 'Number of shots with zero distances but nonzero coordiantes: %d' % (hole.filter_out_shots_with_zero_distances_from_hole_but_not_last_shot(),)
        hole.filter_out_shots_with_zero_distances_from_hole_but_not_last_shot()
        usable1,usable2,usable3 = False,False,False
        usable1 = hole.are_there_any_non_zero_distances_from_the_hole_after_the_shot()
        if usable1:
            usable2 = hole.find_the_hole(.5,3)
            if usable2:
                usable3 = hole.find_the_tee_box(5,15)
        if usable1 and usable2 and usable3:
            hole_locations[(year,)+tup] = (hole.hole_x,hole.hole_y)
            tee_box_locations[(year,)+tup] = (hole.tee_box_x,hole.tee_box_y)
            hole.set_z_of_closest()
            z_of_closest[(year,)+tup] = hole.z_of_closest
        tuples_to_remove.update(hole.tuples_to_remove)


    before = len(data)
    inds = [u for u,i in enumerate(data[Impute_a_Hole.id_cols].values.astype(int).tolist()) if tuple(i) not in tuples_to_remove]
    data = data.iloc[inds]
    after = len(data)
    shrinkage = float(before-after)/before * 100
    print 'Data has been shrunk by %g percent.' % shrinkage


    data.insert(len(data.columns),'Shots_taken_after',data.Hole_Score-data.Shot)
    data = data.sort_values('Shots_taken_after')
    cols = ['Year','Course_#','Round','Hole']
    data.insert(len(data.columns),'Went_to_X',[hole_locations[tuple(tup)][0] for tup in data[data.Shots_taken_after==0][cols].values.astype(int).tolist()]
                    +data[data.Shots_taken_after>0].X_Coordinate.tolist())
    data.insert(len(data.columns),'Went_to_Y',[hole_locations[tuple(tup)][1] for tup in data[data.Shots_taken_after==0][cols].values.astype(int).tolist()]
                    +data[data.Shots_taken_after>0].Y_Coordinate.tolist())
    data.insert(len(data.columns),'Went_to_Z',[z_of_closest[tuple(tup)] for tup in data[data.Shots_taken_after==0][cols].values.astype(int).tolist()]
                    +data[data.Shots_taken_after>0].Z_Coordinate.tolist())
    data =  data.sort_values('Shot')
    all_cols = ['Course_#','Player_#','Hole','Round','Shot','X_Coordinate','Y_Coordinate','Z_Coordinate']
    cols2 = ['Course_#','Player_#','Hole','Round','Shot']
    my_big_dict = {tuple(tup[0:5]):tuple(tup[5:8]) for tup in data[all_cols].values.tolist()}
    data.insert(len(data.columns),'Started_at_X',[tee_box_locations[tuple(tup)][0] for tup in data[data.Shot==1][cols].values.astype(int).tolist()]
                    +[my_big_dict[tuple(tup[0:4]+[tup[4]-1])][0] for tup in data[data.Shot!=1][cols2].values.astype(int).tolist()])
    data.insert(len(data.columns),'Started_at_Y',[tee_box_locations[tuple(tup)][1] for tup in data[data.Shot==1][cols].values.astype(int).tolist()]
                    +[my_big_dict[tuple(tup[0:4]+[tup[4]-1])][1] for tup in data[data.Shot!=1][cols2].values.astype(int).tolist()])
    data.insert(len(data.columns),'Started_at_Z',[np.nan for _ in xrange(len(data[data.Shot==1]))]
                    +[my_big_dict[tuple(tup[0:4]+[tup[4]-1])][2] for tup in data[data.Shot!=1][cols2].values.astype(int).tolist()])
    data.insert(len(data.columns),'Distance_with_new_coordinates',((data.Started_at_X.values - data.Went_to_X.values)**2 + 
                                                                (data.Started_at_Y.values - data.Went_to_Y.values)**2)**.5)
    data.insert(len(data.columns),'Dist_diff',np.abs(data.Distance_with_new_coordinates.values - data.Distance.values / 12.0) / 3)

    tuples_to_remove = set(tuple(tup) for tup in data[data.Dist_diff>15][Impute_a_Hole.id_cols].values.astype(int).tolist())
    before = len(data)
    inds = [u for u,i in enumerate(data[Impute_a_Hole.id_cols].values.astype(int).tolist()) if tuple(i) not in tuples_to_remove]
    data = data.iloc[inds]
    after = len(data)
    shrinkage = float(before-after)/before * 100
    print 'Data has been shrunk by %g percent.' % shrinkage
    print len(data)
    cols_to_remove = ['Dist_diff','Distance_with_new_coordinates','X_Coordinate','Y_Coordinate','Z_Coordinate','Date','Lie',
                    'Tour_Code','Tour_Description','In_the_Hole_Flag','Slope','Distance_from_Center','Distance_from_Edge']

    for column in cols_to_remove:
        data = data.drop(column,axis=1)
    with open('data/data.csv','a') as f:
        if year>2003:
            data.to_csv(f,mode='a',header=False,index=False)
        else:
            data.to_csv(f,mode='a',index=False)