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
        if bool(len(self.df[self.df.Distance_to_Hole_after_the_Shot!=0])):
            return True
        else:
            self.tuples_to_remove.update(tuple(j) for j in self.df[Impute_a_Hole.id_cols].as_matrix().tolist())
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
            print len(self.df), np.mean(abs_deviation)
            c += 1
            if c>=25:
                print '\033[1m','25 attempts were made to remove outliers. Imputation failed. Removed whole hole.','\033[0m'
                self.tuples_to_remove.update(self.df[Impute_a_Hole.id_cols].as_matrix().tolist())
                return False
            mean_abs_deviation = np.mean(abs_deviation); std_abs_deviation = np.std(abs_deviation)
            self.tuples_to_remove.update(tuple(j) for j in self.without_holed_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.without_holed_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()
            abs_deviation = np.abs(((self.x - hole_x)**2 + (self.y - hole_y)**2)**.5 - self.d)

        if np.amax(abs_deviation) > max_allowable_max_abs_deviation:
            self.tuples_to_remove.update(tuple(j) for j in self.without_holed_shots[abs_deviation > max_allowable_max_abs_deviation][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.without_holed_shots[abs_deviation > max_allowable_max_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_hole()
            hole_x, hole_y = self._find_the_hole()

        after = len(self.tuples_to_remove)
        if (after-before)>0:
            print '\033[91m','There were %d player-holes removed during imputation of hole location.' % (after-before,),'\033[0m'
        else:
            print '\033[92m', 'Imputation Successful, no shots removed.','\033[0m'
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
            print len(self.df), np.mean(abs_deviation)
            c += 1
            if c>=25:
                print '\033[1m','25 attempts were made to remove outliers. Imputation failed. Removed whole hole.','\033[0m'
                self.tuples_to_remove.update(self.df[Impute_a_Hole.id_cols].as_matrix().tolist())
                return False
            mean_abs_deviation = np.mean(abs_deviation); std_abs_deviation = np.std(abs_deviation)
            self.tuples_to_remove.update(tuple(j) for j in self.only_tee_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.only_tee_shots[abs_deviation > mean_abs_deviation + 2.5*std_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_hole()
            tee_box_x, tee_box_y = self._find_the_hole()
            abs_deviation = np.abs(((self.x - tee_box_x)**2 + (self.y - tee_box_y)**2)**.5 - self.d)

        if np.amax(abs_deviation) > max_allowable_max_abs_deviation:
            self.tuples_to_remove.update(tuple(j) for j in self.only_tee_shots[abs_deviation > max_allowable_max_abs_deviation][Impute_a_Hole.id_cols].as_matrix().tolist())
            self.df = self.df.drop(self.only_tee_shots[abs_deviation > max_allowable_max_abs_deviation].index,axis=0)
            self._set_attributes_before_finding_the_tee_box()
            tee_box_x, tee_box_y = self._find_the_hole()

        after = len(self.tuples_to_remove)
        if (after-before)>0:
            print '\033[91m','There were %d player-holes removed during imputation of hole location.' % (after-before,),'\033[0m'
        else:
            print '\033[92m','Imputation Successful, no shots removed.','\033[0m'
        if len(self.df)==0: 
            return False
        else: 
            self.tee_box_x = tee_box_x
            self.tee_box_y = tee_box_y
            return True


data = e.make_df(2003)
uCRHtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))
hole_locations = {}
tee_box_locations = {}
tuples_to_remove = set()
for tup in uCRHtps:
    hole = Impute_a_Hole(tup[0],tup[1],tup[2],data)
    print tup
    print 'Number of shots with zero coordinates but nonzero distances: %d' % (hole.filter_out_shots_with_zero_coordinates_but_nonzero_distances(),)
    print 'Number of shots with zero distances but nonzero coordiantes: %d' % (hole.filter_out_shots_with_zero_distances_from_hole_but_not_last_shot(),)
    if hole.are_there_any_non_zero_distances_from_the_hole_after_the_shot():
        if hole.find_the_hole(3.5,10):
            hole_locations[tup] = (hole.hole_x,hole.hole_y)
            if hole.find_the_tee_box(310,720):
                tee_box_locations[tup] = (hole.tee_box_x,hole.tee_box_y)
    tuples_to_remove.update(hole.tuples_to_remove)