import pandas as pd
import numpy as np
from scipy.optimize import fmin_tnc

class Imputer():
    """Represents process of imputing missing hole and tee box locations for a one single hole"""
    def __init__(self,tee_box_flag,df):
        self.tee_box_flag = tee_box_flag
        if tee_box_flag:
            self.df = df[df.Shot==1].copy()
            self.dist_col = 'Distance'
            self.coord_col = 'End'
        else:
            last_shot_mask = df.groupby('Player_#')['Shot'].transform(lambda x: x==x.max())
            self.df = df[last_shot_mask].copy()
            self.dist_col = 'Distance'
            self.coord_col = 'Start'
        self.x_col, self.y_col, self.z_col = [self.coord_col + '_' + s for s in ['X_Coordinate','Y_Coordinate','Z_Coordinate']]
        mask = (self.df[self.dist_col]!=0) & (self.df[self.x_col]!=0) & (self.df[self.y_col]!=0) & (self.df[self.z_col]!=0)
        self.df = self.df[mask].copy()
        
    def __str__(self):
        return self.df[['Course_#','Round','Hole','Hole_Score','Shot','Distance_to_Hole_after_the_Shot']].__str__()

    @staticmethod
    def _f(a,x,y,d):
        """Returns mean squared error from location to guess for hole or tee box location.
        Function to be minimized for imputing locations."""
        x0,y0 = a[0],a[1]
        return sum((((x-x0)**2 + (y-y0)**2)**.5-d)**2)/len(x)

    def _find_loc(self):
        """Finds best estimate for hole or tee box

        Precondition: The following variables are defined for the instance
        x - x coordinates of the shots
        y - y coordinates of the shots
        d - distance recorded of shots from hole (or tee box)
        x0 - initial guess for location x coordinate
        y0 - initial guess for location y coordinate
        """
        xopt = fmin_tnc(Imputer._f,[self.x0,self.y0],args=(self.x,self.y,self.d),approx_grad=1,maxfun=1000,disp=0)[0].tolist()
        return xopt

    id_cols = ['Course_#','Player_#','Round','Hole']

    def _set_attributes_before_finding_loc(self):
        self.d = self.df[self.dist_col].values / 12.0
        self.x, self.y = self.df[self.x_col].values, self.df[self.y_col].values

        min_dist = self.df[self.dist_col].min()
        min_shot = self.df[self.df[self.dist_col]==min_dist].iloc[0]
        self.x0, self.y0 = min_shot[self.x_col], min_shot[self.y_col]

    def find_location(self,max_allowable_mean_abs_deviation,iter=25):
        self._set_attributes_before_finding_loc()
        loc_x, loc_y = self._find_loc()
        abs_deviation = np.abs(((self.x - loc_x)**2 + (self.y - loc_y)**2)**.5 - self.d)
        
        c = 0
        while np.mean(abs_deviation) > max_allowable_mean_abs_deviation:
            #print c, len(self.df), np.mean(abs_deviation)
            c += 1
            if c>=25:
                #print '\033[91m','25 attempts were made to remove outliers. Imputation failed. Removed all shots.','\033[0m'
                return False
            self.df = self.df[abs_deviation!=np.amax(abs_deviation)]
            self._set_attributes_before_finding_loc()
            loc_x, loc_y = self._find_loc()
            abs_deviation = np.abs(((self.x - loc_x)**2 + (self.y - loc_y)**2)**.5 - self.d)
     
        # print '\033[91m',\
        #       '%d of the %d shots removed from consideration during imputation of %s location.' % (c,len(self.df),'tee box' if self.tee_box_flag else 'hole'),\
        #       'Final Error: %g' % np.mean(abs_deviation),\
        #       '\033[0m'
        
        if len(self.df)==0: 
            return False
        else: 
            self.loc_x, self.loc_y = loc_x, loc_y
            if not self.tee_box_flag:
                self.set_z_of_closest()
            return True

    def set_z_of_closest(self):
        """This sets an attribute for the z coordinate of the closest shot recorded to the hole. Since imputation
        of the z coordinate of the hole is impossible, I'll use this as an approximation for the z coordinate of the hole. """
        min_dist = self.df[self.dist_col].min()
        min_shot = self.df[self.df[self.dist_col]==min_dist].iloc[0]
        self.loc_z = min_shot[self.z_col]
