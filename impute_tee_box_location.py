import pandas as pd
import numpy as np
import itertools
from scipy.optimize import fmin_tnc

year = 2014

data = pd.read_csv('data/'+str(year)+'_with_hole_coordinates.csv', sep = ',')


#unique Course-Round-Hole Tuples
uCRHtps = list(itertools.product(np.unique(data['Course Name']),np.unique(data['Round']),np.unique(data['Hole'])))

# goal is to impute tee box locations. will use all shots with 'Shot' column equal to 1
# will do the same process of optimizing a guess as when imputing the location of the hole.
# will use a randomly selected shot as initial guess, then run optimization, then filter out
# shots over a certian threshold of implausibility, then rereun the optimization.

def f (a):
    x0,y0,z0 = a[0],a[1],a[2]
    return sum((((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**.5-d)**2)/len(x)

def find_the_tee_box ():
    xopt = fmin_tnc(f,[x0,y0,z0],approx_grad=1,maxfun=1000)[0].tolist()
    return xopt

# initializing new data frame with two rows which will be deleted after
newdata = pd.DataFrame(data.loc[1:2,:])
for u,i in enumerate(uCRHtps):
    if u%50==1:
        print u, newdata.shape
    subset = data[(data['Course Name']==i[0]) & (data['Round']==int(i[1])) & (data['Hole']==int(i[2]))]
    before = subset.shape[0]
    if subset[subset['Distance to Hole after the Shot']!=0].shape[0] == 0:
        continue
    d = subset[subset['Shot']==1]['Distance'].values/12.0
    x = subset[subset['Shot']==1]['X Coordinate'].values
    y = subset[subset['Shot']==1]['Y Coordinate'].values
    z = subset[subset['Shot']==1]['Z Coordinate'].values
    rand_ind = np.random.choice(range(subset[subset['Shot']==1].shape[0]),size=1)
    rand_shot = subset[subset['Shot']==1][['X Coordinate','Y Coordinate','Z Coordinate']].values[rand_ind,:].tolist()[0]
    x0,y0,z0 = rand_shot[0],rand_shot[1],rand_shot[2]
    a = find_the_tee_box()
    subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Shot']!=1].shape[0]                   + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
    subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset['dist_w_impute'].values[j] -                   subset['Distance'].values[j]/12) if subset['Shot'].values[j]==1 else 0 for j in range(subset.shape[0])]))
    mean_err = subset[subset['dist_diff']>0]['dist_diff'].mean()
    std_err = subset[subset['dist_diff']>0]['dist_diff'].std()
    c=0
    while mean_err>252:
        c+=1
        if c>=25:
            break
        print u,mean_err
        subset = subset.drop(subset[subset['dist_diff'] > mean_err + 2.5*std_err].index,axis=0)
        subset = subset.drop('dist_w_impute',axis=1)
        subset = subset.drop('dist_diff',axis=1)
        d = subset[subset['Shot']==1]['Distance'].values/12.0
        x = subset[subset['Shot']==1]['X Coordinate'].values
        y = subset[subset['Shot']==1]['Y Coordinate'].values
        z = subset[subset['Shot']==1]['Z Coordinate'].values
        rand_ind = np.random.choice(range(subset[subset['Shot']==1].shape[0]),size=1)
        rand_shot = subset[subset['Shot']==1][['X Coordinate','Y Coordinate','Z Coordinate']].values[rand_ind,:].tolist()[0]
        x0,y0,z0 = rand_shot[0],rand_shot[1],rand_shot[2]
        a = find_the_tee_box()
        subset.insert(len(subset.columns),'dist_w_impute',np.array([0]*subset[subset['Shot']!=1].shape[0]                   + (((x-a[0])**2 + (y-a[1])**2 + (z-a[2])**2)**.5).tolist()))
        subset.insert(len(subset.columns),'dist_diff',np.array([abs(subset['dist_w_impute'].values[j] -                   subset['Distance'].values[j]/12) if subset['Shot'].values[j]==1 else 0 for j in range(subset.shape[0])]))
        mean_err = subset[subset['dist_diff']>0]['dist_diff'].mean()
        std_err = subset[subset['dist_diff']>0]['dist_diff'].std()
    
    if c==25:
        print 'Skipping ', u, len(subset)
        continue
    subset = subset.drop('dist_w_impute',axis=1)
    subset = subset.drop('dist_diff',axis=1)
    subset.insert(len(subset.columns),'Tee Box X Coordinate',np.array([a[0]]*subset.shape[0]))
    subset.insert(len(subset.columns),'Tee Box Y Coordinate',np.array([a[1]]*subset.shape[0]))
    subset.insert(len(subset.columns),'Tee Box Z Coordinate',np.array([a[2]]*subset.shape[0]))
    after = subset.shape[0]
    if before-after>0:
        print u, before-after
    newdata = newdata.append(subset)

newdata.drop(newdata.head(2).index, inplace=True)

newdata.to_csv('data/'+str(year)+'_with_teebox_coordinates.csv',index=False)


print data.shape[0]/(data.shape[0]-newdata.shape[0])




