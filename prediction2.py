import pandas as pd
import numpy as np
import itertools
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import multiprocessing

##training data
inds_chosen = {}
for year in range(2003,2009):
    print year
    cols = ['Course_#','Round','Hole','Permanent_Tournament_#']
    data = pd.read_csv('data/%d.csv' % (year,))
    if year==2003:
        df = data.loc[0:2,:]
    inds_chosen[year] = np.random.choice(range(len(data.drop_duplicates(cols))),size=50,replace=False)
    samp = set([tuple(i) for i in data.drop_duplicates(cols).iloc[inds_chosen[year]][cols].values.astype(int).tolist()])
    inds = [u for u,i in enumerate(data[cols].values.astype(int).tolist()) if tuple(i) in samp]
    df = df.append(data.iloc[inds])
    if year==2003:
        df = df.drop(data.iloc[0:2].index,axis=0)
data = df.reset_index()


def convert_cats(cat,dist):
    if cat in ['Green Side Bunker','Fairway Bunker']:
        return 'Bunker'
    elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
        return 'Other'
    elif cat=='Fringe' and dist>120:
        return 'Intermediate Rough'
    else:
        return cat

data.insert(len(data.columns),'Cat',[convert_cats(c,d) for c,d in zip(data['From_Location(Scorer)'],data['Distance_from_hole'])])
overall_models = {}
cats = ['Bunker','Other','Green','Fairway','Fringe','Primary Rough','Intermediate Rough']
for cat in cats:
    overall_models[cat] = IsotonicRegression(out_of_bounds='clip')
    overall_models[cat].fit(data[data.Cat==cat].Distance_from_hole,data[data.Cat==cat].Shots_taken_from_location)

##testing data
inds_chosen = {}
for year in range(2003,2009):
    print year
    cols = ['Course_#','Round','Hole','Permanent_Tournament_#']
    data = pd.read_csv('data/%d.csv' % (year,))
    if year==2003:
        df = data.loc[0:2,:]
    inds_available = list(set(range(len(data.drop_duplicates(cols)))) - set(inds_chosen[year]))
    test_inds = np.random.choice(inds_available,size=50,replace=False)
    samp = set([tuple(i) for i in data.drop_duplicates(cols).iloc[test_inds][cols].values.astype(int).tolist()])
    inds = [u for u,i in enumerate(data[cols].values.astype(int).tolist()) if tuple(i) in samp]
    df = df.append(data.iloc[inds])
    if year==2003:
        df = df.drop(data.iloc[0:2].index,axis=0)
data = df.reset_index()

uYCRHtps = list(itertools.product(pd.unique(data.Year),pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))
overall_iso = []
just_hole_iso = []
def run_a_slice(slice):
    n = 12
    little_dict = {}
    for year,course,round,hole in slice:
        df = data[(data.Year==year) & (data['Course_#']==course) & (data.Round==round) & (data.Hole==hole)]
        for p in pd.unique(df['Player_#']):
            sub = df[df['Player_#']!=p]
            range_covered = {}
            baby_models = {}
            for cat in cats:
                if len(sub[sub.Cat==cat])>n:
                    s = sub[sub.Cat==cat]
                    baby_models[cat] = IsotonicRegression()
                    baby_models[cat].fit(s.Distance_from_hole,s.Shots_taken_from_location)
                    range_covered[cat] = (s.Distance_from_hole.min(),s.Distance_from_hole.max())
            for ind,shot in df[df['Player_#']==p].iterrows():
                if shot.Cat in baby_models and shot.Distance_from_hole>=range_covered[shot.Cat][0] and shot.Distance_from_hole<=range_covered[shot.Cat][1]:
                    little_dict[(year,course,round,hole,p,shot.Shot)] = (overall_models.predict([shot.Distance_from_hole])[0],(overall_models.predict([shot.Distance_from_hole])[0] + baby_models[cat].predict([shot.Distance_from_hole])[0])/2)
                else:
                    little_dict[(year,course,round,hole,p,shot.Shot)] = (overall_models.predict([shot.Distance_from_hole])[0],overall_models.predict([shot.Distance_from_hole])[0])
    return little_dict

def partition (lst, n):
    return [lst[i::n] for i in xrange(n)]

num_cores = multiprocessing.cpu_count()
slices = partition(uYCRHtps,num_cores)
pool = multiprocessing.Pool(num_cores)
results = pool.map(run_a_slice, slices)

big_dict = {key:value for little_dict in results for key,value in little_dict.iteritems()}
cols = ['Course_#','Round','Hole','Year','Player_#','Shot']
errors_baseline = [data[(data.Year==year) & (data['Course_#']==course) & (data.Round==round)
        & (data.Hole==hole) & (data['Player_#']==player) & (data.Shot==shot)].Shots_taken_from_location -\
        big_dict[(year,course,round,hole,player,shot)[0] for year,course,round,hole,player,shot in big_dict]
errors_ensemble = [data[(data.Year==year) & (data['Course_#']==course) & (data.Round==round)
        & (data.Hole==hole) & (data['Player_#']==player) & (data.Shot==shot)].Shots_taken_from_location -\
        big_dict[(year,course,round,hole,player,shot)[1] for year,course,round,hole,player,shot in big_dict]

print (pd.Series(errors_baseline)**2).describe()
print (pd.Series(errors_ensemble)**2).describe()
