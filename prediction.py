import pandas as pd
import numpy as np
import itertools
from sklearn.isotonic import IsotonicRegression
import multiprocessing

for year in range(2003,2009):
    print year
    cols = ['Course_#','Round','Hole']
    data = pd.read_csv('data/%d.csv' % (year,))
    if year==2003:
        df = data.loc[0:2,:]
    np.random.seed(20)
    samp = set([tuple(i) for i in data.drop_duplicates(cols).iloc[np.random.choice(range(len(data.drop_duplicates
                                        (cols))),size=200,replace=False)][cols].values.astype(int).tolist()])
    inds = [u for u,i in enumerate(data[cols].values.astype(int).tolist()) if tuple(i) in samp]
    df = df.append(data.iloc[inds])
    if year==2003:
        df = df.drop(data.iloc[0:2].index,axis=0)
data = df.reset_index()
print len(data)

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


uCRHYtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole),pd.unique(data.Year)))

data = data[['Course_#','Round','Hole','Player_#','Hole_Score','Shot','Cat','Shots_taken_from_location',
            'Distance_from_hole','Started_at_X','Started_at_Y','Went_to_X','Went_to_Y','Year']].values


errors = []
strokes_gained_per_cat = {'Bunker':[],'Other':[],'Green':[],'Fairway':[],'Fringe':[],'Primary Rough':[],
                            'Intermediate Rough':[], 'Tee Box':[]}
overall_models = {}
for cat in strokes_gained_per_cat:
    overall_models[cat] = IsotonicRegression(out_of_bounds='clip')
    overall_models[cat].fit(data[np.where(data[:,6]==cat)][:,8],data[np.where(data[:,6]==cat)][:,7])

overall_just_dist = IsotonicRegression(out_of_bounds='clip')
overall_just_dist.fit(data[:,8],data[:,7])

def run_a_slice(inds):
    for ind in inds:
        crhytup = uCRHYtps[ind]
        subset = data[np.where((data[:,0]==crhytup[0]) & (data[:,1]==crhytup[1]) & (data[:,2]==crhytup[2]) & (data[:,13]==crhytup[3]))]
        if subset.shape[0]==0:
            continue
        players = pd.unique(subset[:,3])
        if len(players)<=1:
            continue
        scores = {player:int(subset[np.where(subset[:,3]==player)][0,4]) for player in players}
        ave_score = np.mean(np.array([scores.get(player) for player in players]))
        for player in players:
            sub = subset[np.where(subset[:,3]!=player)]
            #print len(sub)
            models = {}
            for cat in strokes_gained_per_cat:
                if len(sub[np.where(sub[:,6]==cat)])>0:
                    models[cat] = IsotonicRegression()
                    models[cat].fit(sub[np.where(sub[:,6]==cat)][:,8],sub[np.where(sub[:,6]==cat)][:,7])

            just_dist_model = IsotonicRegression(out_of_bounds='clip')
            just_dist_model.fit(sub[:,8],sub[:,7])

            tot_strokes_gained = ave_score - scores[player]

            model_predicted_strokes_gained = 0

            sub = subset[np.where(subset[:,3]==player)]

            for row_ind in range(2,scores[player]+1):
                if scores[player]!=sub.shape[0]:
                    print 'hmmm'
                shot = sub[np.where(sub[:,5]==row_ind)]
                cat = shot[0,6]
                dist = shot[0,8]
                shot_before = sub[np.where(sub[:,5]==row_ind-1)]
                cat_before = shot_before[0,6]
                dist_before = shot_before[0,8]
                if row_ind==2:
                    if cat not in models or np.isnan(models[cat].predict([dist])[0]):
                        normal_cat_diff = overall_models[cat].predict([dist])[0] - overall_just_dist.predict([dist])[0]
                        difficulty_end = just_dist_model.predict([dist])[0] + normal_cat_diff
                    else:
                        difficulty_end = models[cat].predict([dist])[0]
                    model_predicted_strokes_gained += ave_score - difficulty_end - 1
                    strokes_gained_per_cat[cat_before].append(ave_score - difficulty_end - 1)
                else:
                    if cat not in models or np.isnan(models[cat].predict([dist])[0]):
                        normal_cat_diff = overall_models[cat].predict([dist])[0] - overall_just_dist.predict([dist])[0]
                        difficulty_end = just_dist_model.predict([dist])[0] + normal_cat_diff
                    else:
                        difficulty_end = models[cat].predict([dist])[0]
                    if cat_before not in models or np.isnan(models[cat_before].predict([dist_before])[0]):
                        normal_cat_diff = overall_models[cat_before].predict([dist_before])[0] - overall_just_dist.predict([dist_before])[0]
                        difficulty_start = just_dist_model.predict([dist_before])[0] + normal_cat_diff
                    else:
                        difficulty_start = models[cat_before].predict([dist_before])[0]
                    model_predicted_strokes_gained += difficulty_start - difficulty_end - 1
                    strokes_gained_per_cat[cat_before].append(difficulty_start - difficulty_end - 1)

            cat_last = sub[np.where(sub[:,5]==scores[player])][0,6]
            dist_last = sub[np.where(sub[:,5]==scores[player])][0,8]
            if cat_last not in models or np.isnan(models[cat_last].predict([dist_last])[0]):
                normal_cat_diff = overall_models[cat_last].predict([dist_last])[0] - overall_just_dist.predict([dist_last])[0]
                difficulty_last = just_dist_model.predict([dist_last])[0] + normal_cat_diff
            else:
                difficulty_last = models[cat_last].predict([dist_last])[0]
            model_predicted_strokes_gained += difficulty_last - 1
            strokes_gained_per_cat[cat_last].append(difficulty_last - 1)

            errors.append((model_predicted_strokes_gained - tot_strokes_gained))
    return (errors,strokes_gained_per_cat)

def hypo_test_different_than_zero(cats,its=10000):
    for cat in cats:
        sample = strokes_gained_per_cat[cat]
        c = 0
        m = abs(np.mean(sample))
        for _ in xrange(its):
            strap = np.random.choice(sample,len(sample))
            if abs(np.mean(strap)) >= m:
                c += 1
        print cat, ' '*(20-len(cat)), np.mean(sample), ' '*(20-len(str(np.mean(sample)))), float(c)/its
def partition (lst, n):
    return [lst[i::n] for i in xrange(n)]

num_cores = multiprocessing.cpu_count()
slices = partition(range(len(uCRHYtps)),num_cores)
pool = multiprocessing.Pool(num_cores)
results = pool.map(run_a_slice, slices)

errors = []
for i in results:
    errors += i[0]
    for cat in i[1]:
        strokes_gained_per_cat[cat] += i[1][cat]

print sum(pd.Series(errors).isnull())
tot = 0
for cat in strokes_gained_per_cat:
    tot += len(strokes_gained_per_cat[cat])
print tot
slices = partition(strokes_gained_per_cat.keys(),num_cores)
pool.map(hypo_test_different_than_zero, slices)
