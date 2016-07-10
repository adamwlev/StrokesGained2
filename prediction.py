import pandas as pd
import numpy as np
import itertools
from sklearn.isotonic import IsotonicRegression

for year in range(2003,2006):
    print year
    cols = ['Course_#','Round','Hole']
    data = pd.read_csv('data/%d.csv' % (year,))
    if year==2003:
        df = data.loc[0:2,:]
    samp = set([tuple(i) for i in data.drop_duplicates(cols).iloc[np.random.choice(range(len(data.drop_duplicates
                                        (cols))),size=500,replace=False)][cols].values.astype(int).tolist()])
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


uCRHtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))

data = data[['Course_#','Round','Hole','Player_#','Hole_Score','Shot','Cat','Shots_taken_from_location',
            'Distance_from_hole','Started_at_X','Started_at_Y','Went_to_X','Went_to_Y']].values



errors = []
strokes_gained_per_cat = {'Bunker':[],'Other':[],'Green':[],'Fairway':[],'Fringe':[],'Primary Rough':[],
                            'Intermediate Rough':[], 'Tee Box':[]}
overall_models = {}
for cat in strokes_gained_per_cat:
    overall_models[cat] = IsotonicRegression()
    overall_models[cat].fit(data[np.where(data[:,6]==cat)][:,8],data[np.where(data[:,6]==cat)][:,7])

overall_just_dist = IsotonicRegression()
overall_just_dist.fit(data[:,8],data[:,7])

for crhtup in uCRHtps:
    subset = data[np.where((data[:,0]==crhtup[0]) & (data[:,1]==crhtup[1]) & (data[:,2]==crhtup[2]))]
    if subset.shape[0]==0:
        continue
    players = pd.unique(subset[:,3])
    scores = {player:int(subset[np.where(subset[:,3]==player)][0,4]) for player in players}
    ave_score = np.mean(np.array([scores.get(player) for player in players]))
    for player in players:
        sub = subset[np.where(subset[:,3]!=player)]
        #print len(sub)
        models = {}
        for cat in strokes_gained_per_cat:
            models[cat] = IsotonicRegression()
            if len(sub[np.where(sub[:,6]==cat)])>0:
                models[cat].fit(sub[np.where(sub[:,6]==cat)][:,8],sub[np.where(sub[:,6]==cat)][:,7])

        just_dist_model = IsotonicRegression()
        just_dist_model.fit(sub[:,8],sub[:,9])

        tot_strokes_gained = ave_score - scores[player]

        model_predicted_strokes_gained = 0

        sub = subset[np.where(subset[:,3]==player)]

        for row_ind in range(2,scores[player]+1):
            shot = sub[np.where(sub[:,5]==row_ind)]
            cat = shot[0,6]
            dist = shot[0,8]
            shot_before = sub[np.where(sub[:,5]==row_ind-1)]
            cat_before = shot_before[0,6]
            dist_before = shot_before[0,8]
            if row_ind==2:
                if models[cat].predict([dist]) != np.nan:
                    model_predicted_strokes_gained += ave_score - models[cat].predict([dist])[0] - 1
                    strokes_gained_per_cat[cat_before].append(ave_score - models[cat].predict([dist])[0] - 1)
                else:
                    normal_cat_diff = overall_models[cat].predict([dist]) - overall_just_dist.predict([dist])
                    model_predicted_strokes_gained += ave_score - (just_dist_model.predict([dist])[0] + normal_cat_diff) - 1
                    strokes_gained_per_cat[cat_before].append(ave_score - (just_dist_model.predict([dist])[0] + normal_cat_diff) - 1)
            else:
                if models[cat].predict([dist]) != np.nan:
                    model_predicted_strokes_gained += models[cat_before].predict([dist_before])[0] - models[cat].predict([dist])[0] - 1
                    strokes_gained_per_cat[cat_before].append(models[cat_before].predict([dist_before])[0] - models[cat].predict([dist])[0] - 1)
                ***below here***
                else:
                    normal_cat_diff = overall_models[cat].predict([dist]) - overall_just_dist.predict([dist])
                    model_predicted_strokes_gained += ave_score - (just_dist_model.predict([dist])[0] + normal_cat_diff) - 1
                    strokes_gained_per_cat[cat_before].append(ave_score - (just_dist_model.predict([dist])[0] + normal_cat_diff) - 1)

        cat_last = sub[np.where(sub[:,5]==scores[player])][0,6]
        dist_last = sub[np.where(sub[:,5]==scores[player])][0,8]
        model_predicted_strokes_gained += model.predict([dist_last])[0] - 1
        strokes_gained_per_cat[cat_last].append(model.predict([dist_last])[0] - 1)

        errors.append((model_predicted_strokes_gained - tot_strokes_gained))


def hypo_test_above_below_zero(sample,its):
    above_0,below_0 = 0,0
    for _ in xrange(its):
        samp = np.random.choice(sample,len(sample))
        if np.mean(samp)>0:
            above_0 +=1
        else:
            below_0 +=1
    return (above_0-below_0)/float(its)

print pd.Series(errors).describe()
for cat in strokes_gained_per_cat:
    print cat, hypo_test_above_below_zero(strokes_gained_per_cat[cat],10000) 


