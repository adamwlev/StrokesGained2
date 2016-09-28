import pandas as pd
import numpy as np
import itertools
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import multiprocessing

def convert_cats(cat,dist):
    if cat in ['Green Side Bunker','Fairway Bunker']:
        return 'Bunker'
    elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
        return 'Other'
    elif cat=='Fringe' and dist>120:
        return 'Intermediate Rough'
    else:
        return cat

def run_a_slice(slice):
	little_dict = {}
	for crhytup in slice:
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
	        ranges_covered = {}
	        for cat in cats:
	            if len(sub[np.where(sub[:,6]==cat)])>6:
	                models[cat] = LinearRegression()
	                models[cat].fit(sub[np.where(sub[:,6]==cat)][:,8][:,None],sub[np.where(sub[:,6]==cat)][:,7])
	                ranges_covered[cat] = (np.amin(sub[np.where(sub[:,6]==cat)][:,8]),np.amax(sub[np.where(sub[:,6]==cat)][:,8]))

	        just_dist_model = LinearRegression()
	        just_dist_model.fit(sub[:,8][:,None],sub[:,7])

	        sub = subset[np.where(subset[:,3]==player)]
	        
	        if scores[player]!=sub.shape[0]:
	            print 'hmmm'

	        for row_ind in range(2,scores[player]+1):
	            shot = sub[np.where(sub[:,5]==row_ind)]
	            cat = shot[0,6]
	            dist = shot[0,8]
	            shot_before = sub[np.where(sub[:,5]==row_ind-1)]
	            cat_before = shot_before[0,6]
	            dist_before = shot_before[0,8]
	            if row_ind==2:
	                if cat not in models or dist<ranges_covered[cat][0] or dist>ranges_covered[cat][1]:
	                    normal_cat_diff = overall_models[cat].predict([dist])[0] - overall_just_dist.predict([dist])[0]
	                    difficulty_end = just_dist_model.predict(np.array([dist])[:,None])[0] + normal_cat_diff
	                else:
	                    difficulty_end = models[cat].predict(np.array([dist])[:,None])[0]
	            	little_dict[crhytup+(player,row_ind-1)] = ave_score - difficulty_end - 1
	            else:
	                if cat not in models or dist<ranges_covered[cat][0] or dist>ranges_covered[cat][1]:
	                    normal_cat_diff = overall_models[cat].predict([dist])[0] - overall_just_dist.predict([dist])[0]
	                    difficulty_end = just_dist_model.predict(np.array([dist])[:,None])[0] + normal_cat_diff
	                else:
	                    difficulty_end = models[cat].predict(np.array([dist])[:,None])[0]
	                if cat_before not in models or dist_before<ranges_covered[cat_before][0] or dist_before>ranges_covered[cat_before][1]:
	                    normal_cat_diff = overall_models[cat_before].predict([dist_before])[0] - overall_just_dist.predict([dist_before])[0]
	                    difficulty_start = just_dist_model.predict(np.array([dist_before])[:,None])[0] + normal_cat_diff
	                else:
	                    difficulty_start = models[cat_before].predict(np.array([dist_before])[:,None])[0]
	                
	            	little_dict[crhytup+(player,row_ind-1)] = difficulty_start - difficulty_end - 1

	        cat_last = sub[np.where(sub[:,5]==scores[player])][0,6]
	        dist_last = sub[np.where(sub[:,5]==scores[player])][0,8]
	        if cat_last not in models or dist_last<ranges_covered[cat_last][0] or dist_last>ranges_covered[cat_last][1]:
	            normal_cat_diff = overall_models[cat_last].predict([dist_last])[0] - overall_just_dist.predict([dist_last])[0]
	            difficulty_last = just_dist_model.predict(np.array([dist_last])[:,None])[0] + normal_cat_diff
	        else:
	            difficulty_last = models[cat_last].predict(np.array([dist_last])[:,None])[0]
	        little_dict[crhytup+(player,scores[player])] = difficulty_last - 1
	return little_dict

def partition (lst, n):
	return [lst[i::n] for i in xrange(n)]

for YEAR in range(2004,2017):
	data = pd.read_csv('./../data/%d.csv' % YEAR)

	#data.insert(len(data.columns),'Cat',[convert_cats(c,d) for c,d in zip(data['From_Location(Scorer)'],data['Distance_from_hole'])])

	uCRHYtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole),pd.unique(data.Year)))

	data = data[['Course_#','Round','Hole','Player_#','Hole_Score','Shot','Cat','Shots_taken_from_location',
	            'Distance_from_hole','Started_at_X','Started_at_Y','Went_to_X','Went_to_Y','Year']].values

	cats = ['Bunker','Other','Green','Fairway','Fringe','Primary Rough','Intermediate Rough']

	overall_models = {}
	for cat in cats:
	    overall_models[cat] = IsotonicRegression(out_of_bounds='clip')
	    overall_models[cat].fit(data[np.where(data[:,6]==cat)][:,8],data[np.where(data[:,6]==cat)][:,7])

	overall_just_dist = IsotonicRegression(out_of_bounds='clip')
	overall_just_dist.fit(data[:,8],data[:,7])

	num_cores = multiprocessing.cpu_count()
	slices = partition(uCRHYtps,num_cores)
	pool = multiprocessing.Pool(num_cores)
	results = pool.map(run_a_slice, slices)

	big_dict = {key:value for little_dict in results for key,value in little_dict.iteritems()}
	cols = ['Course_#','Round','Hole','Year','Player_#','Shot']
	data = pd.read_csv('data/%d.csv' % YEAR)
	print len(big_dict),len(data)
	data['Strokes_Gained_h'] = [big_dict[tuple(tup)] if tuple(tup) in big_dict else np.nan for tup in data[cols].values.astype(int).tolist()]

	data.to_csv('./../data/%d.csv' % YEAR,index=False)