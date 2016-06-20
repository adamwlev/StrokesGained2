import pandas as pd
import numpy as np
import itertools
from sklearn.isotonic import IsotonicRegression


data = pd.read_csv('data/2015.csv')
def convert_cats(cat):
    if cat in ['Green Side Bunker','Fairway Bunker']:
        return 'Bunker'
    elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
        return 'Other'
    else:
        return cat

if 'Cat' in data.columns:
    data = data.drop('Cat',axis=1)
data.insert(len(data.columns),'Cat',[convert_cats(c) for c in data['From_Location(Scorer)'].tolist()])


uCRHtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole)))

data = data[['Course_#','Round','Hole','Player_#','Hole_Score','Shot','Cat','Shots_taken_from_location',
			'Distance_from_hole','Started_at_X','Started_at_Y','Went_to_X','Went_to_Y']].values



errors = []
strokes_gained_per_cat = {'Bunker':[],'Other':[],'Green':[],'Fairway':[],'Fringe':[],'Primary Rough':[],
							'Intermediate Rough':[], 'Tee Box':[]}


for crhtup in uCRHtps[0:1500]:
	subset = data[np.where((data[:,0]==crhtup[0]) & (data[:,1]==crhtup[1]) & (data[:,2]==crhtup[2]))]
	if subset.shape[0]==0:
		continue
	players = pd.unique(subset[:,3])
	scores = {player:int(subset[np.where(subset[:,3]==player)][0,4]) for player in players}
	ave_score = np.mean(np.array([scores.get(player) for player in players]))

	for player in players:
		sub = subset[np.where(subset[:,3]!=player)]
		model = IsotonicRegression(out_of_bounds='clip')
		model.fit(sub[:,8],sub[:,7])

		tot_strokes_gained = ave_score - scores[player]

		model_predicted_strokes_gained = 0

		sub = subset[np.where(subset[:,3]==player)]

		for row_ind in range(2,scores[player]+1):
			if row_ind==2:
				model_predicted_strokes_gained += ave_score - model.predict([sub[np.where(sub[:,5]==row_ind)][0,8]])[0] - 1
				strokes_gained_per_cat[sub[np.where(sub[:,5]==row_ind-1)][0,6]].append(ave_score - model.predict([sub[np.where(sub[:,5]==row_ind)][0,8]])[0] - 1)
			else:
				model_predicted_strokes_gained += model.predict([sub[np.where(sub[:,5]==row_ind-1)][0,8]])[0] - model.predict([sub[np.where(sub[:,5]==row_ind)][0,8]])[0] - 1
				strokes_gained_per_cat[sub[np.where(sub[:,5]==row_ind-1)][0,6]].append(model.predict([sub[np.where(sub[:,5]==row_ind-1)][0,8]])[0] - model.predict([sub[np.where(sub[:,5]==row_ind)][0,8]])[0] - 1)

		model_predicted_strokes_gained += model.predict([sub[np.where(sub[:,5]==scores[player])][0,8]])[0] - 1
		strokes_gained_per_cat[sub[np.where(sub[:,5]==scores[player])][0,6]].append(model.predict([sub[np.where(sub[:,5]==scores[player])][0,8]])[0] - 1)

		errors.append((model_predicted_strokes_gained - tot_strokes_gained))


print pd.Series(errors).describe()
for cat in strokes_gained_per_cat:
	print cat, pd.Series(strokes_gained_per_cat[cat]).describe()


