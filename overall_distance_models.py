import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import pickle

def convert_cats(cat,dist,shot):
    if cat in ['Green Side Bunker','Fairway Bunker']:
        return 'Bunker'
    elif cat not in ['Green','Fairway','Fringe','Primary Rough','Intermediate Rough','Tee Box']:
        return 'Other'
    elif cat=='Fringe' and dist>120:
        return 'Intermediate Rough'
    elif cat=='Tee Box' and shot!=1:
    	return 'Fairway'
    else:
        return cat


data = pd.concat([pd.read_csv('data/%d.csv' % (year,)) for year in range(2003,2017)])
data = data[data.Shot!=1]
print data.shape
data.insert(len(data.columns),'Cat',[convert_cats(c,d) for c,d in zip(data['From_Location(Scorer)'],data['Distance_from_hole'],data.Shot)])

cats = ['Bunker','Other','Green','Fairway','Fringe','Primary Rough','Intermediate Rough']

overall_models = {}
for cat in cats:
    overall_models[cat] = IsotonicRegression(out_of_bounds='clip')
    overall_models[cat].fit(data[np.where(data[:,6]==cat)][:,8],data[np.where(data[:,6]==cat)][:,7])

for cat in overall_models:
	with open('overall_distance_models/%s.pkl' % cat,'w') as pickleFile:
		pickle.dump(overall_models[cat],pickleFile)
