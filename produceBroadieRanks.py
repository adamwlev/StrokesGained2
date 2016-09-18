import pandas as pd
import numpy as np
import pickle
from collections import defaultdict


def convert_broadie_cats(cat,dist,par):
    if cat=='Green':
    	return 'Putting'
    elif cat=='Tee Box' and (par==4 or par==5):
    	return 'Off-the-Tee'
    elif dist>135:
    	return 'Approach-the-Green'
    else:
    	return 'Around-the-Green'


data = pd.concat([pd.read_csv('data/%d.csv' % year) for year in range(2003,2005)])
data.insert(len(data.columns),'Broadie_cat',[convert_broadie_cats(cat,dist,par) for cat,dist,par in zip(data.Cat,data.Distance_from_hole,data.Par_Value)])

field_for_cat = data.groupby(['Year','Course_#','Round','Broadie_cat'])
d = field_for_cat.Strokes_Gained.mean().to_dict()
data.insert(len(data.columns),'SG_of_Field',[d[tup] for tup in zip(data.Year,data['Course_#'],data.Round,data.Broadie_cat)])
print data.SG_of_Field.describe()
