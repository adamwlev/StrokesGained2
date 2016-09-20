import pandas as pd
import numpy as np
import math
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

data = pd.concat([pd.read_csv('./../data/%d.csv' % year) for year in range(2003,2004)])
data.insert(len(data.columns),'Broadie_cat',[convert_broadie_cats(cat,dist,par) for cat,dist,par in zip(data.Cat,data.Distance_from_hole,data.Par_Value)])
field_for_cat = data.groupby(['Year','Course_#','Round','Broadie_cat'])
d = field_for_cat.Strokes_Gained.mean().to_dict()
data.insert(len(data.columns),'SG_of_Field',[d[tup] for tup in zip(data.Year,data['Course_#'],data.Round,data.Broadie_cat)])
data.insert(len(data.columns),'Strokes_Gained_Broadie',data.Strokes_Gained-data.SG_of_Field)

with open('./../hole_tups.pkl','r') as pickleFile:
    hole_tups = pickle.load(pickleFile)

with open('./../num_to_ind.pkl','r') as pickleFile:
    num_to_ind = pickle.load(pickleFile)

ind_to_num = {value:key for key,value in num_to_ind.iteritems()}
n_players = len(num_to_ind)
n_tournaments = len(pd.DataFrame(np.array(hole_tups))[[0,1]].drop_duplicates())

bin_size = 4
window_size = 28
current_group = 0
n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))
tournament_groups = defaultdict(set)
tournaments = set()
for tup in hole_tups:
    tournaments.add(tuple(tup[0:2]))
    tournament_group = (len(tournaments)-1)/bin_size
    if tournament_group>current_group:
        current_group = tournament_group
        holes_to_inflate = []
    tournament_groups[current_group].add(tuple(tup[0:2]))

ave_perfs,counts = {},{}
broadie_cats = ['Putting','Off-the-Tee','Approach-the-Green','Around-the-Green']
for group in tournament_groups:
    years = set(x[0] for x in tournament_groups[group])
    t_ids = set(x[1] for x in tournament_groups[group])
    for cat in broadie_cats:
        d = data[(data.Year.isin(years)) & (data['Permanent_Tournament_#'].isin(t_ids)) & (data.Broadie_cat==cat)].groupby('Player_#').Strokes_Gained_Broadie.mean().to_dict()
        if len(d)==0:
            continue
        dc = data[(data.Year.isin(years)) & (data['Permanent_Tournament_#'].isin(t_ids)) & (data.Broadie_cat==cat)].groupby('Player_#').Strokes_Gained_Broadie.count().to_dict()
        if cat not in ave_perfs:
            ave_perfs[cat] = np.zeros((n_players,n_tournament_groups))
            counts[cat] = np.zeros((n_players,n_tournament_groups))
        ave_perfs[cat][:,group] += np.array([d[ind_to_num[ind]] if ind_to_num[ind] in d else 0 for ind in range(len(num_to_ind))])
        counts[cat][:,group] += np.array([dc[ind_to_num[ind]] if ind_to_num[ind] in dc else 0 for ind in range(len(num_to_ind))])
for cat in ave_perfs:
    np.save('Broadie_aves/%s_ave.npy' % cat,ave_perfs[cat])
    np.save('Broadie_aves/%s_count.npy' % cat,counts[cat])


