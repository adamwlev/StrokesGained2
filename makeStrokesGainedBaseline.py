import pandas as pd
import numpy as np
import itertools
from math import atan2,radians
from sklearn.isotonic import IsotonicRegression
import multiprocessing
import gzip
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

def partition (lst, n):
    return [lst[i::n] for i in xrange(n)]

def get_green_to_work_with(df,points,slack):
    df.insert(len(df.columns),'angle',[atan2(y,x) for x,y in zip(df.Started_at_X,
                                                                 df.Started_at_Y)])
    ww = []
    def get_sub(df,angle,slack):
        return df[(df.angle>angle-slack) & (df.angle<angle+slack)]
    for point in points:
        angle = atan2(point[1],point[0])
        sub = get_sub(df,angle,slack)
        c = 0
        while len(sub)==0:
            c += 1
            if c==15:
                break
            sub = get_sub(df,angle,slack+2)
        if c==15:
            ww.append(np.nan)
        else:
            ww.append(sub.Distance_from_hole.max())
    return ww

def predict_using_dist(list_of_tuples):
    preds = []
    for cat,dist in list_of_tuples:
        preds.append(overall_models[cat].predict([dist])[0])
    return preds

def run_a_slice(slice):
    to_work_with_dict = {}
    baseline_dict = {}
    for course,round,hole,year in slice:
        subset = data[(data['Course_#']==course) & (data.Round==round) & (data.Hole==hole) & (data.Year==year) & (data.Shot!=1)]
        if subset.shape[0]==0:
            continue
        baseline = predict_using_dist(zip(subset.Cat,subset.Distance_from_hole))
        baseline_dict[(course,round,hole,year)] = {(pl,sh):bd for pl,sh,bd in zip(subset['Player_#'],subset.Shot,baseline)}
        green = subset[subset.Cat=='Green']
        non_green = subset[subset.Cat!='Green'] 
        to_work_with = get_green_to_work_with(green,zip(non_green.Started_at_X,non_green.Started_at_Y),slack=radians(25))
        to_work_with_dict[(course,round,hole,year)] = {(pl,sh):ww for pl,sh,ww in zip(non_green['Player_#'],non_green.Shot,to_work_with)}
    return (baseline_dict,to_work_with_dict)

cats = ['Bunker','Other','Green','Fairway','Fringe','Primary Rough','Intermediate Rough']
overall_models = {}
for cat in cats:
    with gzip.open('overall_distance_models/%s.pkl.gz', 'rb') as pickleFile:
        overall_models[cat] = pickle.load(pickleFile)

for YEAR in range(2016,2017):
    print YEAR
    data = pd.read_csv('data/%d.csv' % YEAR)

    data.insert(len(data.columns),'Cat',[convert_cats(c,d,s) for c,d,s in zip(data['From_Location(Scorer)'],data['Distance_from_hole'],data.Shot)])

    uCRHYtps = list(itertools.product(pd.unique(data['Course_#']),pd.unique(data.Round),pd.unique(data.Hole),pd.unique(data.Year)))

    cats = ['Bunker','Other','Green','Fairway','Fringe','Primary Rough','Intermediate Rough']

    num_cores = multiprocessing.cpu_count()
    slices = partition(uCRHYtps,num_cores)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(run_a_slice, slices)

    big_baseline_dict = {key:value for tup in results for key,value in tup[0].iteritems()}
    big_work_with_dict = {key:value for tup in results for key,value in tup[1].iteritems()}
    cols = ['Course_#','Round','Hole','Year','Player_#','Shot']
    ww = [big_work_with_dict[(course,round,hole,year)][(player,shot)] if (course,round,hole,year) in big_work_with_dict 
            and (player,shot) in big_work_with_dict[(course,round,hole,year)] else np.nan for course,round,hole,year,player,shot 
            in data[cols].values.tolist()]
    baseline = [big_baseline_dict[(course,round,hole,year)][(player,shot)] for course,round,hole,year,player,shot in data[cols].values.tolist()]
    data.insert(len(data.columns),'Green_to_work_with',ww)
    data.insert(len(data.columns),'Difficulty_Baseline',baseline)
    print data[(data.Shot!=1) & (data.Cat!='Green')].Green_to_work_with.describe()
    #data['Strokes_Gained'] = [big_dict[tuple(tup)] if tuple(tup) in big_dict else np.nan for tup in data[cols].values.astype(int).tolist()]
    #data.to_csv('data/%d.csv' % YEAR,index=False)
