import numpy as np
import pandas as pd
import pickle
import copy
from collections import defaultdict
from scipy.stats import norm,gaussian_kde,spearmanr

cols = ['Distance_from_hole','Cat','Par_Value','Year','Course_#','Round','Player_#','Strokes_Gained','Permanent_Tournament_#']
data = pd.concat([pd.read_csv('./../data/%d.csv' % year)[cols] for year in range(2003,2017)])
rdata = pd.read_csv('./../data/round.csv')

with open('./../PickleFiles/tourn_order.pkl','r') as pickleFile:
    tourn_order = pickle.load(pickleFile)

with open('./../PickleFiles/course_order.pkl','r') as pickleFile:
    course_order = pickle.load(pickleFile)


def make_specific_cats(distance,cat,par):
    if cat=='Tee Box':
        if par==3:
            return 'Tee-3'
        else:
            return 'Tee-45'
    elif cat=='Bunker' or cat=='Other':
        return cat
    elif cat=='Green' or cat=='Fringe':
        if distance<5:
            return 'Green-0'
        elif distance<10:
            return 'Green-5'
        elif distance<20:
            return 'Green-10'
        else:
            return 'Green-20'
    elif cat=='Fairway':
        if distance<300:
            return 'Fairway-0'
        elif distance<540:
            return 'Fairway-300'
        else:
            return 'Fairway-540'
    elif cat=='Primary Rough' or cat=='Intermediate Rough':
        if distance<90:
            return 'Rough-0'
        elif distance<375:
            return 'Rough-90'
        else:
            return 'Rough-375'

data.insert(len(data.columns),'Specific_Cat',
            [make_specific_cats(tup[0],tup[1],tup[2]) 
             for tup in data[['Distance_from_hole','Cat','Par_Value']].values.tolist()])

course_profiles = defaultdict(lambda: defaultdict(list))
cats = pd.unique(data.Specific_Cat)
for year,tourn in tourn_order:
    for (course,round),day in data[(data.Year==year) & (data['Permanent_Tournament_#']==tourn)].groupby(['Course_#','Round']):
        rday = rdata[(rdata.Tournament_Year==year) & (rdata['Course_#']==course) & (rdata.Round_Number==round)]
        if len(rday)==0 or len(day)==0:
            continue
        players = pd.unique(day['Player_#'])
        cat_aves = {}
        for cat in cats:
            df = day[day.Specific_Cat==cat]
            if len(df)==0:
                cat_aves[cat] = [0.0]*len(players)
            player_map = df.groupby('Player_#').Strokes_Gained.mean().to_dict()
            cat_aves[cat] = np.array([player_map[player] if player in player_map else 0.0 for player in players])
        ave_score = rday.groupby('Player_Number').Round_Score.mean().mean()
        score_map = rday.groupby('Player_Number').Round_Score.mean().to_dict()
        score_vec = np.array([score_map[player]-ave_score if player in score_map else np.nan for player in players])
        for cat in cats:
            cat_aves[cat] = cat_aves[cat][~np.isnan(score_vec)]
            course_profiles[course][cat].append(spearmanr(score_vec[~np.isnan(score_vec)],cat_aves[cat])[0])

priors = defaultdict(None)
for cat in cats:
    x = []
    for course in course_profiles:
        if np.isnan(np.mean(course_profiles[course][cat])):
            continue
        x.append(np.mean(course_profiles[course][cat]))
    priors[cat] = (np.mean(x),np.std(x))

for cat in cats:
    mean,std = priors[cat][0],priors[cat][1]
    priors[cat] = [np.linspace(mean-3*std,mean+3*std,100),
                   norm.pdf(np.linspace(mean-3*std,mean+3*std,100),mean,std)/
                    norm.pdf(np.linspace(mean-3*std,mean+3*std,100),mean,std).sum()]

def update(x,p_x,r,n):
    se = 1.06/(n-3.0)**.5
    p_x *= norm.pdf(np.arctanh(x)-np.arctanh(r),0,se)
    p_x /= p_x.sum()
    return p_x

course_posteriors = {course:copy.deepcopy(priors) for course in course_order}
to_return = {cat:np.zeros((len(tourn_order),len(course_order))) for cat in cats}
for u,(year,tourn) in enumerate(tourn_order):
    for cat in cats:
        to_return[cat][u,:] = np.array([np.dot(course_posteriors[course][cat][0],course_posteriors[course][cat][1])
                                        for course in course_order])
    for (course,round),day in data[(data.Year==year) & (data['Permanent_Tournament_#']==tourn)].groupby(['Course_#','Round']):
        rday = rdata[(rdata.Tournament_Year==year) & (rdata['Course_#']==course) & (rdata.Round_Number==round)]
        if len(rday)==0 or len(day)==0:
            continue
        players = pd.unique(day['Player_#'])
        cat_aves = {}
        for cat in cats:
            df = day[day.Specific_Cat==cat]
            if len(df)==0:
                cat_aves[cat] = [0.0]*len(players)
            player_map = df.groupby('Player_#').Strokes_Gained.mean().to_dict()
            cat_aves[cat] = np.array([player_map[player] if player in player_map else 0.0 for player in players])
        ave_score = rday.groupby('Player_Number').Round_Score.mean().mean()
        score_map = rday.groupby('Player_Number').Round_Score.mean().to_dict()
        score_vec = np.array([score_map[player]-ave_score if player in score_map else np.nan for player in players])
        for cat in cats:
            cat_aves[cat] = cat_aves[cat][~np.isnan(score_vec)]
            if np.isnan(spearmanr(score_vec[~np.isnan(score_vec)],cat_aves[cat])[0]):
            	continue
            course_posteriors[course][cat][1] = update(course_posteriors[course][cat][0],
                                                       course_posteriors[course][cat][1],
                                                       spearmanr(score_vec[~np.isnan(score_vec)],cat_aves[cat])[0],
                                                       cat_aves[cat].shape[0])

def return_matrix():
	return to_return