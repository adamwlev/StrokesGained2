import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist,squareform
import math,os,sys,multiprocessing,gc,pickle,itertools
from collections import defaultdict
    
cols = ('Cat','Year','Round','Permanent_Tournament_#','Course_#','Hole','Start_X_Coordinate','tourn_num',
        'Start_Y_Coordinate','Distance_from_hole','Strokes_Gained','Time','Par_Value','Player_#')
data = pd.concat([pd.read_csv('data/%d.csv' % year,usecols=cols) for year in range(2003,2018)])
len_before = len(data)
data = data.dropna(subset=['Strokes_Gained'])
print 'Dropped %d shots for missing strokes gained.' % (len_before-len(data),)

e_d,e_t,w_d,p_mult = .8,.7,.8,1.9

cats = {}
cats['green0'] = 'Cat=="Green" & Distance_from_hole<5'
cats['fringe0'] = 'Cat=="Fringe" & Distance_from_hole<5'
cats['green5'] = 'Cat=="Green" & Distance_from_hole>=5 & Distance_from_hole<10'
cats['fringe5'] = 'Cat=="Fringe" & Distance_from_hole>=5 & Distance_from_hole<10'
cats['green10'] = 'Cat=="Green" & Distance_from_hole>=10 & Distance_from_hole<20'
cats['fringe10'] = 'Cat=="Fringe" & Distance_from_hole>=10 & Distance_from_hole<20'
cats['green20'] = 'Cat=="Green" & Distance_from_hole>=20'
cats['fringe20'] = 'Cat=="Fringe" & Distance_from_hole>=20'
cats['prough0'] = 'Cat=="Primary Rough" & Distance_from_hole<90'
cats['irough0'] = 'Cat=="Intermediate Rough" & Distance_from_hole<90'
cats['prough90'] = 'Cat=="Primary Rough" & Distance_from_hole>=90 & Distance_from_hole<375'
cats['irough90'] = 'Cat=="Intermediate Rough" & Distance_from_hole>=90 & Distance_from_hole<375'
cats['prough375'] = 'Cat=="Primary Rough" & Distance_from_hole>=375'
cats['irough375'] = 'Cat=="Intermediate Rough" & Distance_from_hole>=375'
cats['fairway0'] = 'Cat=="Fairway" & Distance_from_hole<300'
cats['fairway300'] = 'Cat=="Fairway" & Distance_from_hole>=300 & Distance_from_hole<540'
cats['fairway540'] = 'Cat=="Fairway" & Distance_from_hole>=540'
cats['bunker'] = 'Cat=="Bunker"'
cats['tee3'] = 'Cat=="Tee Box" & Par_Value==3'
cats['tee45'] = 'Cat=="Tee Box" & (Par_Value==4 | Par_Value==5)'
cats['other'] = 'Cat=="Other"'

meta_cats = {}
meta_cats['tee3'] = ['tee3']
meta_cats['tee45'] = ['tee45']
meta_cats['green0'] = ['green0','fringe0']
meta_cats['green5'] = ['green5','fringe5']
meta_cats['green10'] = ['green10','fringe10']
meta_cats['green20'] = ['green20','fringe20']
meta_cats['rough0'] = ['prough0','irough0']
meta_cats['rough90'] = ['prough90','irough90']
meta_cats['rough375'] = ['prough375','irough375']
meta_cats['fairway0'] = ['fairway0']
meta_cats['fairway300'] = ['fairway300']
meta_cats['fairway540'] = ['fairway540']
meta_cats['bunker'] = ['bunker']
meta_cats['other'] = ['other']

p_map = {mini_cat:(p_mult/data.query(cats[mini_cat])['Strokes_Gained'].std()
                   if not np.isnan(data.query(cats[mini_cat])['Strokes_Gained'].std()) else 3.)
         for mini_cat in cats}

print p_map

def partition (lst, n):
    return [lst[i::n] for i in xrange(n)]

def run_a_slice(slice):
    def sigmoid(x,sig_p):
        m,r = sig_p, sig_p/10.
        return (1./(1. + np.exp(m)**(-x)) + (np.tanh(r*x) + 1.)/2.)/2.

    def get_matrix(tournament,conditon,sig_p):
        arr,arr1 = np.zeros((n_players,n_players)),np.zeros((n_players,n_players))
        for (round,course,hole),df in data[data.tourn_num==tournament].groupby(['Round','Course_#','Hole']):
            subset = df.query(condition)[['Start_X_Coordinate','Start_Y_Coordinate','Distance_from_hole',
                                          'Strokes_Gained','Time','Player_Index']].values
            num_shots = subset.shape[0]
            dists = squareform(pdist(subset[:,0:2]))
            w_1 = w_1 = 1/(dists/(np.add.outer(subset[:,2],subset[:,2])/2) + .01)**e_d
            w_2 = 1/((np.abs(np.subtract.outer(subset[:,4],subset[:,4]))+5)/100.0)**e_t
            w = w_1*w_d + w_2*(1-w_d)
            np.fill_diagonal(w,0)
            w = np.squeeze(w.reshape(-1,1))
            vals = sigmoid(np.subtract.outer(subset[:,3],subset[:,3]),10.)
            np.fill_diagonal(vals,0)
            vals = np.squeeze(vals.reshape(-1,1))
            inds = (np.repeat(subset[:,5],num_shots).astype(int),
                    np.tile(subset[:,5],num_shots).astype(int))
            np.add.at(arr,inds,w*vals)
            np.add.at(arr1,inds,w*.5)
        mat,mat1 = csc_matrix(arr),csc_matrix(arr1)
        return (mat,mat1)

    def save_sparse_csc(filename,array):
        np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
        return

    for tournament in slice:
        print tournament
        #tournament += run_a_slice.base_number_tournaments ## for incremental
        for big_cat in meta_cats:
            # if os.path.exists('cats/cats_w-%g-%g-%g/%s_%d.npz' % (e_d,e_t,w_d,big_cat,tournament)):
            #     continue
            mat,mat1 = None,None
            for small_cat in meta_cats[big_cat]:
                sig_p = p_map[small_cat]
                condition = cats[small_cat] 
                try:
                    mat.data
                except:
                    mat,mat1 = get_matrix(tournament,condition,sig_p)
                    gc.collect()
                else:
                    res = get_matrix(tournament,condition,sig_p)
                    gc.collect()
                    mat += res[0]
                    mat1 += res[1]
            save_sparse_csc('cats/cats_w-%g-%g-%g/%s_%d' % (e_d,e_t,w_d,big_cat,tournament),mat)
            save_sparse_csc('cats/cats_w-%g-%g-%g/%s_%d_g' % (e_d,e_t,w_d,big_cat,tournament),mat1)
            #cmd = "rsync -avL --progress -e \"ssh -i /home/ubuntu/aws_ds8key.pem\" /home/ubuntu/project/Rank_a_Golfer/cats/cats_w%g-%g-%g-%g ubuntu@ec2-54-162-31-22.compute-1.amazonaws.com:~/project/Rank_a_Golfer/cats/" % (epsilon*100,e_d,e_t,w_d)
            #os.system(cmd)
    return

if not os.path.exists('cats/cats_w-%s-%s-%s' % (e_d,e_t,w_d)):
    os.makedirs('cats/cats_w-%s-%s-%s' % (e_d,e_t,w_d))
e_d,e_t,w_d = tuple(map(float,[e_d,e_t,w_d]))

with open('PickleFiles/num_to_ind_shot.pkl','rb') as pickle_file:
    num_to_ind = pickle.load(pickle_file)

for player_num in data['Player_#'].drop_duplicates():
    if player_num not in num_to_ind:
        num_to_ind[player_num] = len(num_to_ind)

with open('PickleFiles/num_to_ind_shot.pkl','wb') as pickle_file:
    pickle.dump(num_to_ind,pickle_file)

data.insert(5,'Player_Index',[num_to_ind[num] for num in data['Player_#']])
n_players = len(num_to_ind)
print n_players
data.Time = data.Time.values/100 * 60 + data.Time.values%100

n_tournaments = len(pd.unique(data.tourn_num))

#num_cores = multiprocessing.cpu_count()-2
num_cores = 3
slices = partition(range(n_tournaments),num_cores)
pool = multiprocessing.Pool(num_cores)
results = pool.map(run_a_slice, slices)
pool.close()

