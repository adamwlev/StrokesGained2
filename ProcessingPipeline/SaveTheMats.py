import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
import gc
import os,sys
import pickle

if __name__=='__main__':
    _,epsilon,e_d,e_t,w_d,beta = sys.argv
    if os.path.exists('./../mats%s-%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d,beta)):
        pass
    else:
        os.makedirs('./../mats%s-%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d,beta))

    beta = float(beta)

    cats = ['tee3','tee45','green0','green5','green10','green20','rough0','rough90',
            'rough375','fairway0','fairway300','fairway540','bunker','other']

    def load_sparse_csc(filename):
        loader = np.load(filename)
        return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

    def save_sparse_csc(filename,array):
        np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)
        return

    with open('./../hole_tups.pkl','r') as pickleFile:
        hole_tups = pickle.load(pickleFile)
    
    n_tournaments = len(pd.DataFrame(np.array(hole_tups))[[0,1]].drop_duplicates())

    bin_size = 4
    window_size = 28
    n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))

    for cat in cats:
        A = bmat([[load_sparse_csc('./../cats_w%s-%s-%s-%s/%s_%d.npz' % ((epsilon,e_d,e_t,w_d,cat,group))) for group in range(1,n_tournament_groups)]],format='csc')
        save_sparse_csc('./../mats%s-%s-%s-%s-%g/%s_A' % (epsilon,e_d,e_t,w_d,beta,cat),A)
        A = None
        gc.collect()
        G = bmat([[load_sparse_csc('./../cats_w%s-%s-%s-%s/%s_%d_g.npz' % ((epsilon,e_d,e_t,w_d,cat,group))) for group in range(1,n_tournament_groups)]],format='csc')
        save_sparse_csc('./../mats%s-%s-%s-%s-%g/%s_G' % (epsilon,e_d,e_t,w_d,beta,cat),G)
        G = None
        gc.collect()
