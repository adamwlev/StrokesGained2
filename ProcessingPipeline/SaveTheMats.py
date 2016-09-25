import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,bmat
import gc
import os,sys

if __name__=='__main__':
    _,epsilon,e_d,e_t,w_d,beta = sys.argv
    if os.path.isfile('./../mats%s-%s-%s-%s-%s' % (epsilon,e_d,e_t,w_d,beta)):
        sys.exit('File already exists.')
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

    for cat in cats:
        A = bmat([[load_sparse_csc('./../cats_w%s-%s-%s-%s/%s_%d.npz' % ((epsilon,e_d,e_t,w_d,cat,group))) for group in range(1,n_tournament_groups)]],format='csc')
        save_sparse_csc('./../mats%s-%s-%s-%s-%g/%s_A' % (epsilon,e_d,e_t,w_d,beta,cat),A)
        A = None
        gc.collect()
        G = bmat([[load_sparse_csc('./../cats_w%s-%s-%s-%s/%s_%d_g.npz' % ((epsilon,e_d,e_t,w_d,cat,group))) for group in range(1,n_tournament_groups)]],format='csc')
        save_sparse_csc('./../mats%s-%s-%s-%s-%g/%s_G' % (epsilon,e_d,e_t,w_d,beta,cat),G)
        G = None
        gc.collect()
