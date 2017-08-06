import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,eye,bmat
from scipy.sparse.linalg import eigs,gmres
from scipy.stats import norm
import multiprocessing,os,sys,pickle

if __name__=="__main__":
    def my_norm(x,BETA):
        return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

    def alpha(A,a):
        w,v = eigs(A,k=1,which='LM')
        return a/w[0].real

    def solve(mat,mat_1,a,x_guess=None,x_guess1=None):
        print mat.shape
        alpha_ = alpha(mat,a)
        S = eye(mat.shape[0],format='csc')-alpha_*mat
        w_a = gmres(S,mat.sum(1),x0=x_guess)[0]
        
        print mat_1.shape
        alpha_ = alpha(mat_1,a)
        S = eye(mat_1.shape[0],format='csc')-alpha_*mat_1 
        w_g = gmres(S,mat_1.sum(1),x0=x_guess1)[0]
        
        solve.w_a = w_a
        solve.w_g = w_g
        return ((w_a/w_g)[-num_players:],w_g[-num_players:])

    def load_sparse_csc(filename):
        loader = np.load(filename)
        return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

    with open('PickleFiles/num_to_ind_shot.pkl','r') as pickleFile:
        num_to_ind = pickle.load(pickleFile)

    _,cat,e_d,e_t,w_d,a,beta,block_size,window_size = sys.argv

    if not os.path.exists('cats/cats_w-%s-%s-%s' % (e_d,e_t,w_d)):
        sys.exit('No cats found.')
    print cat
    if not os.path.exists('ranks/ranks-%s-%s-%s-%s-%s' % (e_d,e_t,w_d,a,beta)):
        os.makedirs('ranks/ranks-%s-%s-%s-%s-%s' % (e_d,e_t,w_d,a,beta))

    a, beta = map(float,[a,beta])
    block_size, window_size = map(int,[block_size,window_size])
    files = os.listdir('cats/cats_w-%s-%s-%s' % (e_d,e_t,w_d))
    num_tournaments = max(int(fn.split('_')[-1].split('.')[0]) 
                          for fn in files if '_' in fn and 'g.npz' not in fn)
    fn = 'cats/cats_w-%s-%s-%s/%s_' % (e_d,e_t,w_d,cat,) + '%d.npz'
    fn_g = 'cats/cats_w-%s-%s-%s/%s_' % (e_d,e_t,w_d,cat,) + '%d_g.npz'
    num_players = load_sparse_csc(fn % (0,)).shape[0]
    
    ranks, reps = [], []
    for tournament in range(num_tournaments):
        tournament_group = tournament/block_size
        min_tournament_group = max(0,tournament_group-window_size+1)
        max_tournament_group = tournament_group+1
        range_tournament_group = range(min_tournament_group,max_tournament_group)
        A = bmat([[
                   bmat([[reduce(lambda x,y: x+y, [load_sparse_csc(fn % (i)) 
                                                   for i in range(row_ind*block_size,(row_ind+1)*block_size)])\
                          *my_norm(abs(row_ind-col_ind),beta)]
                          for col_ind in range_tournament_group],format='csc')
                   for row_ind in range_tournament_group
                 ]],format='csc')
        G = bmat([[
                   bmat([[reduce(lambda x,y: x+y, [load_sparse_csc(fn_g % (i)) 
                                                   for i in range(row_ind*block_size,(row_ind+1)*block_size)])\
                          *my_norm(abs(row_ind-col_ind),beta)]
                          for col_ind in range_tournament_group],format='csc')
                   for row_ind in range_tournament_group
                 ]],format='csc')
        
        if tournament==0:
            res = solve(A,G,a)
            ranks.append(res[0])
            reps.append(res[1])
        else:
            if A.shape[0]>solve.w_a.shape[0]:
                solve.w_a = np.concatenate([solve.w_a,solve.w_a[-num_players:]])
                solve.w_g = np.concatenate([solve.w_g,solve.w_g[-num_players:]])
            print np.isnan(A.data).sum()
            print A.shape
            res = solve(A,G,a,solve.w_a,solve.w_g)
            ranks.append(res[0])
            reps.append(res[1])
        print pd.Series(res[0]).describe()
        print pd.Series(res[1]).describe()
    #np.save('ranks/ranks-%s-%s-%s-%g-%g/%s_ranks.npy' % (e_d,e_t,w_d,a,beta,cat), np.array(ranks).T)
    #np.save('ranks/ranks-%s-%s-%s-%g-%g/%s_reps.npy' % (e_d,e_t,w_d,a,beta,cat), np.array(reps).T)

    