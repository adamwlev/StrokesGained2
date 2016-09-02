import pandas as pd
import numpy as np
import math
from scipy.sparse import csc_matrix,csr_matrix,eye,bmat
from scipy.sparse.linalg import eigs,inv,gmres
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import sys


def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((loader['data'],loader['indices'],loader['indptr']),shape = loader['shape'])

def my_norm(x,BETA):
    return norm.pdf(x,0,BETA)/norm.pdf(0,0,BETA)

def inflate(tournament_group,rounds_to_inflate,n_tournament_groups,BETA,n_players,window_size=28):
    mat = csc_matrix((n_players*n_tournament_groups,n_players),dtype=float)
    mat_1 = csc_matrix((n_players*n_tournament_groups,n_players),dtype=float)
    for j in rounds_to_inflate:
        mat += bmat([[load_sparse_csc('rounds/%d.npz' % j)*my_norm(tournament_group-k,BETA)] for k in range(1,n_tournament_groups+1)],format='csc')
        mat_1 += bmat([[(load_sparse_csc('rounds/%d.npz' % j)!=0).astype(float)*my_norm(tournament_group-k,BETA)] for k in range(1,n_tournament_groups+1)],format='csc')
    if tournament_group>window_size:
        del inflate.__dict__[tournament_group-window_size]
    inflate.__dict__[tournament_group] = (mat,mat_1)
    out_mat = bmat([[inflate.__dict__[i][0][max(0,tournament_group-window_size)*n_players:n_players*tournament_group] for i in range(max(1,tournament_group-window_size+1),tournament_group+1)]],format='csc')
    out_mat1 = bmat([[inflate.__dict__[i][1][max(0,tournament_group-window_size)*n_players:n_players*tournament_group] for i in range(max(1,tournament_group-window_size+1),tournament_group+1)]],format='csc')
    return (out_mat,out_mat1)

def alpha(A,a):
    A.data[A.data<1e-6] = 0
    A.data[np.isnan(A.data)]=0
    w,v = eigs(A,k=1,which='LM')
    return a/w[0].real

def solve(mat,mat_1,a,min_reps,n_players,x_guess=None,x_guess1=None):
    mat.data[mat_1.data<1e-6] = 0
    mat_1.data[mat_1.data<1e-6] = 0
    mat.data[np.isnan(mat.data)] = 0
    mat_1.data[np.isnan(mat_1.data)] = 0
    
    S = eye(mat.shape[0],format='csc')-alpha(mat,a)*mat
    w_a = gmres(S,mat.sum(1),x0=x_guess)[0]
    
    S = eye(mat_1.shape[0],format='csc')-alpha(mat_1,a)*mat_1 
    w_g = gmres(S,mat_1.sum(1),x0=x_guess1)[0]
    
    solve.__dict__[a] = (w_a,w_g)
    w_a[w_g<min_reps]=0
    
    return ((w_a/w_g)[-n_players:],w_g[-n_players:])

def main(args):
    data = pd.read_csv('data/round.csv')
    inds = {num:ind for ind,num in enumerate(pd.unique(data.Player_Number))}
    data.insert(5,'Player_Index',[inds[num] for num in data.Player_Number])
    rounds = data.groupby(['Tournament_Year','Permanent_Tournament_#','Round_Number','Course_#'])

    n_players = len(pd.unique(data.Player_Index))
    n_rounds = len(rounds)
    n_tournaments = len(data.groupby(['Tournament_Year','Permanent_Tournament_#']))
    
    
    args = args[1:][0]
    BETA = float(args.split(':')[0])
    As = map(float,args.split(':')[1].split(','))
    ranks,reps = {},{}
    bin_size = 4
    window_size = 28
    n_tournament_groups = int(math.ceil(n_tournaments/float(bin_size)))
    current_group = 0
    tournament_groups=[set()]
    tournaments = set()
    rounds_to_inflate = []
    for round_ind,df in enumerate(rounds):
        df = df[1]
        tournament_groups[current_group].add(','.join(map(str,[df.iloc[0].Tournament_Year,df.iloc[0]['Permanent_Tournament_#']])))
        tournaments.add(','.join(map(str,[df.iloc[0].Tournament_Year,df.iloc[0]['Permanent_Tournament_#']])))
        tournament_group = len(tournaments)/bin_size
        rounds_to_inflate.append(round_ind)
        # if tournament_group>=3:
        #     continue
        if tournament_group>current_group:
            A,G = inflate(tournament_group,rounds_to_inflate,n_tournament_groups,BETA,n_players)
            if current_group==0:
                for a in As:
                    res = solve(A,G,a,1,n_players)
                    ranks[a] = []
                    reps[a] = []
                    ranks[a].append(res[0])
                    reps[a].append(res[1])
                print 'Tournament Group %d done' % tournament_group
                current_group = tournament_group
                tournament_groups.append(set())
                rounds_to_inflate = []
            else:
                for a in As:
                    w_a_approx = np.append(solve.__dict__[a][0][0 if tournament_group<=window_size else n_players:],solve.__dict__[a][0][-n_players:])
                    w_g_approx = np.append(solve.__dict__[a][1][0 if tournament_group<=window_size else n_players:],solve.__dict__[a][1][-n_players:])
                    res = solve(A,G,a,1,n_players,w_a_approx,w_g_approx)
                    ranks[a].append(res[0])
                    reps[a].append(res[1])
                print 'Tournament Group %d done' % tournament_group
                current_group = tournament_group
                tournament_groups.append(set())
                rounds_to_inflate = []

    a_to_score = {}
    ols = LinearRegression()
    for a in As:

        master_df = pd.DataFrame({'Player_Index':[],'Permanent_Tournament_#':[],'Course_#':[],
                                  'Finishing_Pct':[],'Rating':[],'Reps':[],'Pct_Reps':[]})
        for j in range(len(ranks[a])):
            df = pd.DataFrame({'player_ind':range(n_players),
                               'rank':ranks[a][j],
                               'reps':reps[a][j]}).dropna()
            pct_reps = pd.Series(df.reps[df.reps!=0]).rank(pct=True)
            df.insert(len(df.columns),'pct_reps',[0]*len(df))
            df.ix[df.reps!=0,'pct_reps'] = pct_reps
            rank_dict,reps_dict,pct_reps_dict = df['rank'].to_dict(),df['reps'].to_dict(),df['pct_reps'].to_dict()
            years = [int(i.split(',')[0]) for i in tournament_groups[j+1]]
            t_ids = [int(i.split(',')[1]) for i in tournament_groups[j+1]]
            df2 = data[data['Tournament_Year'].isin(years) & data['Permanent_Tournament_#'].isin(t_ids)]
            grouped = df2.groupby(['Player_Index','Permanent_Tournament_#','Course_#'],as_index=False)
            df3 = grouped['Finishing_Pct'].mean()
            df3['Rating'] = df3['Player_Index'].map(rank_dict)
            df3['Reps'] = df3['Player_Index'].map(reps_dict)
            df3['Pct_Reps'] = df3['Player_Index'].map(pct_reps_dict)
            master_df = pd.concat([master_df,df3])

        master_df = master_df.replace([np.inf, -np.inf], np.nan).dropna()

        X,y = master_df.Rating.values,master_df.Finishing_Pct.values
        kfold = KFold(len(y),n_folds=10,shuffle=True,random_state=45)
        scores = []
        for train,test in kfold:
            ols.fit(X[train,None],y[train],sample_weight=master_df.Reps.values[train])
            predictions = ols.predict(X[test,None])
            squared_residuals = (predictions - y[test])**2
            weighted_squared_residuals = squared_residuals * master_df.Reps.values[test] / master_df.Reps.values[test].sum()
            RMWSR = np.mean(weighted_squared_residuals)**.5
            scores.append(RMWSR)
        
        a_to_score[a] = np.mean(scores)+np.std(scores)/10**.5

    with open('outFiles/%g:' % (BETA,) + ','.join(map(str,As)), 'w') as outFile:
        outFile.write('BETA=%g' % (BETA,) + '\n')
        for a in As:
            outFile.write('%g = %g' % (a,a_to_score[a]) + '\n')

if __name__ == '__main__':
    main(sys.argv)
