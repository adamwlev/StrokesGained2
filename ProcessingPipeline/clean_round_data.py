import pandas as pd
import numpy as np

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def make_finishing_pos(m):
	A = pd.DataFrame(m).transpose()
	A.columns = ['End_of_Event_Pos._(text)','Total_Strokes','Round_1_Score','Round_2_Score','Round_3_Score','Round_4_Score',
				'Round_5_Score','Round_6_Score']

	A.loc[A[A['End_of_Event_Pos._(text)']=='1'].index,'Total_Strokes'] -= 1

	A.insert(len(A.columns),'ranked',np.zeros(len(A)))
	A = A.sort_values('Total_Strokes',ascending=False)
	rank_of_everyone = np.array([])
	for r in range(6,0,-1):
		to_be_ranked = A[(A['Round_%d_Score' % (r,)]!=0) & (A.ranked!=1)]
		rank = to_be_ranked.rank(axis=0).Total_Strokes.values + len(rank_of_everyone[np.where(rank_of_everyone!=0)])
		rank_of_everyone = np.append(rank_of_everyone,rank)
		A.loc[to_be_ranked.index,'ranked'] = 1
	A.insert(len(A.columns),'Finishing_Pos',rank_of_everyone)
	return {player:A[A.index==player].Finishing_Pos.values.tolist()[0] for player in A.index}


def make_df(year):

	datar = pd.read_csv('data/rawdata/%dr.txt' % (year,),sep=';')
	datae = pd.read_csv('data/rawdata/%de.txt' % (year,),sep=';')

	datar.columns = np.array([str(i).strip().replace(' ','_') for i in list(datar.columns.values)])
	datae.columns = np.array([str(i).strip().replace(' ','_') for i in list(datae.columns.values)])

	datar = datar[['Tournament_Year','Tournament_#','Permanent_Tournament_#','Course_#','Player_Number','Player_Name','Round_Number',
				'Tee_Time','Round_Score','End_of_Event_Pos._(text)']]
	datae = datae[['Permanent_Tournament_Number','Player_Number','Total_Strokes','Round_1_Score','Round_2_Score','Round_3_Score','Round_4_Score',
				'Round_5_Score','Round_6_Score']]
	datae.columns = [['Permanent_Tournament_#','Player_Number','Total_Strokes','Round_1_Score','Round_2_Score','Round_3_Score','Round_4_Score',
				'Round_5_Score','Round_6_Score']]

	data = datar.merge(datae,'left',on=['Permanent_Tournament_#','Player_Number'])
	data['End_of_Event_Pos._(text)'] = data['End_of_Event_Pos._(text)'].str.strip()

	inds_to_drop = set(data[data['End_of_Event_Pos._(text)'].isin(['W/D','DQ'])].index.tolist())
	inds_to_drop.update(data[data.Total_Strokes==0].index.tolist())
	inds_to_drop.update(data[data.Round_1_Score==0].index.tolist())
	inds_to_drop.update(data[data.Round_2_Score==0].index.tolist())
	inds_to_drop.update(data[(data.Round_3_Score!=0) & (data.Round_3_Score<55)].index.tolist())
	inds_to_drop.update(data[(data.Round_4_Score!=0) & (data.Round_4_Score<55)].index.tolist())
	inds_to_drop.update(data[(data.Round_5_Score!=0) & (data.Round_5_Score<55)].index.tolist())
	inds_to_drop.update(data[(data.Round_6_Score!=0) & (data.Round_6_Score<55)].index.tolist())
	inds_to_drop = pd.Index(inds_to_drop)
	data = data.drop(inds_to_drop,axis=0)

	assert np.sum(data['Tournament_#'] == data.sort_values('Tournament_#')['Tournament_#']) == len(data)

	tournaments = pd.unique(data['Tournament_#'])
	for u,t in enumerate(tournaments):
		subset = data[data['Tournament_#']==t]
		finishing_pos_map = {player:subset[subset.Player_Number==player][['End_of_Event_Pos._(text)','Total_Strokes','Round_1_Score','Round_2_Score',
		'Round_3_Score','Round_4_Score','Round_5_Score','Round_6_Score']].values[0,:].tolist() for player in pd.unique(subset.Player_Number)}
		finishing_pos_map = make_finishing_pos(finishing_pos_map)
		subset = subset.drop(['Total_Strokes','Round_1_Score','Round_2_Score','Round_3_Score','Round_4_Score','Round_5_Score','Round_6_Score'],axis=1)
		subset.insert(len(subset.columns),'Finishing_Pos',[finishing_pos_map[player] for player in subset.Player_Number])
		subset.insert(len(subset.columns),'Finishing_Pct',subset.rank(pct=True).Finishing_Pos)
		if u==0:
			newdata = subset
		else:
			newdata = newdata.append(subset)

	return newdata

f = open('data/round.csv','w')

for year in range(2003,2017):
	if year==2015:
		continue
	if year==2003:
		make_df(year).to_csv(f,index=False,mode='a')
	else:
		make_df(year).to_csv(f,header=False,index=False,mode='a')

f.close()