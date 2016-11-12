import pandas as pd

for year in range(2003,2017):
	data = pd.read_csv('./../data/%d.csv' % (year))
	score_min,score_max = int(data.Hole_Score.min()),int(data.Hole_Score.max())
	if 'Strokes_Gained' in data.columns:
		data = data.drop('Strokes_Gained',axis=1)
	data.insert(len(data.columns),'Strokes_Gained',[0]*len(data))
	data = data.sort_values(['Year','Player_#','Course_#','Round','Hole'])
	for i in range(score_min,score_max+1):
		for j in range(1,i+1):
			data.loc[(data.Hole_Score==i) & (data.Shot==j),'Strokes_Gained'] = data.loc[(data.Hole_Score==1) & (data.Shot==j),'Difficulty_Start'] - \
																			   data.loc[(data.Hole_Score==1) & (data.Shot==j+1),'Difficulty_Start'] - 1
	data.to_csv('./../data/%d.csv',index=False)