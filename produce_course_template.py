import pandas as pd

data = pd.concat([pd.read_csv('data/%d.csv' % year, usecols=['Course_#','Course_Name'])
                  for year in range(2003,2018)])

data.drop_duplicates().to_csv('courses_template.csv',index=False)