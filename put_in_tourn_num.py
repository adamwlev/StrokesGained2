import pandas as pd
import numpy as np
from datetime import datetime

if __name__=='__main__':
    cols = ('Year','Permanent_Tournament_#','Date')
    data = pd.concat([pd.read_csv('../GolfData/Shot/%d.csv.gz' % year,usecols=cols) for year in range(2003,2019)])
    data.Date = pd.to_datetime(data.Date)
    data = data.sort_values('Date')
    start = datetime(2003,1,1,0)
    data = data[data.Date>start]

    tourn_order = data[['Year','Permanent_Tournament_#']].drop_duplicates().values.tolist()
    tourn_seq = {tuple(tup):u for u,tup in enumerate(tourn_order)}

    for year in range(2003,2019):
        data = pd.read_csv('../GolfData/Shot/%d.csv.gz' % (year,))
        data['tourn_num'] = [tourn_seq[tuple(tup)]
                             if tuple(tup) in tourn_seq else np.nan
                             for tup in data[['Year','Permanent_Tournament_#']].values]
        data = data.dropna(subset=['tourn_num'])
        data.to_csv('../GolfData/Shot/%d.csv.gz' % (year,), compression='gzip', index=False)