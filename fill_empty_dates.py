import requests
import pandas as pd
from datetime import date, timedelta

months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
          'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

def doit():
    for year in range(2003,2018):
        data = pd.read_csv('data/%d.csv' % year,usecols=['Date'])
        data.columns = [col.strip().replace(' ','_') for col in data.columns]
        if (data.Date.str.strip()=='00/00/0000').sum()==0:
            print year,'is good.'
            continue
        r = requests.get('http://www.pgatour.com/tournaments/schedule.history.%d.html' % year)
        if r.status_code!=200:
            raise RuntimeError('error while grabbing %d page' % year)
        df = pd.read_html(r.content)[1]
        df.columns = [col.strip() for col in df.columns]
        df = df[~df.Dates.str.startswith('new pgatour')]
        df = df.drop('Champion',axis=1)
        df['tourn'] = [s.split('  ')[0] for s in df.Tournament]
      
        date_dict = {}
        for row_index,row in df.iterrows():
          date_ = ' '.join(row.Dates.split(' ')[1:]).replace('  ',' ')
          date_1 = date_.split('-')[0].strip()
          date_2 = date_.split('-')[1].strip()
          date_1 = date(year,int(months[date_1.split(' ')[0]]),int(date_1.split(' ')[1]))
          if len(date_2)>2:
              date_2 = date(year,date_1.month+1,int(date_2.split(' ')[1]))
          else:
              date_2 = date(year,date_1.month,int(date_2))
          for i in range((date_2-date_1).days + 1):
              date_dict[(row.tourn,i+1)] = (date_1 + timedelta(days=i)).strftime('%m/%d/%Y')

        data = pd.read_csv('data/%d.csv' % year)
        print year, sum(tuple(tup) in date_dict 
                          for tup in data[['Tournament_Name','Round']].values)/float(len(data))

        data.loc[data.Date.str.strip()=='00/00/0000','Date'] = [date_dict[(tuple(tup))] for tup in 
                                                                data.loc[data.Date.str.strip()=='00/00/0000',
                                                                         ['Tournament_Name','Round']].values]
        print year, (data.Date.str.strip()=='00/00/0000').sum()
        data.to_csv('data/%d.csv' % year, index=False)

