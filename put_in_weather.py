import requests, time, json
from datetime import date, datetime
import pandas as pd
import numpy as np

def doit():
    data = pd.concat([pd.read_csv('data/%d.csv' % year, usecols=['Year','Course_#','Permanent_Tournament_#',
                                                                 'Round','Hole','Player_#','Shot','Date','Time'])
                      for year in range(2003,2018)])

    courses = pd.read_csv('courses.csv')

    loc_dict = dict((c,(la,lo)) for c,la,lo in zip(courses['Course_#'],courses['Latitude'],courses['Longitude']))

    data['nearest_hour'] = (data['Time']/100.0).apply(round)
    data.loc[data.nearest_hour==24,'nearest_hour'] = 23
    data.loc[data.nearest_hour==0,'nearest_hour'] = 12

    d = data.drop_duplicates(subset=['Course_#','Date'])

    weather_dict = {}
    for row_ind,row in d.iterrows():
        la,lo = loc_dict[row['Course_#']]
        month,day,year = map(int,row['Date'].split('/'))
        date_ = date(year,month,day)
        try:
            timestamp = time.mktime(date_.timetuple()) - 7*60*60 ##pacific time
        except:
            print row['Date']
            continue
        request = 'https://api.darksky.net/forecast/412f70f063a8ba73446829ee76c02d9a/%g,%g,%d' % (la,lo,timestamp)
        response = requests.get(request)
        if response.status_code!=200:
            print 'Error', (row['Course_#'],row['Date']), response.status_code
            continue
        weather_dict[(row['Course_#'],row['Date'])] = json.loads(response.content)['hourly']

    d = data.drop_duplicates(subset=['Course_#','Date','nearest_hour'])
    hourly_dict = {}
    keys = set(key_ for key in weather_dict for dict_ in weather_dict[key]['data'] for key_ in dict_ )
    keys = sorted([key for key in keys if key not in ('time','icon')])
    for row_ind,row in d.iterrows():
        hour = row['nearest_hour']
        if (row['Course_#'],row['Date']) not in weather_dict:
            print (row['Course_#'],row['Date'])
            continue
        dict_ = weather_dict[(row['Course_#'],row['Date'])]['data'][int(hour)]
        hourly_dict[(row['Course_#'],row['Date'],row['nearest_hour'])] = [dict_[key] 
                                                                          if key in dict_ else np.nan
                                                                          for key in keys]

    for year in range(2003,2018):
        data = pd.read_csv('data/%d.csv' % year)
        data['nearest_hour'] = (data['Time']/100.0).apply(round)
        data.loc[data.nearest_hour==24,'nearest_hour'] = 23
        data.loc[data.nearest_hour==0,'nearest_hour'] = 12
        for u,key in enumerate(keys):
            data[key] = [hourly_dict[(course,date_,hour)][u]
                         if (course,date_,hour) in hourly_dict else np.nan
                         for course,date_,hour in zip(data['Course_#'],data['Date'],data['nearest_hour'])]
        data.to_csv('data/%d.csv' % year, index=False)