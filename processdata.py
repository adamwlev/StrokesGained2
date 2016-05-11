
# coding: utf-8

# In[81]:

import pandas as pd
import numpy as np
import itertools


# In[66]:

data = pd.read_csv('rawdata/2014.txt', sep = ';')


# In[67]:

data.columns = np.array([str(i).strip() for i in list(data.columns.values)]) #remove space in col names


# In[68]:

data.columns


# In[69]:

data['X Coordinate'] = [str(i).replace(' ','') for i in data['X Coordinate']] #remove space in coordinates cols
data['Y Coordinate'] = [str(i).replace(' ','') for i in data['Y Coordinate']]
data['Z Coordinate'] = [str(i).replace(' ','') for i in data['Z Coordinate']]


# In[70]:

data['X Coordinate'] = [str(i).replace(',','') for i in data['X Coordinate']] #remove commas in coordinates cols
data['Y Coordinate'] = [str(i).replace(',','') for i in data['Y Coordinate']]
data['Z Coordinate'] = [str(i).replace(',','') for i in data['Z Coordinate']]


# In[71]:

data['X Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['X Coordinate']] #putting negative in front
data['Y Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Y Coordinate']]
data['Z Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Z Coordinate']]


# In[74]:

data['X Coordinate'] = pd.to_numeric(data['X Coordinate'])
data['Y Coordinate'] = pd.to_numeric(data['Y Coordinate'])
data['Z Coordinate'] = pd.to_numeric(data['Z Coordinate'])


# In[77]:

#d=data['Distance to Hole after the Shot']


# In[86]:

uCRHtps = list(itertools.product(np.unique(data['Course Name']),np.unique(data['Round']),np.unique(data['Hole']))) #unique Course-Round-Hole Tuples


# In[ ]:



