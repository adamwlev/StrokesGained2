import pandas as pd
import numpy as np


def create_df (year):
    data = pd.read_csv('data/rawdata/' + str(year) + '.txt', sep = ';')

    #remove space in col names
    data.columns = np.array([str(i).strip() for i in list(data.columns.values)])

    #remove space in coordinates cols
    data['X Coordinate'] = [str(i).replace(' ','') for i in data['X Coordinate']]
    data['Y Coordinate'] = [str(i).replace(' ','') for i in data['Y Coordinate']]
    data['Z Coordinate'] = [str(i).replace(' ','') for i in data['Z Coordinate']]

    #remove commas in coordinates cols
    data['X Coordinate'] = [str(i).replace(',','') for i in data['X Coordinate']] 
    data['Y Coordinate'] = [str(i).replace(',','') for i in data['Y Coordinate']]
    data['Z Coordinate'] = [str(i).replace(',','') for i in data['Z Coordinate']]

    #putting negative in front
    data['X Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['X Coordinate']]
    data['Y Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Y Coordinate']]
    data['Z Coordinate'] = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data['Z Coordinate']]

    data['X Coordinate'] = pd.to_numeric(data['X Coordinate'])
    data['Y Coordinate'] = pd.to_numeric(data['Y Coordinate'])
    data['Z Coordinate'] = pd.to_numeric(data['Z Coordinate'])

    return data