import pandas as pd
import numpy as np

def make_df (y,verbose=True):
    data = pd.read_csv('data/rawdata/'+str(y)+'.txt', sep = ';') ## read in data
    before = len(data) ##len before
    
    ##processing
    data.columns = np.array([str(i).strip().replace(' ','_') for i in list(data.columns.values)])
    data.X_Coordinate = [str(i).replace(' ','') for i in data.X_Coordinate]
    data.Y_Coordinate = [str(i).replace(' ','') for i in data.Y_Coordinate]
    data.Z_Coordinate = [str(i).replace(' ','') for i in data.Z_Coordinate]
    data.X_Coordinate = [str(i).replace(',','') for i in data.X_Coordinate] 
    data.Y_Coordinate = [str(i).replace(',','') for i in data.Y_Coordinate]
    data.Z_Coordinate = [str(i).replace(',','') for i in data.Z_Coordinate]
    data.X_Coordinate = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data.X_Coordinate]
    data.Y_Coordinate = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data.Y_Coordinate]
    data.Z_Coordinate = ['-' + str(i)[:len(str(i))-1] if str(i)[len(str(i))-1]=='-' else i for i in data.Z_Coordinate]
    data.X_Coordinate = pd.to_numeric(data.X_Coordinate)
    data.Y_Coordinate = pd.to_numeric(data.Y_Coordinate)
    data.Z_Coordinate = pd.to_numeric(data.Z_Coordinate)
    
    cols = ['Year','Course_#','Player_#','Round','Hole'] 
    problems = set(tuple(i) for i in data[data.Shot>data.Hole_Score][cols].as_matrix().tolist())
    problems.update(tuple(i) for i in data[(data.Shot!=data.Hole_Score) & ((data.X_Coordinate==0) | (data.Y_Coordinate==0) | \
                                         (data.Z_Coordinate==0))][cols].as_matrix().tolist())
    inds = [u for u,i in enumerate(data[cols].as_matrix().tolist()) if tuple(i) not in problems]
    data = data.iloc[inds]
    after = len(data)
    shrinkage = float(before-after)/before * 100
    if verbose:
        print 'Data has been shrunk by %g percent.' % shrinkage
    return data