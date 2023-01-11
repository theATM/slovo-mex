import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def reshape_dataframe(df):
    ''' Helper method for reshaping dataframe with PCA feature values of recording window from 5x10 to 1x50 '''
    return df.reshape(1, -1)

'''Task 3.3 helper methods'''

class Normalizer:
    def __init__(self) -> None:
        self.min= None
        self.max = None

    def fit(self, data) -> None:
        '''Concatenate all dataframes to one to calculate min and max'''
        samples_concatenated = np.concatenate(data['df_merged'].values,axis=0)
        self.min = np.min(samples_concatenated, axis=0).reshape(1,-1)
        self.max = np.max(samples_concatenated, axis=0).reshape(1,-1)

    def normalize_row(self,row):
        row['df_normalized'] = (row['df_merged'] - self.min) / (self.max - self.min)
        return row

    def transform(self, data):
        normalized_data = deepcopy(data)
        normalized_data = normalized_data.apply(self.normalize_row, axis=1)
        return normalized_data


def merge_dataframes(row):
    '''Merge PCA of from accelerometer and depth sensor to one feature vector with 86 fetures'''
    row['df_merged'] = np.concatenate([row['df_x'], row['df_y']], axis=1)
    return row