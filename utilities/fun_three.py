import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def reshape_dataframe(df):
    # reshape dataframe with PCA feature values of recording window from 5x10 to 1x50
    return df.reshape(1, -1)


class Normalizer:
    '''
    Normalizes the data among train and test sets
    '''
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit(self, data) -> None:
        '''
        Fits the normalizer to the given training data. Sets self.min and self.max
        :param data: nested pandas dataframe, eg. training_data
        :return: None
        '''
        # to fit dataset, all data frames of column 'df' are concatenated to one
        samples_concatenated = np.concatenate(data['df_merged'].values, axis=0)
        # min and max are arrays. Values are calculated for each feature.
        self.min = np.min(samples_concatenated, axis=0).reshape(1, -1)
        self.max = np.max(samples_concatenated, axis=0).reshape(1, -1)

    def normalize_row(self, row):
        row['df_normalized'] = (row['df_merged'] - self.min) / (self.max - self.min)
        return row

    def transform(self, data):
        '''
        Applies the normalization to the given data
        :param data: nested pandas dataframe, eg. testing_data
        :return: standardized data
        '''
        normalized_data = deepcopy(data)
        normalized_data = normalized_data.apply(self.normalize_row, axis=1)
        return normalized_data


def merge_dataframes(row):
    # Merge PCA of from accelerometer and depth sensor to one feature vector with 86 fetures
    row['df_merged'] = np.concatenate([row['df_x'], row['df_y']], axis=1)
    return row
