# By: Kevin PULIDO

import pandas as pd
import numpy as np


class EDA(object):
    """docstring for EDA."""

    def __init__(self, df):
        super(EDA, self).__init__()
        self.df = df
        self.shape = self.df.shape

    def details(self):
        print(self.shape)

    def describe(self):
        print(self.df.describe())

    def sum_nan(self):
        print(self.df.isna().sum())

    def sum_null(self):
        print(self.df.isnull().sum())

    def dataset_mean(self):
        print(self.df.mean(axis=0))

    def fill_mean(self, col_name):
        col_mean = self.df[col_name].astype('float').mean(axis=0)
        self.df[col_name].replace(np.nan, col_mean, inplace=True)

    def drop(self, columns, axis=1):
        self.df.drop([columns], axis=axis, inplace=True)
        self.shape = self.df.shape

    def drop_nan(self, axis=0, subset=None):
        '''simply drop whole rows/columns with NaN in selected column

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.
        subset : [array-like] Labels along other axis to consider,
                 e.g. if you are dropping rows these would be a list
                 of columns to include.'''

        self.df.dropna(subset=subset, axis=axis, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.shape = self.df.shape

    def copy(self):
        return self.df.copy()


df = pd.read_csv('C:/Users/Kevin Pulido/github/outil/Customers.csv')

print(df.head())

dataset = EDA(df)
dataset.details()
dataset.sum_null()
# dataset.fill_mean('Age')
dataset.drop_nan()
df2 = dataset.copy()
print(df2.shape)
