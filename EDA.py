# By: Kevin PULIDO

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA(object):
    """
    docstring for EDA.
    """

    def __init__(self, df):
        super(EDA, self).__init__()
        self.df = df

    @property
    def shape(self):
        return self.df.shape

    def update(self):
        self.shape = self.df.shape

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

    def dropper(self, columns, axis=1):
        self.df.drop(columns, axis=axis, inplace=True)
        self.update()

    def drop_nan(self, axis=0, subset=None):
        '''simply drop whole rows/columns with NaN in selected column

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.
        subset : [array-like] Labels along other axis to consider,
                 e.g. if you are dropping rows these would be a list
                 of columns to include.'''

        self.df.dropna(subset=subset, axis=axis, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.update()

    def GraphDistributions(self):
        plt.figure(dpi=120)
        rws = int(self.df.shape[1]/2)

        for i, name_col in enumerate(self.df.columns):
            try:
                if(len(self.df[name_col].unique()) > 1):
                    mean = self.df[name_col].mean()
                    std = self.df[name_col].std()
                    title = f'Mean: {mean:.2f} | Std: {std:.2f}'
                    plt.subplot(rws+1, 3, i+1).set_title(title)
                    sns.distplot(self.df[name_col])
                    plt.axvline(mean, c='red')
    #                 plt.axvline(mean+std*3,c='r')
            except:
                pass
        plt.tight_layout()
        plt.show()

    def corr_heatmap(self):
        self.corr = self.df.corr()
        mask = np.zeros_like(self.corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        matrix = np.triu(self.corr)
        # plt.figure(figsize=(15, 10))
        plt.figure(num='Correlation heatmap', dpi=120)

        ax = sns.heatmap(
            self.corr,
            vmin=-1, vmax=1, center=0,
            cmap='coolwarm',
            mask=matrix,
            square=True,
            annot=True,
            fmt='.3g',
            linewidths=1.25, linecolor='white'
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            horizontalalignment='center'
        )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            horizontalalignment='center'
        )
        plt.show()

    def high_corr(self, threshold=0.9, drop=False):
        """Calculates and return the more correlated columns that pass the threshold

        Args:
            threshold (float, optional): [description]. Defaults to 0.9.
            drop (bool, optional): [description]. Defaults to False.

        Returns:
            [list]: [Columns that reach the threshold]
        """

        # ! Create correlation matrix
        corr_matrix = self.df.corr().abs()

        # todo Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # ? Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]

        if drop:
            self.dropper(to_drop, axis=1)

        return to_drop #* Done

    def copy(self):
        return self.df.copy()
        