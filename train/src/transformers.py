import sys
import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

sys.path.append('././')
from serve.src.functions import get_title

class ReplaceQM(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        """ Replace the values '?' with numpy.nan.
        It identifies all values contained in the data frame with value
        '?' and replaces them with numpy.nan.
        Parameters
        ----------
        X : pd.DataFrame
            The pandas.DataFrame that will be transformed
        Returns
        -------
        pd.DataFrame
            The pd.DataFrame Transformed.
        """ 
        return X.replace('?',np.nan)

class CleanData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X:np.ndarray, y=None)->np.ndarray:
        """ Cleans numeric columns in position index 1 and 4.
        Converts column index 1 as type float, then, rounds it to its
        closest integer and then converts it into an integer.
        Then converts column index 4 as type float, and finlly rounds it to its
        closest integer.
        Parameters
        ----------
        X : np.ndarray
            The np.ndarray that will be transformed
        Returns
        -------
        np.ndarray
            The np.ndarray transformed.
        """ 
        X[:,1] =  X[:,1].astype('float')
        X[:,1] = list(map(lambda x: round(x, 0),  X[:,1]))
        X[:,1] =  X[:,1].astype('int')

        X[:,4] =  X[:,4].astype('float')
        X[:,4] = list(map(lambda x: round(x, 0),  X[:,4]))

        return X

class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None)->pd.DataFrame:
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        """ Create the title and cabin_letter columns.
        Creates the title column by mapping the get_title() function
        to the column name.
        Converts the cabin columnt astype str and extrancts the first
        character.
        If the first character is 'n', it is repalced by 'U'
        Parameters
        ----------
        X : pd.DataFrame
            The pandas.DataFrame that will be transformed
        Returns
        -------
        pd.DataFrame
            The pd.DataFrame Transformed.
        """ 
        X['title'] = list(map(lambda x: get_title(x), X['name']))
        X['cabin'] = X['cabin'].astype('str')
        X['cabin_letter'] = list(map(lambda x: x[0], X['cabin']))
        X['cabin_letter'] = X['cabin_letter'].replace('n','U')
        return X