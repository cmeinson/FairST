from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class AdultData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, preproc :str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the data folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train
        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        super().__init__('data/adult.csv', preproc, test_ratio)

    def _clean_data(self, dataset):
        dataset = dataset.dropna()

        # Binarize sex, race and income (probability)
        dataset['sex'] = np.where(dataset['sex'] == 'Male', 1, 0)
        dataset['race'] = np.where(dataset['race'] != 'White', 0, 1)
        dataset['Probability'] = np.where(dataset['Probability'] == '<=50K', 0, 1)

        # Discretize age
        dataset['age'] = np.where(dataset['age'] >= 70, 70, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 60 ) & (dataset['age'] < 70), 60, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 50 ) & (dataset['age'] < 60), 50, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 40 ) & (dataset['age'] < 50), 40, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 30 ) & (dataset['age'] < 40), 30, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 20 ) & (dataset['age'] < 30), 20, dataset['age'])
        dataset['age'] = np.where((dataset['age'] >= 10 ) & (dataset['age'] < 10), 10, dataset['age'])
        dataset['age'] = np.where(dataset['age'] < 10, 0, dataset['age'])
        return dataset

    def _get_columns(self, dataset: pd.DataFrame, preproc: str) -> pd.DataFrame:
        """Select used columns"""
        if preproc == "FairBalance":
            dataset = self.fairbalance_columns(dataset)
        elif preproc == "FairMask":
            dataset = self.fairmask_columns(dataset)
        else:
            dataset = self.my_columns(dataset)

        return self.std_data_transform(dataset)
        
    def _get_labels(self, dataset: pd.DataFrame) -> np.array:
        """Select Y label column"""
        return dataset['Probability'].to_numpy()
    
    def fairmask_columns(self, X):
        return X.drop(['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'Probability'], axis=1)

    def fairbalance_columns(self, X):
        return X.drop(['fnlwgt', 'education', 'Probability'], axis=1)
    
    def my_columns(self, X):
        return X.drop(['fnlwgt',  'Probability'], axis=1)
        # TODO: ADD MORE COLUMNS ONCE WE HAVE ONE HOT
        #return X.drop(['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'Probability'], axis=1)

    
    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        return ['race', 'sex']