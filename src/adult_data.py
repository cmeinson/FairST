from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
FairBalance Adult Data columns: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, 
                    race, sex, capital-gain, capital-loss, hours-per-week, native-country

FairMask Adult Data columns: age, education-num, race, sex, capital-gain, capital-loss, hours-per-week
"""

class AdultData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, preprocessing:str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the data folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train
        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        self._test_ratio = test_ratio
        self.dataset_orig = pd.read_csv('data/adult.csv')

        # Do default pre-processing from Preprocessing.ipynb
        self.pre_processing()

        # Split into input and output
        self._X = pd.DataFrame(self.dataset_orig)
        self._y = self.dataset_orig['Probability'].to_numpy()

        if preprocessing == "FairBalance":
            self._X = self.fairbalance_columns(self._X)
        else:
            self._X = self.fairmask_columns(self._X)

        # Create train-test split
        self.new_data_split()

    def pre_processing(self):
        self.dataset_orig = self.dataset_orig.dropna()

        # Binarize sex, race and income (probability)
        self.dataset_orig['sex'] = np.where(self.dataset_orig['sex'] == 'Male', 1, 0)
        self.dataset_orig['race'] = np.where(self.dataset_orig['race'] != 'White', 0, 1)
        self.dataset_orig['Probability'] = np.where(self.dataset_orig['Probability'] == '<=50K', 0, 1)

        # Discretize age
        self.dataset_orig['age'] = np.where(self.dataset_orig['age'] >= 70, 70, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 60 ) & (self.dataset_orig['age'] < 70), 60, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 50 ) & (self.dataset_orig['age'] < 60), 50, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 40 ) & (self.dataset_orig['age'] < 50), 40, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 30 ) & (self.dataset_orig['age'] < 40), 30, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 20 ) & (self.dataset_orig['age'] < 30), 20, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where((self.dataset_orig['age'] >= 10 ) & (self.dataset_orig['age'] < 10), 10, self.dataset_orig['age'])
        self.dataset_orig['age'] = np.where(self.dataset_orig['age'] < 10, 0, self.dataset_orig['age'])

    def fairmask_columns(self, X):
        return X.drop(['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'Probability'], axis=1)

    def fairbalance_columns(self, X):
        return X.drop(['fnlwgt', 'education', 'Probability'], axis=1)
    
    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        
        #return ['sex', 'race', 'age', 'Probability'] (is age, income a sensitive attribute?)
        return ['race', 'sex']