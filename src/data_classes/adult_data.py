from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
age,            [continuous]    keep
workclass,      [merge 8 -> 5 cat]  
fnlwgt,         -[continuous]    DROP    basically contains sensitive data as far as i understand final weight: The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian noninstitutional population of the US.
education,      -[16 cat]        DROP. it is a one to one mapping basically with education-num which is continuous
education-num,  [continuous]    keep
marital-status, [merge 7 -> binary]  
occupation,     [15 cat] keep all for now but could merge some into ?
relationship,   [6 -> 4 cat] Other-relative, Own-child, Spouse (Husband/Wife), None (THe rest)  
race,
sex,
capital-gain,   bin ??? seems like a weird param
capital-loss,   bin ??? seems like a weird param
hours-per-week, [continuous]    keep
native-country, - I kinda want to leave it out as it feels susss and potentially race encoding

Probability - the Y label
"""
"""

DROP: fnlwgt, education,  native-country

"""

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
        dataset['race'] = np.where(dataset['race'] == 'White', 1, 0)
        dataset['Probability'] = np.where(dataset['Probability'] == '<=50K', 0, 1)

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
        X = X.drop(['fnlwgt', 'education', 'native-country', 'Probability'], axis=1)
        
        X['workclass'] = X['workclass'].replace(['Never-worked', 'Without-pay'], '?')

        X['marital-status'] = X['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 1)
        X['marital-status'] = np.where(X['marital-status'] == 1, 1, 0)

        X['relationship'] = X['relationship'].replace(['Wife', 'Husband'], 'Spouse') # TODO: to test the masks ability, see if it swithces this upon swithcing gender!!!!!!!!!
        X['relationship'] = X['relationship'].replace(['Not-in-family', 'Unmarried'], 'No') 
        return X
 
    
    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        return ['race', 'sex']