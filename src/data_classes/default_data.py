from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd



class DefaultData(Data):
    def __init__(self, preproc=None, test_ratio=0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        super().__init__('data/UCI_Credit_Card.csv', preproc, test_ratio)

    def _get_columns(self, dataset: pd.DataFrame, preproc: str) -> pd.DataFrame:
        """Select used columns"""
        dataset = self.my_columns(dataset)

        return dataset
    
    def _get_labels(self, dataset: pd.DataFrame) -> np.array:
        """Select Y label column"""
        return dataset['Probability'].to_numpy()

    def my_columns(self, X):
        return X.drop(['ID', 'Probability'], axis=1)
        return X[['sex', 'age', 'race',
                  'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'priors_count', 'c_charge_degree', "decile_score.1", "v_decile_score"]] ##, ?? NOTE:  "v_decile_score"
        # TODO: ADD MORE COLUMNS ONCE WE HAVE ONE HOT
        #return X[["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"]]

    def _clean_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # preprocessing done according to preprocessing.ipynb
        dataset = dataset.dropna()
        dataset['SEX'] = np.where(dataset['SEX'] == 2, 0, 1)
        dataset.rename(index=str, columns={"SEX": "sex"}, inplace=True)

        # label 1 = defaulted on payment. 0 initially preferred
        dataset.rename(index=str, columns={"default.payment.next.month": "Probability"}, inplace=True)
        dataset['Probability'] = np.where(dataset['Probability'] == 0, 1, 0)
        
        dataset['MARRIAGE'] = np.where(dataset['MARRIAGE'] == 1, 'married', dataset['MARRIAGE'])
        dataset['MARRIAGE'] = np.where(dataset['MARRIAGE'] == 2, 'single', dataset['MARRIAGE'])
        dataset['MARRIAGE'] = np.where(dataset['MARRIAGE'] == 3, 'other', dataset['MARRIAGE'])
        
        return dataset
        

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex'] # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
        # raise NotImplementedError

