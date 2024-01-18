from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd



class CompasData(Data):
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
        super().__init__('data/compas-scores-two-years.csv', preproc, test_ratio)

    def _get_columns(self, dataset: pd.DataFrame, preproc: str) -> pd.DataFrame:
        """Select used columns"""
        if preproc == "FairBalance":
            dataset = self.fairbalance_columns(dataset)
        elif preproc == "FairMask":
            dataset = self.fairmask_columns(dataset)
        else:
            dataset = self.my_columns(dataset)

        return dataset
    
    def _get_labels(self, dataset: pd.DataFrame) -> np.array:
        """Select Y label column"""
        return dataset['Probability'].to_numpy()

    def my_columns(self, X):
        return X[['sex', 'age', 'race',
                  'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'priors_count', 'c_charge_degree', "decile_score.1", "v_decile_score"]] ##, ?? NOTE:  "v_decile_score"
        # TODO: ADD MORE COLUMNS ONCE WE HAVE ONE HOT
        #return X[["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"]]

                                                                                   
    def fairbalance_columns(self, X):
        return X[['sex', 'age', 'age_cat', 'race',
                  'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'priors_count', 'c_charge_degree', 'c_charge_desc']]
    
    def fairmask_columns(self, X):
        return X[["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"]]
    
    def _clean_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # preprocessing done according to preprocessing.ipynb
        dataset = dataset.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'decile_score',
             'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text', 'screening_date',
             'v_type_of_assessment', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
             'start', 'end', 'event'], axis=1)
        
        dataset = dataset.dropna()

        dataset['race'] = np.where(dataset['race'] != 'Caucasian', 0, 1)
        dataset['sex'] = np.where(dataset['sex'] == 'Female', 0, 1)
        dataset['age_cat'] = np.where(dataset['age_cat'] == 'Greater than 45', 45, dataset['age_cat'])
        dataset['age_cat'] = np.where(dataset['age_cat'] == '25 - 45', 25, dataset['age_cat'])
        dataset['age_cat'] = np.where(dataset['age_cat'] == 'Less than 25', 0, dataset['age_cat'])
        dataset['c_charge_degree'] = np.where(dataset['c_charge_degree'] == 'F', 1, 0)

        dataset.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
        dataset['Probability'] = np.where(dataset['Probability'] == 0, 1, 0)

        return dataset

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex', 'race'] # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
        # raise NotImplementedError

