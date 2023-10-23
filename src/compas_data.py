from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CompasData(Data):
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    # does reading and cleaning go here or do we add extra functions for that?
    def __init__(self, preprocessing=None, test_ratio=0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        self._test_ratio = test_ratio
        self.data = pd.read_csv('data/compas-scores-two-years.csv')

        # Do default preprocessing
        self.pre_processing()

        self._X = pd.DataFrame(self.data)
        self._y = self.data['Probability'].to_numpy()

        if preprocessing == "FairBalance":
            self._X = self.fairbalance_columns(self._X)
        else:
            self._X = self.fairmask_columns(self._X)

        # Create train-test split
        self.new_data_split()                                                                                 

    def fairbalance_columns(self, X):
        return X[['sex', 'age', 'age_cat', 'race',
                  'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                  'priors_count', 'c_charge_degree', 'c_charge_desc']]
    
    def fairmask_columns(self, X):
        return X[["sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1"]]
    
    def pre_processing(self):
        # preprocessing done according to preprocessing.ipynb
        self.data = self.data.drop(
            ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'decile_score',
             'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text', 'screening_date',
             'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
             'start', 'end', 'event'], axis=1)

        self.data = self.data.dropna()

        self.data['race'] = np.where(self.data['race'] != 'Caucasian', 0, 1)
        self.data['sex'] = np.where(self.data['sex'] == 'Female', 0, 1)
        self.data['age_cat'] = np.where(self.data['age_cat'] == 'Greater than 45', 45, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == '25 - 45', 25, self.data['age_cat'])
        self.data['age_cat'] = np.where(self.data['age_cat'] == 'Less than 25', 0, self.data['age_cat'])
        self.data['c_charge_degree'] = np.where(self.data['c_charge_degree'] == 'F', 1, 0)

        self.data.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
        self.data['Probability'] = np.where(self.data['Probability'] == 0, 1, 0)

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex', 'race'] # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError

# compas = CompasData()
