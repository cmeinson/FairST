from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MEPSData(Data):
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
        self.data = pd.read_csv('data/MEPS.csv')

        # Do default preprocessing
        self.pre_processing()

        self._X = pd.DataFrame(self.data)
        # print(self._X)
        self._y = self.data['Probability'].to_numpy()

        if preprocessing == "FairBalance":
            self._X = self.fairbalance_columns(self._X)
            # pass
        else:
            self._X = self.fairmask_columns(self._X)
        # print(self._X)
        # Create train-test split
        self.new_data_split()

    def fairbalance_columns(self, X):
        return X[["REGION", "AGE", "sex", "race", "MARRY", "FTSTU", "ACTDTY", "HONRDC", "RTHLTH", "MNHLTH", "HIBPDX",
                  "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "EMPHDX", "CHBRON", "CHOLDX", "CANCERDX", "DIABDX",
                  "JTPAIN", "ARTHDX", "ARTHTYPE", "ASTHDX", "ADHDADDX", "PREGNT", "WLKLIM", "ACTLIM", "SOCLIM",
                  "COGLIM", "DFHEAR42", "DFSEE42", "ADSMOK42", "PCS42", "MCS42", "K6SUM42", "PHQ242", "EMPST", "POVCAT", "INSCOV","PERWT15F"]]

    def fairmask_columns(self, X):
        return X[["REGION", "AGE", "sex", "race", "MARRY", "FTSTU", "ACTDTY", "HONRDC", "RTHLTH", "MNHLTH", "HIBPDX",
                  "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "EMPHDX", "CHBRON", "CHOLDX", "CANCERDX", "DIABDX",
                  "JTPAIN", "ARTHDX", "ARTHTYPE", "ASTHDX", "ADHDADDX", "PREGNT", "WLKLIM", "ACTLIM", "SOCLIM", "COGLIM",
                  "DFHEAR42", "DFSEE42", "ADSMOK42", "PCS42", "MCS42", "K6SUM42", "PHQ242", "EMPST", "POVCAT", "INSCOV", "PERWT15F"]]

    def pre_processing(self):
        # preprocessing done according to preprocessing.ipynb
        self.data = self.data.dropna()

        self.data = self.data.rename(
            columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                     'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                     'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                     'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                     'POVCAT15': 'POVCAT', 'INSCOV15': 'INSCOV'})

        self.data = self.data[self.data['PANEL'] == 19]
        self.data = self.data[self.data['REGION'] >= 0]  # remove values -1
        self.data = self.data[self.data['AGE'] >= 0]  # remove values -1
        self.data = self.data[self.data['MARRY'] >= 0]  # remove values -1, -7, -8, -9
        self.data = self.data[self.data['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
        self.data = self.data[
            (self.data[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        self.data['RACEV2X'] = np.where((self.data['HISPANX'] == 2) & (self.data['RACEV2X'] == 1), 1, self.data['RACEV2X'])
        self.data['RACEV2X'] = np.where(self.data['RACEV2X'] != 1, 0, self.data['RACEV2X'])
        self.data = self.data.rename(columns={"RACEV2X": "RACE"})

        self.data['SEX'] = np.where(self.data['SEX'] != 1, 0, self.data['SEX'])

        # self.data['UTILIZATION'] = np.where(self.data['UTILIZATION'] >= 10, 1, 0)

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        self.data['TOTEXP15'] = self.data.apply(lambda row: utilization(row), axis=1)
        lessE = self.data['TOTEXP15'] < 10.0
        self.data.loc[lessE, 'TOTEXP15'] = 0.0
        moreE = self.data['TOTEXP15'] >= 10.0
        self.data.loc[moreE, 'TOTEXP15'] = 1.0

        self.data = self.data.rename(columns={'TOTEXP15': 'UTILIZATION'})

        self.data = self.data[['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                         'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                         'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                         'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                         'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                         'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT15F']]

        self.data = self.data.rename(columns={"UTILIZATION": "Probability", "RACE": "race", "SEX": "sex"})

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['race', 'sex']
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError