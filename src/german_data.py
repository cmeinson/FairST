from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class GermanData(Data):
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
        self.data = pd.read_csv('data/GermanData.csv')

        # Do default preprocessing
        self.pre_processing()

        self._X = pd.DataFrame(self.data)
        self._y = self.data['Probability'].to_numpy()
        print(self._X)

        if preprocessing == "FairBalance":
            # self._X = self.fairbalance_columns(self._X)
            pass
        else:
            self._X = self.fairmask_columns(self._X)
            # pass
        print(self._X)

        # Create train-test split
        self.new_data_split()

    def fairbalance_columns(self, X):
        return X[[]]

    def fairmask_columns(self, X):
        #NOT COMPLETE
        return X[["sex", "age",	"Probability",
                  "credit_history=Delay", "credit_history=None/Paid", "credit_history=Other",
                "savings=500+",	"savings=<500",	"savings=Unknown/None",
                "employment=1-4 years",	"employment=4+ years",	"employment=Unemployed"]]

    def pre_processing(self):
        # preprocessing done according to preprocessing.ipynb
        self.data = self.data.drop(
            ['1', '2', '4', '5', '8', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20'], axis=1)

        ## Drop NULL values
        self.data = self.data.dropna()

        ## Change symbolics to numerics
        self.data['sex'] = np.where(self.data['sex'] == 'A91', 1, self.data['sex'])
        self.data['sex'] = np.where(self.data['sex'] == 'A92', 0, self.data['sex'])
        self.data['sex'] = np.where(self.data['sex'] == 'A93', 1, self.data['sex'])
        self.data['sex'] = np.where(self.data['sex'] == 'A94', 1, self.data['sex'])
        self.data['sex'] = np.where(self.data['sex'] == 'A95', 0, self.data['sex'])
        self.data['age'] = np.where(self.data['age'] >= 25, 1, 0)
        self.data['credit_history'] = np.where(self.data['credit_history'] == 'A30', 1, self.data['credit_history'])
        self.data['credit_history'] = np.where(self.data['credit_history'] == 'A31', 1, self.data['credit_history'])
        self.data['credit_history'] = np.where(self.data['credit_history'] == 'A32', 1, self.data['credit_history'])
        self.data['credit_history'] = np.where(self.data['credit_history'] == 'A33', 2, self.data['credit_history'])
        self.data['credit_history'] = np.where(self.data['credit_history'] == 'A34', 3, self.data['credit_history'])

        self.data['savings'] = np.where(self.data['savings'] == 'A61', 1, self.data['savings'])
        self.data['savings'] = np.where(self.data['savings'] == 'A62', 1, self.data['savings'])
        self.data['savings'] = np.where(self.data['savings'] == 'A63', 2, self.data['savings'])
        self.data['savings'] = np.where(self.data['savings'] == 'A64', 2, self.data['savings'])
        self.data['savings'] = np.where(self.data['savings'] == 'A65', 3, self.data['savings'])

        self.data['employment'] = np.where(self.data['employment'] == 'A72', 1, self.data['employment'])
        self.data['employment'] = np.where(self.data['employment'] == 'A73', 1, self.data['employment'])
        self.data['employment'] = np.where(self.data['employment'] == 'A74', 2, self.data['employment'])
        self.data['employment'] = np.where(self.data['employment'] == 'A75', 2, self.data['employment'])
        self.data['employment'] = np.where(self.data['employment'] == 'A71', 3, self.data['employment'])

        self.data['credit_history=Delay'] = 0
        self.data['credit_history=None/Paid'] = 0
        self.data['credit_history=Other'] = 0

        self.data['credit_history=Delay'] = np.where(self.data['credit_history'] == 1, 1, self.data['credit_history=Delay'])
        self.data['credit_history=None/Paid'] = np.where(self.data['credit_history'] == 2, 1, self.data['credit_history=None/Paid'])
        self.data['credit_history=Other'] = np.where(self.data['credit_history'] == 3, 1, self.data['credit_history=Other'])

        self.data['savings=500+'] = 0
        self.data['savings=<500'] = 0
        self.data['savings=Unknown/None'] = 0

        self.data['savings=500+'] = np.where(self.data['savings'] == 1, 1, self.data['savings=500+'])
        self.data['savings=<500'] = np.where(self.data['savings'] == 2, 1, self.data['savings=<500'])
        self.data['savings=Unknown/None'] = np.where(self.data['savings'] == 3, 1, self.data['savings=Unknown/None'])

        self.data['employment=1-4 years'] = 0
        self.data['employment=4+ years'] = 0
        self.data['employment=Unemployed'] = 0

        self.data['employment=1-4 years'] = np.where(self.data['employment'] == 1, 1, self.data['employment=1-4 years'])
        self.data['employment=4+ years'] = np.where(self.data['employment'] == 2, 1, self.data['employment=4+ years'])
        self.data['employment=Unemployed'] = np.where(self.data['employment'] == 3, 1, self.data['employment=Unemployed'])

        self.data = self.data.drop(['credit_history', 'savings', 'employment'], axis=1)

        self.data['Probability'] = np.where(self.data['Probability'] == 2, 0, 1)

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        # returns a list of names
        return ['sex']  # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
        # raise NotImplementedError

    # def transform(self): # LATER
    #    # will probably rename later. but something for merging attributes into binary ones?
    #    raise NotImplementedError