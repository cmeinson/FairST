from .data_interface import Data
from typing import List
import numpy as np
import pandas as pd


class GermanData(Data):
    def __init__(self, preproc :str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        super().__init__('data/GermanData.csv', preproc, test_ratio)

    def _get_labels(self, dataset: pd.DataFrame) -> np.array:
        """Select Y label column"""
        return dataset['Probability'].to_numpy()
    
    def _get_columns(self, dataset: pd.DataFrame, preproc: str) -> pd.DataFrame:
        """Select used columns"""
        if preproc == "FairMask":
            dataset = self.fairmask_columns(dataset)
        else:
            dataset = self.my_columns(dataset)

        return dataset

    def fairmask_columns(self, X):
        #NOT COMPLETE
        return X[["sex", "age",	"Probability",
                  "credit_history=Delay", "credit_history=None/Paid", "credit_history=Other",
                "savings=500+",	"savings=<500",	"savings=Unknown/None",
                "employment=1-4 years",	"employment=4+ years",	"employment=Unemployed"]]
    
    def my_columns(self, X):
        return X.drop(['Probability'], axis=1)


    def _clean_data(self, dataset):
        """
        Index(['1', '2', 'credit_history', '4', '5', 'savings', 'employment', '8',
            'sex', '10', '11', '12', 'age', '14', '15', '16', '17', '18', '19',
            '20', 'Probability'],
            dtype='object')
        """
        dataset = dataset.rename(columns={"1": "checkings", "2": "duration", "5": "credit_amount", "8": "credit_amount_%_income", 
                                "10": "guarantors", 
                                "12": "property", "15": "housing", "17": "job" ,"18": "nr_dependants"})       

        dataset = dataset.drop(
            ['4', '11', '14', '16', '19', '20'], axis=1)

        dataset = dataset.dropna()
        
        dataset['credit_history'] = dataset['credit_history'].replace(['A30'], 'None') 
        dataset['credit_history'] = dataset['credit_history'].replace(['A31', 'A32'], 'Paid') 
        dataset['credit_history'] = dataset['credit_history'].replace(['A33'], 'Delay') 
        dataset['credit_history'] = dataset['credit_history'].replace(['A34'], 'Other') 
        
        dataset['checkings'] = np.where(dataset['checkings'] == 'A11', 0, dataset['checkings'])
        dataset['checkings'] = np.where(dataset['checkings'] == 'A12', 100, dataset['checkings'])
        dataset['checkings'] = np.where(dataset['checkings'] == 'A13', 300, dataset['checkings'])
        dataset['checkings'] = np.where(dataset['checkings'] == 'A14', 0, dataset['checkings'])
         
        dataset['savings'] = np.where(dataset['savings'] == 'A61', 50, dataset['savings'])
        dataset['savings'] = np.where(dataset['savings'] == 'A62', 250, dataset['savings'])
        dataset['savings'] = np.where(dataset['savings'] == 'A63', 750, dataset['savings'])
        dataset['savings'] = np.where(dataset['savings'] == 'A64', 1500, dataset['savings'])
        dataset['savings'] = np.where(dataset['savings'] == 'A65', 0, dataset['savings'])
    
        dataset['employment'] = np.where(dataset['employment'] == 'A71', 0, dataset['employment'])
        dataset['employment'] = np.where(dataset['employment'] == 'A72', 1, dataset['employment'])
        dataset['employment'] = np.where(dataset['employment'] == 'A73', 3, dataset['employment'])
        dataset['employment'] = np.where(dataset['employment'] == 'A74', 6, dataset['employment'])
        dataset['employment'] = np.where(dataset['employment'] == 'A75', 10, dataset['employment'])
        
        dataset['sex'] = np.where(dataset['sex'] == 'A91', 1, dataset['sex'])
        dataset['sex'] = np.where(dataset['sex'] == 'A92', 0, dataset['sex'])
        dataset['sex'] = np.where(dataset['sex'] == 'A93', 1, dataset['sex'])
        dataset['sex'] = np.where(dataset['sex'] == 'A94', 1, dataset['sex'])
        dataset['sex'] = np.where(dataset['sex'] == 'A95', 0, dataset['sex'])
        
        dataset[["sex","checkings", "savings", "employment"]] = dataset[["sex","checkings", "savings", "employment"]].apply(pd.to_numeric)

        
        dataset['co-applicant'] = 0
        dataset['co-applicant'] = np.where(dataset['guarantors'] == 'A102', 1, 0)
        dataset['guarantors'] = np.where(dataset['guarantors'] == 'A103', 1, 0)


        dataset['Probability'] = np.where(dataset['Probability'] == 2, 0, 1)
        return dataset

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        return ['sex']  # For now removed the age cause it eas not used in a ny papers so not relevant in replication ['sex', 'age_cat', 'race']
       
       
       
"""
Description of the German credit dataset.

1. Title: German Credit data

2. Source Information

Professor Dr. Hans Hofmann  
Institut f"ur Statistik und "Okonometrie  
Universit"at Hamburg  
FB Wirtschaftswissenschaften  
Von-Melle-Park 5    
2000 Hamburg 13 

3. Number of Instances:  1000

Two datasets are provided.  the original dataset, in the form provided
by Prof. Hofmann, contains categorical/symbolic attributes and
is in the file "german.data".   
 
For algorithms that need numerical attributes, Strathclyde University 
produced the file "german.data-numeric".  This file has been edited 
and several indicator variables added to make it suitable for 
algorithms which cannot cope with categorical variables.   Several
attributes that are ordered categorical (such as attribute 17) have
been coded as integer.    This was the form used by StatLog.


6. Number of Attributes german: 20 (7 numerical, 13 categorical)
   Number of Attributes german.numer: 24 (24 numerical)


7.  Attribute description for german

Attribute 1:  (qualitative)
	       Status of existing checking account
               A11 :      ... <    0 DM
	       A12 : 0 <= ... <  200 DM
	       A13 :      ... >= 200 DM /
		     salary assignments for at least 1 year
               A14 : no checking account

Attribute 2:  (numerical)
	      Duration in month

Attribute 3:  (qualitative)
	      Credit history
	      A30 : no credits taken/
		    all credits paid back duly
          A31 : all credits at this bank paid back duly
	      A32 : existing credits paid back duly till now
          A33 : delay in paying off in the past
	      A34 : critical account/
		    other credits existing (not at this bank)

Attribute 4:  (qualitative)
	      Purpose
	      A40 : car (new)
	      A41 : car (used)
	      A42 : furniture/equipment
	      A43 : radio/television
	      A44 : domestic appliances
	      A45 : repairs
	      A46 : education
	      A47 : (vacation - does not exist?)
	      A48 : retraining
	      A49 : business
	      A410 : others

Attribute 5:  (numerical)
	      Credit amount

Attibute 6:  (qualitative)
	      Savings account/bonds
	      A61 :          ... <  100 DM
	      A62 :   100 <= ... <  500 DM
	      A63 :   500 <= ... < 1000 DM
	      A64 :          .. >= 1000 DM
              A65 :   unknown/ no savings account

Attribute 7:  (qualitative)
	      Present employment since
	      A71 : unemployed
	      A72 :       ... < 1 year
	      A73 : 1  <= ... < 4 years  
	      A74 : 4  <= ... < 7 years
	      A75 :       .. >= 7 years

Attribute 8:  (numerical)
	      Installment rate in percentage of disposable income

Attribute 9:  (qualitative)
	      Personal status and sex
	      A91 : male   : divorced/separated
	      A92 : female : divorced/separated/married
              A93 : male   : single
	      A94 : male   : married/widowed
	      A95 : female : single

Attribute 10: (qualitative)
	      Other debtors / guarantors
	      A101 : none
	      A102 : co-applicant
	      A103 : guarantor

Attribute 11: (numerical)
	      Present residence since

Attribute 12: (qualitative)
	      Property
	      A121 : real estate
	      A122 : if not A121 : building society savings agreement/
				   life insurance
          A123 : if not A121/A122 : car or other, not in attribute 6
	      A124 : unknown / no property

Attribute 13: (numerical)
	      Age in years

Attribute 14: (qualitative)
	      Other installment plans 
	      A141 : bank
	      A142 : stores
	      A143 : none

Attribute 15: (qualitative)
	      Housing
	      A151 : rent
	      A152 : own
	      A153 : for free

Attribute 16: (numerical)
              Number of existing credits at this bank

Attribute 17: (qualitative)
	      Job
	      A171 : unemployed/ unskilled  - non-resident
	      A172 : unskilled - resident
	      A173 : skilled employee / official
	      A174 : management/ self-employed/
		     highly qualified employee/ officer

Attribute 18: (numerical)
	      Number of people being liable to provide maintenance for

Attribute 19: (qualitative)
	      Telephone
	      A191 : none
	      A192 : yes, registered under the customers name

Attribute 20: (qualitative)
	      foreign worker
	      A201 : yes
	      A202 : no



8.  Cost Matrix

This dataset requires use of a cost matrix (see below)


      1        2
----------------------------
  1   0        1
-----------------------
  2   5        0

(1 = Good,  2 = Bad)

the rows represent the actual classification and the columns
the predicted classification.

It is worse to class a customer as good when they are bad (5), 
than it is to class a customer as bad when they are good (1).


"""