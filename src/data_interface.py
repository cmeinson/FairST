from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

from skorch import NeuralNetClassifier
from torch import nn, float32
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense




class Data:

    def __init__(self, file_name: str, preprocessing:str = None, test_ratio = 0.2) -> None:
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

        self.dataset_orig = self._clean_data(pd.read_csv(file_name))
        # base preproc - remove nans, any feature eng i need 
        # Do default pre-processing


        self._y = self._get_labels(self.dataset_orig)

        # selected preproc columns   
        # fit and apply trainsformer normalisation, one hot encoding (NOTE: keep track of sensitive columns_)
        # TODO; func that would reverse one hot enc for diplay opurposes?
        self._X = self._get_columns(self.dataset_orig, preprocessing)
  
        # data split
        self.new_data_split()

    
    def _clean_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning (e.g. remove nans, any feature eng i need )"""
        raise NotImplementedError
    
    def _get_columns(self, dataset: pd.DataFrame, preprocessing: str) -> pd.DataFrame:
        """Select used columns"""
        raise NotImplementedError
    
    def _get_labels(self, dataset: pd.DataFrame) -> np.array:
        """Select Y label column"""
        raise NotImplementedError
    
    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        raise NotImplementedError
    
    def new_data_split(self) -> None:
        """Changes the data split"""
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=0.1) # TOD): PUT IT BACKKKKKKKKKKKKK to self.test_size

    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        """Returns the training data where
        X: is the df with all attriutes, with accordig column names
        y: the outcome for each row (e.g. the default credit, is income above 50k, did reoffend?)
        Duplicares everything for safety reasons:)

        :return: training data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_train.copy(), self._y_train.copy())

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        """
        :return: test data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_test.copy(), self._y_test.copy())
    
    def _get_transformer_scaler(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)

        numerical_columns = numerical_columns_selector(X)
        non_binary_numerical_columns = [
            column for column in numerical_columns
            if len(X[column].unique()) > 2 or sum(abs(X[column].unique())) > 2
        ]

        numerical_processor = StandardScaler()
        transformer = ColumnTransformer([
            ('StandardScaler', numerical_processor, non_binary_numerical_columns)], remainder="passthrough", verbose_feature_names_out = False)
        return transformer
    
    def std_data_transform(self, X):
        # just normalisation no one hot
        transformer = self._get_transformer_scaler(X)
        transformer.fit_transform(X)
        X = transformer.transform(X)
        transformed_column_names = transformer.get_feature_names_out()
        X = pd.DataFrame(X, columns=transformed_column_names, dtype=np.float32)
        print(X)
        return X



class DummyData(Data):
    def __init__(self, preprocessing = None, test_ratio=0.2) -> None:
        data = [[0.31, 0, 0], [0.41, 0, 1], [0.51, 0, 0], [0.71, 0, 1], [0.81, 0, 0], [0.91, 0, 1], 
                [0.2, 1, 0], [0.3, 1, 1], [0.4, 1, 0], [0.5, 1, 1], [0.6, 1, 0], [0.7, 1, 1]]
        self._X = pd.DataFrame(data, columns=["attr", "sensitive", "sensitive2"])
        self._y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,])

        self.my_experiment()
        self._test_ratio = test_ratio
        self.new_data_split()        

    def get_sensitive_column_names(self) -> List[str]:
        return ["sensitive", "sensitive2"]
    
    def my_experiment(self):
        n = 10000

        data = []
        ys = []

        for i in range(n):
            privilege = np.random.uniform(0,1)
            merit = np.random.uniform(0,1)

            luck = np.random.uniform(-5, 4) 

            money = np.exp(luck + privilege*0.7 + merit*0.3) #(0 - 155) 55 for unprivileged

            education = ((luck+5)/10 + merit + privilege)**2 # 0-9
            first_job = education/5 + merit + money/100 + np.random.uniform(0,1) # - 5
            something = np.random.uniform(0,1) + money/120 # up to 2
            something2 = np.random.uniform(0,1)*0.2 + privilege*0.8 # up to 2
            something3 = np.random.uniform(0,1)*0.3 + merit*0.5 + privilege*0.2 # up to 2

            y = 1
            if merit<0.6:
                y =0
            
            sensitive = 1
            if privilege < 0.4:
                sensitive = 0

            data.append([sensitive, education/8, first_job/5, something/2, something2/2, something3/2])
            ys.append(y)
        self._X = pd.DataFrame(data, columns=["sensitive", "edu", "job", "idkmoney", "idkprivi", "idkmerit"])
        self._y = np.array(ys)




