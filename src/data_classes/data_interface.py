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
    name_split_symbol = ": cat="
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
        self._binary_columns = None # col_name: [og values]
        self._one_hot_columns = None # og_col_name: [new cols]
        self._transformed_column_names = None
        
        self._test_ratio = test_ratio
        
        self._raw = pd.read_csv(file_name)

        self.dataset_orig = self._clean_data(self._raw)
        # base preproc - remove nans, any feature eng i need 
        # Do default pre-processing


        self._y = self._get_labels(self.dataset_orig)

        # selected preproc columns   
        # fit and apply trainsformer normalisation, one hot encoding (NOTE: keep track of sensitive columns_)
        # TODO; func that would reverse one hot enc for diplay opurposes?
        self._untransformed_cols = self._get_columns(self.dataset_orig, preprocessing)
        self._X = self._std_data_transform(self._untransformed_cols)
  
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
    
    def _get_transformer_scaler(self, X, onehot = True):
        numerical_columns_selector = selector(dtype_exclude=object)
        numerical_columns = numerical_columns_selector(X)
        non_binary_numerical_columns = [
            column for column in numerical_columns
            if len(X[column].unique()) > 2 or sum(abs(X[column].unique())) > 2
        ]
        numerical_processor = StandardScaler()

        if not onehot:
            return ColumnTransformer([('StandardScaler', numerical_processor, non_binary_numerical_columns)], remainder="passthrough", verbose_feature_names_out = False, sparse_threshold=0)
        
        categorical_columns_selector = selector(dtype_include=object)
        categorical_columns = categorical_columns_selector(X)
        name_combiner = lambda col, cat: col+self.name_split_symbol+str(cat)
        categorical_processor = OneHotEncoder(handle_unknown = 'infrequent_if_exist', feature_name_combiner=name_combiner) # NOTE: has inverse_transform

        transformer = ColumnTransformer([
            ('StandardScaler', numerical_processor, non_binary_numerical_columns),
            ('OneHotEncoder', categorical_processor, categorical_columns)
            ], remainder="passthrough", verbose_feature_names_out = False, sparse_threshold=0)
        
        return transformer
    
    def _std_data_transform(self, X):      
        transformer = self._get_transformer_scaler(X)
        transformer.fit_transform(X)
        X_new = transformer.transform(X)
        
        self._transformed_column_names = transformer.get_feature_names_out()
        self._save_transformer_column_types(transformer, X_new)

        X_new = pd.DataFrame(X_new, columns=self._transformed_column_names, dtype=np.float32)
        return X_new
    
      
    def post_mask_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col_name, og_values in self._binary_columns.items():
            X[col_name] = X[col_name].map(lambda x: self._round_to_closest(x, og_values))
            
        for og_name, new_cols in self._one_hot_columns.items():
            max_values = X[new_cols].max(axis=1)
            X[new_cols] = np.where(X[new_cols].eq(max_values, axis=0), 1, 0)
        return X
            
    
    def _save_transformer_column_types(self, transformer, X_new):
        bin_slice = transformer.output_indices_["remainder"]
        hot_slice = transformer.output_indices_["OneHotEncoder"]
        
        self._binary_columns = {} # col_name: [og values]
        for i in range(bin_slice.start, bin_slice.stop):
            col_name = self._transformed_column_names[i]
            self._binary_columns[col_name] = np.unique(X_new[:, i])
        
        self._one_hot_columns = {} # og_col_name: [new cols]
        for i in  range(hot_slice.start, hot_slice.stop):
            new_col_name = self._transformed_column_names[i]
            og_name, cat = new_col_name.split(self.name_split_symbol)
            if og_name not in self._one_hot_columns:
                self._one_hot_columns[og_name] = []
            self._one_hot_columns[og_name].append(new_col_name)
        
  
    def _round_to_closest(self, value, rounding_values):
        return min(rounding_values, key=lambda x: abs(x - value))
            
        
        



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
        return ["sensitive"]
    
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




