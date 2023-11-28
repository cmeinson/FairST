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

class Model:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    LG_R = "LogisticRegression" 
    DT_R = "DecisionTreeRegressor"
    DT_C = "DecisionTreeClassifier"
    RF_C = "RandomForestClassifier"
    KN_C = "KNearestNeighbours"
    SV_C = "SupportVectorClassifier"
    NN_C = "nn keras"
    NN_old = "MLPClassifier"
    NB_C = "NaiveBayes"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyperparams we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        raise NotImplementedError

    def train(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        :param sensitive_atributes: names of sensitive attributes to be protected
        :type sensitive_atributes: List[str]
        :param method:  ml algo name to use for the main model training
        :type method: _type_
        :param method_bias: method name if needed for the bias mitigation, defaults to None
        :type method_bias: _type_, optional
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        raise NotImplementedError
    
    def _get_transformer(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_processor = OneHotEncoder(handle_unknown = 'infrequent_if_exist')
        numerical_processor = StandardScaler()
        transformer = ColumnTransformer([
            ('OneHotEncoder', categorical_processor, categorical_columns)], remainder="passthrough") 
        
        return transformer
    
        
    def _get_model(self, method, other={}):
        if method == self.SV_C:
            return SVC()
        elif method == self.KN_C:
            # example of how to pass hyperparams through "other"
            k = 3 if ("KNN_k" not in other) else other["KNN_k"] 
            return KNeighborsClassifier(k)
        elif method == self.NN_C:
            model = Sequential()
            model.add(Dense(45, input_dim=other["input_dim"] ,
                activation="relu"))
            model.add(Dense(45, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(30, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(15, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(1))
            model.add(Activation("sigmoid"))
            sgd = SGD(learning_rate=0.001, clipnorm=1)
            model.compile(loss="binary_crossentropy", optimizer=sgd,
                metrics=["accuracy"])
            return model
        elif method == self.NN_old:
            iter = 500 if ("iter_1" not in other) else other["iter_1"] 
            ws = False if ("warm_start" not in other) else other["warm_start"] 
            return MLPClassifier( max_iter= iter, warm_start=ws)
        elif method == self.NB_C:
            return GaussianNB()
        elif method == self.RF_C:
            return RandomForestClassifier()
        elif method == self.DT_C:
            return DecisionTreeClassifier()
        elif method == self.DT_R:
            return DecisionTreeRegressor()
        elif method == self.LG_R:
            return LogisticRegression(max_iter=100000)
        else:
            raise RuntimeError("Invalid ml method name: ", method)
        
    def _is_regression(self, method):
        return method in [self.LG_R, self.DT_R]
    


class BaseModel(Model):
    OPT_FBALANCE = "Apply fairbalance column transform"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        self._use_transformer = (self.OPT_FBALANCE in other) and (other[self.OPT_FBALANCE])    
        self._model = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        self._model = self._get_model(method, other | {"input_dim":X.shape[1]})  
        self.transformer = self._get_transformer(X)   
        if self._use_transformer:     
            self._model.fit(self.transformer.fit_transform(X), y)
        else:
            self._model.fit(X.to_numpy(), y)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        if self._use_transformer:  
            y = self._model.predict(self.transformer.transform(X))
        else:
            y = self._model.predict(X)

        y[y>=0.5] = 1
        y[y<=0.5] = 0
        return y
    
