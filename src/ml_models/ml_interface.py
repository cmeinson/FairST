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
from ..utils import TestConfig

class Model:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    LG_R = "LogisticRegression" 
    DT_R = "DecisionTreeRegressor"
    DT_C = "DecisionTreeClassifier"
    RF_C = "RandomForestClassifier"
    KN_C = "KNearestNeighbours"
    SV_C = "SupportVectorClassifier"
    NN_C = "nn keras"
    MLP_C = "MLPClassifier"
    NB_C = "NaiveBayes"

    def __init__(self, config: TestConfig) -> None:
        """
        """
        self._config = config
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.array):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, binary = True) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame

        :return: predictions for each row of X
        :rtype: np.array
        """
        if binary:
            return self._model.predict(X)
        # TODO: make this into ._predict functionality
        # TODO: not sure predict proba is the way to go for all modes!!!!
        return self._model.predict_proba(X)[:,1]
    
        
    def _get_model(self, method = None):
        other = self._config.other
        if not method:
            method = self._config.ml_method

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
        elif method == self.MLP_C:
            iter = 500 if ("iter_1" not in other) else other["iter_1"] 
            return MLPClassifier( max_iter= iter)
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
    
    def _binarise(self, y: np.array) -> np.array:
        y[y>=0.5] = 1
        y[y<=0.5] = 0
        return y
    


class BaseModel(Model):
    def fit(self, X: pd.DataFrame, y: np.array):
        # if self._model if the base model was not previously fitted.
        if self._model is None:
            self._model = self._get_model() #  | {"input_dim":X.shape[1]} 
            self._model.fit(X, y)
        # TODO: rethink the base model logic so that could use preproc and in proc as base models
