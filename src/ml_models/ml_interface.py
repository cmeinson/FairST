from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
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
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras.layers import Dense
from ..utils import TestConfig

class Model:
    LG_R = "LogisticRegression" 
    DT_R = "DecisionTreeRegressor"
    DT_C = "DecisionTreeClassifier"
    RF_C = "RandomForestClassifier"
    KN_C = "KNearestNeighbours"
    SV_C = "SupportVectorClassifier"
    NN_C = "nn keras"
    MLP_C = "MLPClassifier"
    NB_C = "NaiveBayes"
    EN_R = "ElasticNet"

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

    def _fit(self, model, X, y, sample_weight = None, epochs = 1000):
        if isinstance(model, Sequential):
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X, y, sample_weight = sample_weight, epochs = epochs, callbacks=[early_stopping], verbose=0)
        elif isinstance(model, MLPClassifier):
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight = sample_weight)
        
    
    def predict(self, X: pd.DataFrame, binary = True) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame

        :return: predictions for each row of X
        :rtype: np.array
        """
        return self._predict(self._model, X, binary)
    
    def _predict(self, model, X: pd.DataFrame, binary = True) -> np.array:
        if isinstance(model, Sequential):
            preds = np.squeeze(model.predict(X))
        elif isinstance(model, ElasticNet) or isinstance(model, DecisionTreeRegressor):
            preds = model.predict(X)
        elif isinstance(model, BaseModel):
            preds = model.predict(X, binary = binary)
        else:
            if binary:
                return model.predict(X)
            # NOTE: mentioned that pfor NB predict proba is not the best!
            return model.predict_proba(X)[:,1]

        if binary:
            return self._binarise(preds)
        return preds  
    
        
    def _get_model(self, method = None):
        other = self._config.other
        if not method:
            method = self._config.ml_method

        if method == self.SV_C:
            return SVC(probability=True)
        elif method == self.KN_C:
            # example of how to pass hyperparams through "other"
            k = 3 if ("KNN_k" not in other) else other["KNN_k"] 
            return KNeighborsClassifier(k)
        elif method == self.NN_C:
            model = Sequential()
            model.add(Dense(128, input_dim=self._config.n_cols,
                activation="relu"))
            model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
            model.add(Dense(1))
            model.add(Activation("sigmoid"))
            sgd = Adam(learning_rate=0.001, clipnorm=1)
            model.compile(loss="binary_crossentropy", optimizer=sgd,
                metrics=["accuracy"])
            return model
        elif method == self.MLP_C:
            iter = 500 if ("iter_1" not in other) else other["iter_1"] 
            return MLPClassifier(max_iter=iter, hidden_layer_sizes=(128, 64, 32, 16))
        elif method == self.NB_C:
            return GaussianNB()
        elif method == self.RF_C:
            return RandomForestClassifier()
        elif method == self.DT_C:
            return DecisionTreeClassifier()
        elif method == self.EN_R:
            return ElasticNet(alpha=0.05)
        elif method == self.DT_R:
            return DecisionTreeRegressor()
        elif method == self.LG_R:
            return LogisticRegression(max_iter=100000)
        else:
            raise RuntimeError("Invalid ml method name: ", method)
        
    def _is_regression(self, method):
        return method in [self.LG_R, self.DT_R, self.EN_R]
    
    def _binarise(self, y: np.array) -> np.array:
        y[y>=0.5] = 1
        y[y<0.5] = 0
        return y.astype(int)
    


class BaseModel(Model):
    def fit(self, X: pd.DataFrame, y: np.array):
        # if self._model if the base model was not previously fitted.
        if self._model is None:
            self._model = self._get_model() #  | {"input_dim":X.shape[1]} 
            self._model.fit(X, y)
        # TODO: rethink the base model logic so that could use preproc and in proc as base models
