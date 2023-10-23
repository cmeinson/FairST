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



class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, data, *a):
        print("fwd", data)
        data = data.to(float32)
        X = self.nonlin(self.dense0(data))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X
    
class MyNet(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        # make sure to set reduce=False in your criterion, since we need the loss
        # for each sample so that it can be weighted
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
         # override get_loss to use the sample_weight from X
         loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
         sample_weight = X['sample_weight']
         loss_reduced = (sample_weight * loss_unreduced).mean()
         return loss_reduced


class Model:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder
    LG_R = "LogisticRegression" 
    DT_R = "DecisionTreeRegressor"
    DT_C = "DecisionTreeClassifier"
    RF_C = "RandomForestClassifier"
    KN_C = "KNearestNeighbours"
    SV_C = "SupportVectorClassifier"
    NN_C = "MLPClassifier"
    NB_C = "NaiveBayes"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyperparams we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        raise NotImplementedError
    
    def _get_model(self, method, other={}):
        if method == self.SV_C:
            return SVC()
        elif method == self.KN_C:
            # example of how to pass hyperparams through "other"
            k = 3 if ("KNN_k" not in other) else other["KNN_k"] 
            return KNeighborsClassifier(k)
        elif method == self.NN_C:
            print("OTHE INPUT",  other)
            iter = 500 if ("iter_1" not in other) else other["iter_1"] 
            ws = False if ("warm_start" not in other) else other["warm_start"] 
            return MyNet(MyModule, max_epochs = iter, warm_start=ws)
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

        categorical_processor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_processor = StandardScaler()
        transformer = ColumnTransformer([
            ('OneHotEncoder', categorical_processor, categorical_columns),
            ('StandardScaler', numerical_processor, numerical_columns)])
        return transformer


class BaseModel(Model):
    OPT_FBALANCE = "Apply fairbalance column transform"

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        self._use_transformer = (self.OPT_FBALANCE in other) and (other[self.OPT_FBALANCE])    
        self._model = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_atributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        self._model = self._get_model(method, other)  
        self.transformer = self._get_transformer(X)   
        if self._use_transformer:     
            self._model.fit(self.transformer.fit_transform(X), y)
        else:
            self._model.fit(X.to_numpy(), y)

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        if self._use_transformer:  
            return self._model.predict(self.transformer.transform(X))
        else:
            return self._model.predict(X)
    
