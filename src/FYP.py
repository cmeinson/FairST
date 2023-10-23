from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.base import clone


from sklearn.neural_network import MLPClassifier




class FypModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._models = []
        self._sensitive = None
        self._method_bias = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_attributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        :param sensitive_attributes: names of sensitive attributes to be protected
        :type sensitive_attributes: List[str]
        :param method:  ml algo name to use for the main model training
        :type method: _type_
        :param method_bias: method name if needed for the bias mitigation, defaults to None
        :type method_bias: _type_, optional
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional
        """
        if method != Model.NN_C:
            raise RuntimeError("attempted to run FYP with method:", method)
        
        # train a general model
        # for each subgroup train another model

        self._method_bias = method_bias
        self._sensitive = sensitive_attributes

        model_1 = self._get_model(method, other | {"warm_start":True})
        model_1.fit(X, y)
        print("FIRST FIT FINE")

        model_1.set_params(max_epochs = other["iter_2"]) # !!!

        model_0 = clone(model_1)


        mask = (X[self._sensitive[0]] == 1)

        model_0.fit(X[~mask],y[~mask])
        model_1.fit(X[mask], y[mask])


        self._models = [model_0, model_1]
             

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        # choose the right model to predict

        y = self._models[0].predict(X)
        y_1 = self._models[1].predict(X)

        mask = (X[self._sensitive[0]] == 1)

        y[mask] = y_1[mask]
        return y
        

# add reweighing betweeen positive and negative labels

# WHAT IF I MIX REWEIGHING and my thing

    