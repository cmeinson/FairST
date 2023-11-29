from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any



class FairMaskModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._mask_models = None
        self._model = None
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
        self._method_bias = method_bias
        self._mask_models = {}
        self._sensitive = sensitive_attributes

        X_non_sens = X.copy().drop(columns=self._sensitive)

        # Build the mask_model for predicting each protected attribute
        for attr in sensitive_attributes: # ngl this code very sketchy but whatever will just copy what they did for now 
            mask_model = self._get_model(method_bias, other)

            if not self._is_regression(method_bias): # if a classifier
                mask_model.fit(X_non_sens, X[attr])
            else: # if regression
                clf = self._get_model(method, other)
                clf.fit(X_non_sens, X[attr])
                y_proba = clf.predict_proba(X_non_sens)
                # TODO: seems sus
                y_proba = [each[1] for each in y_proba]
                mask_model.fit(X_non_sens, y_proba)
            self._mask_models[attr] = mask_model 
            
        # Build the model for the actual prediction
        self._model = self._get_model(method, other)
        self._model.fit(X, y)
             

    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        X_masked = self._mask(X)
        return self._model.predict(X_masked)
        
    def _mask(self, X: pd.DataFrame):
        threshold = 0.5

        X_out = X.copy()
        X_non_sens = X.copy().drop(columns=self._sensitive)

        for attr in self._sensitive: 
            mask_model = self._mask_models[attr]
            mask = mask_model.predict(X_non_sens)
            if self._is_regression(self._method_bias): # if regression
                mask = np.where(mask >= threshold, 1, 0) # substitute to the reg2clf function
            X_out[attr] = mask

    
        return X_out


    