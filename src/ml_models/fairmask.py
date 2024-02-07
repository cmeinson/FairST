from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..utils import TestConfig


class FairMaskModel(Model):
    def __init__(self, config: TestConfig, base_model: Model) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)        
        self._model = base_model
        self._mask_models = {}

    def fit(self, X: pd.DataFrame, y: np.array):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """

        X_non_sens = X.copy().drop(columns=self._config.sensitive_attr)

        # Build the mask_model for predicting each protected attribute
        for attr in self._config.sensitive_attr: # ngl this code very sketchy but whatever will just copy what they did for now 
            mask_model = self._get_model(self._config.bias_ml_method)

            if not self._is_regression(self._config.bias_ml_method): # if a classifier
                mask_model.fit(X_non_sens, X[attr])
            else: # if regression
                clf = self._get_model()
                clf.fit(X_non_sens, X[attr])
                y_proba = clf.predict_proba(X_non_sens)
                y_proba = [each[1] for each in y_proba]
                mask_model.fit(X_non_sens, y_proba)
            self._mask_models[attr] = mask_model 
            
        # Build the model for the actual prediction
        #self._model = self._get_model(method, other)
        #self._model.fit(X, y)
             

    def predict(self, X: pd.DataFrame, binary = True) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
       
        :return: predictions for each row of X
        :rtype: np.array
        """
        X_masked = self._mask(X)
        preds = self._model.predict(X_masked, binary=False)
        if not binary:
            return preds
        return self._binarise(preds)
        
    def _mask(self, X: pd.DataFrame):
        threshold = 0.5

        X_out = X.copy()
        X_non_sens = X.copy().drop(columns=self._config.sensitive_attr)

        for attr in self._config.sensitive_attr: 
            mask_model = self._mask_models[attr]
            mask = mask_model.predict(X_non_sens)
            if self._is_regression(self._config.bias_ml_method): # if regression
                mask = np.where(mask >= threshold, 1, 0) # substitute to the reg2clf function
            X_out[attr] = mask

        return X_out


    