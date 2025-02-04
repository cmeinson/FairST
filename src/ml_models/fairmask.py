from .ml_interface import Model
import pandas as pd
import numpy as np
from ..test_config import TestConfig


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
        for attr in self._config.sensitive_attr:
            mask_model = self._get_model(self._config.bias_ml_method)

            self._fit(mask_model, X_non_sens, X[attr])
     
            self._mask_models[attr] = mask_model 
            

    def predict(self, X: pd.DataFrame, binary = True) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
       
        :return: predictions for each row of X
        :rtype: np.array
        """
        X_masked = self._mask(X)
        return self._predict(self._model, X_masked, binary=binary)
        
    def _mask(self, X: pd.DataFrame):
        threshold = 0.5

        X_out = X.copy()
        X_non_sens = X.copy().drop(columns=self._config.sensitive_attr)

        for attr in self._config.sensitive_attr: 
            mask_model = self._mask_models[attr]
            mask = mask_model.predict(X_non_sens)
            if self._is_regression(self._config.bias_ml_method): # if regression
                mask = np.where(mask >= threshold, 1, 0) 
            X_out[attr] = mask

        return X_out


    