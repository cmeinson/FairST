from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .auto_encoder import *
from .utils import TestConfig


class FypMaskModel(Model):
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

        columns = X.shape[1]
        my_other = self._config.other | {"input_dim":columns}

        #print(X)

        self.position_to_insert = X.shape[1] -1  # Number of existing columns -1
        sensitive = self._config.sensitive_attr

        #print(self.position_to_insert, self._sensitive[0])
        X.insert(self.position_to_insert, sensitive[0], X.pop(sensitive[0]))
        
        # Build the mask_model for predicting each protected attribute
        self.build_mask_model( torch.tensor(X.values, dtype=torch.float32), my_other)
            
        # Build the model for the actual prediction
        #self._model = self._get_model(method, my_other)
        #self._model.fit(X, y)
             

    def predict(self, X: pd.DataFrame) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        sensitive = self._config.sensitive_attr
        X.insert(self.position_to_insert, sensitive[0], X.pop(sensitive[0]))

        X_masked = self._mask(torch.tensor(X.values, dtype=torch.float32)).detach().numpy()
        return self._model.predict(X_masked)
        
    def _mask(self, X):
        print("MASKING")
        print("BEFORE MASK")
        print(X)
        self._mask_models.eval()
        
        X_out = self._mask_models.forward(X,1) # TODO PUT , 1  
        print("AFTER MASK")
        print(X_out)  

        return X_out
    
    def build_mask_model(self, X, other):
        self._mask_models = train_ae(X, self.position_to_insert+1, other["mask_epoch"])


"""
as it is a post processing approach i only rly need to compare to likes of fairmask and not reweighing

"""