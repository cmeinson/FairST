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


class FypMaskModel(Model):
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

        columns = X.shape[1]
        my_other = other | {"input_dim":columns}

        #print(X)
        
        self._method_bias = method_bias
        self._sensitive = sensitive_attributes

        self.position_to_insert = X.shape[1] -1  # Number of existing columns -1

        #print(self.position_to_insert, self._sensitive[0])
        X.insert(self.position_to_insert, self._sensitive[0], X.pop(self._sensitive[0]))

        print("DATA INPUT TRAINI*NG")
        print(X)
        
        # Build the mask_model for predicting each protected attribute
        self.build_mask_model( torch.tensor(X.values, dtype=torch.float32), my_other)
            
        # Build the model for the actual prediction
        self._model = self._get_model(method, my_other)
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
        
        X.insert(self.position_to_insert, self._sensitive[0], X.pop(self._sensitive[0]))

        X_masked = self._mask( torch.tensor(X.values, dtype=torch.float32)).detach().numpy()
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


    # TODO:
    # normalization
    # 

"""
as it is a post processing approach i only rly need to compare to likes of fairmask and not reweighing

"""