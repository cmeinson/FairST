from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import torch.nn.functional as F

def reconstruction_loss(prediction, target):
    
    #solution
    recon_loss = F.mse_loss(prediction, target)
    #recon_loss = F.binary_cross_entropy(prediction, target, reduction="mean")
    #end_solution

    # disentangelment

    
    return recon_loss

def kl_divergence_loss(mu,std):
    
    #solution
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(std) + mu**2 - 1. - std, 1))
    #end_solution

    
    return kl_loss

class AutoEncoder(Module):
    
    def __init__(self, bottleneck_size = 10, input_dim=1):
        super().__init__()
        
        #solution
        encoding_dim = bottleneck_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim-1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        #end_solution

    
    def encoder(self,image):
        
        #solution
        code = self._encoder(image)    
        #end_solution

        
        return code
    
    def decoder(self,code):
        
        #solution
        decoded_image = self._decoder(code)
        #end_solution

        
        return decoded_image
    
    def forward(self,image, attr = None):
        # keeps the og attr if attr not specifies

        attr_col = image[:, -1]
        if attr:
            attr_col = np.array([attr]*len(attr_col))

        #solution
        x = self.encoder(image)
        #print("encodewd image sixe", x.size())
        x_np = x.numpy()
        X_with_last_column = np.column_stack((x_np, attr_col)) 
        decoded_image = self.decoder(torch.tensor(X_with_last_column))
        #end_solution

        
        return decoded_image




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

        self.transform=transforms.Compose([
            transforms.ToTensor()
        ])
        

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
        
        self._sensitive = sensitive_attributes

        X.insert(-1, self._sensitive[0], X.pop(self._sensitive[0]))

        # Build the mask_model for predicting each protected attribute
        self.build_mask_model(self.trnsform(X))
            
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
        X.insert(-1, self._sensitive[0], X.pop(self._sensitive[0]))

        X_masked = self._mask(self.trnsform(X)).to_numpy()
        return self._model.predict(X_masked)
        
    def _mask(self, X):
        self._mask_models.eval()
        
        X_out = self._mask_models.forward(X, 1)    
        
        return X_out
    
    def build_mask_model(self, X):
        model = AutoEncoder(len(X.columns), 5)
        criterion = reconstruction_loss
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        epochs = 15
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, X)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        self._mask_models = model


    