from ..ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .auto_encoder import *
from ...utils import TestConfig


# TODO: no longer attr col last

class FypMaskModel(Model):
    USE_LATENT_S_ADV = "latent stpace sensitive attr discr TODO"
    USE_FLIPPED_ADV = "flipped sens attr adv TODO"
    USE_KL_DISCR = "KL loss between the distributions of z for all samples with s=0 and s=1"
    USE_FAIRNESS_VEC = "Y direction discr"

    def __init__(self, config: TestConfig, base_model: Model) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)        
        self._model = base_model
        self._mask_models = {}
        self._column_names = None


    def fit(self, X: pd.DataFrame, y: np.array):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """
        self._column_names = X.columns.tolist()
        self._config.other[VAE.INPUT_DIM] = len(self._column_names)


        # Build the mask_model for predicting each protected attribute
        tensor = torch.tensor(X.values, dtype=torch.float32)
        self._mask_models = self.train_ae(tensor)
            
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

        X_masked = self._mask(torch.tensor(X.values, dtype=torch.float32)).detach().numpy()
        return self._model.predict(X_masked)
        
    def _mask(self, X):
        print("BEFORE MASK")
        print(X)
        self._mask_models.eval()
        
        X_out = self._mask_models.forward(X,1) # TODO PUT , 1  
        print("AFTER MASK")
        print(X_out)  

        return X_out
    
    def train_ae(self, X, input_dim = 0, epochs = 1000):
        COUNTER = 0
        ####################### init
        model = VAE(input_dim)
        discr = Discriminator()

        optimizer = optim.Adam(model.parameters(), lr=0.007)#, momentum=0.3)
        optimizer_discr = optim.Adam(discr.parameters(), lr=0.05)#, momentum=0.3)
        
        model.train()
        discr.train()
        for epoch in range(epochs):
            COUNTER +=1
            # predict on main model -> mu, std, z, decoded
            mu, std, attr_col = model.encoder(X) # keep og attr
            z = model.reparametrization_trick(mu, std)
            full_z = model.add_attr_col(z, attr_col)
            outputs = model.decoder(full_z)

            
            ####################### train each loss model and get loss 

            # discr loss predict -
            optimizer_discr.zero_grad()
            discr_pred = discr.forward(mu) 
        
            # calc discr LOSS
            discr_loss = (nn.MSELoss()(discr_pred,attr_col.detach().clone()))

            # trains discr
            discr_loss.backward(retain_graph=True)
            optimizer_discr.step()


            # calculate total loss based on main model preds and discr LOSS
            optimizer.zero_grad()
            loss = vae_loss(outputs, X, mu, std, discr_loss.detach().clone(), COUNTER)
            # train main model    
            loss.backward()
            optimizer.step()

            # print al used losses
            if COUNTER%50==0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, discr loss: {discr_loss.item():.4f}')
        # TODO: should prob set em all to eval
        return model
        
