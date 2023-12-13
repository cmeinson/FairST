from ..ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .models import *
from .losses import *
from ...utils import TestConfig
from .configs import VAEMaskConfig



# TODO: no longer attr col last
PRINT_FREQ = 50

class VAEMaskModel(Model):
    VAE_MASK_CONFIG = "my model config"

    COUNTER = 0

    def __init__(self, config: TestConfig, base_model: Model) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)        
        self._model = base_model
        self._mask_models = {}
        self._column_names = None
        self._vm_config = config.other[self.VAE_MASK_CONFIG]


    def fit(self, X: pd.DataFrame, y: np.array):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """
        self._column_names = X.columns.tolist()
        input_dim = len(self._column_names)
        sens_column_ids = [self._column_names.index(col) for col in self._config.sensitive_attr]

        self._vm_config.set_input_dim_and_sens_column_ids(input_dim, sens_column_ids)

        # Build the mask_model for predicting each protected attribute
        tensor = torch.tensor(X.values, dtype=torch.float32)
        self._mask_models = self.train_vae(tensor, y, self._vm_config.epochs)


    def predict(self, X: pd.DataFrame) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        X_masked = self._mask(torch.tensor(X.values, dtype=torch.float32)).detach().numpy()
        return self._model.predict(X_masked)
        
    def _mask(self, X):
        print("BEFORE MASK")
        print(X)
        self._mask_models.eval()
        
        X_out, *_ = self._mask_models.forward(X,[1]) # TODO PUT , 1  
        print("AFTER MASK")
        print(X_out)  

        return X_out
    
    def train_vae(self, X: Tensor, y: np.array, epochs = 500):
        self.COUNTER = 0
        model = VAE(self._vm_config)
        optimizer = optim.Adam(model.parameters(), lr=self._vm_config.lr)#, momentum=0.3)
        model.train()

        loss_models = [self._get_loss_model(name) for name in self._vm_config.lossed_used]
        
        for epoch in range(epochs):
            self.COUNTER +=1
            # predict on main model -> mu, std, z, decoded
            outputs, mu, std, z, attr_cols = model.forward(X)

            # train each loss model and get loss 
            loss = self._get_total_loss(loss_models, X, outputs, mu, std, z, attr_cols, model, y)

            optimizer.zero_grad()
            # train main model    
            loss.backward() 
            optimizer.step()

            # print al used losses
            if self.COUNTER%PRINT_FREQ==0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        model.eval()
        return model
    
    def _get_total_loss(self, loss_models, original_X, decoded_X, mu, std, z, attr_cols, model, y):
        losses = []
        for model in loss_models:
            model.set_current_state(original_X, decoded_X, mu, std, z, attr_cols, model, y)
            losses.append(model.get_loss())
            if self.COUNTER%PRINT_FREQ==0:
                print('------------------',model,'loss:  ', losses[-1])

        return sum(losses)

    def _get_loss_model(self, name) -> LossModel:
        classes = {
            VAEMaskConfig.KL_DIV_LOSS: KLDivergenceLoss,
            VAEMaskConfig.RECON_LOSS: ReconstructionLoss,
            VAEMaskConfig.LATENT_S_ADV_LOSS: LatentDiscrLoss,
            VAEMaskConfig.FLIPPED_ADV_LOSS: FlippedDiscrLoss,
            VAEMaskConfig.KL_SENSITIVE_LOSS: SensitiveKLLoss
        }
        
        loss_config = self._vm_config.loss_configs[name]
        return classes[name](loss_config)



        
