import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .configs import VAEMaskConfig



class VAE(Module):
    def __init__(self, config: VAEMaskConfig, input_dim):
        super().__init__()

        # TODO: take the layers from the config
        layers = [90, 60, 30]
        sens_attr_nr = 1
        
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.LeakyReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.LeakyReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.LeakyReLU()
        )
        self._decoder = nn.Sequential(
            nn.Linear(config.latent_dim, layers[2]),
            nn.LeakyReLU(),
            nn.Linear(layers[2], layers[1]),
            nn.LeakyReLU(),
            nn.Linear(layers[1], layers[0]),
            nn.LeakyReLU(),
            nn.Linear(layers[0], input_dim),
        )

        self.fc_mu = nn.Linear(layers[2], config.latent_dim - sens_attr_nr )
        self.fc_std = nn.Linear(layers[2], config.latent_dim - sens_attr_nr)
        #end_solution

    def add_attr_col(self, x, attr_col):
        return torch.cat((x, attr_col), dim=1)
        x_np = x.detach().numpy()
        out = np.column_stack((x_np, attr_col))  
        #print("latent space", torch.tensor(out, dtype=torch.float32))
        return torch.tensor(out, dtype=torch.float32)

    
    def encoder(self, image, attr = None):
        # keeps the og attr if attr not specifies
        #print("og en", image)
        attr_col = image[:, -1:]
        if attr:
            attr_col = torch.full_like(attr_col, fill_value=attr) 

        #solution
        x = self._encoder(image)
        mu = self.fc_mu(x).nan_to_num()
        std = self.fc_std(x).nan_to_num()
        #print("encoded image sixe", x.size())

        return mu, std, attr_col
    
    def decoder(self,code):
        #solution
        decoded_image = self._decoder(code)
        #end_solution
        #print("decoded", decoded_image)
        return decoded_image
    
    def reparametrization_trick(self,mu,std):
        
        #solution
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #end_solution
        return z
    
    def forward(self, image, attr = None):
        
        #solution
        mu, std, attr_col = self.encoder(image, attr)
        z = self.reparametrization_trick(mu, std)
        full_z = self.add_attr_col(z, attr_col)
        decoded_image = self.decoder(full_z)
        #end_solution

        return decoded_image, mu, std, z, attr_col



class Discriminator(nn.Module):
    def __init__(self, config: Dict):
        # TODO: take the layers from the config
        super(Discriminator, self).__init__()
        self._fwd = nn.Sequential(
            nn.Linear(config['latent_dim']-1, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._fwd(x)