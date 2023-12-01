import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from .configs import VAEMaskConfig



class VAE(Module):
    def __init__(self, config: VAEMaskConfig, input_dim: int, sens_column_ids: List[int]):
        super().__init__()
        self._sens_column_ids = sens_column_ids
        sens_attr_nr = len(sens_column_ids)

        layers = [input_dim] + list(config.vae_layers)
        
        # Encoder input dim -> last layer size
        enc_layers = []
        for i in range(len(layers)-1):
            enc_layers.append(nn.Linear(layers[i], layers[i+1]))
            enc_layers.append(nn.LeakyReLU())
        self._encoder = nn.Sequential(*enc_layers)

        # Decoder latent_dim -> input dim
        dec_layers = [nn.Linear(config.latent_dim, layers[-1]), nn.LeakyReLU()]
        for i in range(len(layers)-1, 0, -1):
            dec_layers.append(nn.Linear(layers[i], layers[i-1]))
            dec_layers.append(nn.LeakyReLU())
        dec_layers.append(nn.Linear(layers[0], input_dim))
        self._decoder = nn.Sequential(*dec_layers)
        

        self.fc_mu = nn.Linear(layers[2], config.latent_dim - sens_attr_nr )
        self.fc_std = nn.Linear(layers[2], config.latent_dim - sens_attr_nr)
        #end_solution

    def add_attr_cols(self, x, attr_cols: List[Tensor]):
        return torch.cat([x] + attr_cols, dim=1)
        x_np = x.detach().numpy()
        out = np.column_stack((x_np, attr_cols))  
        #print("latent space", torch.tensor(out, dtype=torch.float32))
        return torch.tensor(out, dtype=torch.float32)

    
    def encoder(self, image, attrs = None):
        # keeps the og attr if attr not specifies
        #print("og en", image)
        # TODO: will probably have to pass the indxs of sensitive columns
        attr_cols = [image[:, i] for i in self._sens_column_ids]
        if attrs:
            for i in range(len(attr_cols)):
                attr_cols[i] = torch.full_like(attr_cols[i], fill_value=attrs[i]) 


        x = self._encoder(image)
        mu = self.fc_mu(x).nan_to_num()
        std = self.fc_std(x).nan_to_num()
        #print("encoded image sixe", x.size())

        return mu, std, attr_cols
    
    def decoder(self,code):
        decoded_image = self._decoder(code)
        return decoded_image
    
    def reparametrization_trick(self,mu,std):
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, image, attr = None):
        mu, std, attr_cols = self.encoder(image, attr)
        z = self.reparametrization_trick(mu, std)
        full_z = self.add_attr_cols(z, attr_cols)
        decoded_image = self.decoder(full_z)

        return decoded_image, mu, std, z, attr_cols



class Discriminator(nn.Module):
    def __init__(self, config: Dict):
        super(Discriminator, self).__init__()
        layers = [config['latent_dim']-1] + list(config['layers'])

        # latent dim-1  ->  1
        sq_layers = []
        for i in range(len(layers)-1):
            sq_layers.append(nn.Linear(layers[i], layers[i+1]))
            sq_layers.append(nn.LeakyReLU())
        sq_layers.append(nn.Linear(layers[-1], 1))
        sq_layers.append(nn.Sigmoid())

        self._fwd = nn.Sequential(*sq_layers)

    def forward(self, x):
        return self._fwd(x)