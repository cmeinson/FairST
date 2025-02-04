from typing import List, Dict
import torch
from torch.nn import Module
import torch.nn as nn
from torch import Tensor

from .configs import VAEMaskConfig


class VAE(Module):
    def __init__(self, config: VAEMaskConfig):
        super().__init__()
        self._sens_column_ids = config.sens_column_ids
        input_dim = config.input_dim
        sens_attr_nr = len(config.sens_column_ids)

        layers = [input_dim] + list(config.vae_layers)

        # Encoder input dim -> last layer size
        enc_layers = []
        for i in range(len(layers) - 1):
            enc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            enc_layers.append(nn.LeakyReLU())
        self._encoder = nn.Sequential(*enc_layers)

        # Decoder latent_dim -> input dim
        dec_layers = [nn.Linear(config.latent_dim, layers[-1]), nn.LeakyReLU()]
        for i in range(len(layers) - 1, 0, -1):
            dec_layers.append(nn.Linear(layers[i], layers[i - 1]))
            dec_layers.append(nn.LeakyReLU())
        dec_layers.append(nn.Linear(layers[0], input_dim))
        self._decoder = nn.Sequential(*dec_layers)

        self.fc_mu = nn.Linear(layers[-1], config.latent_dim - sens_attr_nr)
        self.fc_std = nn.Linear(layers[-1], config.latent_dim - sens_attr_nr)

    def add_attr_cols(self, X, attr_cols: List[Tensor]):
        return torch.cat([X] + attr_cols, dim=1)

    def encoder(self, X, attrs=None):
        # keeps the og attr if attr not specifies
        attr_cols = [X[:, i, None] for i in self._sens_column_ids]
        if attrs:
            for i in range(len(attr_cols)):
                attr_cols[i] = torch.full_like(attr_cols[i], fill_value=attrs[i])

        x = self._encoder(X)
        mu = self.fc_mu(x).nan_to_num()
        std = self.fc_std(x).nan_to_num()

        return mu, std, attr_cols

    def decoder(self, code):
        decoded = self._decoder(code)
        return decoded

    def reparametrization_trick(self, mu, std):
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, X, attrs=None, use_std=True):
        mu, std, attr_cols = self.encoder(X, attrs)
        z = mu
        if use_std:
            z = self.reparametrization_trick(mu, std)
        full_z = self.add_attr_cols(z, attr_cols)
        decoded_X = self.decoder(full_z)

        return decoded_X, mu, std, z, attr_cols


class Discriminator(nn.Module):
    def __init__(self, config: Dict):
        super(Discriminator, self).__init__()
        layers = [config["input_dim"]] + list(config["layers"])
        print("DISCR LAYERS", layers)

        # (latent dim)-(no of sens cols)  ->  1
        sq_layers = []
        for i in range(len(layers) - 1):
            sq_layers.append(nn.Linear(layers[i], layers[i + 1]))
            sq_layers.append(nn.LeakyReLU())
        sq_layers.append(nn.Linear(layers[-1], 1))
        sq_layers.append(nn.Sigmoid())

        self._fwd = nn.Sequential(*sq_layers)

    def forward(self, x):
        return self._fwd(x)
