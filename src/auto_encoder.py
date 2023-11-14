import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


LATENT_DIM = 10

def ae_loss(prediction, target, discr_loss):
    
    #solution
    recon_loss = F.mse_loss(prediction, target)
    #recon_loss = F.binary_cross_entropy(prediction, target, reduction="mean")
    #end_solution

    # disentangelment

    return recon_loss + torch.pow(discr_loss, -1)


class VAE(Module):
    
    def __init__(self, input_dim=1,  bottleneck_size = LATENT_DIM):
        super().__init__()
        
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, bottleneck_size-1),
            nn.ReLU()
        )
        self._decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim),
            nn.Tanh()
        )
        #end_solution

    
    def encoder(self,image, attr = None):
        # keeps the og attr if attr not specifies
        attr_col = image[:, -1]
        if attr:
            attr_col = np.array([attr]*len(attr_col))

        #solution
        x = self._encoder(image)
        #print("encoded image sixe", x.size())
        x_np = x.detach().numpy()
        code = np.column_stack((x_np, attr_col))         
        return code, x, attr_col
    
    def decoder(self,code):
        #solution
        decoded_image = self._decoder(code)
        #end_solution
        
        return decoded_image
    
    def forward(self,image, attr = None):
        code, _, __ = self.encoder(image, attr)
        return self.decoder(torch.tensor(code, dtype=torch.float32))
        #end_solution


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._fwd = nn.Sequential(
            nn.Linear(LATENT_DIM-1, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._fwd(x)
    
def train_ae(X, input_dim, epochs = 1000):
    optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.3)
    optimizer_discr = optim.SGD(model.parameters(), lr=0.007, momentum=0.3)
    model = VAE(input_dim)
    discr = Discriminator()
    model.train()
    discr.train()
    for epoch in range(epochs):
        code, x_no_attr, attr = model.encoder(X)
        outputs = model.decoder(code)

        optimizer_discr.zero_grad()
        discr_pred = discr.forward(x_no_attr) 
        discr_loss = F.binary_cross_entropy(discr_pred, attr, reduction="mean") # we are aiming to increase this loss
        discr_loss.backward()
        optimizer_discr.step()

        optimizer.zero_grad()
        loss = ae_loss(outputs, X, discr_loss)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, discr loss: {discr_loss.item():.4f}')
    return model