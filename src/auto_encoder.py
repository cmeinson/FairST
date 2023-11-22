import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


LATENT_DIM = 10

def kl_divergence_loss(mu,std):
    
    #solution
    #kl_loss = torch.mean(0.5 * torch.sum(torch.exp(std) + mu**2 - 1. - std, 1))

    kl_loss = - torch.mean(0.5 * torch.sum(1. + torch.log(std.pow(2) + 1e-10 ) - mu.pow(2) - std.pow(2), 1))
    #kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(std + 1e-8) - mu.pow(2) - std)

    #end_solution

    
    return kl_loss


def vae_loss(prediction, target, mu, std, discr_loss = None, COUNTER = 0):
    
    #solution
    #recon_loss =nn.MSELoss()(prediction,target)
    recon_loss = F.smooth_l1_loss(prediction, target)
    #print("calculating mse")
    #print(prediction)
    #print(target)
    #recon_loss = F.binary_cross_entropy(prediction, target, reduction="mean")
    #end_solution

    # disentangelment
    if not discr_loss:
        print("\t\t\t\t\t\t\t\t\t",recon_loss, kl_divergence_loss(mu,std))

        return recon_loss +  kl_divergence_loss(mu,std)*0.01
    
    discr = torch.clamp(torch.pow(discr_loss, -1)-2, min=0) 
    
    if COUNTER%50==0:
        print("\t\t\t\t\t\t\t\t\t",recon_loss ,discr, kl_divergence_loss(mu,std))
    #return recon_loss + torch.pow(discr_loss, -1)*(0.1) + kl_divergence_loss(mu,std)*0.01
    return recon_loss + discr + kl_divergence_loss(mu,std)*0.01


class VAE(Module):
    
    def __init__(self, input_dim=1,  bottleneck_size = LATENT_DIM):
        super().__init__()

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
            nn.Linear(bottleneck_size, layers[2]),
            nn.LeakyReLU(),
            nn.Linear(layers[2], layers[1]),
            nn.LeakyReLU(),
            nn.Linear(layers[1], layers[0]),
            nn.LeakyReLU(),
            nn.Linear(layers[0], input_dim),
        )

        self.fc_mu = nn.Linear(layers[2], bottleneck_size - sens_attr_nr )
        self.fc_std = nn.Linear(layers[2], bottleneck_size - sens_attr_nr)
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

        return decoded_image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._fwd = nn.Sequential(
            nn.Linear(LATENT_DIM-1, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._fwd(x)
    
def train_ae(X, input_dim, epochs = 1000):
    COUNTER = 0
    model = VAE(input_dim)
    discr = Discriminator()

    optimizer = optim.Adam(model.parameters(), lr=0.007)#, momentum=0.3)
    optimizer_discr = optim.Adam(discr.parameters(), lr=0.05)#, momentum=0.3)
    
    model.train()
    discr.train()
    for epoch in range(epochs):
        COUNTER +=1
        mu, std, attr_col = model.encoder(X) # keep og attr
        z = model.reparametrization_trick(mu, std)
        full_z = model.add_attr_col(z, attr_col)
        outputs = model.decoder(full_z)

        
        optimizer_discr.zero_grad()
        discr_pred = discr.forward(mu) 
     
        #discr_loss = F.binary_cross_entropy((discr_pred), attr_col, reduction="mean") # we are aiming to increase this loss
        discr_loss = (nn.MSELoss()(discr_pred,attr_col.detach().clone()))

        discr_loss.backward(retain_graph=True)
        optimizer_discr.step()


        optimizer.zero_grad()
        loss = vae_loss(outputs, X, mu, std, discr_loss.detach().clone(), COUNTER)
        #print(loss)
        loss.backward()
        optimizer.step()
        if COUNTER%50==0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, discr loss: {discr_loss.item():.4f}')
    return model