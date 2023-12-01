import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .models import *


LATENT_DIM = 10


class LossModel:
    def __init__(self, loss_config: Dict) -> None:
        # init model, optimiser, and set it to train()
        self._loss_config = loss_config

    def set_current_state(self, original_X, decoded_X, mu, std, z, attr_col):
        self.original_X = original_X
        self.decoded_X = decoded_X
        self.mu = mu
        self.std = std
        self.z = z
        self.attr_col = attr_col

    def get_loss(self):
        # train the model and return the loss for this iteration
        raise NotImplementedError


class KLDivergenceLoss(LossModel):
    def get_loss(self):
        kl_loss = - torch.mean(0.5 * torch.sum(1. + torch.log(self.std.pow(2) + 1e-10 ) - self.mu.pow(2) - self.std.pow(2), 1))
        return kl_loss * self._loss_config["weight"]
    

class ReconstructionLoss(LossModel):
    def get_loss(self):
        #recon_loss =nn.MSELoss()(prediction,target)
        #recon_loss = F.smooth_l1_loss(prediction, target)
        #recon_loss = F.binary_cross_entropy(prediction, target, reduction="mean")
        return F.smooth_l1_loss(self.decoded_X, self.original_X) * self._loss_config["weight"]



class LatentDiscrLoss(LossModel):
    def __init__(self, loss_config) -> None:
        self._loss_config = loss_config # TODO: init through super clas?

        self.discr = Discriminator(loss_config)
        # TODO: optim from config
        self.optimizer = optim.Adam(self.discr.parameters(), lr=loss_config["lr"])#, momentum=0.3)
        self.discr.train()
    
    def get_loss(self):
        # discr loss predict -
        self.optimizer.zero_grad()
        discr_pred =  self.discr.forward(self.mu) 
    
        # calc discr LOSS
        # TODO: !!!!!!!!!!!!!!!! only predicts first attr
        discr_loss = (nn.MSELoss()(discr_pred,self.attr_col[0].detach().clone()))

        # trains discr
        discr_loss.backward(retain_graph=True)
        self.optimizer.step()

        loss = discr_loss.detach().clone()
        return torch.clamp(torch.pow(loss, -1)-2, min=0) * self._loss_config["weight"]
        


