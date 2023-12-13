import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .models import *
from itertools import combinations



class LossModel:
    def __init__(self, loss_config: Dict) -> None:
        # init model, optimiser, and set it to train()
        self._loss_config = loss_config

    def set_current_state(self, original_X, decoded_X, mu, std, z, attr_col, model, y):
        self.original_X = original_X
        self.decoded_X = decoded_X
        self.mu = mu
        self.std = std
        self.z = z
        self.attr_col = attr_col
        self.model = model
        self.y = y

    def get_loss(self):
        # train the model and return the loss for this iteration
        raise NotImplementedError
    
    def get_subgroup_ids(self):
        subgroups = set(tuple(row.tolist()) for row in self.attr_col)
        indices_of_subgroups = {comb: [] for comb in subgroups}
        for idx, row in enumerate(self.attr_col):
            indices_of_subgroups[tuple(row.tolist())].append(idx)
        return indices_of_subgroups



class KLDivergenceLoss(LossModel):
    def get_loss(self):
        kl_loss = - torch.mean(0.5 * torch.sum(1. + torch.log(self.std.pow(2) + 1e-10 ) - self.mu.pow(2) - self.std.pow(2), 1))
        return kl_loss * self._loss_config["weight"]
    

class ReconstructionLoss(LossModel):
    def get_loss(self):
        #recon_loss = nn.MSELoss()(self.decoded_X, self.original_X)
        #recon_loss = F.binary_cross_entropy(self.decoded_X, self.original_X, reduction="mean")
        recon_loss = F.smooth_l1_loss(self.decoded_X, self.original_X)
        return recon_loss * self._loss_config["weight"]



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
        labels = self.attr_col[0].detach().clone()
        discr_loss = (nn.MSELoss()(discr_pred, labels))

        # trains discr
        discr_loss.backward(retain_graph=True)
        self.optimizer.step()

        loss = discr_loss.detach().clone()
        return torch.clamp(torch.pow(loss, -1)-2, min=0) * self._loss_config["weight"]
    

class FlippedDiscrLoss(LossModel):
    def __init__(self, loss_config) -> None:
        self._loss_config = loss_config # TODO: init through super clas?

        self.discr = Discriminator(loss_config)
        # TODO: optim from config
        self.optimizer = optim.Adam(self.discr.parameters(), lr=loss_config["lr"])#, momentum=0.3)
        self.discr.train()
    
    def get_loss(self):
        # discr loss predict -
        self.optimizer.zero_grad()
        # flip attr cols
        # run model on fdlipped cols
        # predict on og attr aim label 0
        # predict on flipped cols aim label 1

        og_attr_pred = self.discr.forward(self.decoded_X) 
        flip_attr_pred = self.discr.forward(self.get_flipped_attrs_decoded) 
        # calc discr LOSS
        og_attr_loss = (nn.MSELoss()(og_attr_pred, torch.zeros_like(og_attr_pred)))
        flip_attr_loss = (nn.MSELoss()(flip_attr_pred, torch.ones_like(og_attr_pred)))
        discr_loss = og_attr_loss + flip_attr_loss
        # trains discr
        discr_loss.backward(retain_graph=True)
        self.optimizer.step()

        loss = discr_loss.detach().clone()
        return torch.clamp(torch.pow(loss, -1)-2, min=0) * self._loss_config["weight"]
        
    def get_flipped_attrs_decoded(self):
        X = self.original_X.clone().detach()

        # TODO: !!!!!!!!!!!!!!!! only predicts first attr
        i = self._loss_config["sens_col_ids"][0]
        X[:, i] = 1 - X[:, i]
        outputs, _, _, _, _ = self.model.forward(X)
        return outputs



class SensitiveKLLoss(LossModel):
    def get_loss(self):
        kls = []
        # TODO: for future, maybe weigh the variancc
        # TODO: with multiple attrs will need to make it intersectional???/ tbh not an expensive computation so can just do it pairwise
        # for each dimention and each subgroup get mean and variance
        subgroup_ids = self.get_subgroup_ids()
        subgroup_mus = []
        subgroup_stds = []

        for ids in subgroup_ids:
            subgroup_mus.append(torch.mean(self.z[ids], dim=0))
            subgroup_stds.append(torch.std(self.z[ids], dim=0))
        
        pairs = list(combinations(range(len(subgroup_ids)), 2))
        for p, q in pairs:
            kl = self.kl_div(subgroup_mus[p], subgroup_stds[p], subgroup_mus[q], subgroup_stds[q])
            kls.append(kl)

        return np.mean(kls)

    def kl_div(self, p_mu, p_std, q_mu, q_std):
        return np.log(q_std/p_std) + (p_std**2 + (p_mu - q_mu)**2) / (2*(q_std**2)) - 0.5
    

class PositiveVectorLoss(LossModel):
    def get_loss(self):
        # aim is for the vectors to be as similar as possible
        # loss = cosine distance
        # 0 - identical directions, 
        # 1 - orthogonal directions, 
        # 2 - opposite directions.
        vectors = self.get_vectors()
        losses = []

        pairs = list(combinations(vectors), 2)
        for v, u in pairs:
            losses.append(1 - F.cosine_similarity(v, u, dim=0).item())

        return torch.mean(losses)   


    def get_vectors(self):
        subgroup_ids = self.get_subgroup_ids()
        pos_label_ids = set(np.nonzero(self.y)) 

        vectors = []
        for ids in subgroup_ids:
            subgroup_pos_ids = set(ids) | pos_label_ids
            subgroup_neg_ids = set(ids) - pos_label_ids

            pos_mean = torch.mean(self.z[subgroup_pos_ids], dim=0)
            neg_mean = torch.mean(self.z[subgroup_neg_ids], dim=0)

            vec = pos_mean - neg_mean
            vectors.append(vec / torch.norm(vec))
        return vectors

        



        
