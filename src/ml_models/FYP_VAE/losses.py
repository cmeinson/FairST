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
from itertools import product
import random


def print_grad(loss):
    current_grad_fn = loss.grad_fn
    while current_grad_fn is not None:
        print(current_grad_fn)
        current_grad_fn = current_grad_fn.next_functions[0][0] if current_grad_fn.next_functions else None
    


class LossModel:
    def __init__(self, loss_config: Dict) -> None:
        # init model, optimiser, and set it to train()
        self._loss_config = loss_config
        self._first_init_done = False

    def set_current_state(self, original_X, decoded_X, mu, std, z, attr_cols, vae_model, y):
        self.original_X = original_X
        self.decoded_X = decoded_X
        self.mu = mu
        self.std = std
        self.z = z
        self.attr_cols = attr_cols
        self.attr_tensor = torch.stack([torch.squeeze(col) for col in attr_cols], dim =1)
        self.vae_model = vae_model
        self.y = y
        
        if not self._first_init_done:
           self._first_init_done = True 
           self._first_init()
           
    def _first_init(self):
        # for child classes to run optinally after the first state is given.
        pass

    def get_loss(self):
        # train the model and return the loss for this iteration
        raise NotImplementedError
    
    def get_subgroup_ids(self):
        subgroups, _ = torch.unique(self.attr_tensor , dim=0, return_inverse=True)     
        indices_of_subgroups = {tuple(comb.numpy()): [] for comb in subgroups}
        for idx, row in enumerate(self.attr_tensor):
            indices_of_subgroups[tuple(row.numpy())].append(idx)
        return indices_of_subgroups.values()



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
    # https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826
    # could be that it is tensors left from previous it???
    """
    How to make it multi attribute?
    - a discriminator for each attr.
    
    not going to use subgroup as the goal is for none of them to be predictible.
    NOTE: in the future could try evaluating it based on subgroup somehow similar to DF
    """      
    def _first_init(self):
        self.discrs = []
        self.optims = []
        
        for _ in self.attr_cols:
            discr = Discriminator(self._loss_config) 
            optimizer = optim.Adam(discr.parameters(), lr=self._loss_config["lr"])#, momentum=0.3)
            discr.train()
            self.discrs.append(discr)
            self.optims.append(optimizer)
    
    def get_loss(self):
        total_loss = 0
        for i in range(len(self.attr_cols)):
            labels = self.attr_cols[i].detach().clone()
            total_loss += self._loss_per_attr(labels, self.discrs[i], self.optims[i])
        
        # disciminator loss:
        # 0   - dirctiminator can guess perfectly guess
        # 0.5 - discriminator no better than a random guess - perfect
        # loss:
        # 1/(0)-2 = infinity
        # 1/(0.1)-2 = 8
        # 1/(0.25)-2 = 2
        # 1/(0.5)-2 = 0 
        return (total_loss/len(self.attr_cols)) * self._loss_config["weight"]

    
    def _loss_per_attr(self, labels, discr, optimizer):
        # discr loss predict -
        discr_pred = discr.forward(self.mu.detach().clone()) # !!!!!!!!!?????
        discr_loss = (nn.MSELoss()(discr_pred, labels)) 

        # trains discr
        optimizer.zero_grad()
        discr_loss.backward()
        optimizer.step()

        # NOTE: trying to avoid copy()
        discr_pred = discr.forward(self.mu) # !!!!!!!!!????? TODO: why does it not work with z
        loss = (nn.MSELoss()(discr_pred, labels))
        #loss = discr_loss.detach().clone() # this breaks it
        
        return torch.clamp(torch.pow(loss, -1)-2, min=0)
        
    

class FlippedDiscrLoss(LossModel):
    """
    NOTE:
    How to make it multi attribte?
    (could have a single discriminator. and do all possible flips. but with more attributes the discriminator could just aways preduct fliped....)
    
    ATM:   For half of the points keep the original. for other half flip some attributes randomly. 
    
    Could have a discriminator for each att being flip so could have a 50/50 split but that would take so much longer to train and not ure would be worth
    """
    
    def __init__(self, loss_config) -> None:
        super().__init__(loss_config)

        self.discr = Discriminator(loss_config)
        # TODO: optim from config
        self.optimizer = optim.Adam(self.discr.parameters(), lr=loss_config["lr"])#, momentum=0.3)
        self.discr.train()
    
    def get_loss(self):
        # discr loss predict -
        
        # flip attr cols
        # run model on fdlipped cols
        # predict on og attr aim label 0
        # predict on flipped cols aim label 1

        discr_loss = self._loss(self.decoded_X.detach().clone()) 

        # trains discr

        self.optimizer.zero_grad()
        discr_loss.backward()
        self.optimizer.step()

        loss = self._loss(self.decoded_X)
        return torch.clamp(torch.pow(loss, -1)-2, min=0) * self._loss_config["weight"] 
    
    def _loss(self, decoded_X):
        og_attr_pred = self.discr.forward(decoded_X) 
        flip_attr_pred = self.discr.forward(self.get_flipped_attrs_decoded()) 
        # calc discr LOSS
        og_attr_loss = (nn.MSELoss()(og_attr_pred, torch.zeros_like(og_attr_pred)))
        flip_attr_loss = (nn.MSELoss()(flip_attr_pred, torch.ones_like(og_attr_pred)))
        #print("disrc loss", og_attr_loss,flip_attr_loss, (og_attr_loss + flip_attr_loss) / 2 )
        return (flip_attr_loss + og_attr_loss) / 2

        
    def get_flipped_attrs_decoded(self):
        # flip at least 1 attr
        nr_of_flips = random.randint(1, len(self.attr_cols))
        ids_to_flip = random.sample(self._loss_config["sens_col_ids"], nr_of_flips)
        
        X = self.original_X.clone().detach()

        for i in ids_to_flip:
            X[:, i] = 1 - X[:, i]
            
        outputs, _, _, _, _ = self.vae_model.forward(X)
        return outputs.detach().clone()



class SensitiveKLLoss(LossModel):
    def get_loss(self):
        # TODO: for future, maybe weigh the variancc
        # NOTE: with multiple attrs will need to make it intersectional???/ tbh not an expensive computation so can just do it pairwise
        # for each dimention and each subgroup get mean and variance
        subgroup_ids = self.get_subgroup_ids()
        subgroup_mus = []
        subgroup_stds = []

        # TODO: why this not work from z!?!?!?!?!?!?!
        for ids in subgroup_ids:
            subgroup_mus.append(torch.mean(self.mu[ids], dim=0))
            subgroup_stds.append(torch.std(self.mu[ids], dim=0))
        
        pairs = list(combinations(range(len(subgroup_ids)), 2))
        total = 0
        for p, q in pairs:
            kl = self.kl_div(subgroup_mus[p], subgroup_stds[p], subgroup_mus[q], subgroup_stds[q])
            total += kl

        return total / len(pairs)

    def kl_div(self, p_mu, p_std, q_mu, q_std):
        kls = torch.log(q_std/p_std) + (p_std**2 + (p_mu - q_mu)**2) / (2*(q_std**2)) - 0.5
        return torch.mean(kls)
    

class PositiveVectorLoss(LossModel):
    def get_loss(self):
        # aim is for the vectors to be as similar as possible
        # loss = cosine distance
        # 0 - identical directions, 
        # 1 - orthogonal directions, 
        # 2 - opposite directions.
        vectors = self.get_vectors()
        losses = []

        pairs = list(combinations(vectors, 2))
        for v, u in pairs:
            similarity = F.cosine_similarity(v, u, dim=0)
            # need to clmp due to floating point issue in the torch library
            losses.append(torch.clamp(1 - similarity, min=0))

        tensor_loss = torch.stack(losses)
        return  torch.mean(tensor_loss, dim=0) 


    def get_vectors(self):
        subgroup_ids = self.get_subgroup_ids()
        pos_label_ids = set(np.nonzero(self.y)[0]) 

        vectors = []
        for ids in subgroup_ids:
            subgroup_pos_ids = set(ids) | pos_label_ids
            subgroup_neg_ids = set(ids) - pos_label_ids

            pos_mean = torch.mean(self.mu[list(subgroup_pos_ids)], dim=0)
            neg_mean = torch.mean(self.mu[list(subgroup_neg_ids)], dim=0)

            vec = pos_mean - neg_mean
            vectors.append(vec / torch.norm(vec))
        return vectors

        



        
