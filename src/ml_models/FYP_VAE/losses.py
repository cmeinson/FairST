import numpy as np
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .models import *
from itertools import combinations
import random


def print_grad(loss):
    current_grad_fn = loss.grad_fn
    while current_grad_fn is not None:
        print(current_grad_fn)
        current_grad_fn = (
            current_grad_fn.next_functions[0][0]
            if current_grad_fn.next_functions
            else None
        )


class LossModel:
    def __init__(self, loss_config: Dict) -> None:
        # init model, optimiser, and set it to train()
        self._loss_config = loss_config
        self._first_init_done = False

    def set_current_state(
        self, original_X, decoded_X, mu, std, z, attr_cols, vae_model, y
    ):
        self.original_X = original_X
        self.decoded_X = decoded_X
        self.mu = mu
        self.std = std
        self.z = z
        self.attr_cols = attr_cols
        self.attr_tensor = torch.stack([torch.squeeze(col) for col in attr_cols], dim=1)
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
        subgroups, _ = torch.unique(self.attr_tensor, dim=0, return_inverse=True)
        indices_of_subgroups = {tuple(comb.numpy()): [] for comb in subgroups}
        for idx, row in enumerate(self.attr_tensor):
            indices_of_subgroups[tuple(row.numpy())].append(idx)
        return indices_of_subgroups.values()


class KLDivergenceLoss(LossModel):
    def get_loss(self):
        kl_loss = -torch.mean(
            0.5 * torch.sum(1.0 + torch.log(self.std.pow(2) + 1e-10) - self.mu.pow(2) - self.std.pow(2),1,)
        )
        return kl_loss * self._loss_config["weight"]


class ReconstructionLoss(LossModel):
    def get_loss(self):
        # recon_loss = nn.MSELoss()(self.decoded_X, self.original_X)
        # recon_loss = F.binary_cross_entropy(self.decoded_X, self.original_X, reduction="mean")
        recon_loss = F.smooth_l1_loss(self.decoded_X, self.original_X)
        return recon_loss * self._loss_config["weight"]


class LatentDiscrLoss(LossModel):
    # TODO: what if instead of training a new descriminator for each goup we just train a multiobjectiive one
    def _first_init(self):
        self.discrs = []
        self.optims = []

        for _ in self.attr_cols:
            discr = Discriminator(self._loss_config)
            optimizer = optim.Adam(
                discr.parameters(), lr=self._loss_config["lr"]
            )  
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
        return (total_loss / len(self.attr_cols)) * self._loss_config["weight"]

    def _loss_per_attr(self, labels, discr, optimizer):
        discr_pred = discr.forward(self.mu.detach().clone()) 
        discr_loss = nn.MSELoss()(discr_pred, labels)

        optimizer.zero_grad()
        discr_loss.backward()
        optimizer.step()

        discr_pred = discr.forward(
            self.mu
        ) 
        loss = nn.MSELoss()(discr_pred, labels)

        return torch.clamp(torch.pow(loss, -1) - 2, min=0)


class FlippedDiscrLoss(LossModel):
    def __init__(self, loss_config) -> None:
        super().__init__(loss_config)

        self.discr = Discriminator(loss_config)
        # TODO: optim from config
        self.optimizer = optim.Adam(
            self.discr.parameters(), lr=loss_config["lr"]
        )  # , momentum=0.3)
        self.discr.train()

    def get_loss(self):
        # flip attr cols
        # run model on fdlipped cols
        # predict on og attr aim label 0
        # predict on flipped cols aim label 1

        discr_loss = self._loss(self.decoded_X.detach().clone())

        self.optimizer.zero_grad()
        discr_loss.backward()
        self.optimizer.step()

        loss = self._loss(self.decoded_X)
        return torch.clamp(torch.pow(loss, -1) - 2, min=0) * self._loss_config["weight"]

    def _loss(self, decoded_X):
        og_attr_pred = self.discr.forward(decoded_X)
        flip_attr_pred = self.discr.forward(self.get_flipped_attrs_decoded())
        # calc discr LOSS
        og_attr_loss = nn.MSELoss()(og_attr_pred, torch.zeros_like(og_attr_pred))
        flip_attr_loss = nn.MSELoss()(flip_attr_pred, torch.ones_like(og_attr_pred))
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
    
        subgroup_ids = self.get_subgroup_ids()
        subgroup_mus = []
        subgroup_stds = []

        for ids in subgroup_ids:
            subgroup_mus.append(torch.mean(self.mu[ids], dim=0))
            subgroup_stds.append(torch.std(self.mu[ids], dim=0))

        pairs = list(combinations(range(len(subgroup_ids)), 2))
        total = 0
        for p, q in pairs:
            kl = self.kl_div(
                subgroup_mus[p], subgroup_stds[p], subgroup_mus[q], subgroup_stds[q]
            )
            total += kl

        return total / len(pairs)

    def kl_div(self, p_mu, p_std, q_mu, q_std):
        kls = (
            torch.log(q_std / p_std)
            + (p_std**2 + (p_mu - q_mu) ** 2) / (2 * (q_std**2))
            - 0.5
        )
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
            # need to clamp due to floating point issue in the torch library
            losses.append(torch.clamp(1 - similarity, min=0))

        tensor_loss = torch.stack(losses)
        return torch.mean(tensor_loss, dim=0)

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
