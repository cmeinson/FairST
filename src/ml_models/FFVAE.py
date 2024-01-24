from .ml_interface import Model
from typing import List, Dict, Any
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import pandas as pd
import numpy as np
from ..utils import TestConfig


# NOTE: not functional
class FFVAE(Model):

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
        self._config.other["my model config"].set_input_dim_and_sens_column_ids(input_dim, sens_column_ids)

        
        epochs = 100# self._config.other["my model config"].epochs
        self._model = Ffvae(self._config)
        #print(X)
        #print(self._config.sensitive_attr)
        
        non_sen = X.copy()
        #print(non_sen.columns)
        non_sen.drop(self._config.sensitive_attr, axis=1)
        inputs = torch.tensor(non_sen.values, dtype=torch.float32)
        attrs = torch.tensor(X[self._config.sensitive_attr].values, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        
        non_sen2 = X.copy()
        #print(non_sen.columns)
        non_sen2.drop(self._config.sensitive_attr, axis=1)
        inputs2 = torch.tensor(non_sen2.values, dtype=torch.float32)
        attrs2 = torch.tensor(X[self._config.sensitive_attr].values, dtype=torch.float32)
        labels2 = torch.tensor(y, dtype=torch.long)
        
        for e in range(epochs):
            self._model.forward(inputs, labels, attrs)
            #self._model.forward(inputs2, labels2, attrs2, mode="ffvae_train")
            
            
    def predict(self, X: pd.DataFrame) -> np.array:
        non_sen = X.copy()

        non_sen.drop(self._config.sensitive_attr, axis=1)
        inputs = torch.tensor(non_sen.values, dtype=torch.float32)

        attrs = torch.tensor(X[self._config.sensitive_attr].values, dtype=torch.float32)
        
        pre_softmax, _ = self._model(inputs, None, attrs, mode = "pred")
        
        print(pre_softmax)
        np_preds =  pre_softmax.detach().numpy()
        binary_array = (np_preds > 0).astype(int)
        print(np.sum(binary_array))
        return binary_array


        
    
"""
For reproducing Setting 1, 2 experiments with Ffvae model on Adult dataset
"""



# https://github.com/charan223/FairDeepLearning/blob/main/models/model_ffvae.py 


"""Module for a VAE MLP for Adult dataset
Parameters
----------
num_neurons: list, dtype: int
        Number of neurons in each layer
zdim: int
        length of mu, std of the latent representation
activ: string, default: "leakyrelu"
        Activation function
"""
class MLP(nn.Module):
    def __init__(self, num_neurons, zdim, activ="leakyrelu"):
        print('mlp', num_neurons)
        """Initialized VAE MLP"""
        super(MLP, self).__init__()
        self.num_neurons = num_neurons # TODO:
        self.num_layers = len(self.num_neurons) - 1
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_layers)
            ]
        )
        for hidden in self.hiddens:
            torch.nn.init.xavier_uniform_(hidden.weight)
        self.activ = activ
        self.zdim =zdim

    def forward(self, inputs, mode):
        """Computes forward pass for VAE's encoder, VAE's decoder, classifier, discriminator"""
        L = inputs
        for hidden in self.hiddens:
            L = F.leaky_relu(hidden(L))
            #print("L", L.shape)
            #print( L[:, : self.zdim].shape, L[:, self.zdim :].shape)
            
        if mode == "encode":
            mu, logvar = L[:, : self.zdim], L[:, self.zdim :]
            return mu, logvar
        elif mode == "decode":
            return L.squeeze()
        elif mode == "discriminator":
            logits, probs = L, nn.Softmax(dim=1)(L)
            return logits, probs
        elif mode == "classify":
            return L
        else:
            raise Exception(
                "Wrong mode choose one of encoder/decoder/discriminator/classifier"
            )
        

"""Module for FFVAE network
Parameters
----------
args: ArgumentParser
        Contains all model and shared input arguments
"""
gammas = [10, 50, 100]
alphas = [10, 100, 1000]

widths = [32]
edepths = [2]
ewidths = [32]
adepths = [2]
awidths = [32]
cdepths = [2]
cwidths = [32]
zdims = [16]

class Args:
    def __init__(self, config: TestConfig):
        self.input_dim = config.other['my model config'].input_dim
        self.num_classes = 1
        self.gamma = 10
        self.alpha = 10
        self.zdim = config.other['my model config'].latent_dim
        self.device = "cpu"
        self.adepth = 2
        self.awidths = 32
        self.cdepth = 2
        self.cwidths = 32
        self.edepth = 2
        self.ewidths = 32
        self.batch_size = 29304 # TODO data size
        self.sens_idx = config.other['my model config'].sens_column_ids

class Ffvae(nn.Module):
    """Initializes FFVAE network: VAE encoder, MLP classifier, MLP discriminator"""
    def __init__(self, config):
        args = Args(config)
        super(Ffvae, self).__init__()
        self.input_dim = args.input_dim # vae layers
        self.num_classes = args.num_classes  # 1?
        self.gamma = args.gamma # given above
        self.alpha = args.alpha # given above
        self.zdim = args.zdim # latent dim
        self.device = args.device # cpu
        # args.edepth   # given above
        # args.ewidths  # given above
        
        # index for sensitive attribute
        self.n_sens = len(args.sens_idx)
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = [
            i for i in range(int(self.zdim / 2)) if i not in self.sens_idx
        ]
        self.count = 0
        self.batch_size = args.batch_size

        # VAE encoder
        self.enc_neurons = (
            [self.input_dim] + args.edepth * [args.ewidths] + [2 * args.zdim]
        )
        self.encoder = MLP(self.enc_neurons, self.zdim).to(args.device)

        self.dec_neurons = (
            [args.zdim] + args.edepth * [args.ewidths] + [args.input_dim]
        )
        self.decoder = MLP(self.dec_neurons, self.zdim).to(args.device)

        # MLP Discriminator
        self.adv_neurons = [args.zdim] + args.adepth * [args.awidths] + [2]
        self.discriminator = MLP(self.adv_neurons, args.zdim).to(args.device)

        # MLP Classifier
        self.class_neurons = (
            [args.zdim] + args.cdepth * [args.cwidths] + [args.num_classes]
        )
        self.classifier = MLP(self.class_neurons, args.zdim).to(args.device)

        (
            self.optimizer_ffvae,
            self.optimizer_disc,
            self.optimizer_class,
        ) = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds FFVAE class """
        model = Ffvae(args)
        return model

    def vae_params(self):
        """Returns VAE parameters required for training VAE"""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def discriminator_params(self):
        """Returns discriminator parameters"""
        return list(self.discriminator.parameters())

    def classifier_params(self):
        """Returns classifier parameters"""
        return list(self.classifier.parameters())

    def get_optimizer(self):
        """Returns an optimizer for each network"""
        optimizer_ffvae = torch.optim.Adam(self.vae_params())
        optimizer_disc = torch.optim.Adam(self.discriminator_params())
        optimizer_class = torch.optim.Adam(self.classifier_params())
        return optimizer_ffvae, optimizer_disc, optimizer_class

    def forward(self, inputs, labels, attrs, mode="train", A_wts=None, Y_wts=None, AY_wts=None, reweight_target_tensor=None, reweight_attr_tensors=None):
        """Computes forward pass through encoder ,
            Computes backward pass on the target function"""
        # Make inputs between 0, 1
        self.batch_size = inputs.size(0)
        x = (inputs + 1) / 2

        # encode: get q(z,b|x)
        _mu, _logvar = self.encoder(x, "encode")
        #print(_mu, _logvar)

        # only non-sensitive dims of latent code modeled as Gaussian
        mu = _mu[:, self.nonsens_idx]
        logvar = _logvar[:, self.nonsens_idx]
        zb = torch.zeros_like(_mu).to(self.device)  
        std = (logvar / 2).exp()
        q_zIx = torch.distributions.Normal(mu, std)

        # the rest are 'b', deterministically modeled as logits of sens attrs a
        #b_logits = _mu[:, self.sens_idx] # NOTE: OG
        b_logits = _mu[:, self.sens_idx].clone()
        #print("dwedhcwedcsk",b_logits.squeeze().t())

        # draw reparameterized sample and fill in the code
        z = q_zIx.rsample()
        # reparametrization
        zb[:, self.sens_idx] = b_logits
        zb[:, self.nonsens_idx] = z

        # decode: get p(x|z,b)
        # xIz_params = self.decoder(zb, "decode")  # decoder yields distn params not preds
        xIz_params = self.decoder(zb, "decode")

        p_xIz = torch.distributions.Normal(loc=xIz_params, scale=1.0)
        # negative recon error per example
        logp_xIz = p_xIz.log_prob(x)  

        recon_term = logp_xIz.reshape(len(x), -1).sum(1)

        # prior: get p(z)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # compute analytic KL from q(z|x) to p(z), then ELBO
        kl = torch.distributions.kl_divergence(q_zIx, p_z).sum(1)

        # vector el
        elbo = recon_term - kl  
        
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaerers",b_logits.squeeze().t())
        
     
        # decode: get p(a|b)
        clf_losses = [
            nn.BCEWithLogitsLoss()(_b_logit.to(self.device), _a_sens.to(self.device))
            for _b_logit, _a_sens in zip(
                b_logits.squeeze().t(), attrs.type(torch.FloatTensor).squeeze()#.t()
            )
        ]

        # compute loss
        logits_joint, probs_joint = self.discriminator(zb.detach(), "discriminator")
        
        # compute loss
        fflogits_joint, probs_joint = self.discriminator(zb.detach(), "discriminator")
        #NOTE: logits_joint used in disc loss and ffvae
        fftotal_corr = fflogits_joint[:, 0] - fflogits_joint[:, 1]

        ffvae_loss = (
            -1.0 * elbo.mean()
            + self.gamma * fftotal_corr.mean()
            + self.alpha * torch.stack(clf_losses).mean()
        )

        # shuffling minibatch indexes of b0, b1, z
        z_fake = torch.zeros_like(zb)
        z_fake[:, 0] = zb[:, 0][torch.randperm(zb.shape[0])]
        z_fake[:, 1:] = zb[:, 1:][torch.randperm(zb.shape[0])]
        z_fake = z_fake.to(self.device).detach()

        # discriminator
        logits_joint_prime, probs_joint_prime = self.discriminator(
            z_fake, "discriminator"
        )
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        disc_loss = (
            0.5
            * (
                F.cross_entropy(logits_joint, zeros)
                + F.cross_entropy(logits_joint_prime, ones)
            ).mean()
        )

        encoded_x = _mu.detach()

        # IMPORTANT: randomizing sensitive latent
        encoded_x[:, 0] = torch.randn_like(encoded_x[:, 0])

        pre_softmax = self.classifier(encoded_x, "classify")
        
        cost_dict = dict()
        
        if "train" in mode:
            logprobs = F.log_softmax(pre_softmax, dim=1)
            print(logprobs.t().squeeze())
            print(labels)
            
            class_loss = F.nll_loss(logprobs.t().squeeze(), labels)

            cost_dict = dict(
                ffvae_cost=ffvae_loss, disc_cost=disc_loss, main_cost=class_loss
            )

        # ffvae optimization
        if mode == "train":
            self.optimizer_ffvae.zero_grad()
            ffvae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae_params(), 5.0)
            self.optimizer_ffvae.step()

            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator_params(), 5.0)
            self.optimizer_disc.step()

        # classifier optimization
        #elif mode == "train":
            self.optimizer_class.zero_grad()
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier_params(), 5.0)
            self.optimizer_class.step()

        return pre_softmax, cost_dict

"""Function to resize input
Parameters
----------
size: tuple
        target size to be resized to
tensor: torch tensor
        input tensor
Returns
----------
Resized tensor
"""
class Resize(torch.nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.s)
