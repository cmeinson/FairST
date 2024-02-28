import sys
import os
import itertools

from typing import Optional, List
from .tester import Tester
from .ml_models import FairBalanceModel, FairMaskModel, VAEMaskModel, BaseModel, ReweighingModel, Model, VAEMaskConfig
from .metrics import Metrics
from .test_config import TestConfig


default_models = [Model.LG_R, Model.SV_C, Model.NB_C, Model.RF_C, Model.DT_R, Model.EN_R, Model.NN_C,]

all_losses = [
        [VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.LATENT_S_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.KL_SENSITIVE_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
    ]

def try_test(results_filename, s, dataset, metric_names, mls, n_repetitions, attempts = 3):
    for i in range(attempts):
        try:
            results_file = os.path.join("results",results_filename +"_".join(s)+".csv")
            tester = Tester(results_file, dataset, metric_names)
            tester.run_tests(mls, n_repetitions, save_intermid_results=True)
            return
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print("failed test nr ",i+1, dataset, s)
            #raise e
            
def get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss, loss_params, mask_value = None):
    config = VAEMaskConfig(epochs=epochs, latent_dim=latent_dim, lr=lr, losses_used=loss, vae_layers=vae_layers, mask_values=mask_value)
    config.config_loss(config.KL_DIV_LOSS, weight = loss_params["w_kl_div"])
    config.config_loss(config.RECON_LOSS, weight = loss_params["w_recon"])
    
    config.config_loss(config.LATENT_S_ADV_LOSS, weight = loss_params["w_latent_s"], lr=loss_params["lr_latent_s"], layers=loss_params["layers_latent_s"])
    config.config_loss(config.FLIPPED_ADV_LOSS, weight = loss_params["w_flipped"], lr=loss_params["lr_flipped"], layers=loss_params["layers_flipped"])
    
    config.config_loss(config.KL_SENSITIVE_LOSS, weight = loss_params["w_kl_sens"])
    config.config_loss(config.POS_VECTOR_LOSS, weight = loss_params["w_pos_sens"])
    
    return config


def ml_configs(ml_model, fyp_losses, attrs, epochs, latent_dim, lr, vae_layers, loss_params, baselines=None):
    fyp  = [
        TestConfig(Tester.FYP_VAE, ml_model, sensitive_attr=attrs, other={"c": "FYP", VAEMaskModel.VAE_MASK_CONFIG:  
            get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in fyp_losses
    ]
    #fyp = []
    available_base =  [  
        TestConfig(Tester.BASE_ML, ml_model, sensitive_attr = attrs),   
        TestConfig(Tester.FAIRMASK, ml_model, Model.DT_R, sensitive_attr=attrs),
        TestConfig(Tester.FAIRBALANCE, ml_model, sensitive_attr=attrs),
        TestConfig(Tester.REWEIGHING, ml_model, sensitive_attr=attrs),
        TestConfig(Tester.LFR, ml_model, other={"c":"LFR"}, sensitive_attr = attrs), #base_model_bias_mit=Tester.REWEIGHING
    ] 
    if baselines==None:
        return available_base + fyp
    
    bases = [config for config in available_base if config.bias_mit in baselines]
    return bases + fyp
    
            
def run_all_losses(dataset, epochs, latent_dim, lr, vae_layers, loss_params, results_filename, models = default_models, n_reps = 2, metrics = Metrics.get_all_names()):
    
    for s in [["race","sex"]]: #  ["sex","race"],["race"],["sex"]
        if dataset==Tester.GERMAN_D and "race" in s:
            continue
        # check German only run with sex
        
        
        mls = list(itertools.chain.from_iterable([
            ml_configs(model, all_losses, s, epochs, latent_dim, lr, vae_layers, loss_params)
            for model in models
        ]))
        try_test(results_filename, s, dataset, metrics, mls, n_reps)
        