
import os
from src import *
import torch
import sys

n_repetitions = 4
results_filename = "defualt_"
comment = "FYP"

losses = [ [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS]]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _try_test(results_filename, s, dataset, metric_names, mls, n_repetitions, attempts = 3):
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
            raise e
            
def _get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss, loss_params, mask_value = None):
    config = VAEMaskConfig(epochs=epochs, latent_dim=latent_dim, lr=lr, losses_used=loss, vae_layers=vae_layers, mask_values=mask_value)
    config.config_loss(config.KL_DIV_LOSS, weight = loss_params["w_kl_div"])
    config.config_loss(config.RECON_LOSS, weight = loss_params["w_recon"])
    
    config.config_loss(config.LATENT_S_ADV_LOSS, weight = loss_params["w_latent_s"], lr=loss_params["lr_latent_s"], layers=loss_params["layers_latent_s"])
    config.config_loss(config.FLIPPED_ADV_LOSS, weight = loss_params["w_flipped"], lr=loss_params["lr_flipped"], layers=loss_params["layers_flipped"])
    
    config.config_loss(config.KL_SENSITIVE_LOSS, weight = loss_params["w_kl_sens"])
    config.config_loss(config.POS_VECTOR_LOSS, weight = loss_params["w_pos_sens"])
    
    return config


def _ml_configs(ml_model, fyp_losses, attrs, epochs, latent_dim, lr, vae_layers, loss_params, baselines=None):
    fyp  = [
        TestConfig(Tester.FYP_VAE, ml_model, sensitive_attr=attrs, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
            _get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in fyp_losses
    ]
    #fyp = []
    available_base =  [  
        TestConfig(Tester.BASE_ML, ml_model, sensitive_attr = attrs),   
        TestConfig(Tester.FAIRMASK, ml_model, Model.DT_R, sensitive_attr=attrs),
        TestConfig(Tester.FAIRBALANCE, ml_model, sensitive_attr=attrs),
        TestConfig(Tester.REWEIGHING, ml_model, sensitive_attr=attrs),
        #TestConfig(Tester.LFR, ml_model, other={"c":"LFR"}, sensitive_attr = attrs), #base_model_bias_mit=Tester.REWEIGHING
    ] 
    if baselines==None:
        return available_base + fyp
    
    bases = [config for config in available_base if config.bias_mit in baselines]
    return bases + fyp
    
            
def _run_all_losses(dataset, epochs, latent_dim, lr, vae_layers, loss_params):
    
    for s in [["sex"]]: #  ["sex","race"],["race"],["sex"]
        if dataset==Tester.GERMAN_D and "race" in s:
            continue
        # check German only run with sex
        
        mls = ( _ml_configs(Model.LG_R, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            ) + _ml_configs(Model.NN_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params)) # TODO: need to add input dim
                
        #mls = ml_configs(Model.NN_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params)
        _try_test(results_filename, s, dataset, metric_names, mls, n_repetitions)
        


torch.autograd.set_detect_anomaly(True)
# TODO: redirect output



datasets = [Tester.DEFAULT_D]
defaults = [
    (1500, 20, 0.006, (75, 60, 30, 30)),
]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.F1,Metrics.MEAN_Y, Metrics.ERD, Metrics.AOD, Metrics.EOD, Metrics.A_AOD, Metrics.A_EOD, Metrics.A_SPD, Metrics.SPD, Metrics.DI_FM, Metrics.SF]

metric_names = Metrics.get_all_names()

loss_params = {
    "w_kl_div" : 0.05,
    "w_recon" : 15,
    "w_latent_s" : 0.1, 
    "w_flipped" : 0.01,
    "w_kl_sens" : 9000,
    "w_pos_sens" : 1200000,
    "lr_latent_s" : 0.05, 
    "lr_flipped" : 0.05,
    "layers_latent_s" : (75, 30, 10), 
    "layers_flipped" : (75, 30, 10)
}


for e in range(10):
    for dataset, default_values in zip(datasets, defaults):
        print('~'*1000)
        print(e+5, " iteration of 4x dataset:", dataset)
        print('~'*100)
        _run_all_losses(dataset, default_values[0], default_values[1], default_values[2], default_values[3], loss_params)
