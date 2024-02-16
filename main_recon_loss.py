
import os
from src import *
import torch
import sys

n_repetitions = 1

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
            print("failed test nr ",i+1, dataset, s, e)
            raise e
            
def get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss, loss_params):
    config = VAEMaskConfig(epochs=epochs, latent_dim=latent_dim, lr=lr, losses_used=loss, vae_layers=vae_layers)
    config.config_loss(config.RECON_LOSS, weight = loss_params["w_recon"])
    config.config_loss(config.FLIPPED_ADV_LOSS, weight = loss_params["w_recon"])
    
    
    return config
            
def run_all_losses(dataset, epochs, latent_dim, lr, vae_layers, loss_params):
    
    results_filename = "2recon_loss_"
    comment= "mse2"

    loss_og = [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS]
    loss_mse = [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.FLIPPED_ADV_LOSS, VAEMaskConfig.KL_DIV_LOSS]
       # [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.FLIPPED_ADV_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
  
    for s in [["race"], ["sex"]]: #  ["sex","race"],["race"],["sex"]
        if dataset==Tester.GERMAN_D and "race" in s:
            continue
        # check German only run with sex
        eq_MLP = [
            TestConfig(Tester.EQODDS, Model.MLP_C, other={"c":"EqO2"}, sensitive_attr = s), 
            TestConfig(Tester.EQODDS_ALT, Model.MLP_C, other={"c":"EqO ALT2"}, sensitive_attr = s), 
        ]
        vae_MLP = [
            TestConfig(Tester.FYP_VAE, Model.MLP_C, sensitive_attr=s, other={"c": "FYP", VAEMaskModel.VAE_MASK_CONFIG:  get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss_og, loss_params)}),
            TestConfig(Tester.FYP_VAE, Model.MLP_C, sensitive_attr=s, other={"c": "mse2", VAEMaskModel.VAE_MASK_CONFIG:  get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss_mse, loss_params)})
            ]
        mls_MLP =  vae_MLP + eq_MLP + [
            TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
            TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
            #TestConfig(Tester.LFR, Model.MLP_C, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
            
        ]
        
        
        mls = mls_MLP
        try_test(results_filename, s, dataset, metric_names, mls, n_repetitions)
        


torch.autograd.set_detect_anomaly(True)
# For the experiments with hyperpara I will directly modify the configs.py to have all the hyperparam combos



datasets = [Tester.COMPAS_D, Tester.ADULT_D,   Tester.GERMAN_D]#, Tester.COMPAS_D] #[Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [ 15, 30, 20]
defaults = [
    (1300, 12, 0.01, (50, 45, 35, 30)),
    (1400, 30, 0.014,(75, 60, 30, 30)),
    (1400, 15, 0.01, (50, 45, 35, 30)),

]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.ERD, Metrics.AOD, Metrics.EOD, Metrics.A_AOD, Metrics.A_EOD, Metrics.SPD, Metrics.SF]


og_loss_params = {
    "w_kl_div" : 0.005,
    "w_recon" : 17,
    "w_latent_s" : 0.1, 
    "w_flipped" : 17,#0.01,
    "w_kl_sens" : 9000,
    "w_pos_sens" : 1200000,
    "lr_latent_s" : 0.05, 
    "lr_flipped" : 0.05,
    "layers_latent_s" : (75, 30, 10), 
    "layers_flipped" : (75, 30, 10)
}
  
  
for e in range(10):
    for dataset, dim, default_values in zip(datasets, latent_dims, defaults):
        print(e, dataset)
        run_all_losses(dataset, default_values[0], default_values[1], default_values[2], default_values[3], og_loss_params)

# mse - recon w 40
# mse2 - default w
# FYP - Huber Loss 