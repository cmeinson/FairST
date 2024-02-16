
import os
from src import *
import torch
import sys

n_repetitions = 4

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
            raise e
            
def get_vaemask_config(epochs, latent_dim, lr, vae_layers, loss, loss_params):
    config = VAEMaskConfig(epochs=epochs, latent_dim=latent_dim, lr=lr, losses_used=loss, vae_layers=vae_layers)
    config.config_loss(config.KL_DIV_LOSS, weight = loss_params["w_kl_div"])
    config.config_loss(config.RECON_LOSS, weight = loss_params["w_recon"])
    
    config.config_loss(config.LATENT_S_ADV_LOSS, weight = loss_params["w_latent_s"], lr=loss_params["lr_latent_s"], layers=loss_params["layers_latent_s"])
    config.config_loss(config.FLIPPED_ADV_LOSS, weight = loss_params["w_flipped"], lr=loss_params["lr_flipped"], layers=loss_params["layers_flipped"])
    
    config.config_loss(config.KL_SENSITIVE_LOSS, weight = loss_params["w_kl_sens"])
    config.config_loss(config.POS_VECTOR_LOSS, weight = loss_params["w_pos_sens"])
    
    return config
            
def run_all_losses(dataset, epochs, latent_dim, lr, vae_layers, loss_params):
    
    results_filename = "MAIN_"
    comment= "FYP"

    losses = [
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
    for s in [["race"],["sex"]]: #  ["sex","race"],["race"],["sex"]
        if dataset==Tester.GERMAN_D and "race" in s:
            continue
        # check German only run with sex
        
        vae_LG  = [
            TestConfig(Tester.FYP_VAE, Model.LG_R, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
                get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in losses
            ]
        mls_LG = vae_LG +  [  
            TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
            TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
            TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
            TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),
            #TestConfig(Tester.LFR, Model.LG_R, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
        ]

        vae_SV  = [
            TestConfig(Tester.FYP_VAE, Model.SV_C, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
                get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in losses
            ]
        mls_SV = vae_SV +  [  
            TestConfig(Tester.BASE_ML, Model.SV_C , sensitive_attr = s),   
            TestConfig(Tester.FAIRMASK, Model.SV_C, Model.DT_R, sensitive_attr=s),
            TestConfig(Tester.FAIRBALANCE, Model.SV_C, sensitive_attr=s),
            TestConfig(Tester.REWEIGHING, Model.SV_C, sensitive_attr=s),
            #TestConfig(Tester.LFR, Model.LG_R, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
        ]
        
        
        vae_MLP = [
            TestConfig(Tester.FYP_VAE, Model.MLP_C, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
                get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in losses
            ]
        mls_MLP =  vae_MLP + [
            TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
            TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
            #TestConfig(Tester.LFR, Model.MLP_C, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
            
        ]
        
        mls = mls_SV + mls_LG + mls_MLP
        try_test(results_filename, s, dataset, metric_names, mls, n_repetitions)
        


torch.autograd.set_detect_anomaly(True)
# TODO: redirect output



datasets = [Tester.GERMAN_D, Tester.ADULT_D, Tester.COMPAS_D]
defaults = [
    (1500, 20, 0.006, (75, 60, 30, 15)),
    (1500, 30, 0.006, (75, 60, 30, 30)),
    (1800, 17, 0.006, (100, 75, 50)),
]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.F1,Metrics.MEAN_Y, Metrics.ERD, Metrics.AOD, Metrics.EOD, Metrics.A_AOD, Metrics.A_EOD, Metrics.A_SPD, Metrics.SPD, Metrics.DI_FM, Metrics.SF]


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
        print(e, " iteration of 4x dataset:", dataset)
        print('~'*100)
        run_all_losses(dataset, default_values[0], default_values[1], default_values[2], default_values[3], loss_params)


