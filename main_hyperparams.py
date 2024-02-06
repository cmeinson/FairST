
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
        except Exception:
            print("failed test nr ",i+1, dataset, s)
            
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
    
    results_filename = "methodic_hyperparams_"
    comment= "FYP"

    losses = [
        [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
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
        
        
        vae_MLP = [
            TestConfig(Tester.FYP_VAE, Model.MLP_C, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
                get_vaemask_config(epochs, latent_dim, lr, vae_layers, l, loss_params)}) for l in losses
            ]
        mls_MLP =  vae_MLP + [
            TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
            TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
            #TestConfig(Tester.LFR, Model.MLP_C, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
            
        ]
        
        mls = mls_LG + mls_MLP
        try_test(results_filename, s, dataset, metric_names, mls, n_repetitions)
        


torch.autograd.set_detect_anomaly(True)
# For the experiments with hyperpara I will directly modify the configs.py to have all the hyperparam combos

"""
ADULT
ault wants more epochs
2k still much


COMPAS
more than 10 dims. 13?
LP, KP

{'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=1417, latent_dim=7, mask_values=None vae_layers=(100, 100, 30), lr=0.00378, \nlosses_used=['Sensitive KL loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Sensitive KL loss: {'weight': 8100.0, 'sens_col_ids': [7]}, Pos Y vec loss: {'weight': 672000.0}, Reconstruction loss: {'weight': 18.45}, KL divergence loss: {'weight': 0.0043}"}
{'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=1052, latent_dim=12, mask_values=None vae_layers=(50, 45, 35, 30), lr=0.011619999999999998, \nlosses_used=['Latent sens ADV loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Latent sens ADV loss: {'weight': 0.07150000000000001, 'lr': 0.051500000000000004, 'optimizer': 'Adam', 'layers': (75, 30, 10), 'input_dim': 11}, Pos Y vec loss: {'weight': 1572000.0}, Reconstruction loss: {'weight': 21.6}, KL divergence loss: {'weight': 0.0053}"}


DEFAULTS:
Adult Dataset , LogisticRegression ['sex']
KP: {'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=1300, latent_dim=35, mask_values=None vae_layers=(45, 30, 30, 30), lr=0.014, \nlosses_used=['Sensitive KL loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Sensitive KL loss: {'weight': 11340.0, 'sens_col_ids': [33]}, Pos Y vec loss: {'weight': 1284000.0}, Reconstruction loss: {'weight': 9.9}, KL divergence loss: {'weight': 0.00335}"}
LP: {'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=1300, latent_dim=35, mask_values=None vae_layers=(45, 30, 30, 30), lr=0.014, \nlosses_used=['Latent sens ADV loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Latent sens ADV loss: {'weight': 0.1155, 'lr': 0.015, 'optimizer': 'Adam', 'layers': (50, 30, 10), 'input_dim': 34}, Pos Y vec loss: {'weight': 816000.0}, Reconstruction loss: {'weight': 13.65}, KL divergence loss: {'weight': 0.0039000000000000003}"}

MLP
{'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=831, latent_dim=35, mask_values=None vae_layers=(75, 60, 30, 30), lr=0.011619999999999998, \nlosses_used=['Sensitive KL loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Sensitive KL loss: {'weight': 9900.0, 'sens_col_ids': [33]}, Pos Y vec loss: {'weight': 1080000.0}, Reconstruction loss: {'weight': 8.25}, KL divergence loss: {'weight': 0.0066500000000000005}"}
{'c': 'FYP', 'my model config': "VAEMaskConfig(epochs=1561, latent_dim=23, mask_values=None vae_layers=(75, 60, 30, 30), lr=0.00434, \nlosses_used=['Latent sens ADV loss', 'Pos Y vec loss', 'Reconstruction loss', 'KL divergence loss'])Latent sens ADV loss: {'weight': 0.08689999999999999, 'lr': 0.013500000000000002, 'optimizer': 'Adam', 'layers': (60, 10, 10, 10), 'input_dim': 22}, Pos Y vec loss: {'weight': 1296000.0}, Reconstruction loss: {'weight': 8.55}, KL divergence loss: {'weight': 0.005750000000000001}"}
"""


# TODO: redirect output



datasets = [ Tester.GERMAN_D, Tester.ADULT_D, Tester.COMPAS_D]#, Tester.COMPAS_D] #[Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [ 15, 30, 12]
defaults = [
    (1400, 15, 0.01, (50, 45, 35, 30)),
    (1400, 30, 0.014,(75, 60, 30, 30)),
    (1300, 12, 0.01, (50, 45, 35, 30)),
]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.A_AOD, Metrics.A_EOD, Metrics.SPD, Metrics.DI_FM, Metrics.SF,]


og_loss_params = {
    "w_kl_div" : 0.005,
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
        

vaes = [
    (75, 60, 30, 15),
    (75, 60, 30, 15, 15),
    (75, 60, 30),
    (100, 75, 50),
    (150, 100),
    (100, 100, 30),
    (50, 30, 30),
    (50, 45, 35, 30),
    (50, 45, 35, 30, 20),
    (45, 25, 25, 25),
]

# TODO: del
for e in range(10):
    for dataset, dim, default_values in zip(datasets, latent_dims, defaults):
        for epoch_mult in range(50, 160, 10):
            run_all_losses(dataset, epoch_mult*default_values[0]//100, default_values[1], default_values[2], default_values[3], og_loss_params)
            
        for dims_add in range(-5, 6):
            run_all_losses(dataset, default_values[0], default_values[1]+dims_add, default_values[2], default_values[3], og_loss_params)
            
        for lr_mult in range(50, 160, 10):
            run_all_losses(dataset, default_values[0], default_values[1], lr_mult*default_values[2]/100, default_values[3], og_loss_params)
          
        for layers in vaes:
            run_all_losses(dataset, default_values[0], default_values[1], default_values[2], layers, og_loss_params)



for e in range(10):
    for dataset, dim, default_values in zip(datasets, latent_dims, defaults):
        og_loss_params = {
            "w_kl_div" : 0.005,
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
        
        """
        rocon w *5
            kl w  *3
                pos w *5
                    other lossses w *3
                        lrs *3
                            layers *2

        """
        for w_recon in [8, 11.5, 15, 18.5, 22]:
            for w_kl_div in [0.025, 0.05, 0.075]:
                for w_pos_sens in [600000, 900000, 1200000, 1500000, 1800000]:
                    for w_loss_mult in [0.7, 1, 1.3]:
                        for lr_loss_mult in [0.7, 1, 1.3]:
                            for layers in [(75, 30, 10), (60,10,10,10)]:
                                params = {
                                    "w_kl_div" : w_kl_div,
                                    "w_recon" : w_recon,
                                    "w_latent_s" : og_loss_params["w_latent_s"]*w_loss_mult, 
                                    "w_flipped" : og_loss_params["w_flipped"]*w_loss_mult, 
                                    "w_kl_sens" : og_loss_params["w_kl_sens"]*w_loss_mult, 
                                    "w_pos_sens" : w_pos_sens, 
                                    "lr_latent_s" : og_loss_params["lr_latent_s"]*lr_loss_mult, 
                                    "lr_flipped" : og_loss_params["lr_flipped"]*lr_loss_mult,
                                    "layers_latent_s" : layers,
                                    "layers_flipped" : layers,
                                }
                                run_all_losses(dataset, default_values[0], default_values[1], default_values[2], default_values[3], params)


