
import os
from src import *
import torch
import sys

n_repetitions = 4
#results_filename = "MAIN_all_models_"
results_filename = "MAIN_3_models_"
comment = "FYP"

losses = [
        [VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.KL_SENSITIVE_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.LATENT_S_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.FLIPPED_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
        #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.KL_SENSITIVE_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
    ]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        TestConfig(Tester.FYP_VAE, ml_model, sensitive_attr=attrs, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
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
    
            
def run_all_losses(dataset, epochs, latent_dim, lr, vae_layers, loss_params):
    
    for s in [["race","sex"]]: #  ["sex","race"],["race"],["sex"]
        if dataset==Tester.GERMAN_D and "race" in s:
            continue
        # check German only run with sex
        
        mls = ( ml_configs(Model.LG_R, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            #) + ml_configs(Model.SV_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            #) + ml_configs(Model.NB_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            #) + ml_configs(Model.RF_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            ) + ml_configs(Model.DT_R, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            #) + ml_configs(Model.EN_R, losses, s, epochs, latent_dim, lr, vae_layers, loss_params
            ) + ml_configs(Model.NN_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params)) # TODO: need to add input dim
                
        #mls = ml_configs(Model.NN_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params)
        #mls = ml_configs(Model.NN_C, losses, s, epochs, latent_dim, lr, vae_layers, loss_params)
        try_test(results_filename, s, dataset, metric_names, mls, n_repetitions)
        


torch.autograd.set_detect_anomaly(True)
# TODO: redirect output



datasets = [Tester.COMPAS_D, Tester.GERMAN_D, Tester.ADULT_D]
defaults = [
    (1800, 17, 0.006, (100, 75, 50)),
    (1500, 20, 0.006, (75, 60, 30, 15)),
    (1500, 30, 0.006, (75, 60, 30, 30)),
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
        print(e, " iteration of 2x dataset:", dataset)
        print('~'*100)
        run_all_losses(dataset, default_values[0], default_values[1], default_values[2], default_values[3], loss_params)




"""
RQ1: WHICH LOSS

Rule out based on prediction performance on adult (compas has good performance for all):
- FL the lowest performance on all 4 metrics
- FK, L, VAE, also significantly lowest metrics for adult sex and very high variance
- F 2nd lowest performance and high var on adult race

- FP, LP, P, KL, KP, K all have similar results 

note that all well performing losses seem to use either K or P


BIAS mit performance?

KL, KP, K seem to have consistently good DI, SF, ASPD, for botgh datasets and attributes

PLAN: 
keep going with the KL, KP, K
KL - is advrserial so ideally would rule it out due to time constraints. if K does not reliably perform this will be suggested as the alternative without using Y labels!!!
KP - seems best BUT uses Y. (justify KP by the lower var on adult race aod???? ... sus)
K - seems bit more variance and less debias but will wait for enough evidence

"""


"""
RQ2: how is the bias mitigation????

lol literally outperforms everything in all metrics with some small exceptions
outperforms everything for every example for aspd, spd, di, sf


For Adult sex:
horrible predictive performance but 10x better bias than the best debias...... actually not a usable example
For Adult race:
NOTE: for some models the performance loss seems better than others!!!
--- not as different in performance. still bad tho.
- bias much better  than all others for aspd, spd, di, sf
- FM and RW slightly better AEOD (sliiightly)



TODO: checkout results with just nn
"""



"""
Base performance:
nnk, 85%  65% 
lr,  85%  65%
nb,  78%, 64%. could be nice to use if i can
svm  85%  65%

compare performance of diff models
ADUlt race
nnk - <3% acc loss BUT 5% f1
en - <2% but almost 10 f1
Dt - bad
RF, SVM - 3%, bt 10 f1! 
NB - ACC GAIN and smalllll f1 loss
LR - 2% acc, 5% f1
potential: nnk, lr, nb, svm

ADUlt sex
nnk - 5% acc 10% f1 -----------------okay
en - 3.5% acc 30% f1 !!!!!!  HONESTLY WE CAN JUST REMOVE EN 
Dt - 10% CC, 15% f1 BAD
RF-  6% acc 20% BAD
SVM- 4% acc 14% f1 ---------------tolerable
NB - 5% and 7% F1 !!!!!!------------- good
LR -  5% acc 15% -----------------tolerable
potential: 
"""