import os
from src import *
import torch

torch.autograd.set_detect_anomaly(True)
# For the experiments with hyperpara I will directly modify the configs.py to have all the hyperparam combos


epochs = 0
n_repetitions = 2
results_filename = "hyperparams_"
other = {}

datasets = [Tester.COMPAS_D, Tester.ADULT_D]#, Tester.COMPAS_D] #[Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [12, 30]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI_FM, Metrics.SF]


losses = [
    [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
    [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
    [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]


comment= "FYP"

for e in range(10):
    for dataset, dim in zip(datasets, latent_dims):
        for s in [["race"],["sex"]]: #  ["sex","race"],["race"],["sex"]
   
            vae_LG = [
                TestConfig(Tester.FYP_VAE, Model.LG_R, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
                    VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.014, losses_used=l)}) for l in losses
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
                    VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.014, losses_used=l)}) for l in losses
                ]
            mls_MLP =  vae_MLP + [
                TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
                #TestConfig(Tester.LFR, Model.MLP_C, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
                
            ]
            
            mls = mls_LG + mls_MLP
            results_file = os.path.join("results",results_filename +"_".join(s)+".csv")
            #results_file = os.path.join("results",results_filename +".csv")
            tester = Tester(results_file, dataset, metric_names)
            tester.run_tests(mls, n_repetitions, save_intermid_results=True)
            
    #--------------------------------------------------------------------------------------------------
    dataset, dim = Tester.GERMAN_D, 20
    s = ["sex"]
    vae_LG = [
        TestConfig(Tester.FYP_VAE, Model.LG_R, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
            VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.014, losses_used=l)}) for l in losses
        ]
    mls_LG = vae_LG +  [  
        TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
        TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
        TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
        TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),
        TestConfig(Tester.LFR, Model.LG_R, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
    ]
    
    
    vae_MLP = [
        TestConfig(Tester.FYP_VAE, Model.MLP_C, sensitive_attr=s, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG:  
            VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.014, losses_used=l)}) for l in losses
        ]
    mls_MLP = vae_MLP + [
        TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
        TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
        TestConfig(Tester.LFR, Model.MLP_C, other={"c":"LFR"}, sensitive_attr = s), #base_model_bias_mit=Tester.REWEIGHING
        
    ]
    
    mls = mls_LG + mls_MLP
    results_file = os.path.join("results",results_filename +"_".join(s)+".csv")
    #results_file = os.path.join("results",results_filename +".csv")
    tester = Tester(results_file, dataset, metric_names)
    tester.run_tests(mls, n_repetitions, save_intermid_results=True)

