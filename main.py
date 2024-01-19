import os
from src import *
import torch

torch.autograd.set_detect_anomaly(True)


epochs = 1250
n_repetitions = 1
results_filename = 'test'
results_filename = "before_after_mask_proc_"#"multiple_maps_"
other = {}

#EXPERIMENTS RUNNING:
# og: 
# # 1 has with my new method
# # 2 should have without the std !! SEEMS TO HAVE HELPED
# # 3 SWITH VAE DIMS FROM 15
# JUST ADULT:
# # 4 just adult with same dims as before but min 30 !! SEEMS TO HAVE HELPED
# # 5 just adult more epochs # SEEMS BIT MEH
# # 6 less iterations more lr!!!!  / in new new  2
# # 7 not abs!!!!  new new 3 and 4
# # 8 not abs
# # higher w for just reconstruction error???/
# # SMALLER LATENT SPACE????


# # 3 add more iterations


# _new has without


datasets = [Tester.ADULT_D]#, Tester.COMPAS_D] #[Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [30]#, 10]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI_FM, Metrics.SF, Metrics.DF]



# TODO: fix the predict function! with the ppredict proba equivalent for each model 
# TODO: need fancier benchmarks. esp if the combo with rw works.
# TODO: try add eq weighting for all subgroups

"""
curent oppinions:

BEST: 
LP, P, KP, F (?)
SOLID: 
FP, LK, LF (laybe more weight for L and F)
Try With other weights:
L, (more L weight)
K, (lower K weight)
FK .... meh
"""


losses = [
    #[VAEMaskConfig.KL_DIV_LOSS], # 0.1 - 0.0001 x 0.01 # easy to optim
    #[VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],  # 0.07 - 0.0169 x 10
    #[VAEMaskConfig.LATENT_S_ADV_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0 - 10 ish x 0.1 # : could try to reduce a bit ??? not mucg tho. depends if its the only bias mit??
    ##[VAEMaskConfig.FLIPPED_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],   # 0 - 4 ish x 0.1 # : LESS  weight 100 # TODO: 10 more?? maybe 50
    ##[VAEMaskConfig.KL_SENSITIVE_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0.001 - 0.000 001 tho this is overdoing it x 100 . INCREASE WEIGHT 100
    ##[VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # p should be used together with some biaas mit. it mostly imporves acc
    ####[VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.KL_SENSITIVE_LOSS,      VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    ####[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.KL_SENSITIVE_LOSS,      VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.FLIPPED_ADV_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]

losses = [
    [VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0.000 002 - 0 x 1000 # . INCR WEIGHT 1000
    [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.KL_SENSITIVE_LOSS,      VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]

losses = [
    [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]

comment="incr the weight of recon error"

for l in losses:
    for dataset, dim in zip(datasets, latent_dims):
        #fyp_config.config_loss(VAEMaskConfig.KL_DIV_LOSS, weight = w[0])
        #fyp_config.config_loss(VAEMaskConfig.RECON_LOSS, weight = w[1])
        #fyp_config.config_loss(VAEMaskConfig.LATENT_S_ADV_LOSS, weight = w[2])

        for s in [ ["sex"]]: #  ["sex","race"],["race"],["sex"]
            fyp_config = VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.015, losses_used=l)
            
            print(fyp_config)
            mls = [
                TestConfig(Tester.FYP_VAE, Model.LG_R, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),

                TestConfig(Tester.FYP_VAE, Model.MLP_C, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.MLP_C , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.MLP_C, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.MLP_C, sensitive_attr=s),   
                TestConfig(Tester.BASE_ML, Model.MLP_C, sensitive_attr=s),   

                TestConfig(Tester.FYP_VAE, Model.RF_C, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.RF_C , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.RF_C, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.RF_C, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.RF_C, sensitive_attr=s),
            ]

            mls = [  
                TestConfig(Tester.FYP_VAE, Model.LG_R, other={"c": comment, VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s), #base_model_bias_mit=Tester.REWEIGHING
                TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),
            ]
            
            results_file = os.path.join("results",results_filename +"_".join(s)+".csv")
            tester = Tester(results_file, dataset, metric_names)
            tester.run_tests(mls, n_repetitions, save_intermid_results=True)

