import os
from src import *
import torch

torch.autograd.set_detect_anomaly(True)

epochs = 1250
n_repetitions = 1
results_filename = 'test'
results_filename = "multiple_attributes_"
other = {}
results_file = os.path.join("results",results_filename +".csv")


datasets = [Tester.COMPAS_D] #[Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [10] #[30, 10]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF, Metrics.DF]




"""
curent oppinions:

BEST: 
LP, P, KP, F
SOLID: 
FP, LK, LF (laybe more weight for L and F)
Try With other weights:
L, (more L weight)
K, (lower K weight)
FK .... meh
"""

losses = [
    [VAEMaskConfig.FLIPPED_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],   # 0 - 4 ish x 0.1 # : LESS  weight 100 # TODO: 10 more?? maybe 50
    [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0.000 002 - 0 x 1000 # . INCR WEIGHT 1000
    [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]

losses = [
    #[VAEMaskConfig.KL_DIV_LOSS], # 0.1 - 0.0001 x 0.01 # easy to optim
    #[VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],  # 0.07 - 0.0169 x 10
    [VAEMaskConfig.LATENT_S_ADV_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0 - 10 ish x 0.1 # : could try to reduce a bit ??? not mucg tho. depends if its the only bias mit??
    ##[VAEMaskConfig.FLIPPED_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],   # 0 - 4 ish x 0.1 # : LESS  weight 100 # TODO: 10 more?? maybe 50
    ##[VAEMaskConfig.KL_SENSITIVE_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0.001 - 0.000 001 tho this is overdoing it x 100 . INCREASE WEIGHT 100
    ##[VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # 0.000 002 - 0 x 1000 # . INCR WEIGHT 1000
    #[VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.KL_SENSITIVE_LOSS,      VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.KL_SENSITIVE_LOSS,      VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
    #[VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.FLIPPED_ADV_LOSS,       VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
]

for l in losses:
    for dataset, dim in zip(datasets, latent_dims):
        #fyp_config.config_loss(VAEMaskConfig.KL_DIV_LOSS, weight = w[0])
        #fyp_config.config_loss(VAEMaskConfig.RECON_LOSS, weight = w[1])
        #fyp_config.config_loss(VAEMaskConfig.LATENT_S_ADV_LOSS, weight = w[2])

        for s in [["sex"], ["race"], ["sex", "race"]]: # ["race"]
            fyp_config = VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.011, losses_used=l)
            print(fyp_config)
            mls = [
                TestConfig(Tester.FYP_VAE, Model.LG_R, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),

                TestConfig(Tester.FYP_VAE, Model.NN_old, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.NN_old , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.NN_old, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.NN_old, sensitive_attr=s),   
                TestConfig(Tester.BASE_ML, Model.NN_old, sensitive_attr=s),   

                TestConfig(Tester.FYP_VAE, Model.RF_C, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.RF_C , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.RF_C, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.RF_C, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.RF_C, sensitive_attr=s),
            ]

            mls = [  
                TestConfig(Tester.FYP_VAE, Model.LG_R, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),
            ]

            # NOTE loss atm only single attr

            
            
            #results_file = os.path.join("results",results_filename +"".join(s)+".csv")
            results_file = os.path.join("results",results_filename +".csv")
            # TEST
            tester = Tester(results_file, dataset, metric_names)
            tester.run_tests(mls, n_repetitions, save_intermid_results=True)













# 2. removed native cnty from adult and age non discrete
# 3. changed nn dims and the latent dim

## ISSUES:
"""
the losses work only when calculated from mu not z
how to train the adv losses while keeping torch itact

"""


## RESULTS:
"""
x-post-refactor-prelims-


COMPAS:
best bias mit for RF and NN.
RW anf FB outperform in LR
RF bad acc overall tho
sex     acc loss 1% 
race    acc loss <1% 


ADULT:
best bias mit
very bad recall!
sex     acc loss 6-7%  !!!!!
race    acc loss 4% 

out of 
weights = [ 
    [0.005, 12, 1],   # BB0, BE0
    [0.005, 15, 1.5], # FD0, E20
    [0.01 , 10, 1],   # D30, 850
    [0.005, 10, 2]    # 940, B80
]

[0.005, 12, 1], performed the best

COMPAS:
sex     acc loss 1% 
race    acc loss <1% 


ADULT:
sex     acc loss 6-7%  !!!!! not fit at all
race    acc loss 4%          works greatttt with new lossess


"""


