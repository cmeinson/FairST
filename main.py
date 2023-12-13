import os
from src import *

# Just an example for now

n_repetitions = 4
results_filename = "x-post-refactor-prelims-"
other = {}
results_file = os.path.join("results",results_filename +".csv")


datasets = [Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [30, 10]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF, Metrics.DF]


epochs = 1200

weights = [ 
    [0.005, 12, 1],   # BB0, BE0
    [0.005, 15, 1.5], # FD0, E20
    [0.01 , 10, 1],   # D30, 850
    [0.005, 10, 2]    # 940, B80
]

for w in weights:
    for dataset, dim in zip(datasets, latent_dims):
        fyp_config = VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.007, losses_used=[ VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS, VAEMaskConfig.LATENT_S_ADV_LOSS])
        fyp_config.config_loss(VAEMaskConfig.KL_DIV_LOSS, weight = w[0])
        fyp_config.config_loss(VAEMaskConfig.RECON_LOSS, weight = w[1])
        fyp_config.config_loss(VAEMaskConfig.LATENT_S_ADV_LOSS, weight = w[2])

        for s in [["sex"], ["race"]]:
            mls = [
                TestConfig(Tester.FYP_VAE, Model.LG_R, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.LG_R , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.LG_R, sensitive_attr=s),   

                TestConfig(Tester.FYP_VAE, Model.NN_old, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.NN_old , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.NN_old, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.NN_old, sensitive_attr=s),   

                TestConfig(Tester.FYP_VAE, Model.RF_C, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.RF_C , sensitive_attr = s),   
                TestConfig(Tester.FAIRMASK, Model.RF_C, Model.DT_R, sensitive_attr=s),
                TestConfig(Tester.FAIRBALANCE, Model.RF_C, sensitive_attr=s),
                TestConfig(Tester.REWEIGHING, Model.RF_C, sensitive_attr=s),
                TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=s),   
            ]

            #mls = [  TestConfig(Tester.FYP_VAE, Model.LG_R, other={VAEMaskModel.VAE_MASK_CONFIG: fyp_config}, sensitive_attr=s) ]

            # NOTE loss atm only single attr

            
            
            results_file = os.path.join("results",results_filename +s[0]+".csv")
            # TEST
            tester = Tester(results_file, dataset, metric_names)
            tester.run_tests(mls, n_repetitions, save_intermid_results=True)


# 2. removed native cnty from adult and age non discrete
# 3. changed nn dims and the latent dim

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
sex     acc loss 6-7%  !!!!!
race    acc loss 4% 


"""