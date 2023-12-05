import os
from src import *

# Just an example for now

n_repetitions = 7
results_filename = "x-post-refactor-prelims-"
other = {}
results_file = os.path.join("results",results_filename +".csv")


datasets = [Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 
latent_dims = [30, 10]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF, Metrics.DF]


epochs = 1500


for dataset, dim in zip(datasets, latent_dims):
    fyp_config = VAEMaskConfig(epochs=epochs, latent_dim=dim, lr=0.007, losses_used=[ VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS, VAEMaskConfig.LATENT_S_ADV_LOSS])

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

"""