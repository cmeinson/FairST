from src import *

n_repetitions = 4
results_filename = "EXPERIMENTS_"


# baseline bias mitigations to test
baselines = [Tester.BASE_ML, Tester.FAIRMASK, Tester.FAIRBALANCE, Tester.REWEIGHING, Tester.LFR]

# ml models to test. note that most of the training time is spent on training the vae, so testing multiple ml models at once is more efficent than doing it iteratively
models = [Model.LG_R, Model.SV_C, Model.NB_C, Model.RF_C, Model.DT_R, Model.NN_C]

# datasets to test
datasets = [Tester.COMPAS_D, Tester.ADULT_D, Tester.DEFAULT_D]
# vae training config for each dataset (epochs, latent space size, lr, encoder and decoder layers. )
defaults = [
    (1800, 17, 0.006, (100, 75, 50)),
    (1500, 30, 0.006, (75, 60, 30, 30)),
    (1500, 30, 0.006, (75, 60, 30, 30)),
]
# settings used for losses
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
        

#
all_losses = [
        [VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],  # no disentanglement loss
        [VAEMaskConfig.POS_VECTOR_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], # the setting used in the paper
        [VAEMaskConfig.LATENT_S_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,         VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.LATENT_S_ADV_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS, VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS],
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.POS_VECTOR_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.KL_SENSITIVE_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.FLIPPED_ADV_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
        [VAEMaskConfig.KL_SENSITIVE_LOSS,  VAEMaskConfig.LATENT_S_ADV_LOSS,        VAEMaskConfig.RECON_LOSS, VAEMaskConfig.KL_DIV_LOSS], 
    ]

# list of metrics used for evaluation
metrics = Metrics.get_all_names()

for e in range(10):
    for dataset, default_values in zip(datasets, defaults):
        print('~'*1000)
        print(e, " iteration of 4x dataset:", dataset)
        print('~'*100)
        
        for s in [["sex","race"],["race"],["sex"]]:
            if dataset in [Tester.DEFAULT_D] and "race" in s:
                continue
            
            mls = list(itertools.chain.from_iterable([
                ml_configs(model, all_losses, s, default_values[0], default_values[1], default_values[2], default_values[3], loss_params, baselines=baselines)
                for model in models
            ]))
            try_test(results_filename, s, dataset, metrics, mls, n_repetitions=n_repetitions)

