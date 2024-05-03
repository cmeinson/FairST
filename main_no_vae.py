
import os
from src import *
import torch
import sys

n_repetitions = 4
results_filename = "MAIN_only_NO_VAE_"

# ------- 

torch.autograd.set_detect_anomaly(True)
# TODO: redirect output



datasets = [Tester.COMPAS_D, Tester.ADULT_D, Tester.DEFAULT_D]
defaults = [
    (1800, 17, 0.006, (100, 75, 50)),
    (1500, 30, 0.006, (75, 60, 30, 30)),
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
        
metrics = Metrics.get_all_names()


models = [Model.LG_R, Model.SV_C, Model.NB_C, Model.RF_C, Model.DT_R, Model.NN_C,]

#models = [Model.NN_C,]

all_losses = [
        [], 
    ]

baselines = [Tester.BASE_ML]

models = [Model.LG_R, Model.NN_C,]

for e in range(10):
    for dataset, default_values in zip(datasets, defaults):
        print('~'*1000)
        print(e, " iteration of 4x dataset:", dataset)
        print('~'*100)
        
        for s in [["race","sex"]]: #  ["sex","race"],["race"],["sex"]
            if dataset in [Tester.GERMAN_D, Tester.DEFAULT_D] and "race" in s:
                continue
            
            
            mls = list(itertools.chain.from_iterable([
                ml_configs(model, all_losses, s, default_values[0], default_values[1], default_values[2], default_values[3], loss_params, baselines=baselines)
                for model in models
            ]))
            try_test(results_filename, s, dataset, metrics, mls, n_repetitions=4)




models = [ Model.SV_C, Model.NB_C, Model.RF_C, Model.DT_R,]

for e in range(10):
    for dataset, default_values in zip(datasets, defaults):
        print('~'*1000)
        print(e, " iteration of 3x dataset:", dataset)
        print('~'*100)
        
        for s in [["race","sex"]]: #  ["sex","race"],["race"],["sex"]
            if dataset in [Tester.GERMAN_D, Tester.DEFAULT_D] and "race" in s:
                continue
            
            
            mls = list(itertools.chain.from_iterable([
                ml_configs(model, all_losses, s, default_values[0], default_values[1], default_values[2], default_values[3], loss_params, baselines=baselines)
                for model in models
            ]))
            try_test(results_filename, s, dataset, metrics, mls, n_repetitions=4)

