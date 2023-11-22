import os
from src_old import *

# Just an example for now

n_repetitions = 3
same_data_split = False
results_filename = "vaeeeeer"
other = {Tester.OPT_SAVE_INTERMID: True}

#other_fb = other.copy()
#other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 

base_it = 100
def other_it(it_2, RW = False):
    return {"iter_1": base_it - it_2, "iter_2": int(it_2*1.2), "RW": RW}

mls = [
    (Tester.FYP_VAE, Model.LG_R, None, None, other),
    (Tester.BASE_ML, Model.LG_R, None, None, other | {"iter_1": base_it}),   
    (Tester.FAIRMASK, Model.LG_R, Model.DT_R, None, other | {"iter_1": base_it}),
    (Tester.REWEIGHING, Model.LG_R, None, None, other | {"iter_1": base_it}),]

mls = [
       #(Tester.FYP_VAE, Model.NN_old, None, None, other| {"iter_1": base_it, 'mask_epoch': 1000}),
       (Tester.FYP_VAE, Model.NN_old, None, None, other| {"iter_1": base_it, 'mask_epoch': 4000}),
       (Tester.FYP_VAE, Model.NN_old, None, None, other| {"iter_1": base_it, 'mask_epoch': 1000}),
       (Tester.BASE_ML, Model.NN_old, None, None, other | {"iter_1": base_it}), 
       
       (Tester.FAIRMASK, Model.NN_old, Model.DT_R, None, other | {"iter_1": base_it}),
#       (Tester.REWEIGHING, Model.NN_old, None, None, other | {"iter_1": base_it}),
#       (Tester.FAIRBALANCE, Model.NN_old, None, None, other | {"iter_1": base_it}),

       (Tester.FYP_VAE, Model.LG_R, None, None, other| {"iter_1": base_it, 'mask_epoch': 4000}),
       (Tester.FYP_VAE, Model.LG_R, None, None, other| {"iter_1": base_it, 'mask_epoch': 1000}),
       (Tester.BASE_ML, Model.LG_R, None, None, other | {"iter_1": base_it}), 
       
       (Tester.FAIRMASK, Model.LG_R, Model.DT_R, None, other | {"iter_1": base_it}),
       
 
]




metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for i in range(1):
        for dataset in datasets:
            for bias_mit, method, method2, pre, oth in mls:
                print("RUNIN",bias_mit,oth )
                tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth, sensitive_attr=['race'])