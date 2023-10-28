import os
from src import *

# Just an example for now

n_repetitions = 5
same_data_split = False
results_filename = "rw_all_tensor_sex_same_data"
other = {Tester.OPT_SAVE_INTERMID: True}

#other_fb = other.copy()
#other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [  Tester.ADULT_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 

base_it = 70
def other_it(it_2, RW = False):
    return {"iter_1": base_it - it_2, "iter_2": int(it_2*1.2), "RW": RW}

mls = [
    
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(65, True)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(45, True)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(25, True)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(10, True)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(5, True)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(0, True)),

       #(Tester.FYP, Model.NN_C, None, None, other | other_it(65)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(45)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(25)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(10)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(5)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(0)),

       (Tester.BASE_ML, Model.NN_C, None, None, other | {"iter_1": base_it}), 
       
       (Tester.FAIRMASK, Model.NN_old, Model.DT_R, None, other | {"iter_1": base_it}),
       (Tester.REWEIGHING, Model.NN_C, None, None, other | {"iter_1": base_it}),
       (Tester.FAIRBALANCE, Model.NN_C, None, None, other | {"iter_1": base_it}),
       
       (Tester.BASE_ML, Model.NB_C, None, None, other | {"iter_1": base_it}),   
       (Tester.FAIRMASK, Model.NB_C, Model.DT_R, None, other | {"iter_1": base_it}),
       (Tester.REWEIGHING, Model.NB_C, None, None, other | {"iter_1": base_it}),
       (Tester.FAIRBALANCE, Model.NB_C, None, None, other | {"iter_1": base_it})    
]


metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for i in range(3):
        for dataset in datasets:
            for bias_mit, method, method2, pre, oth in mls:
                print("RUNIN",bias_mit,oth )
                tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth, sensitive_attr=['sex'])