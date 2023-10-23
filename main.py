import os
from src import *

# Just an example for now

n_repetitions = 5 
same_data_split = False
results_filename = "ehhhheeeee"
other = {Tester.OPT_SAVE_INTERMID: False}

#other_fb = other.copy()
#other_fb[BaseModel.OPT_FBALANCE] = True

datasets =  [ Tester.ADULT_D, Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,]

base_it = 300
def other_it(it_2):
    return {"iter_1": base_it - it_2, "iter_2": int(it_2*1.5)}

mls = [
       (Tester.BASE_ML, Model.NN_C, None, None, other | {"iter_1": base_it}), 
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(498)),
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(470)),
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(400)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(300)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(200)),
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(100)),
       (Tester.FYP, Model.NN_C, None, None, other | other_it(50)),
       #(Tester.FYP, Model.NN_C, None, None, other | other_it(10)),
       
       (Tester.FAIRMASK, Model.NN_C, Model.DT_R, None, other | {"iter_1": base_it}),
       (Tester.REWEIGHING, Model.NN_C, None, None, other | {"iter_1": base_it})    
]
metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    tester = Tester(results_file)
    for dataset in datasets:
        for bias_mit, method, method2, pre, oth in mls:
            print("RUNIN",bias_mit,oth )
            tester.run_test(metric_names, dataset, bias_mit, method, method2, n_repetitions, same_data_split, data_preprocessing=pre, other = oth, sensitive_attr=['sex'])