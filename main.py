import os
from src import *

# Just an example for now

n_repetitions = 1
results_filename = "refactor"
other = {}

datasets =  [Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 



mls = [
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["sex"]),   
    TestConfig(Tester.FAIRMASK, Model.RF_C, Model.DT_R, sensitive_attr=["sex"]),
    TestConfig(Tester.REWEIGHING, Model.RF_C, sensitive_attr=["sex"]),
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["sex"]),   
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["race"])
    ]


metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    for dataset in datasets:
        tester = Tester(results_file, dataset, metric_names)
        tester.run_tests(mls, n_repetitions, save_intermid_results=False)