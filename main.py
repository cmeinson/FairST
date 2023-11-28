import os
from src import *

# Just an example for now

n_repetitions = 3
results_filename = "refactor"
other = {}

datasets =  [Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 



mls = [
    TestConfig(Tester.BASE_ML, Model.LG_R, sensitive_attr=["sex"]),   
    TestConfig(Tester.FAIRMASK, Model.LG_R, Model.DT_R, sensitive_attr=["sex"]),
    TestConfig(Tester.REWEIGHING, Model.LG_R, sensitive_attr=["sex"])]


metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    for dataset in datasets:
        tester = Tester(results_file, dataset, metric_names)
        tester.run_tests(mls, n_repetitions, save_intermid_results=True)