import os
from src import *

# Just an example for now

n_repetitions = 1
results_filename = "refactor"
other = {}

datasets = []#  [Tester.ADULT_D,  Tester.COMPAS_D]#, Tester.GERMAN_D, Tester.ADULT_D,], Tester.COMPAS_D, 



mls = [
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["sex"]),   
    TestConfig(Tester.FAIRMASK, Model.RF_C, Model.DT_R, sensitive_attr=["sex"]),
    TestConfig(Tester.REWEIGHING, Model.RF_C, sensitive_attr=["sex"]),
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["sex"]),   
    TestConfig(Tester.BASE_ML, Model.RF_C, sensitive_attr=["race"], other={"sdf":11})
    ]


metric_names = [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.SF]
results_file = os.path.join("results",results_filename +".csv")


if __name__ == "__main__":
    for dataset in datasets:
        tester = Tester(results_file, dataset, metric_names)
        tester.run_tests(mls, n_repetitions, save_intermid_results=False)


    import pandas as pd

    # Create a sample DataFrame
    data = {
        'Attr1': [0, 1, 0, 1, 0, 1, 0, 1, 1],
        'Attr2': [1, 1, 0, 0, 1, 1, 0, 0, 1],
        'Value': [10, 20, 30, 40, 50, 60, 70, 80,2]
    }

    df = pd.DataFrame(data)

    # Group by binary attributes and get indices for each subgroup
    subgroup_indices = df.groupby(['Attr1', 'Attr2']).apply(lambda x: x.index.tolist()).tolist()

    # Display the result
    print(subgroup_indices.tolist())