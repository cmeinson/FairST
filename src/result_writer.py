
from typing import Dict, List
import pandas as pd
from .utils import TestConfig
import numpy as np
from os import path
from .ml_models import  VAEMaskModel

ALTERNATIVE_FILE_NAME_END = "_new"

class ResultsWriter:
    # accumulate all evals and print into file
    def __init__(self, file_name: str, dataset_name: str) -> None:
        # for each config we store a dict of all metrics with all previous results
        self._evals_per_metric_per_config = {}
        self._file_name = file_name
        self._dataset_name = dataset_name

    def add_result(self, test_config, evals, write_to_file = False):
        if write_to_file:
            self._write_evals_to_file(test_config, evals)
        
        if test_config not in self._evals_per_metric_per_config:
            self._evals_per_metric_per_config[test_config] = {key: [evals[key]] for key in evals}
        else:
            for metric in evals:
                self._evals_per_metric_per_config[test_config][metric].append(evals[metric])  

    def write_final_results(self):
        for config, evals in self._evals_per_metric_per_config.items():
            self._write_evals_to_file(config, evals)    
    
    def _write_evals_to_file(self, test_config: TestConfig, evals: Dict[str, List[float]]):
        reps = np.size(list(evals.values())[0])

        entry = {
            "time": [pd.Timestamp.now()],
            "reps": [reps],
            "data": [self._dataset_name],
            "bias mitigation": [test_config.bias_mit],
            "ML method": [test_config.ml_method],
            "bias mit ML method": [test_config.bias_ml_method],
            "sensitive attrs": [test_config.sensitive_attr],
            "other": [test_config.other],
        }
        entry.update({key: [np.average(evals[key])] for key in evals})
        entry.update({"VAR|" + key: [np.var(evals[key])] for key in evals})

        res = pd.DataFrame(entry)
        self._append_to_file(res)

    def _append_to_file(self, res):
        try:
            res_new = res
            if path.exists(self._file_name):
                res_new = pd.concat([res, pd.read_csv(self._file_name)], ignore_index=True)
            res_new.to_csv(self._file_name, index=False)
        except IOError as e:
            root, extension = path.splitext(self._file_name)
            self._file_name = root + ALTERNATIVE_FILE_NAME_END + extension
            print(e)
            print("RENAMING OUTPUT FILE TO:", self._file_name)
            self._append_to_file(res)
            




    