from os import path
import pandas as pd
import numpy as np
import collections
from typing import List, Dict, Any

from .ml_models import FairBalanceModel, FairMaskModel, VAEMaskModel, BaseModel, ReweighingModel, Model
from .data_classes import AdultData, CompasData, GermanData, MEPSData, DummyData, Data
from .metrics import Metrics, MetricException
from .utils import TestConfig
from .result_writer import ResultsWriter

MAX_RETRY = 15


class Tester:
    # Avalable datasets for testing:
    ADULT_D = "Adult Dataset"
    COMPAS_D = "Compas Dataset"
    MEPS_D = "MEPS Dataset"
    GERMAN_D = "German Dataset"
    DUMMY_D = "Dummy Dataset"

    # Available bias mitigation methods
    FAIRBALANCE = "FairBalance Bias Mitigation"
    FAIRMASK = "FairMask Bias Mitigation"
    FYP_VAE = "FYP VAE"
    REWEIGHING = "Reweighing Bias Mitigation"
    BASE_ML = "No Bias Mitigation"

    def __init__(self, file_name: str, dataset_name: str, metric_names: List[str]) -> None:
        # same METRICS, FILENAME, DATASET
        self._exceptions = []
        self._initd_data = {} 
        self._base_models = {} # models with no bias mitigation should be separate for each data preproc and renewes for each datat split

        self._dataset_name = dataset_name
        self._metric_names = metric_names
        self._results_writer = ResultsWriter(file_name, dataset_name)
        
        self._preds = None #????

    def run_tests(self, test_configs: List[TestConfig], repetitions = 1, save_intermid_results=False):    
        if repetitions == 1:
            save_intermid_results=False

        self._check_sensitive_attributes_and_init_datasets(test_configs)

        # for reps
        for _ in range(repetitions):
            # RE TRAIN THE POST PROC no bias mit BASE MODELS! PASS IT INTO THE MODELS ON INIT. save it in self.
            #self._train_base_models(test_configs)
            self._base_models = {} # set to none and train when needed by _get_model

            # run each test config 
            for conf in test_configs:
                #print("run conf", conf)
                self._run_test(conf, save_intermid_results)

            # only update the data split for each data in self._initd_data dict. (the first it will have the same default)
            # -> the data split is only the same if the preproc is also the same (otherwise not comparable) 
            self._update_all_data_splits()
        
        # print all accumulated evals to file
        self._results_writer.write_final_results()
        

    def _run_test(self, config: TestConfig, save_intermid: bool):
        retrys = 0
        model = self._get_model(config)
        data = self._get_dataset(config.preprocessing)

        # RUNNING FOR ONE REP
        # train model
        # tetst model
        # TODO probs change it along with the data class
        while True:
            X, y = data.get_train_data()
            model.fit(X, y)

            X, y = data.get_test_data()
            predict = lambda x: model.predict(x.copy()) # used just for fr
            self._preds = predict(X)
            try:
                evals = self._evaluate(Metrics(X, y, self._preds, predict), config.sensitive_attr)
            except MetricException as e:
                print("invalid metric, attmpt nr:", retrys, e)
                retrys += 1
                if retrys == MAX_RETRY:
                    break
                #   TODO: should i write something to file abt the fail or at least print
            else: 
                self._results_writer.add_result(config, evals, save_intermid)
                break
        

    def get_last_run_preds(self): 
        """for debugging. only really useful when running just a single config and a single rep"""
        return self._preds

    def get_exceptions(self):
        """for debugging: get exceptions from the last run"""
        return self._exceptions

    #def get_last_data_split(self):
    #    """for debugging"""
    #    return *self._data.get_test_data(), *self._data.get_train_data()

    def _evaluate(self, metrics: Metrics, sensitive_attr):
        evals = {}
        for name in self._metric_names:
            try:
                if name in Metrics.get_subgroup_dependant():
                    evals[name] = (metrics.get(name, sensitive_attr))
                elif name in Metrics.get_attribute_dependant():
                    for attr in sensitive_attr:
                        evals[attr + '|' + name] = (metrics.get(name, attr))
                else:
                    evals[name] = (metrics.get(name))
            except MetricException as e:
                self._exceptions.append([e,name])
                raise e
        return evals

    def _get_dataset(self, preproc:str) -> Data:
        # allows for diff preproc
        dataset_descr = self._dataset_name if not preproc else self._dataset_name+preproc
        if dataset_descr in self._initd_data:
            return self._initd_data[dataset_descr]

        data = None
        if self._dataset_name == self.ADULT_D:
            data = AdultData(preproc)
        elif self._dataset_name == self.COMPAS_D:
            data = CompasData(preproc)
        elif self._dataset_name == self.MEPS_D:
            data = MEPSData(preproc)
        elif self._dataset_name == self.GERMAN_D:
            data = GermanData(preproc)
        elif self._dataset_name == self.DUMMY_D:
            data = DummyData(preproc)
        else:
            raise RuntimeError("Incorrect dataset name ", self._dataset_name)

        self._initd_data[dataset_descr] = data
        return data

    def _get_model(self, config: TestConfig) -> Model:
        name = config.bias_mit
        if name == self.BASE_ML:
            return self._get_base_model(config)
        elif name == self.FAIRBALANCE:
            return FairBalanceModel(config)
        elif name == self.REWEIGHING:
            return ReweighingModel(config)
        elif name == self.FAIRMASK:
            return FairMaskModel(config, self._get_base_model(config))
        elif name == self.FYP_VAE:
            return VAEMaskModel(config, self._get_base_model(config))
        else:
            raise RuntimeError("Incorrect method name ", name)
        
    def _get_base_model(self, config: TestConfig) -> Model:
        # return if previously initd
        preproc = config.preprocessing
        base_descr = config.ml_method if not preproc else config.ml_method + preproc
        
        if base_descr not in self._base_models:
            # train a base model with no bias mit (unique for every prproc+ml_method)
            X, y = self._get_dataset(config.preprocessing).get_train_data()
            model =  BaseModel(config)
            model.fit(X, y)
            self._base_models[base_descr] = model

        return self._base_models[base_descr]
        
    def _check_sensitive_attributes_and_init_datasets(self, test_configs: List[TestConfig]):
        """ensurest that a dataset is initialised per each used preproc method
        and that configs with attributes=None are replaced with all the possible columns from the dataset"""
        for conf in test_configs:
            data = self._get_dataset(conf.preprocessing)
            if not conf.sensitive_attr:
                conf.sensitive_attr = data.get_sensitive_column_names()

    def _update_all_data_splits(self):
        for _, data in self._initd_data.items():
            data.new_data_split()

