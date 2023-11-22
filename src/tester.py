from os import path
import pandas as pd
import numpy as np
import collections
from .data_interface import Data, DummyData
from .adult_data import AdultData
from .compas_data import CompasData
from .adult_data import AdultData
from .meps_data import MEPSData
from .german_data import GermanData
from .ml_interface import Model, BaseModel
from .metrics import Metrics, MetricException
from typing import List, Dict, Any
from .fairbalance import FairBalanceModel
from .fairmask import FairMaskModel
from .FYP import FypModel
from .fyp_vae import FypMaskModel
from .reweighing import ReweighingModel
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
        
        self._data = None
        self._preds = None #????
        self._evals = None #????

    def run_tests(self, test_configs: List[TestConfig], repetitions = 1, save_intermid_results=False):        
        # for reps
            # new data split
            # RE TRAIN THE POST PROC no bias mit BASE MODEL! PASS IT INTO THE MODELS ON INIT. save it in self.
            # run each test config with the data split

            # only update the data split for each data in self._initd_data dict. (the first it will have the same default)
            # -> the data split is only the same if the preproc is also the same (otherwise not comparable) 
        
        #print all accumulated evals to file
        pass

    def _run_test(self):
        #TODO THOINK OF MAKS RETRYS

        # RUNNING FOR ONE REP
        # train model
        # tetst model
        # while True:
            # try:
            #   evaluate metrics
            # except:
            #   save excemption
            # else: save evals and break
        # if needed print last evals to file
        pass

    def get_last_run_preds(self): 
        """for debugging"""
        return self._preds

    def get_exceptions(self):
        """for debugging: get exceptions from the last run"""
        return self._exceptions


    def get_last_data_split(self):
        """for debugging"""
        return *self._data.get_test_data(), *self._data.get_train_data()

    def _evaluate(self, metrics: Metrics, metric_names: List[str], sensitive_attr):
        evals = {}
        for name in metric_names:
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

    def _get_dataset(self, name:str, preprocessing:str) -> Data:
        # allows for diff preproc
        dataset_description = name if not preprocessing else name+preprocessing
        if dataset_description in self._initd_data:
            return self._initd_data[dataset_description]

        data = None
        if name == self.ADULT_D:
            data = AdultData(preprocessing)
        elif name == self.COMPAS_D:
            data = CompasData(preprocessing)
        elif name == self.MEPS_D:
            data = MEPSData(preprocessing)
        elif name == self.GERMAN_D:
            data = GermanData(preprocessing)
        elif name == self.DUMMY_D:
            data = DummyData(preprocessing)
        else:
            raise RuntimeError("Incorrect dataset name ", name)

        self._initd_data[dataset_description] = data
        return data

    def _get_model(self, name, other) -> Model:
        if name == self.FAIRMASK:
            return FairMaskModel(other)
        elif name == self.FAIRBALANCE:
            return FairBalanceModel(other)
        elif name == self.BASE_ML:
            return BaseModel(other)
        elif name == self.REWEIGHING:
            return ReweighingModel(other)
        elif name == self.FYP:
            return FypModel(other)
        elif name == self.FYP_VAE:
            return FypMaskModel(other)
        else:
            raise RuntimeError("Incorrect method name ", name)

