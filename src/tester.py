from os import path
import pandas as pd
import numpy as np
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
from .reweighing import ReweighingModel

class Tester:
    OPT_SAVE_INTERMID = "save intermediate results to file"

    # Avalable datasets for testing:
    ADULT_D = "Adult Dataset"
    COMPAS_D = "Compas Dataset"
    MEPS_D = "MEPS Dataset"
    GERMAN_D = "German Dataset"
    DUMMY_D = "Dummy Dataset"

    # Available bias mitigation methods
    FAIRBALANCE = "FairBalance Bias Mitigation"
    FAIRMASK = "FairMask Bias Mitigation"
    FYP = "idk i am trying fyp"
    REWEIGHING = "Reweighing Bias Mitigation"
    BASE_ML = "No Bias Mitigation"

    def __init__(self, output_file) -> None:
        self._exceptions = []
        self._initd_data = {}
        self._file = output_file
        self._preds = None
        self._evals = None
        self._data = None

    def run_test(self, metric_names: List[str], dataset: str, 
                 bias_mit: str, ml_method: str, bias_ml_method: str = None, 
                 repetitions = 1, same_data_split = False,
                 data_preprocessing: str = None, sensitive_attr: List[str] = None, other={}):        
        """Runs the experiments and saves the results into the file given on initialization.
        All the experiments are run with the same data set instances and therefore with the same data splits.

        :param metric_names: names of the metrics to use in evaluation.
        :type metric_names: List[str]
        :param dataset: name of the dataset
        :type dataset: str
        :param bias_mit: nam of the bias mitigation method
        :type bias_mit: str
        :param ml_method: name of the ML method used to preict the final outcom
        :type ml_method: str
        :param bias_ml_method: name of the ML method for bias mitigation, defaults to None
        :type bias_ml_method: str, optional
        :param repetitions: nr of repetitions of the experiment, defaults to 1
        :type repetitions: int, optional
        :param same_data_split: when set to False, each repetition of the experiment will be done with a new data split, defaults to False
        :type same_data_split: bool, optional
        :param data_preprocessing: name of the data preprocessing method, defaults to None
        :type data_preprocessing: str, optional
        :param sensitive_attr: list of attributes to be proteced if different from the default one in the dataset, defaults to None
        :type sensitive_attr: List[str], optional
        :param other: any other params described below, defaults to {}
        :type other: dict, optional
            "other" params:
                "save_intermediate" - if set to True saves to file not only the mean results of all runs but also each intermediate result

        :return: Used testing X and y, along with all the predictions
        :rtype: pd.DataFrame, np.array, List[np.array]
        """
        self._exceptions = []
        model = self._get_model(bias_mit, other)
        self._data = self._get_dataset(dataset,data_preprocessing)

        if not sensitive_attr:
            sensitive_attr = self._data.get_sensitive_column_names()

        self._preds = []
        self._evals = None

        rep = 0
        while (rep < repetitions):
            if not same_data_split: self._data.new_data_split()
            
            X, y = self._data.get_train_data()
            model.train(X, y, sensitive_attr, ml_method, bias_ml_method, other)

            X, y = self._data.get_test_data()
            predict = lambda x: model.predict(x.copy(), other)
            rep_preds = predict(X)
            try:
                evals = self._evaluate(Metrics(X, y, rep_preds, predict), metric_names, sensitive_attr)
            except MetricException as e:
                print("invalid metric on rep ", rep,  e)
            else:
                rep+=1

                self._acc_evals(evals)
                self._preds.append(rep_preds)

                if repetitions==1 or (self.OPT_SAVE_INTERMID in other and other[self.OPT_SAVE_INTERMID]):
                    self.save_test_results(evals, dataset, bias_mit, ml_method, bias_ml_method, sensitive_attr, same_data_split, other)

        if repetitions!=1:
            self.save_test_results(self._evals, dataset, bias_mit, ml_method, bias_ml_method, sensitive_attr, same_data_split, other)

    def get_last_run_preds(self): 
        return self._preds

    def get_exceptions(self):
        return self._exceptions

    def get_last_mean_evals(self):
        return {key: [np.average(self._evals[key])] for key in self._evals}

    def get_last_data_split(self):
        # in case neded for debugging:)
        return *self._data.get_test_data(), *self._data.get_train_data()
    
    def _acc_evals(self, evals):
        if self._evals is None:
            self._evals = {key:[val] for (key,val) in evals.items()}
        else:
            for (key, val) in evals.items():
                self._evals[key].append(val)

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
        else:
            raise RuntimeError("Incorrect method name ", name)

    def save_test_results(self, evals: Dict[str, Any], dataset: str,
                          bias_mit: str, ml_method: str, bias_ml_method: str,
                          sensitive_attr: List[str], same_data_split, other):
        nr_samples = np.size(list(evals.values())[0])

        entry = {
            "timestamp": [pd.Timestamp.now()],
            "nr samples": [nr_samples],
            "same data split": [same_data_split],
            "Dataset": [dataset],
            "Bias Mitigation": [bias_mit],
            "ML method": [ml_method],
            "ML bias mit": [bias_ml_method],
            "Sensitive attrs": [sensitive_attr],
            "other": [other],
        }

        entry.update({key: [np.average(evals[key])] for key in evals})
        entry.update({"VAR|"+key: [np.var(evals[key])] for key in evals})
        res = pd.DataFrame(entry)


        if self._file and path.exists(self._file):
            res = pd.concat([res, pd.read_csv(self._file)], ignore_index=True)
        res.to_csv(self._file, index=False)
