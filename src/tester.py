from typing import List

from .ml_models import (
    FairBalanceModel,
    FairMaskModel,
    VAEMaskModel,
    BaseModel,
    ReweighingModel,
    Model,
    LFRModel,
)
from .data_classes import (
    AdultData,
    CompasData,
    GermanData,
    MEPSData,
    DummyData,
    Data,
    DefaultData,
)
from .metrics import Metrics, MetricException
from .test_config import TestConfig
from .result_writer import ResultsWriter

MAX_RETRY = 15


class Tester:
    # Avalable datasets for testing:
    ADULT_D = "Adult Dataset"
    COMPAS_D = "Compas Dataset"
    MEPS_D = "MEPS Dataset"
    DEFAULT_D = "Default Dataset"
    GERMAN_D = "German Dataset"
    DUMMY_D = "Dummy Dataset"

    # Available bias mitigation methods
    FAIRBALANCE = "FairBalance Bias Mitigation"
    FAIRMASK = "FairMask Bias Mitigation"
    FYP_VAE = "FYP VAE"
    LFR = "LFR"
    REWEIGHING = "Reweighing Bias Mitigation"
    BASE_ML = "No Bias Mitigation"

    POST_PROC_METHODS = [FAIRMASK, FYP_VAE]

    def __init__(
        self, file_name: str, dataset_name: str, metric_names: List[str]
    ) -> None:
        # same METRICS, FILENAME, DATASET to be used for one tester instance
        self._exceptions = []
        self._initd_data = {}
        self._base_models = (
            {}
        )  # models with no bias mitigation should be separate for each data preproc and renewes for each datat split
        self._fyp_vae_models = (
            {}
        )  # if 2 instances of my model are used that differ only in base ml method, it can be reused!

        self._dataset_name = dataset_name
        self._metric_names = metric_names
        self._results_writer = ResultsWriter(file_name, dataset_name)

        self._preds = None 
        self._rep_proba_preds = None
        self._last_test_data = None

    def run_tests(
        self, test_configs: List[TestConfig], repetitions=1, save_intermid_results=False
    ):
        if repetitions == 1:
            save_intermid_results = False

        self._check_sensitive_attributes_and_init_datasets(test_configs)

        # for reps
        for _ in range(repetitions):
            # RE TRAIN THE POST PROC no bias mit BASE MODELS! PASS IT INTO THE MODELS ON INIT. save it in self.
            self._base_models = {}  # set to none and train when needed by _get_model
            self._fyp_vae_models = {}
            self._rep_proba_preds = []

            self._results_writer.change_id()
            # run each test config
            for conf in test_configs:
                print("run", conf)
                self._run_test(conf, save_intermid_results)

            # only update the data split for each data in self._initd_data dict. (the first it will have the same default)
            # -> the data split is only the same if the preproc is also the same (otherwise not comparable)
            self._update_all_data_splits()

        self._results_writer.change_id()
        # print all accumulated evals to file
        self._results_writer.write_final_results()

    def _run_test(self, config: TestConfig, save_intermid: bool):
        retrys = 0
        data = self._get_dataset(config.preprocessing)
        config.n_cols = data.n_cols
        model = self._get_model(config)

        while True:
            X, y = data.get_train_data()
            model.fit(X, y)

            X, y = data.get_test_data()
            predict = lambda x: model.predict(
                x.copy()
            )  # used just for fr metric
            self._preds = predict(X)
            self._rep_proba_preds.append(model.predict(X.copy(), binary=False))
            self._last_test_data = (X, y)
            try:
                evals = self._evaluate(
                    Metrics(X, y, self._preds, predict), config.sensitive_attr
                )
            except MetricException as e:
                print("invalid metric, attmpt nr:", retrys, e)
                retrys += 1
                if retrys == MAX_RETRY:
                    break
                
            else:
                print("adding result")
                self._results_writer.add_result(config, evals, save_intermid)
                break

    def get_last_run(self):
        """for debugging. only really useful when running just a single rep"""
        return self._rep_proba_preds, *self._last_test_data

    def get_exceptions(self):
        """for debugging: get exceptions from the last run"""
        return self._exceptions

    def _evaluate(self, metrics: Metrics, sensitive_attr):
        evals = {}
        for name in self._metric_names:
            try:
                if name in Metrics.get_subgroup_dependant():
                    evals[name] = metrics.get(name, sensitive_attr)
                elif name in Metrics.get_attribute_dependant():
                    for attr in sensitive_attr:
                        evals[attr + "|" + name] = metrics.get(name, attr)
                else:
                    evals[name] = metrics.get(name)
            except MetricException as e:
                self._exceptions.append([e, name])
                raise e
        return evals

    def _get_dataset(self, preproc: str) -> Data:
        # allows for diff preproc
        dataset_descr = (
            self._dataset_name if not preproc else self._dataset_name + preproc
        )
        if dataset_descr in self._initd_data:
            return self._initd_data[dataset_descr]

        data = None
        if self._dataset_name == self.ADULT_D:
            data = AdultData(preproc)
        elif self._dataset_name == self.COMPAS_D:
            data = CompasData(preproc)
        elif self._dataset_name == self.MEPS_D:
            data = MEPSData(preproc)
        elif self._dataset_name == self.DEFAULT_D:
            data = DefaultData(preproc)
        elif self._dataset_name == self.GERMAN_D:
            data = GermanData(preproc)
        elif self._dataset_name == self.DUMMY_D:
            data = DummyData(preproc)
        else:
            raise RuntimeError("Incorrect dataset name ", self._dataset_name)

        self._initd_data[dataset_descr] = data
        return data

    def _get_model(self, config: TestConfig) -> Model:
        # if the same model previously initialized (likely for use as a base model for post proc)
        # if 2 configs differ only in ml_method then can REUSE my method

        descr = self._get_model_descr(config)
        if descr in self._base_models:
            return self._base_models[descr]

        name = config.bias_mit
        if name is None or name == self.BASE_ML:
            return BaseModel(config)
        elif name == self.FAIRBALANCE:
            return FairBalanceModel(config)
        elif name == self.REWEIGHING:
            return ReweighingModel(config)
        elif name == self.FAIRMASK:
            return FairMaskModel(config, self._get_base_model(config))
        elif name == self.LFR:
            return LFRModel(config)
        elif name == self.FYP_VAE:
            custom_hash = config.get_hash_without_ml_method()
            data = self._get_dataset(config.preprocessing)
            model = VAEMaskModel(
                config, self._get_base_model(config), data.post_mask_transform
            )

            if custom_hash in self._fyp_vae_models:
                fitted_model = self._fyp_vae_models[custom_hash]
                model._has_been_fitted = True
                model._mask_model = fitted_model._mask_model
            else:
                self._fyp_vae_models[custom_hash] = model

            return model
        else:
            raise RuntimeError("Incorrect method name ", name)

    def _get_base_model(self, config: TestConfig) -> Model:
        base_descr = self._get_model_descr(config.get_base_model_config())
        self._check_base_model(config)

        if base_descr not in self._base_models:
            # train a base model with no bias mit (unique for every preproc+ml_method)
            X, y = self._get_dataset(config.preprocessing).get_train_data()
            model = self._get_model(config.get_base_model_config())
            model.fit(X, y)
            self._base_models[base_descr] = model

        return self._base_models[base_descr]

    def _check_base_model(self, config: TestConfig):
        # check that the base model is not post-proc
        if config.base_model_bias_mit in self.POST_PROC_METHODS:
            raise RuntimeError(
                "Post processing method used as a base model for another", config
            )

    def _get_model_descr(self, config: TestConfig) -> str:
        model = config.bias_mit if config.bias_mit else self.BASE_ML
        preproc = "" if not config.preprocessing else config.preprocessing
        return "-".join([config.ml_method, preproc, model])

    def _check_sensitive_attributes_and_init_datasets(
        self, test_configs: List[TestConfig]
    ):
        """ensures that a dataset is initialised per each used preproc method
        and that configs with attributes=None are replaced with all the possible columns from the dataset
        """
        for conf in test_configs:
            data = self._get_dataset(conf.preprocessing)
            if conf.sensitive_attr is None:
                conf.sensitive_attr = data.get_sensitive_column_names()

    def _update_all_data_splits(self):
        for _, data in self._initd_data.items():
            data.new_data_split()
