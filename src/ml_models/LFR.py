from .ml_interface import Model
import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import LFR

from ..test_config import TestConfig


class LFRModel(Model):
    def __init__(self, config: TestConfig) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)
        self._TR_model = None

    def fit(self, X: pd.DataFrame, y: np.array):
        """Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array

        """

        if len(self._config.sensitive_attr) > 1:
            print(
                "!!!!!!!!!!!!!!!!!!!! Multiple sensitive attributes not supported for LFR",
                self._config.sensitive_attr,
            )
            return

        copied_df = X.copy().reset_index(drop=True)

        # Assuming 'sensitive_attr' is the column representing sensitive attributes
        dataset = BinaryLabelDataset(
            df=pd.concat([copied_df, pd.DataFrame(y, columns=["label"])], axis=1),
            label_names=["label"],
            protected_attribute_names=self._config.sensitive_attr,
        )

        privileged_groups = [{attr: 1 for attr in self._config.sensitive_attr}]
        unprivileged_groups = [{attr: 0 for attr in self._config.sensitive_attr}]

        self._TR_model = LFR(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            k=10,
            Ax=0.1,
            Ay=1.0,
            Az=2.0,
            verbose=1,
        )
        self._TR_model = self._TR_model.fit(dataset, maxiter=500, maxfun=5000)

        dataset_transf = self._TR_model.transform(dataset)

        self._model = self._get_model()

        self._fit(self._model, dataset_transf.features, y)

    def predict(self, X: pd.DataFrame, binary=True) -> np.array:
        if len(self._config.sensitive_attr) > 1:
            print(
                "!!!!!!!!!!!!!!!!!!!! Multiple sensitive attributes not supported for LFR",
                self._config.sensitive_attr,
            )
            return np.zeros(len(X))

        copied_df = X.copy().reset_index(drop=True)
        y = np.zeros(len(X))

        dataset = BinaryLabelDataset(
            df=pd.concat([copied_df, pd.DataFrame(y, columns=["label"])], axis=1),
            label_names=["label"],
            protected_attribute_names=self._config.sensitive_attr,
        )

        dataset_transf = self._TR_model.transform(dataset)
        return self._predict(self._model, dataset_transf.features, binary=binary)
