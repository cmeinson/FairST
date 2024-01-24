from .ml_interface import Model
from typing import List, Dict, Any
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

from aif360.datasets import BinaryLabelDataset

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing

import numpy as np
from ..utils import TestConfig



class EqOModel(Model):
    def __init__(self, config: TestConfig) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)
        self._TR_model = None
     

     


    def predict(self, X: pd.DataFrame) -> np.array:
        
        copied_df = X.copy().reset_index(drop=True)

             # Assuming 'sensitive_attr' is the column representing sensitive attributes
        dataset = BinaryLabelDataset(
            df=pd.concat([copied_df, pd.DataFrame(y, columns=["label"])], axis=1),
            label_names=["label"],
            protected_attribute_names=self._config.sensitive_attr,
        )

        privileged_groups = {attr: 1 for attr in self._config.sensitive_attr}
        unprivileged_groups = {attr: 0 for attr in self._config.sensitive_attr}
        
        # cost constraint of fnr will optimize generalized false negative rates, that of
        # fpr will optimize generalized false positive rates, and weighted will optimize
        # a weighted combination of both
        cost_constraint = "weighted" # "fnr", "fpr", "weighted"

        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint)
        
        
        
        #preds = self._model.predict(dataset_transf.features)
        
        #cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        
        

        self._TR_model = self._TR_model.fit(dataset, maxiter=500, maxfun=5000)

        dataset_transf = self._TR_model.transform(dataset)
        

        dataset_transf = self._TR_model.transform(dataset)
        preds = self._model.predict(dataset_transf.features)
        return self._binarise(preds)
