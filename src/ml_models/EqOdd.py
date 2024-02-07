from .ml_interface import Model
from typing import List, Dict, Any
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

from aif360.datasets import BinaryLabelDataset

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing

import numpy as np
from sklearn.model_selection import train_test_split

from ..utils import TestConfig



class EqOModel(Model):
    def __init__(self, config: TestConfig, base_model: Model = None) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        super().__init__(config)
        self._calibrator = None
        self._use_base_model = True
        self._model = base_model
        self._trainset_model = None
        if base_model is None:
            self._use_base_model = False
        
    def fit(self, X: pd.DataFrame, y: np.array):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
        X_val = X_val.reset_index(drop=True)

        self._trainset_model = self._get_model()
        self._trainset_model.fit(X_train, y_train)
        
        unprivileged_groups = [{attr: 1 for attr in self._config.sensitive_attr}]
        privileged_groups = [{attr: 0 for attr in self._config.sensitive_attr}]
        
        
        val_dataset = BinaryLabelDataset(
            df=pd.concat([X_val, pd.DataFrame(y_val, columns=["label"])], axis=1),
            label_names=["label"],
            protected_attribute_names=self._config.sensitive_attr,
        )
        
        
        preds = self._trainset_model.predict(X_val)

        predictions_val_df = pd.DataFrame(data={'label': preds})
        val_dataset_pred = BinaryLabelDataset(df=pd.concat([X_val, predictions_val_df], axis=1), label_names=['label'], protected_attribute_names=self._config.sensitive_attr)

        # Count the number of samples at each intersection of values
        #intersection_counts = pd.concat([X_val, predictions_val_df], axis=1).groupby(['label', 'race']).size().reset_index(name='count')

        # Print the result
        #print("TRAIN")
        #print(intersection_counts)
        
        # cost constraint of fnr will optimize generalized false negative rates, that of
        # fpr will optimize generalized false positive rates, and weighted will optimize
        # a weighted combination of both
        cost_constraint = "weighted" # "fnr", "fpr", "weighted"
        
        self._calibrator = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint)
        
        
        # Apply CalibratedEqOddsPostprocessing
        self._calibrator.fit(val_dataset, val_dataset_pred)



    def predict(self, X: pd.DataFrame, binary= True) -> np.array: 
        if self._use_base_model:
            preds = self._model.predict(X)
        else:
            preds = self._trainset_model.predict(X)
            
                    
        predictions_df = pd.DataFrame(data={'label': preds})
        
        # Count the number of samples at each intersection of values
        #intersection_counts = pd.concat([X.reset_index(drop=True), pd.DataFrame(data={'label': self._binarise(preds)})], axis=1).groupby(['label', 'race']).size().reset_index(name='count')

        # Print the result
        #print("TEST")
        #print(intersection_counts)
        
        dataset_pred = BinaryLabelDataset(df=pd.concat([X.reset_index(drop=True), predictions_df], axis=1), 
                                          label_names=['label'], protected_attribute_names=self._config.sensitive_attr)

        calibrated_preds = self._calibrator.predict(dataset_pred)
        
        intersection_counts = pd.concat([X.reset_index(drop=True), pd.DataFrame(data={'label': self._binarise(calibrated_preds.labels.T[0])})], axis=1).groupby(['label', 'race']).size().reset_index(name='count')

        # Print the result
        print("TEST AFTER")
        print(intersection_counts)

        preds = calibrated_preds.labels.T[0]
        if not binary:
            return preds
        return self._binarise(preds)
