from .ml_interface import Model
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import pandas as pd

class ReweighingModel(Model):

    def fit(self, X: pd.DataFrame, y: np.array):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        """
        self._model = self._get_model()
        
        self._model.fit(X, y, sample_weight=self.Reweighing(X, y))

    def Reweighing(self, X, y):
        groups_class = {}
        group_weight = {}
        X.reset_index(drop=True, inplace=True)

        A = self._config.sensitive_attr
        for i in range(len(y)):
            key_class = tuple([X[a][i] for a in A]+[y[i]])
            key = key_class[:-1]

            if key not in group_weight:
                group_weight[key]=0
            group_weight[key]+=1
            if key_class not in groups_class:
                groups_class[key_class]=[]
            groups_class[key_class].append(i)
        class_weight = Counter(y)
        sample_weight = np.array([1.0]*len(y))
        for key in groups_class:
            weight = class_weight[key[-1]]*group_weight[key[:-1]]/len(groups_class[key])
            for i in groups_class[key]:
                sample_weight[i] = weight
        # Rescale the total weights to len(y)
        sample_weight = sample_weight * len(y) / sum(sample_weight)
        return sample_weight

   



        


   

    


