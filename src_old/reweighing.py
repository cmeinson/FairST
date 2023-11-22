from .ml_interface import Model
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import pandas as pd

class ReweighingModel(Model):

    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._model = None

    def train(self, X: pd.DataFrame, y: np.array, sensitive_attributes: List[str], method, method_bias = None, other: Dict[str, Any] = {}):
        """ Trains an ML model

        :param X: training data
        :type X: pd.DataFrame
        :param y: training data outcomes
        :type y: np.array
        :param sensitive_attributes: names of sensitive attributes to be protected
        :type sensitive_attributes: List[str]
        :param method:  ml algo name to use for the main model training
        :type method: _type_
        :param method_bias: method name if needed for the bias mitigation, defaults to None
        :type method_bias: _type_, optional
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional
        """
        print("rw other,", other)
        self._model = self._get_model(method, other | {"input_dim":X.shape[1]})
        self.transformer = self._get_transformer(X)
        
        sample_weight = self.Reweighing(X, y, sensitive_attributes)
        #X1 = {'data': self.transformer.fit_transform(X)}
        # add sample_weight to the X dict. All ones so should get similar result as above
        #X1['sample_weight'] = sample_weight

        if method != Model.NN_C:
            self._model.fit(self.transformer.fit_transform(X), y, sample_weight=sample_weight)
        else:
            self._model.fit(self.transformer.fit_transform(X), y, epochs=other["iter_1"], sample_weight=sample_weight)


    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other params that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        y = self._model.predict(self.transformer.transform(X))
        y[y>0.5] = 1
        y[y<=0.5] = 0
        return y

    def Reweighing(self, X, y, A):
        groups_class = {}
        group_weight = {}
        X.reset_index(drop=True, inplace=True)

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

   



        


   

    


