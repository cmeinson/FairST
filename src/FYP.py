from .ml_interface import Model
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random
import copy
import keras

from collections import Counter

SPEED_BOOST_A_BIT = 5

class FypModel(Model):
    def __init__(self, other: Dict[str, Any] = {}) -> None:
        """Idk does not really do much yet I think:)

        :param other: any hyper params we need to pass, defaults to {}
        :type other: Dict[str, Any], optional
        """
        self._models = [None, None]
        self._sensitive = None
        self._method_bias = None

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
        if method != Model.NN_C:
            raise RuntimeError("attempted to run FYP with method:", method)
        
        self.transformer = self._get_transformer(X)
        self.transformer.fit_transform(X)
        
        # train a general model
        # for each subgroup train another model

        self._method_bias = method_bias
        self._sensitive = sensitive_attributes
        attr = sensitive_attributes[0]
        self.calc_counts(X,y,attr)

        self._models[1] = self._get_model(method, other | {"warm_start":True, "input_dim":X.shape[1]})

        self.train1(X, y ,other)

        self._models[0] = self._get_model(method, other | {"warm_start":True, "input_dim":X.shape[1]})
        self._models[0].set_weights(copy.deepcopy(self._models[1].get_weights()))

        if other["iter_2"]>0:
            self.train2(X, y ,other, attr)



    def predict(self, X: pd.DataFrame, other: Dict[str, Any] = {}) -> np.array:
        """ Uses the previously trained ML model

        :param X: testing data
        :type X: pd.DataFrame
        :param other: dictionary of any other parms that we might wish to pass?, defaults to {}
        :type other: Dict[str, Any], optional

        :return: predictions for each row of X
        :rtype: np.array
        """
        # choose the right model to predict

        y = self._models[0].predict(self.transformer.transform(X))
        y_1 = self._models[1].predict(self.transformer.transform(X))

        mask = (X[self._sensitive[0]] == 1)

        y[mask] = y_1[mask]

        y[y>0.5] = 1
        y[y<=0.5] = 0
        return y
        
    def silly_reweighing(self, X, y):
        idxs = list(random.sample(self.mask_11, self.c_11))
        idxs +=  list(random.sample(self.mask_01, self.c_01))
        idxs +=  list(random.sample(self.mask_00, self.c_00))
        idxs +=  list(random.sample(self.mask_10, self.c_10))
        #print("DATA TYPES OF X AND Y", type(X[idxs]), type(y[idxs]) )
        return X.iloc[idxs], y[idxs]
    
    def calc_counts(self, X, y, attr):
        X_p = X.copy()
        X_p['y'] = y
        
        self.mask_11 = list(np.where((X_p[attr]==1) & (X_p['y'] == 1))[0])
        self.mask_10 = list(np.where((X_p[attr]==1) & (X_p['y'] == 0))[0])
        self.mask_01 = list(np.where((X_p[attr]==0) & (X_p['y'] == 1))[0])
        self.mask_00 = list(np.where((X_p[attr]==0) & (X_p['y'] == 0))[0])

        self.c_11 = len(self.mask_11)
        self.c_00 = len(self.mask_00)
        self.c_10 = len(self.mask_10)
        self.c_01 = len(self.mask_01)

        ratio = (self.c_01 + self.c_11) / (self.c_00 + self.c_10)

        ratio_0 = (self.c_01) / (self.c_00)
        ratio_1 = (self.c_11) / (self.c_10)

        if ratio_0 > ratio:
            self.c_01 = int(self.c_00*ratio)
        else:
            self.c_00 = int(self.c_01/ratio)

        if ratio_1 > ratio:
            self.c_11 = int(self.c_10*ratio)
        else:
            self.c_10 = int(self.c_11/ratio)

            

        print("counts 01 00 11 10", self.c_01,self.c_00,self.c_11,self.c_10 )
        #print( self.count_0, self.count_1)

        
    def train1(self, X, y, other):
        if not ("RW" in other and other["RW"]):
            for _ in range(other["iter_1"]//SPEED_BOOST_A_BIT):
                X_p, y_p = self.silly_reweighing(X,y)
                self._models[1].fit(self.transformer.transform(X_p), y_p, epochs=SPEED_BOOST_A_BIT)
        else:
            sample_weight = self.Reweighing(X, y, self._sensitive)
            self._models[1].fit(self.transformer.transform(X),y, epochs=other["iter_1"], sample_weight=sample_weight)

    def train2(self,X,y,other, attr):
        if "RW" not in other or not other["RW"]:
            for _ in range(other["iter_2"]//SPEED_BOOST_A_BIT):
                
                X_rw, y_rw = self.silly_reweighing(X,y)
                mask = (X_rw[attr] == 1)

                self._models[0].fit(self.transformer.transform(X_rw[~mask]),y_rw[~mask], epochs=SPEED_BOOST_A_BIT)
                self._models[1].fit(self.transformer.transform(X_rw[mask]),y_rw[mask], epochs=SPEED_BOOST_A_BIT)
        else:
            sample_weight = self.Reweighing(X, y, self._sensitive)
            mask = (X[attr] == 1)
            self._models[0].fit(self.transformer.transform(X[~mask]),y[~mask], epochs=other["iter_2"], sample_weight=sample_weight[~mask])
            self._models[1].fit(self.transformer.transform(X[mask]),y[mask], epochs=other["iter_2"], sample_weight=sample_weight[mask])


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
        

# concl: the reweighing equivalent kinda works but becomes useless cause you might as well just apply out of the box rw. it does improve some disparity between groups tho when works as intended

# I NEED A CONCRETE COMPARISON WITH REWEIGHING in the nn
# next: add a loss bias mitigation into play.