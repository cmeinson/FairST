import numpy as np
import pandas as pd
from typing import List, Callable
import warnings
import math
from itertools import permutations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


class MetricException(Exception):
    pass

class Metrics:
    # All available metrics:
    #performance metrics
    ACC = "accuracy"
    PRE = "precision"
    REC = "recall"
    F1 = "f1score"

    #fairness metrics
    AOD = "[AOD] Average Odds Difference"
    EOD = "[EOD] Equal Opportunity Difference"
    SPD = "[SPD] Statistical Parity Difference"
    DI = "[DI] Disparate Impact"
    DI_FM = "[DI_FM] Disparate Impact the way it was implemented in FairMask"
    FR = "[FR] Flip Rate"

    SF = "[SF] Statistical Parity Subgroup Fairness"
    SF_INV = "[SF] Statistical Parity Subgroup Fairness if 0 was the positive label"
    ONE_SF = "[SF] Statistical Parity Subgroup Fairness for One Attribute"

    DF = "[DF] Differential Fairness"
    DF_INV = "[DF] Differential Fairness if 0 was the positive label"
    ONE_DF = "[DF] Differential Fairness for One Attribute"

    M_EOD = "[MEOD] M Equal Opportunity Difference"
    M_AOD = "[MAOD] M Average Odds Difference"

    warnings.simplefilter("ignore")

    def __init__(self, X: pd.DataFrame, y: np.array, preds: np.array, predict: Callable[[pd.DataFrame], np.array]) -> None:
        # might need more attributes idk
        self._X = X
        self._X.reset_index(drop=True, inplace=True) # TODO: would it be better for everyone if this was done in the data class?
        self._y = y
        self._data_size = X.shape[0]
        self._preds = preds
        self._predict = predict
        self._computed_cms = {}

        self._last_call = None


    @classmethod
    def get_all_names(cls):
        return Metrics.get_attribute_dependant() + Metrics.get_attribute_independant() + Metrics.get_subgroup_dependant()

    @classmethod
    def get_subgroup_dependant(cls):
        # metrics that need a list of attributes as input to create subgroups
        return [Metrics.SF, Metrics.SF_INV, Metrics.DF, Metrics.DF_INV, Metrics.M_EOD, Metrics.M_AOD]

    @classmethod
    def get_attribute_dependant(cls):
        # metrics that need a single attribute as input
        return [Metrics.AOD, Metrics.EOD, Metrics.SPD, Metrics.DI, Metrics.DI_FM, Metrics.FR, Metrics.ONE_SF, Metrics.ONE_DF]

    @classmethod
    def get_attribute_independant(cls):
        # metrics independant of attributes
        return [Metrics.ACC, Metrics.PRE, Metrics.REC, Metrics.F1]
    
    def get(self, metric_name, attr: List[str] or str = None):
        self._last_call = metric_name

        metrics_dict = {
            self.ACC:   lambda _ : accuracy_score(self._y, self._preds),
            self.PRE:   lambda _ : precision_score(self._y, self._preds),
            self.REC:   lambda _ : recall_score(self._y, self._preds),
            self.F1:    lambda _ : f1_score(self._y, self._preds),
            self.AOD:   self.aod,
            self.EOD:   self.eod,
            self.SPD:   self.spd,
            self.DI:    self.di,
            self.DI_FM: self.di_fm,
            self.M_EOD: self.meod,
            self.M_AOD: self.maod,
            self.FR:    self.fr,
            self.SF:    self.sf,
            self.SF_INV:lambda a: self.sf(a, outcome='n'),
            self.ONE_SF:lambda a: self.sf([a]),
            self.DF:    self.df,
            self.DF_INV:lambda a: self.df(a, outcome='n'),
            self.ONE_DF:lambda a: self.df([a]),
        }

        if metric_name in metrics_dict:
            return metrics_dict[metric_name](attr)
        
        raise RuntimeError("Invalid metric name:", metric_name)

    def _div_guard(self, value):
        if value == 0:
            raise MetricException("Undifined metric ", self._last_call, " due to division by 0")
        return value
    
    def _confusion_matrix(self, idxs=None):
        if idxs is None:
            idxs = range(self._data_size)

        key = tuple(sorted(idxs))
        # avoid recalculating same cms 
        if key not in self._computed_cms:
            tn, fp, fn, tp = confusion_matrix(self._y[idxs], self._preds[idxs]).ravel()
            cm = {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'p': tp+fp, 'n':tn+fn}
            # TODO: what if there are no positive / negative labels at all
            cm['tpr'] = tp / self._div_guard(tp + fn)
            cm['fpr'] = fp / self._div_guard(fp + tn)
            cm['pr'] = cm['p'] / (cm['p'] + cm['n'])
            self._computed_cms[key] = cm

        return self._computed_cms[key]
    
    def _get_attribute_cms(self, attribute: str):
        # get separate cm for 0 and 1 labels of the attribute
        idx0 = np.where(self._X[attribute] == 0)[0]
        idx1 = np.where(self._X[attribute] == 1)[0]
        return self._confusion_matrix(idx0), self._confusion_matrix(idx1)
    
    def aod(self, attribute: str):
        # note that it is the abs aod
        cm0, cm1 = self._get_attribute_cms(attribute)
        diff = (cm0['fpr'] - cm1['fpr']) + (cm0['tpr'] - cm1['tpr'])
        return abs(0.5 * diff)
    
    def eod(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        return abs(cm0['tpr'] - cm1['tpr'])
    
    def spd(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        return abs(cm1['pr'] - cm0['pr'])
    
    def di(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        if cm0['pr'] == 0  and cm1['pr'] == 0:
            return 1 # NOTE! 1 if they are the same!!
        if cm1['pr'] == 0:
            raise MetricException("DI fail pr1 = 0", attribute)
        return cm0['pr']/cm1['pr']
    
    def di_fm(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        if cm0['pr'] == 0  and cm1['pr'] == 0:
            return 1 # NOTE! 1 if they are the same!!
        if cm0['pr'] == 0:
            raise MetricException("DI FM fail pr0 = 0", attribute)
        return abs(1-cm1['pr']/cm0['pr'])
    
    def fr(self, attribute):
        preds_flip = self._predict(self.flip_X(attribute))
        same = np.count_nonzero(self._preds==preds_flip)
        return (self._data_size-same)/self._data_size
    

    def _get_subgroup_indices(self, attributes: List[str]) -> List[List[int]]:
        subgroup_indices = self._X.groupby(attributes).apply(lambda x: x.index.tolist())
        return subgroup_indices.tolist()


    def sf(self, attributes: List[str], outcome = 'p') -> float:
        # max over all subgroups ABS[ P(Y=1) - P(Y=1|subgroup) ] * P(subgroup)
        # where P (Y=1) = p / (p+n)
        cm = self._confusion_matrix()
        PY1 = cm[outcome] / self._data_size

        diffs = []
        subgroups = self._get_subgroup_indices(attributes)
        for subgroup in subgroups:
            sg_cm = self._confusion_matrix(subgroup)
            group_size = len(subgroup)

            if group_size!=0:
                PY1_sg = sg_cm[outcome] / group_size
                P_sg = group_size / self._data_size
                diffs.append(abs(PY1 - PY1_sg)*P_sg)

        return max(diffs)
    
    def _get_subgrups_df(self, subgroup1, subgroup2, outcome):
        cm1 = self._confusion_matrix(subgroup1)
        cm2 = self._confusion_matrix(subgroup2)
        prob_sg1 = cm1[outcome] / len(subgroup1)
        prob_sg2 = cm2[outcome] / len(subgroup2)
        if (prob_sg1 == prob_sg2):
            return 1
        elif prob_sg1==0 or prob_sg2==0:            
            raise MetricException("DF fail")
        return prob_sg1 / prob_sg2


    def df(self, attributes: List[str], outcome = 'p') -> float:
        subgroups = self._get_subgroup_indices(attributes)

        ratios = []
        for subgroup1, subgroup2 in list(permutations(subgroups, 2)):
            if len(subgroup1)!=0 and len(subgroup2)!=0:
                # TODO should i do both outcomes???????
                ratio = self._get_subgrups_df(subgroup1, subgroup2, outcome)
                ratios.append(ratio)

        ans = max(math.log(max(ratios)), -math.log(min(ratios)))
        return ans
    
    def meod(self, attributes):
        tprs = []
        subgroups = self._get_subgroup_indices(attributes)
        for subgroup in subgroups:
            cm = self._confusion_matrix(subgroup)
            tprs.append(cm['tpr'])

        return max(tprs) - min(tprs)

    def maod(self, attributes):
        prs = []

        subgroups = self._get_subgroup_indices(attributes)
        for subgroup in subgroups:
            cm = self._confusion_matrix(subgroup)
            prs.append(cm['tpr'] + cm['fpr'])
        # TODO: THIS SEEMS OFF 
        return 0.5*(max(prs) - min(prs))

    def flip_X(self,attribute):
        X_flip = self._X.copy()
        X_flip[attribute] = np.where(X_flip[attribute]==1, 0, 1)
        return X_flip
