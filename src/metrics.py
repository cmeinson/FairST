import numpy as np
import pandas as pd
from typing import List, Callable
import warnings
from sklearn import metrics
from itertools import combinations
import copy
import math
from collections import defaultdict
from itertools import permutations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


# how do we wanna do metrics?
class MetricException(Exception):
    pass
# we can do just a simple metrcs class with all the mathy functions and then a separate evaluator class?

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
        self.groups = defaultdict(list)
        self._computed_cms = {}
        self._round = lambda x: round(x,5) # TODO: del

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
        # -------------------------------------------------------------
        is_old = (metric_name[:3] == 'OLD')
        if is_old:
            metric_name = metric_name[3:] 
            if metric_name == self.ACC:
                return self.accuracy()
            elif metric_name == self.PRE:
                return self.precision()
            elif metric_name == self.REC:
                return self.recall()
            elif metric_name == self.F1:
                return self.f1score()
            elif metric_name == self.AOD:
                return self.old_aod(attr)
            elif metric_name == self.EOD:
                return self.old_eod(attr)
            elif metric_name == self.SPD:
                return self.old_spd(attr)
            elif metric_name == self.DI:
                return self.old_di(attr) 
            elif metric_name == self.DI_FM:
                return self.old_di_fm(attr)  
            elif metric_name == self.SF:
                return self.old_sf(attr)
            elif metric_name == self.SF_INV:
                return self.old_sf(attr, outcome='n') 
            elif metric_name == self.ONE_SF:
                return self.old_sf([attr])
            elif metric_name == self.DF:
                return self.old_df(attr) 
            elif metric_name == self.DF_INV:
                return self.old_df(attr, outcome='n')
            elif metric_name == self.ONE_DF:
                return self.old_df([attr])
            elif metric_name == self.M_EOD:
                return self.old_meod(attr) 
            elif metric_name == self.M_AOD:
                return self.old_maod(attr) 
            elif metric_name == self.FR:
                return self.old_fr(attr)
            else:
                raise RuntimeError("Invalid metric name: ", metric_name)
        # ------------------------------------------------------------- NEW
        if metric_name == self.ACC:
            return accuracy_score(self._y, self._preds)
        elif metric_name == self.PRE:
            return precision_score(self._y, self._preds)
        elif metric_name == self.REC:
            return recall_score(self._y, self._preds)
        elif metric_name == self.F1:
            return f1_score(self._y, self._preds)
        elif metric_name == self.AOD:
            return self.aod(attr)
        elif metric_name == self.EOD:
            return self.eod(attr) 
        elif metric_name == self.SPD:
            return self.spd(attr) 
        elif metric_name == self.DI:
            return self.di(attr)
        elif metric_name == self.DI_FM:
            return self.di_fm(attr)
        elif metric_name == self.SF:
            return self.sf(attr)
        elif metric_name == self.SF_INV:
            return self.sf(attr, outcome='n') 
        elif metric_name == self.ONE_SF:
            return self.sf([attr])
        elif metric_name == self.DF:
            return self.df(attr) 
        elif metric_name == self.DF_INV:
            return self.df(attr, outcome='n')
        elif metric_name == self.ONE_DF:
            return self.df([attr])
        elif metric_name == self.M_EOD:
            return self.meod(attr)
        elif metric_name == self.M_AOD:
            return self.maod(attr)
        elif metric_name == self.FR:
            return self.fr(attr)
        else:
            raise RuntimeError("Invalid metric name: ", metric_name)

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
        pr0 = cm0['p'] / (cm0['p'] + cm0['n'])
        pr1 = cm1['p'] / (cm1['p'] + cm1['n'])
        return abs(pr1 - pr0)
    
    def di(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        pr0 = cm0['p'] / (cm0['p'] + cm0['n'])
        pr1 = cm1['p'] / (cm1['p'] + cm1['n'])
        if pr0 == 0  and pr1 == 0:
            return 1 # NOTE! 1 if they are the same!!
        if pr1 == 0:
            raise MetricException("DI fail pr1 = 0", attribute)
        return pr0/pr1
    
    def di_fm(self, attribute: str) -> float:
        cm0, cm1 = self._get_attribute_cms(attribute)
        pr0 = cm0['p'] / (cm0['p'] + cm0['n'])
        pr1 = cm1['p'] / (cm1['p'] + cm1['n'])
        if pr0 == 0  and pr1 == 0:
            return 1 # NOTE! 1 if they are the same!!
        if pr0 == 0:
            raise MetricException("DI FM fail pr0 = 0", attribute)
        return abs(1-pr1/pr0)
    
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
                # TODO shoul i do both outcomes???????
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


    # __________________________________________________ OLD ___________________________________________

    def accuracy(self) -> float:
        return accuracy_score(self._y, self._preds)

    def precision(self) -> float:
        per = self._round(precision_score(self._y, self._preds))
        if per == 0: raise MetricException("percision fail val = 0")
        return per

    def recall(self) -> float:
        return self._round(recall_score(self._y, self._preds))

    def f1score(self) -> float:
        return self._round(f1_score(self._y, self._preds))

    def old_aod(self, attribute):
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        if (conf0['tp'] + conf0['fn']) == 0:
            tpr0 = 0
        else:
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])

        if (conf1['tp'] + conf1['fn']) == 0:
            tpr1 = 0
        else:
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])

        if (conf0['fp'] + conf0['tn']) == 0:
            fpr0 = 0
        else:
            fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])

        if (conf1['fp'] + conf1['tn']) == 0:
            fpr1 = 0
        else:
            fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
        return abs(self._round(0.5 * (tpr1 + fpr1 - tpr0 - fpr0)))

    def old_eod(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        if (conf1['tp'] + conf1['fn']) == 0:
            tpr1 = 0
        else:
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])

        if (conf0['tp'] + conf0['fn']) == 0:
            tpr0 = 0
        else:
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
        return abs(self._round(tpr1 - tpr0))

    def old_spd(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        if len(ind0) == 0:
            pr0 = 0
        else:
            pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
        if len(ind1) == 0:
            pr1 = 0
        else:
            pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
        return abs(self._round(pr1 - pr0))

    def old_di(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        if len(ind0) == 0 or len(ind1)==0:
            raise MetricException("DI fail attribute has only 1 val", attribute)
        pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
        pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
        if pr0==0:
            return 0
        if pr1 == 0:
            raise MetricException("DI fail pr1 = 0", attribute)
        di = pr0/pr1
        return self._round(di)
    
    def old_di_fm(self, attribute) -> float:
        for i in range(len(self._y)):
            group = tuple([self._X[attribute][i]])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        ind0 = np.where(self._X[attribute] == 0)[0]
        ind1 = np.where(self._X[attribute] == 1)[0]
        conf0 = self.confusionMatrix(ind0)
        conf1 = self.confusionMatrix(ind1)
        if len(ind0) == 0 or len(ind1)==0:
            raise MetricException("DI fail attribute has only 1 val", attribute)
        pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
        pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
        if pr1==0:
            return 0
        if pr0 == 0:
            raise MetricException("DI fm fail pr0 = 0", attribute)
        di = pr1/pr0
        return self._round(abs(1-di))

    def old_fr(self, attribute):
        X_flip = self.flip_X(attribute)
        preds_flip = self._predict(X_flip)
        total = self._X.shape[0]
        same = np.count_nonzero(self._preds==preds_flip)
        return self._round((total-same)/total)
    
    def old_get_subgroup_attr_vals(self, attrs_unique_vals):
        subgroups = [[]]
        for attr_vals in attrs_unique_vals:
            new_subgroups = []
            for subg in subgroups:
                for attr_val in attr_vals:
                    new_subgroups.append(subg+ [attr_val])
            subgroups = new_subgroups
        return subgroups
    
    def old_get_subgroup_conf_and_size(self, attributes, attribute_vals):
        ind = set(range(len(self._y)))
        for i in range(len(attributes)):
            ind = ind & set(np.where(self._X[attributes[i]] == attribute_vals[i])[0])
        return self.confusionMatrix(sorted(ind)), len(ind)  # TODO: do indices have to be sorted?            

    def old_sf(self, attributes: List[str], outcome = 'p') -> float:
        attr_values = [np.unique(self._X[a]) for a in attributes]
        subgroups = self.old_get_subgroup_attr_vals(attr_values)

        sample_size = len(self._y)
        conf = self.confusionMatrix()
        prob_pos = conf[outcome] / sample_size
        ans = 0
        for subgroup in subgroups:
            group_conf, group_size = self.old_get_subgroup_conf_and_size(attributes, subgroup)
            if group_size!= 0:
                group_prob_pos = group_conf[outcome] / group_size
                group_prob = group_size / sample_size

                group_ans = abs(prob_pos - group_prob_pos)*group_prob
                ans = max(group_ans, ans)
        return ans
    
    def old_df(self, attributes: List[str], outcome = 'p') -> float:
        attr_values = [np.unique(self._X[a]) for a in attributes]
        subgroups = self.old_get_subgroup_attr_vals(attr_values)

        ans_min, ans_max = 1, 1
        for subgroup1 in subgroups:
            conf1, size1 = self.old_get_subgroup_conf_and_size(attributes, subgroup1)
            for subgroup2 in subgroups:
                conf2, size2 = self.old_get_subgroup_conf_and_size(attributes, subgroup2)
                if size1!=0 and size2!=0:
                    prob_pos1 = conf1[outcome] / size1
                    prob_pos2 = conf2[outcome] / size2
                    if (prob_pos1 == prob_pos2):
                        ans = 1
                    else:
                        if prob_pos1==0 or prob_pos2==0:            
                            raise MetricException("DF fail", attributes)
                        ans = prob_pos1 / prob_pos2
                    ans_max = max(ans, ans_max)
                    ans_min = min(ans, ans_min)            
        ans = max(math.log(ans_max), -math.log(ans_min))
        return ans

    def old_meod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        tprs = self.tprs()
        return abs(self._round(max(tprs.values())-min(tprs.values())))

    def old_maod(self, attributes, a=None):
        for i in range(len(self._y)):
            group = tuple([self._X[attr][i] for attr in attributes])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)
        aos = self.aos()
        return abs(self._round(max(aos.values()) - min(aos.values())))

    def confusionMatrix(self, sub=None):
        if sub is None:
            sub = range(len(self._y))
        y = self._y[sub]
        y_pred = self._preds[sub]
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(y)):
            if y[i]==0 and y_pred[i]==0:
                conf['tn']+=1
            elif y[i]==1 and y_pred[i]==1:
                conf['tp'] += 1
            elif y[i]==0 and y_pred[i]==1:
                conf['fp'] += 1
            elif y[i]==1 and y_pred[i]==0:
                conf['fn'] += 1
        conf["p"] = conf['tp'] + conf['fp']
        conf["n"] = conf['tn'] + conf['fn']
        return conf

    def tprs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            if (conf['tp'] + conf['fn']) == 0:
                tpr = 0
            else:
                tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def fprs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            if (conf['fp'] + conf['tn']) == 0:
                fpr = 0
            else:
                fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            fprs[group] = fpr
        return fprs

    def aos(self):
        tprs = self.tprs()
        fprs = self.fprs()
        aos = {key: (tprs[key]+fprs[key])/2 for key in tprs}
        return aos

    def prs(self):
        prs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            if len(sub) == 0:
                pr = 0
            else:
                pr = (conf['tp']+conf['fp']) / len(sub)
            prs[group] = pr
        return prs
    
    def sgf(self):
        sgf = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusionMatrix(sub)
            if (conf['tp']+conf['fp']+conf['tn']+conf['fn']) == 0:
                sg = 0 
            else:
                sg = conf['tp']/(conf['tp']+conf['fp']+conf['tn']+conf['fn'])
            sgf[group] = sg
        return sgf

    def flip_X(self,attribute):
        X_flip = self._X.copy()
        X_flip[attribute] = np.where(X_flip[attribute]==1, 0, 1)
        return X_flip
