from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    # NB: if ur implementation of the class takes more than one file pls put it all into sub folder

    def __init__(self, preprocessing:str = None, test_ratio = 0.2) -> None:
        """
        - reads the according dataset from the ata folder,
        - runs cleaning and preprocessing methods, chosen based on the preprocessing param
        - splits the data into test and train

        :param preprocessing: determines the preprocessing method, defaults to None
        :type preprocessing: str, optional
        :param tests_ratio: determines the proportion of test data, defaults to 0.2
        :type tests_ratio: float, optional
        """
        raise NotImplementedError
    
    def new_data_split(self) -> None:
        """Changes the data split"""
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=self._test_ratio)

    def get_train_data(self) -> Tuple[pd.DataFrame, np.array]:
        """Returns the training data where
        X: is the df with all attriutes, with accordig column names
        y: the outcome for each row (e.g. the default credit, is income above 50k, did reoffend?)
        Duplicares everything for safety reasons:)

        :return: training data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_train.copy(), self._y_train.copy())

    def get_test_data(self) -> Tuple[pd.DataFrame, np.array]:
        """
        :return: test data (X, y)
        :rtype: Tuple[pd.DataFrame, np.array]
        """
        return (self._X_test.copy(), self._y_test.copy())

    def get_sensitive_column_names(self) -> List[str]:
        """
        :return: column names (in the X above) of all sensitive attributes in the given dataset
        :rtype: List[str]
        """
        raise NotImplementedError
    


class DummyData(Data):
    def __init__(self, preprocessing = None, test_ratio=0.2) -> None:
        data = [[0.31, 0, 0], [0.41, 0, 1], [0.51, 0, 0], [0.71, 0, 1], [0.81, 0, 0], [0.91, 0, 1], 
                [0.2, 1, 0], [0.3, 1, 1], [0.4, 1, 0], [0.5, 1, 1], [0.6, 1, 0], [0.7, 1, 1]]
        self._X = pd.DataFrame(data, columns=["attr", "sensitive", "sensitive2"])
        self._y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,])
        self._test_ratio = test_ratio
        self.new_data_split()        

    def get_sensitive_column_names(self) -> List[str]:
        return ["sensitive", "sensitive2"]


