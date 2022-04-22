from typing import List

import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRFClassifier, XGBRFRegressor

def get_models(subtask: int) -> List:
    """
    Fetches a list of models to be evaluated for a specific subtask.
    Put DummyClassifier first model in the list for desired behaviour. 
    """
    random_state = 1
    
    if subtask == 1 or subtask == 2: 
        models = [
            (DummyClassifier, dict()),
            (LogisticRegression, dict()),
            *[(XGBRFClassifier, {
                    "objective":'binary:logistic', 
                    "eval_metric" : roc_auc_score,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "use_label_encoder":False, 
                    'random_state':random_state
                }
            ) for max_depth in [4, 6] for n_estimators in [50, 100]],
            *[(RandomForestClassifier, {
                "max_depth": max_depth,
                "n_estimators": n_estimators, 
                "random_state": random_state}) 
                for max_depth in [4, 5, 10] for n_estimators in [50, 100]],
            *[(AdaBoostClassifier, {"random_state": random_state, "n_estimators": n_estimators}) for n_estimators in [25, 50, 75]],
            (MLPClassifier, dict())
        ]

        return models 

    if subtask == 3:
        models = [
            (DummyRegressor, dict()),
            (LinearRegression, dict()),
            (RidgeCV, {"scoring": "r2"}),
            *[(XGBRFRegressor, {
                "eval_metric": roc_auc_score,
                "max_depth": max_depth,
                "n_estimators": n_estimators, 
                "random_state": random_state
            }) for max_depth in [2, 4, 6] for n_estimators in [50, 100, 250]],
            (SVR, {"kernel": "poly", }),
            (KNeighborsRegressor, dict()),
            (MLPRegressor, dict())
        ]

        return models