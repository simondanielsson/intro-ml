from typing import List

import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRFClassifier

def get_models(subtask: int) -> List:
    random_state = 1
    
    if subtask == 1: 
        models = [
            DummyClassifier(),
            LinearRegression(),
            LassoCV(random_state=random_state),
            AdaBoostClassifier(random_state=random_state), 
            GridSearchCV(
                RandomForestClassifier(random_state=random_state), 
                param_grid={
                    "min_samples_split": [2, 5, 10, 50, 100],
                },
                scoring="roc_auc"
            )
        ]
        
        #XGBRFClassifier(objective='binary:logistic', eval_metric = 'auc', use_label_encoder=False, random_state=random_state)
        #SVC(kernel="poly"), 
        #SVC()
        #*[SVC(C=c) for c in np.logspace(-2, 1, num=3)],
        #*[SVC(C=c, kernel="poly") for c in np.logspace(-2, 1, num=3)]

        return models 

    if subtask == 2:
        raise NotImplementedError()

    if subtask == 3:
        raise NotImplementedError()