from typing import List

from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import numpy as np

def select_feature(X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
    """
    Select the most important features in X_train based on importance in RandomForestRegressor. 
    Returns list of indeces of features to use. 
    """

    sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=38) , threshold='median' )
    sfm.fit(X_train, y_train) 
    #X_train_sfm = sfm.transform(X_train) 

    mask_sfm = sfm.get_support(indeces=True)

    #X_train_selected = X_train.iloc[:,mask_sfm]
    
    return mask_sfm

def select_labels(subtask: int) -> List[str]:
    # TODO: implement depeneding on subtask

    labels = {
        "1": "LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2".split(", "),
        "2": None,
        "3": None 
    }

    labels_ = labels[str(subtask)]
    print(f"Chose labels {labels_}" )
    
    return labels_ 