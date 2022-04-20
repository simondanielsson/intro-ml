from asyncio.proactor_events import _ProactorBaseWritePipeTransport
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

def select_labels(subtask: int, y_train: pd.DataFrame, y_val: pd.DataFrame) -> List[pd.Series]:
    """Selects or transforms label dataframe into a single target column, depending on subtask"""
    # TODO: implement depeneding on subtask

    LABELS = {
        "1": "LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2".split(", "),
        "2": None,
        "3": None 
    }

    labels = LABELS[str(subtask)]
    
    if subtask == 1:
        def get_target(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[:, labels].apply(lambda row : row.any(), axis=1)

        # Construct new boolean column: 1 if any of labels are 1, else 0
        target_train, target_val = get_target(y_train), get_target(y_val)

        return target_train, target_val

    if subtask == 2:
        pass

    if subtask == 3:
        pass
    

    return y_train, y_val