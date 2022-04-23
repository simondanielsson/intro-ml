from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from re import X
from typing import List, Tuple

import pandas as pd

def predict_labels(best_models: List[Tuple[str, object]], subtask: int, X_test: pd.DataFrame, X_val) -> pd.DataFrame:
    """
    Takes the best model and predicts the probabilities on the test set on the label, 
    outputs a resulting predicted dataframe with all the label columns
    """
    labels_pred_test = pd.DataFrame()
    labels_pred_val = pd.DataFrame()

    for X, labels_pred in zip([X_test, X_val], [labels_pred_test, labels_pred_val]):
        for label_name, model in best_models:
            
            if subtask == 1 or subtask == 2:
                # Outputs two columns: fetch the one corresponding to class "1"
                pred_ = pd.DataFrame(
                    model.predict_proba(X)
                )
                pred = pred_.loc[:, 1]

            elif subtask == 3:
                pred = pd.DataFrame(
                    model.predict(X)
                )
                
            labels_pred[label_name] = pred
    
    return labels_pred_test, labels_pred_val
