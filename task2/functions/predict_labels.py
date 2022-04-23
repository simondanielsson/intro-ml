from typing import List, Tuple

import pandas as pd

def predict_labels(best_models: List[Tuple[str, object]], subtask: int, X_test) -> pd.DataFrame:
    """
    Takes the best model and predicts the probabilities on the test set on the label, 
    outputs a resulting predicted dataframe with all the label columns
    """
    labels_pred = pd.DataFrame()

    for i in range(len(best_models)):
        label_name = best_models[i][0]
        
        if subtask == 1 or subtask == 2:
            label_p = copy.copy(best_models[i][1].predict_proba(X_test)[:, 1])
        else:
            label_p = copy.copy(best_models[i][1].predict(X_test))
            
        labels_pred[label_name] = label_p   # there is no binding between labels_pred[label_name] and label_p
    
    return labels_pred
