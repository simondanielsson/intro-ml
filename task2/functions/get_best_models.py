from typing import List, Tuple

def get_best_models(scores: dict, subtask: int) -> List[Tuple[str, object]]:
    """
    Gets a dictionary of models and their scores and outputs the best models per label column.
    For subtask 1 and 2 the best is the one with the largest average roc auc across all labels
    Also prints its average roc auc.
    """

    best_models = []
    list_of_models = list(scores)
    list_of_labels = list(scores[list_of_models[0]])

    for label in list_of_labels:
        dict_of_scores = {}
        for model in list_of_models:
            trained_model = scores[model][label][0]
            dict_of_scores[trained_model] = scores[model][label][1][1]
        best_models.append((label, max(dict_of_scores, key=dict_of_scores.get)))
        

    return best_models


    """
    scores = {"<model_name>": {
            "<label_name1>": (trained_model_object, (train_score1, val_score1)),
            "<label_name2>": (trained_model_object2, (train_score2, val_score2)),
        },
        ...
    }

    best_models = [(label_name, trained_model_object), ...]
    """