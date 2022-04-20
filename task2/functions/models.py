from typing import List

def get_models(subtask: int) -> List:
    
    if subtask == 1: 
        models = [LogisticRegression(max_iter=500), LogisticRegression(solver="saga", max_iter=500), AdaBoostClassifier(random_state=random_state), LogisticRegressionCV()]

        return models 

    if subtask == 2:
        raise NotImplementedError()

    if subtask == 3:
        raise NotImplementedError()