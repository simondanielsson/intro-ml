#### SCORER ###
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

def scorer(data: pd.DataFrame, model, splitter: float) -> tuple:
    """Fits a given model to a given dataset and 
    returns the score on both training and testing as a tuple."""

    # Find the row to split dataset in training and testing set
    split = int(data.shape[0] * splitter)

    # Sclice the dataframe accordingly
    features_train = data.iloc[1:split,:-1]
    labels_train = data.iloc[1:split,-1]

    features_test = data.iloc[split:,:-1]
    labels_test = data.iloc[split:,-1]

    # Fit the model onto training data
    model.fit(features_train, labels_train)

    # Compute score for both training and testing
    training_score = model.score(features_train, labels_train)
    test_score = model.score(features_test, labels_test)

    return (training_score, test_score)


def evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict[str, Tuple]:
    """Evaluates performance of a model wrt a specific label column"""

    model.fit(X_train, y_train)

    training_score = model.score(X_train, y_train)
    test_score = model.score(X_val, y_val)

    return (training_score, test_score)


def scoreboard(models: list, X_train, y_trains: List[pd.Series], X_val, y_vals: List[pd.Series]) -> dict:
    """Takes a list of several models and runs them on the same dataset.
    Trains each model wrt to each label column in the list of label columns. 
    Training and Test scores are written to a dictionaru per model for each label column."""

    # Initializing output dict
    scores = dict()
    scores_model = dict()
    best_score = {y_train.name: 0 for y_train in y_trains}
    progress = dict() 

    # Train each model on each label column
    for model in models:
        model_name = type(model).__name__
        print(f"Evaluating {model_name}...")

        for y_train, y_val in zip(y_trains, y_vals):
            label = y_train.name
            prev_best_score = best_score[label]
            
            score = evaluate_model(model, X_train, y_train, X_val, y_val)
            scores_model[label] = score
            
            # Printing progress
            val_score = score[1]
            if val_score > prev_best_score:
                print(f"[Progress] Best validation score on column {label}:")
                print(f"    {model_name}: {val_score:.4f}")

                best_score[label] = val_score

                if "DummyClassifier" in list(scores.keys()):
                    val_score_dummy = scores["DummyClassifier"][label][1]
                    print(f"    DummyClassifier: {val_score_dummy:.4f}\n")
                    print(f"    Progress: {val_score - val_score_dummy}\n")

                    progress[label] = val_score - val_score_dummy
        
        scores[model_name] = scores_model

    print(f"Total progress: {progress}")
    return scores