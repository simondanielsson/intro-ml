from typing import List
import argparse

from random import random
import pandas as pd
from pprint import pprint

from functions.scoring import scoreboard
from functions.select_feature import select_feature, select_labels
from functions.impute import impute, load_imputed_data
from functions.preprocess import preprocess 
from functions.models import get_models 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier


def load_data(paths: List[str]) -> List[pd.DataFrame]:
    """Loads data from paths and returns the data as a list of dataframes"""
    dataframes = []

    for path in paths:
        dataframes.append(pd.read_csv(path))

    return dataframes


def present_results(scores: dict) -> None: 
    """Presents scores"""    
    # TODO: implement something more verbose, graphs etc. 
    pprint(scores)


def main(in_paths: str, subtask: int, update: bool = False, verbose: int = 0) -> None:
    """Evaluate models on data set
    in_path: path of original data
    If update, then imputation is re-done and saved as csv. Otherwise, load imputed data from csv"""
    random_state = 1
    
    if update:
        # Load train and test data
        X, y, X_test = load_data(in_paths)

        # Split data into train and validation set
        train_size = 0.7 # TODO: set

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=random_state)
    
        # Impute and load data
        threshold = 0.7 # TODO set
        max_iter = 10
        
        impute(
            X_train, 
            X_val, 
            X_test, 
            y_train, 
            y_val, 
            threshold=threshold, 
            max_iter=max_iter, 
            verbose=verbose
        )

    X_train, X_val, X_test, y_train, y_val = load_imputed_data()

    # Select features
    # TODO: implement
    if verbose:
        print("Selecting features")
    """
    feature_indeces_to_select = select_feature(X_train, y_train)
    X_train, X_val, X_test= (
        X_train.iloc[:,feature_indeces_to_select],
        X_val.iloc[:,feature_indeces_to_select],
        X_test.iloc[:,feature_indeces_to_select],
    )
    """

    # Select label features for this specific problem 
    label_indeces_to_select = select_labels(subtask)
    y_train, y_val = (
        y_train.loc[:, label_indeces_to_select],
        y_val.loc[:, label_indeces_to_select]
    )

    # Preprocess data
    print("Preprocessing data")
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    # Evaluate models
    models = get_models(subtask)
    print(f"Testing models {models}")

    print(f"Evaluating models {models}")
    scores = scoreboard(models, X_train, y_train, X_val, y_val)

    # Present scores
    present_results(scores)

    # Make predictions on test set and save as csv 
    # TODO
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model pipeline for subtask 1')
    parser.add_argument("subtask", metavar="N", type=int, nargs=1, help="Choice of subtask to be run")
    parser.add_argument("-u", "--update-imputation", action="store_true", help="Update the imputated dataset, and save as csv")
    parser.add_argument("-v", "--verbose", action="store_const", const=2, help="Verbose output")
    args = vars(parser.parse_args())

    subtask = args["subtask"][0]
    update_imputation = args["update_imputation"]
    verbose = args["verbose"] if not "None" else 0

    # Paths for loading original data used for imputation
    in_paths =["data/train_features_ts.csv", "data/train_labels.csv", "data/test_features_ts.csv"]

    main(in_paths, subtask=subtask, update=update_imputation, verbose=verbose)