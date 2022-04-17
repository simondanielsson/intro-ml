from typing import List

import pandas as pd 

from functions.single_impute import single_impute
from functions.multiple_impute import multivariate_impute

OUT_PATHS = (
        "data/X_train_imputed.csv", 
        "data/X_val_imputed.csv",
        "data/X_test_imputed.csv",
        "data/y_train.csv",
        "data/y_val.csv"
)


def impute_single_df(df: pd.DataFrame, threshold: float, max_iter: int, verbose: int = 0) -> pd.DataFrame:
    """Perform single imputation if proportion of missing values of a column is 
    less than threshold, and performs multivariate imputation otherwise.
    
    Note that mutlivariate imputation takes very long to converge; you'll likely want to 
    perform it once and save it to disk."""

    df_single, single_imputers = single_impute(df, threshold) 
    df_multi, multi_imputer = multivariate_impute(df_single, max_iter=max_iter, verbose=verbose)

    # You'll need the imptuter objects when imputing any new data, e.g. the test set 
    return df_multi, (single_imputers, multi_imputer)


def impute_new_df(df: pd.DataFrame, single_imputers: dict, multi_imputer) -> pd.DataFrame:
    """Impute df using given imputers""" 
    df = df.copy()

    print("Imputing new df")
    # Single impute columns
    for feature, imputer in single_imputers.items():
        df[feature] = imputer.transform(df[feature].values.reshape(-1, 1))

    # Multiple impute the remaining columns
    df = pd.DataFrame(multi_imputer.transform(df))

    return df 


def impute(X_train, X_val, X_test, y_train, y_val, threshold, max_iter, verbose) -> None:
    """Impute data on basis of training set and save as csv"""   

    X_train, (single_imputers, multi_imputer) = impute_single_df(X_train, threshold, max_iter, verbose)

    X_val, X_test = (
        impute_new_df(X_val, single_imputers, multi_imputer), 
        impute_new_df(X_test, single_imputers, multi_imputer)
    )

    for path, data in zip(OUT_PATHS, [X_train, X_val, X_test, y_train, y_val]):
        if verbose != 0:
            print(f"Saving data to {path}")
        data.to_csv(path)


def load_imputed_data() -> List[pd.DataFrame]:
    data = [pd.read_csv(path) for path in OUT_PATHS]

    return data
