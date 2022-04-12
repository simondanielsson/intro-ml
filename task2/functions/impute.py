import pandas as pd 

from single_impute import single_impute
from multiple_impute import multivariate_impute

def impute(df: pd.DataFrame, threshold: float, max_iter: int) -> pd.DataFrame:
    """Perform single imputation if proportion of missing values of a column is 
    less than threshold, and performs multivariate imputation otherwise.
    
    Note that mutlivariate imputation takes very long to converge; you'll likely want to 
    perform it once and save it to disk."""

    df_single, single_imputers = single_impute(df, threshold) 
    df_multi, multi_imputer = multivariate_impute(df_single, max_iter=max_iter)

    # You'll need the imptuter objects when imputing any new data, e.g. the test set 
    return df_multi, (single_imputers, multi_imputer)
