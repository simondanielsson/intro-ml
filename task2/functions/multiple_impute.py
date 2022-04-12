from typing import Tuple
import pandas as pd

from sklearn.linear_model import BayesianRidge

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def multivariate_impute(df: pd.DataFrame, 
                        max_iter: int = 10, 
                        estimator = BayesianRidge()) -> Tuple[pd.DataFrame, IterativeImputer]:
    
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter)
    df_imputed = imputer.fit_transform(df) 
    
    return df_imputed, imputer