import pandas as pd
from sklearn.preprocessing import StandardScaler 

def preprocess(X_train, X_val, X_test):
    """Preprocess on the basis of training data"""
    # TODO: implement depending on what type of preprocessing we want to do (might depend on subtask)

    scaler = StandardScaler().fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train))
    X_val = pd.DataFrame(scaler.transform(X_val))

    print("Train data stats:")
    print("avg mean", X_train.describe().loc["mean", :10].mean())
    print("avg std", X_train.describe().loc["std", :10].mean())

    print("Val data stats:")
    print("avg mean", X_val.describe().loc["mean", :10].mean())
    print("avg std", X_val.describe().loc["std", :10].mean())

    return (
        scaler.transform(X_train), 
        scaler.transform(X_val),
        scaler.transform(X_test)
    )





