from sklearn.preprocessing import StandardScaler 

def preprocess(X_train, X_val, X_test):
    """Preprocess on the basis of training data"""
    # TODO: implement depending on what type of preprocessing we want to do (might depend on subtask)
    
    scaler = StandardScaler().fit(X_train)

    return (
        scaler.transform(X_train), 
        scaler.transform(X_val),
        scaler.transform(X_test)
    )





