from subtask1 import PATHS_PRED

import pandas as pd

OUT_PATH = "/predictions/submission.csv"

if __name__ == "__main__":
    predictions = [pd.read_csv(path, index_col=0) for _, path in PATHS_PRED.items()]
    prediction = pd.concat(predictions)

    prediction.to_csv(OUT_PATH)



