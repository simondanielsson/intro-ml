from create_submission import OUT_PATH_VAL
from score_submission import get_score

import pandas as pd

def estimate_score():
    """Estimate final score based on validation set predictions"""

    prediction_val = pd.read_csv(OUT_PATH_VAL)

    val_label_path = "data/y_val.csv"
    true_val = pd.read_csv(val_label_path)

    score = get_score(true_val, prediction_val)
    print(f"Mean score {score:.4f}")


if __name__ == "__main__":
    estimate_score()