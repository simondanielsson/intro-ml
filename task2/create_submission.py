from subtask1 import PATHS_PRED_TEST, PATHS_PRED_VAL
import os 
import pandas as pd

OUT_PATH = "./predictions/submission.csv"
OUT_PATH_VAL = "./predictions/val_predictions.csv"

def main():
    # Save test and val set predictions

    for in_paths, out_path in zip([PATHS_PRED_TEST, PATHS_PRED_VAL], [OUT_PATH, OUT_PATH_VAL]):

        pred_1 = pd.read_csv(in_paths["1"], index_col=0)
        pred_2 = pd.DataFrame(pd.read_csv(in_paths["2"])["LABEL_Sepsis"])
        pred_3 = pd.read_csv(in_paths["3"]).iloc[:, 1:]

        prediction = pd.concat([pred_1, pred_2, pred_3], axis=1)

        pathfile = os.path.normpath(os.path.join(out_path))
        file_present = os.path.isfile(pathfile) 
        if file_present:
            print(f"Overwriting file {out_path}...")
            os.remove(pathfile)

        prediction.to_csv(out_path)
    
    print("Submissions created successfully")


if __name__ == "__main__":
    main()





