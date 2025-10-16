import os
import sys

import joblib
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append("..")
from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD
import util_wsa_uncertainty


def main():
    tasks = []

    for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
        tasks.append(
            joblib.delayed(percentile_analysis_baseline)(
                real=0,
                daysahead=daysahead,
                verbose=0,
            )
        )

    n_jobs = 7

    with joblib_progress("Calculating baseline percentiles...", total=len(tasks)):
        joblib.Parallel(n_jobs=n_jobs, verbose=1000)(tasks)


def percentile_analysis_baseline(real, daysahead, prefix=None, verbose=1):
    # Return if already processed -----------------------------------------------
    prefix = prefix or ""
    out_file = (
        f"data/processed/baseline/percentiles_daysahead{daysahead}_R{real:03d}.csv"
    )

    if os.path.exists(out_file):
        return

    # Load dataframe ---------------------------------------------------------
    dfs = {}

    for i in [daysahead]:
        knn_dataset = util_wsa_uncertainty.KnnUncertaintyDataset(
            input_map="AGONG",
            sat="ACE",
            real=real,
            daysahead=daysahead,
        )
        dfs[i] = pd.read_csv(knn_dataset.file_name, index_col=0).dropna()

    if verbose:
        print(dfs[1].head().to_string())

    # Calculate percentiles --------------------------------------------------
    percentiles = list(range(0, 100, 5))
    daysahead_cols = {daysahead: "ObservedPercentile"}
    records = {}

    for daysahead, colname in daysahead_cols.items():
        if verbose:
            print(colname)

        baseline_sigma = np.sqrt(
            np.mean(np.square(dfs[daysahead]["Vp_pred"] - dfs[daysahead]["Vp_obs"]))
        )

        for percentile in percentiles:
            records[colname, percentile] = []

            for _, row in dfs[daysahead].iterrows():
                Vp_pred = row["Vp_pred"]
                Vp_obs = row["Vp_obs"]
                left, right = norm(loc=Vp_pred, scale=baseline_sigma).interval(
                    percentile / 100
                )

                records[colname, percentile].append(
                    bool(Vp_obs > left and Vp_obs < right)
                )

    # Create output data frame -----------------------------------------------
    df_rows = []
    df_cols = None

    for idx, percentile in enumerate(percentiles):
        df_row = [percentile]
        df_cols = ["TruePercentile"]

        for colname in daysahead_cols.values():
            df_row.append(100 * np.mean(records[colname, percentile]))
            df_cols.append(colname)

        df_rows.append(df_row)

    df_output = pd.DataFrame(df_rows, columns=df_cols)

    if verbose:
        pd.options.display.float_format = "{:.3}%".format
        print(df_output.to_string())

    dir_name = os.path.dirname(out_file)
    os.makedirs(dir_name, exist_ok=True)

    df_output.to_csv(out_file, index=False)

    if verbose:
        print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
