import os
import sys

import joblib
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append("..")
from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD
from grid_definition import define_grid


def main():
    # Test Code
    percentile_analysis(real=0, tag="test", prefix="test")

    # Grid Search --------------------------------
    tasks = []
    tags = set()

    for k, method, delta_window, daysahead, tag in define_grid():
        tags.add(tag)

    for tag in tags:
        tasks.append(
            joblib.delayed(percentile_analysis)(
                real=0,
                tag=tag,
                prefix=tag,
            )
        )

    n_jobs = 45
    with joblib_progress("Calculating percentile...", total=len(tasks)):
        joblib.Parallel(n_jobs=n_jobs, verbose=1000)(tasks)


@profile
def percentile_analysis(real, tag=None, prefix=None, verbose=0):
    # Return if already processed -----------------------------------------------
    prefix = prefix or ""
    out_file = f"data/processed/{prefix}/percentiles_R{real:03d}.csv"

    # if os.path.exists(out_file):
    #    return

    # Load dataframe ---------------------------------------------------------
    dfs = {}
    tag = tag or ""

    # for i in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
    for i in [3]:
        dfs[i] = pd.read_csv(
            f"data/processed/{tag}/processed_daysahead{i}_R{real:03d}.csv"
        )

    if verbose:
        print(dfs[1].head().to_string())

    # Calculate percentiles --------------------------------------------------
    percentiles = list(range(0, 100, 5))
    daysahead_cols = {daysahead: f"{daysahead} Days" for daysahead in dfs.keys()}
    records = {}

    for daysahead, colname in daysahead_cols.items():
        if verbose:
            print(colname)

        for percentile in percentiles:
            records[colname, percentile] = []

        for _, row in dfs[daysahead].iterrows():
            Vp_pred = row["forward_Vp_pred"]
            Vp_obs = row["forward_Vp_obs"]
            sigma = row["forward_sigma"]
            dist = norm(loc=Vp_pred, scale=sigma)

            for percentile in percentiles:
                left, right = dist.interval(percentile / 100)
                records[colname, percentile].append(
                    bool(Vp_obs > left and Vp_obs < right)
                )

    # Create output data frame -----------------------------------------------
    df_rows = []
    df_cols = None

    for idx, percentile in enumerate(percentiles):
        df_row = [percentile]
        df_cols = ["percentile"]

        for colname in daysahead_cols.values():
            df_row.append(100 * np.mean(records[colname, percentile]))
            df_cols.append(colname)

        df_rows.append(df_row)

    df_output = pd.DataFrame(df_rows, columns=df_cols)

    if verbose:
        pd.options.display.float_format = "{:.3}%".format
        print(df_output.to_string())

    df_output.to_csv(out_file, index=False)

    print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
