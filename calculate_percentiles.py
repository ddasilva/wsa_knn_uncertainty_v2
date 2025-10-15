import os
import sys

import joblib
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd
from scipy.stats import norm, skewnorm
from tqdm import tqdm

sys.path.append("..")
from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD
from grid_definition import define_grid


def main():
    # Test Code
    #percentile_analysis(real=0, daysahead=3, tag="test", prefix="test", verbose=1)
    #return
    
    # Grid Search --------------------------------
    tasks = []
    items = set()

    for k, method, delta_window, daysahead, tag in define_grid():
        items.add((daysahead, tag))

    for daysahead, tag in items:
        tasks.append(
            joblib.delayed(percentile_analysis)(
                real=0,
                tag=tag,
                daysahead=daysahead,
                prefix=tag,
            )
        )

    n_jobs = 70
    
    with joblib_progress("Calculating percentiles...", total=len(tasks)):
        joblib.Parallel(n_jobs=n_jobs, verbose=1000)(tasks)


def percentile_analysis(real, daysahead, tag=None, prefix=None, verbose=0):
    # Return if already processed -----------------------------------------------
    prefix = prefix or ""
    out_file = f"data/processed/{prefix}/percentiles_daysahead{daysahead}_R{real:03d}.csv"

    if os.path.exists(out_file):
        return

    # Load dataframe ---------------------------------------------------------
    df = pd.read_csv(
        f"data/processed/{tag}/processed_daysahead{daysahead}_R{real:03d}.csv"
    )

    if verbose:
        print(df.head().to_string())

    # Calculate percentiles --------------------------------------------------
    percentiles = list(range(0, 100, 5))
    daysahead_cols = {daysahead: "ObservedPercentile"}
    records = {}

    for daysahead, colname in daysahead_cols.items():
        for percentile in percentiles:
            records[colname, percentile] = []

        rows = list(df.iterrows())
        if verbose:
            iterator = tqdm(rows)
        else:
            iterator = rows
            
        for _, row in iterator:
            Vp_pred = row["forward_Vp_pred"]
            Vp_obs = row["forward_Vp_obs"]
            mean = row["forward_mean"]
            sigma = row["forward_sigma"]
            skew = row["forward_skew"]

            if np.isnan(skew):
                dist = norm(loc=Vp_pred+mean, scale=sigma)
            else:
                dist = skewnorm(skew, loc=Vp_pred+mean, scale=sigma)

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
        df_cols = ["TruePercentile"]

        for colname in daysahead_cols.values():
            df_row.append(100 * np.mean(records[colname, percentile]))
            df_cols.append(colname)

        df_rows.append(df_row)

    df_output = pd.DataFrame(df_rows, columns=df_cols)

    if verbose:
        pd.options.display.float_format = "{:.3}%".format
        print(df_output.to_string())

    df_output.to_csv(out_file, index=False)

    if verbose:
        print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
