import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append("..")
from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD


def main():
    percentile_analysis_baseline(
        real=0,
        tag="method2/k20/delta_window4",
        prefix="baseline",
        verbose=1,
    )


def percentile_analysis_baseline(real, tag=None, prefix=None, verbose=1):
    # Return if already processed -----------------------------------------------
    prefix = prefix or ""
    out_file = f"data/processed/{prefix}/percentiles_R{real:03d}.csv"

    # if os.path.exists(out_file):
    #    return

    # Load dataframe ---------------------------------------------------------
    dfs = {}
    tag = tag or ""

    for i in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
        # for i in [1]:
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

        baseline_sigma = np.sqrt(
            np.mean(
                np.square(dfs[daysahead]["forward_Vp_pred"] - dfs[daysahead]["forward_Vp_obs"])
            )
        )

        for percentile in percentiles:
            records[colname, percentile] = []

            for _, row in dfs[daysahead].iterrows():
                Vp_pred = row["forward_Vp_pred"]
                Vp_obs = row["forward_Vp_obs"]
                left, right = norm(loc=Vp_pred, scale=baseline_sigma).interval(percentile / 100)

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

    dir_name = os.path.dirname(out_file)
    os.makedirs(dir_name, exist_ok=True)

    df_output.to_csv(out_file, index=False)

    print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
