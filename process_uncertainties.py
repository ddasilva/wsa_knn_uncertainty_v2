import os
import random
import sys

import joblib
from joblib_progress import joblib_progress
import numpy as np
import pandas as pd
import properscoring as ps
from termcolor import colored
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), ".."))
import util_wsa_uncertainty

from grid_definition import define_grid


def main():
    # Test code -------------------------------------------------------
    # do_processing(real=0, daysahead=3, method='skew_gaussian', tag='test', k=100, delta_window=8, verbose=1)
    # return

    # Grid search calibration ----------------------------------------
    tasks = []

    for k, method, delta_window, daysahead, tag in define_grid():
        tasks.append(
            joblib.delayed(do_processing)(
                real=0,
                daysahead=daysahead,
                method=method,
                tag=tag,
                k=k,
                delta_window=delta_window,
                verbose=0,
            )
        )

    n_jobs = 65

    with joblib_progress("Processing Uncetainties...", total=len(tasks)):
        joblib.Parallel(n_jobs=n_jobs, verbose=1000)(tasks)


def do_processing(real, daysahead, method, tag, k, delta_window, verbose=1):
    if tag:
        tag = f"{tag}/"
    else:
        tag = ""

    out_file = f"data/processed/{tag}processed_daysahead{daysahead}_R{real:03d}.csv"

    if os.path.exists(out_file):
        return

    # Print status message
    if verbose:
        print(
            colored(
                f"Woking on dayshead={daysahead} and real={real} and "
                f"delta_window={delta_window} tag={tag}",
                "green",
            )
        )

    # Create k-NN dataset for nearest neighbor queries
    knn_dataset = util_wsa_uncertainty.KnnUncertaintyDataset(
        input_map="AGONG",
        sat="ACE",
        real=real,
        daysahead=daysahead,
        delta_window=delta_window,
    )

    # Load binned dataset to run through code. The code is smart enough to
    # not use neighbors close the target data we ask to predict uncertianty
    # for.
    df_dataset = pd.read_csv(knn_dataset.file_name, index_col=0)
    df_dataset.index = pd.to_datetime(df_dataset.index)
    df_dataset = df_dataset.interpolate()

    if verbose:
        print("Loaded data:")
        print(df_dataset)

    # Do main loop ----------------------------------------------------------
    df_rows = []

    inds = range(len(df_dataset) - knn_dataset.npred)
    # sample = random.sample(inds, 10)
    sample = inds
    cols = None

    if k is None:
        k = util_wsa_uncertainty.DEFAULT_K

    if verbose:
        iterator = tqdm(sample)
    else:
        iterator = sample

    for i in iterator:
        times = df_dataset.iloc[i : i + knn_dataset.npred].index
        Vp_obs = df_dataset.Vp_obs.iloc[i : i + delta_window]
        Vp_pred = df_dataset.Vp_pred.iloc[i : i + knn_dataset.npred]

        sigma_time, mean, sigma, skew = (
            util_wsa_uncertainty.calculate_uncertainty_gaussian(
                knn_dataset=knn_dataset,
                times=times,
                Vp_pred=Vp_pred,
                Vp_obs=Vp_obs,
                method=method,
                daysahead=daysahead,
                k=k,
                verbose=0,
            )
        )

        current_time = df_dataset.index[i + delta_window]
        Vp_pred_nom = df_dataset.Vp_pred.iloc[i + knn_dataset.npred]
        Vp_obs_nom = df_dataset.Vp_obs.iloc[i + knn_dataset.npred]
        crps = ps.crps_gaussian(Vp_obs_nom, mu=Vp_pred_nom + mean, sig=sigma)

        df_row = [
            current_time,
            sigma_time,
            Vp_pred_nom,
            Vp_obs_nom,
            mean,
            sigma,
            skew,
            crps,
        ]

        df_rows.append(df_row)

    # Write to disk ------------------------------
    cols = [
        "current_time",
        "forward_time",
        "forward_Vp_pred",
        "forward_Vp_obs",
        "forward_mean",
        "forward_sigma",
        "forward_skew",
        "forward_crps",
    ]

    if not os.path.exists(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df = pd.DataFrame(df_rows, columns=cols)
    df.to_csv(out_file, index=0)

    if verbose:
        print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
