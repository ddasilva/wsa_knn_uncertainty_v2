#!/bin/env python
"""Prepare training dataset for the Uncertainty KNN algorithm

Written by: Daniel da Silva <Daniel.e.daSilva@nasa.gov>
"""

import argparse
from datetime import datetime
import glob
import os

from astropy.time import Time
import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import tqdm


from constants import (
    BIN_FREQ,
    MIN_DAYSAHEAD,
    MAX_DAYSAHEAD,
    N_REALS,
)

# Julian dates stored in text files must have this bias added to them when they
# are converted to conventional julian dates (eg, with AstroPy).
JULDATE_BIAS = 2440000.0


def main():
    # Get command line arguments
    args = get_parser().parse_args()

    cprint("Arguments", "green")
    print(args)
    print()

    # Do work for each value of daysahead
    for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
        do_daysahead(args, daysahead)


def do_daysahead(args, daysahead):
    """Process a batch of files for a given daysahead subset.

    Args
      args: command line argumnets
      daysahead: int
    """

    # Get files to parse
    files_per_real = get_files_per_real(args, daysahead)

    # Get single (long) dataframe with all realizations mixed
    dfs_unbinned = load_unbinned_dataframe(files_per_real, args)

    # Bin the dataframe
    dfs_binned = bin_dataframes(dfs_unbinned, args)

    # Load observations
    df_obs = get_obs_data(args, dfs_binned)

    # Add observations
    dfs_final = add_obs_data(dfs_binned, df_obs)

    # Write output to disk
    cprint("Writing to disk", "green")

    for real, df_final in dfs_final.items():
        out_dir = os.path.join(
            "data/",
            "WSA_DATA/",
            "UNCERTAINTY/",
            "KNN_DATASET/",
            args.input_map.upper(),
            f"R{real:03d}",
        )

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(
            out_dir,
            f"{args.input_map}_{args.sat.upper()}_daysahead{daysahead}_R{real:03d}_knn_dataset.csv",
        )

        df_final.to_csv(out_path, na_rep="NaN")

        print(f"Wrote to {out_path}")

    print()


def get_obs_data(args, dfs_binned):
    """Load solar wind observations

    Args
      args: command line arguments
      df_binned: dict, int to binned predictions data frame
    Returns
      df_obs: Observation data frame
    """
    cprint("Loading solar wind observations...", "green")

    obs_data_path = f"data/{args.sat.lower()}_hr.dat"

    df_obs = pd.read_csv(obs_data_path, sep="\\s+", comment="#")
    df_obs = df_obs[df_obs.year >= dfs_binned[0].index.min().year]
    df_obs = df_obs[df_obs.year <= dfs_binned[0].index.max().year]
    df_obs.index = [
        datetime(int(row.year), int(row.month), int(row.day), int(row.hour))
        for _, row in df_obs.iterrows()
    ]

    df_obs = df_obs[["Vp_obs"]]
    df_obs.loc[df_obs.Vp_obs < 0.1, "Vp_obs"] = np.nan
    df_obs["Vp_obs"] = df_obs["Vp_obs"].resample(args.bin_freq).mean()

    return df_obs


def add_obs_data(dfs_binned, df_obs):
    """Merge observations and binned predictions

    Args
      dfs_binned: binned dataframe to regular cadence with
         averaging for all realizations
      df_obs: Observation data frame
    Returns
      df_final: Final datafrmae to write to disk
    """
    dfs_final = {}

    for real, df_binned in dfs_binned.items():
        # Join dataframes on the binned indeces
        dfs_final[real] = df_binned.join(df_obs, how="left")

        # Interpolate over single missing row of NaN's
        for col in dfs_final[real].columns:
            dfs_final[real][col] = fill_single_gaps(dfs_final[real][col])

    cprint("Final dataframe (with observations)", "green")

    return dfs_final


def bin_dataframes(dfs_unbinned, args):
    """Bin the data into regular cadence with mean and std
    calculated.

    Args
      dfs_unbinned: dict, int to unbinned dataframe
      args: command line arguments
    Returns
      dict, int to binned dataframe
    """
    # Do resampling
    dfs_binned = {}

    for real, df_unbinned in dfs_unbinned.items():
        Vp_pred_mean = df_unbinned.Vp.resample(args.bin_freq).mean()

        df_contents = {"Vp_pred": Vp_pred_mean}
        dfs_binned[real] = pd.DataFrame(df_contents, index=Vp_pred_mean.index)

    # Print output
    cprint("Binned dataframes", "green")

    return dfs_binned


def load_unbinned_dataframe(files_per_real, args):
    """Real all files per realization and concat them into a
    single dataframe.

    Return
      Dataframe
    """
    cprint("Loading csv files...", "green")

    dfs_unbinned = {}

    for real, file_list in tqdm(list(files_per_real.items())):
        dfs = []

        for file_path in file_list:
            df = read_file(file_path, args)
            df["real"] = real
            dfs.append(df)

        dfs_unbinned[real] = pd.concat(dfs)

    print()

    return dfs_unbinned


def read_file(file_path, args):
    """Return parsed pred file.

    Args
       pred_file: Path to pred file
       no_date_parse: (optional) set to true to disable parsing dates,
         which is much faster.
    Returns
       Pandas data frame with the juldate column replaced with datetime
       instances.
    """
    df = pd.read_csv(file_path, sep="\\s+", comment="#")
    df.index = [
        Time(float(s) + JULDATE_BIAS, format="jd").to_datetime() for s in df["juldate"]
    ]
    df = df[df.year >= args.min_year]
    df = df[df.year <= args.max_year]

    return df


def get_files_per_real(args, daysahead):
    """Get dictionary that maps integer realization to list of
    field_line files for that realization.

    Returns
       dict, int -> list[str]
    """
    files_per_real = {}

    for real in range(N_REALS):
        files_per_real[real] = glob.glob(
            f"{args.data_root}/*/PREDSOLARWIND/"
            f"{args.input_map}field_line{daysahead}R{real:03d}.dat"
        )

        cprint(f"Files for realization {real}", "green")
        for line in files_per_real[real]:
            print(line)

    return files_per_real


def fill_single_gaps(series):
    """Fill single NaN gaps in a time series with interpolated values

    Args
      series: pandas series to fix
    Returns
      series: pandas series with inteprolated values
    """
    # Mask marking single NaNs (not adjacent to another NaN)
    is_na = series.isna()
    # Shift the mask to the left and right
    left_na = is_na.shift(1, fill_value=False)
    right_na = is_na.shift(-1, fill_value=False)
    # Single gap: NaN where neither left nor right is NaN
    single_gap = is_na & (~left_na) & (~right_na)
    # Interpolate as usual
    interpolated = series.interpolate()
    # Fill only single-gap NaNs with interpolated values
    return series.where(~single_gap, interpolated)


def get_parser():
    """Define command line arguments

    Returns
      parser: instance of ArgumentParser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="/data/dedasilv/wsa/LearnedUncertainty")
    parser.add_argument("--sat", default="ACE")
    parser.add_argument("--input_map", default="AGONG")
    parser.add_argument("--bin-freq", default=BIN_FREQ)
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-year", type=int, default=2020)

    return parser


if __name__ == "__main__":
    main()
