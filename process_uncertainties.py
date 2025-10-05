import os
import random
import sys

import numpy as np
import pandas as pd
import properscoring as ps
from termcolor import colored
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), ".."))
import util_wsa_uncertainty
from util_wsa_uncertainty import NPRED, NOBS

from constants import (
    MIN_DAYSAHEAD,
    MAX_DAYSAHEAD,
    N_REALS,
    NOBS,
    NPRED,
)


def main():
    do_processing(0, 3)

    #window_size_days = 4.5
    #do_processing(0, 3, nobs=int(4*window_size_days), npred=int(4*2*window_size_days), tag=f'windowSize{window_size_days}')

    
    #for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
    ##    do_processing(0, daysahead)
    #    for real in range(N_REALS):
    #        do_processing(real, daysahead)


def do_processing(real, daysahead, nobs=NOBS, npred=NPRED, tag=None):
    # Print status message
    print(colored(
        f"Woking on dayshead={daysahead} and real={real} and "
        f"npred={npred} and nobs={nobs} and tag={tag}", "green"))

    # Create k-NN dataset for nearest neighbor queries
    knn_dataset = util_wsa_uncertainty.KnnUncertaintyDataset(
        input_map="AGONG", sat="ACE", real=real, daysahead=daysahead,
        npred=npred, nobs=nobs,
    )

    # Load binned dataset to run through code. The code is smart enough to
    # not use neighbors close the target data we ask to predict uncertianty
    # for.
    df_dataset = pd.read_csv(knn_dataset.file_name, index_col=0)
    df_dataset.index = pd.to_datetime(df_dataset.index)
    df_dataset = df_dataset.interpolate()

    print("Loaded data:")
    print(df_dataset)

    # Do main loop ----------------------------------------------------------
    times_total = []
    sigmas_total = []
    Vp_pred_total = []
    Vp_obs_total = []
    crps_total = []
    
    inds = range(len(df_dataset) - npred)
    #sample = random.sample(inds, 100)
    sample = inds
    tasks = []

    for i in tqdm(sample):
        times = df_dataset.iloc[i : i + npred].index
        Vp_obs = df_dataset.Vp_obs.iloc[i : i + nobs]
        Vp_pred = df_dataset.Vp_pred.iloc[i : i + npred]

        sigma = util_wsa_uncertainty.calculate_uncertainty_gaussian(
            knn_dataset=knn_dataset,
            times=times,
            Vp_pred=Vp_pred,
            Vp_obs=Vp_obs,
            verbose=0,
        )

        time_nom = df_dataset.index[i + nobs]
        Vp_pred_nom = df_dataset.Vp_pred.iloc[i + nobs]
        Vp_obs_nom = df_dataset.Vp_obs.iloc[i + nobs]

        crps = ps.crps_gaussian(Vp_obs_nom, mu=Vp_pred_nom, sig=sigma)
        
        times_total.append(time_nom)
        sigmas_total.append(sigma)
        Vp_pred_total.append(Vp_pred_nom)
        Vp_obs_total.append(Vp_obs_nom)
        crps_total.append(crps)
        
    # Write to disk ------------------------------
    if tag:
        tag = f'{tag}/'
    else:
        tag = ''

    out_file = f"data/processed/{tag}processed_daysahead{daysahead}_R{real:03d}.csv"

    if not os.path.exists(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df_dict = dict(sigma=sigmas_total, Vp_pred=Vp_pred_total, Vp_obs=Vp_obs_total, crps=crps_total)
    df = pd.DataFrame(df_dict, index=times_total)
    df.to_csv(out_file)

    print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
