import os
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), ".."))
import util_wsa_uncertainty
from util_wsa_uncertainty import NPRED, NOBS

BASE_DATASET = "data/WSA_DATA/UNCERTAINTY/KNN_DATASET/AGONG/R000/AGONG_ACE_daysahead3_R000_knn_dataset.csv"


def main():
    # Load binned dataset to run through code. The code is smart enough to
    # not use neighbors close the target data we ask to predict uncertianty
    # for.
    df_dataset = pd.read_csv(BASE_DATASET, index_col=0)
    df_dataset.index = pd.to_datetime(df_dataset.index)

    # Create k-NN dataset for nearest neighbor queries
    knn_dataset = util_wsa_uncertainty.KnnUncertaintyDataset(
        input_map="AGONG", sat="ACE", real=0, daysahead=3
    )

    # Do main loop ----------------------------------------------------------
    sigmas = []

    inds = range(len(df_dataset) - util_wsa_uncertainty.NPRED)
    # sample = random.sample(inds, 5000)
    sample = inds

    for i in tqdm(sample):
        times = df_dataset.iloc[i : i + NPRED].index
        Vp_obs = df_dataset.Vp_obs.iloc[i : i + NOBS]
        Vp_pred = df_dataset.Vp_pred.iloc[i : i + NPRED]

        if not (np.isfinite(Vp_obs).all() and np.isfinite(Vp_pred).all()):
            continue

        sigma = util_wsa_uncertainty.calculate_uncertainty_gaussian(
            knn_dataset=knn_dataset,
            times=times,
            Vp_pred=Vp_pred,
            Vp_obs=Vp_obs,
            verbose=0,
        )
        sigmas.append(sigma)

    # Write to disk ------------------------------
    out_file = "data/histogram_plot_data.csv"
    pd.DataFrame(dict(sigma=sigmas)).to_csv(out_file, index=False)

    print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()
