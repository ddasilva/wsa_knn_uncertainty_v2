#!/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD
from grid_definition import define_grid


def main():
    # Search for minimum score
    scores = {}
    score_tags = {}

    for k, method, delta_window, daysahead, tag in tqdm(define_grid()):
        if daysahead not in scores:
            scores[daysahead] = []
            score_tags[daysahead] = []

        score = get_score(0, tag, daysahead)
        scores[daysahead].append(score)
        score_tags[daysahead].append(tag)

    for i in range(len(scores[1])):
        print(f"Rank {i}-------------------")
        df_rows = []

        for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
            I = np.argsort(scores[daysahead])
            df_rows.append(
                [
                    daysahead,
                    scores[daysahead][I[i]],
                    score_tags[daysahead][I[i]],
                ]
            )

        df = pd.DataFrame(df_rows, columns=["Days Ahead", "Score", "Tag"])

        print(df.to_string())


def get_score(real, tag, daysahead):
    percentiles_path = (
        f"data/processed/{tag}/percentiles_daysahead{daysahead}_R{real:03d}.csv"
    )
    df = pd.read_csv(percentiles_path)

    percentiles_true = np.array(df["TruePercentile"].tolist() + [100])
    score = 0

    percentiles_pred = np.array(df["ObservedPercentile"].tolist() + [100])
    score += np.trapz((percentiles_true - percentiles_pred) ** 2, percentiles_true)

    return score


if __name__ == "__main__":
    main()
