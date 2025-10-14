 #!/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD, BIN_FREQ_PER_DAY
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
        print(f'Rank {i}-------------------')
        df_rows = []
        
        for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
            I = np.argsort(scores[daysahead])
            df_rows.append([
                daysahead,
                scores[daysahead][I[i]],
                score_tags[daysahead][I[i]],
            ])

        df = pd.DataFrame(df_rows, columns=['Days Ahead', 'Score', 'Tag'])

        print(df.to_string())


def get_score(real, tag, daysahead):

# Load dataframe ---------------------------------------------------------
    df = pd.read_csv(f"data/processed/{tag}/processed_daysahead{daysahead}_R{real:03d}.csv")

    i = BIN_FREQ_PER_DAY * daysahead - 1
    score = df[f'crps{i}'].mean()
    #score = np.median(df[f'crps{i}'])
    
    
    return score

    
if __name__ == '__main__':
    main()
