import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append('..')
from constants import MIN_DAYSAHEAD, MAX_DAYSAHEAD, BIN_FREQ_PER_DAY


def main():
    #percentile_analysis(0, prefix='baseline_', verbose=1, do_baseline=True)
    percentile_analysis(0, verbose=1)
    
    # tasks = []

    # for delta_window in range(1, 27):
    #     for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
    #         prefix = f'delta_window{delta_window}/'
    #         tasks.append(joblib.delayed(percentile_analysis)(
    #             0, prefix=prefix
    #         ))

    # joblib.Parallel(n_jobs=36, verbose=1000)(tasks)


def percentile_analysis(real, prefix=None, verbose=0, do_baseline=False):
    # Load dataframe ---------------------------------------------------------
    dfs = {}
    
    for i in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
        out_file = f"data/processed/processed_daysahead{i}_R{real:03d}.csv"
        dfs[i] = pd.read_csv(out_file)

    if verbose:
        print(dfs[1].head().to_string())


    # Calculate percentiles --------------------------------------------------
    percentiles = list(range(0, 100, 10))
    daysahead_cols = {daysahead: f'{daysahead} Days' for daysahead in dfs.keys()}
    records = {}

    for daysahead, colname in daysahead_cols.items():
        if verbose:
            print(colname)

        i = BIN_FREQ_PER_DAY * daysahead - 1
        baseline_sigma = np.sqrt(np.mean(np.square(
            dfs[daysahead][f'Vp_pred{i}'] - dfs[daysahead][f'Vp_obs{i}']
        )))
            
        for percentile in percentiles:
            records[colname, percentile] = []

            for _, row in dfs[daysahead].iterrows():
                Vp_pred = row[f'Vp_pred{i}']
                Vp_obs = row[f'Vp_obs{i}']

                if do_baseline:
                    sigma = baseline_sigma
                else:                
                    sigma = row[f'sigma{i}']
                    
                left, right = norm(loc=Vp_pred, scale=sigma).interval(percentile/100)
                
                records[colname, percentile].append(
                    bool(Vp_obs > left and Vp_obs < right)
                )

    # Set prefix for filename bit -----------------------------------------------
    if not prefix:
        prefix = ""
                
    # Create output data frame -----------------------------------------------
    df_rows = []
    df_cols = None
    
    for idx, percentile in enumerate(percentiles):
        df_row = [percentile]
        df_cols = ['percentile']
        
        for colname in daysahead_cols.values():
            df_row.append(100*np.mean(records[colname, percentile]))
            df_cols.append(colname)

        df_rows.append(df_row)
    
    df_output = pd.DataFrame(df_rows, columns=df_cols)

    if verbose:
        pd.options.display.float_format = '{:.3}%'.format
        print(df_output.to_string())

    out_file = f"data/processed/{prefix}percentiles_R{real:03d}.csv"
    df_output.to_csv(out_file, index=False)
    print(f'Wrote to {out_file}')


if __name__ == '__main__':
    main()
