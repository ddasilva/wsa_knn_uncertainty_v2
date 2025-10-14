import numpy as np

from constants import (
    BIN_FREQ_PER_DAY,
    MIN_DAYSAHEAD,
    MAX_DAYSAHEAD,
)

def define_grid():
    grid = []
    
    for k in [25, 30, 35]:
        for method in ['method2']:
            for delta_window in [4]:
                for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
                    tag = f'{method}/k{k}/delta_window{delta_window}'
                    grid.append((k, method, delta_window, daysahead, tag))

    return grid
