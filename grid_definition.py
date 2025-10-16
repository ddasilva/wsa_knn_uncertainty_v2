import numpy as np

from constants import (
    BIN_FREQ_PER_DAY,
    MIN_DAYSAHEAD,
    MAX_DAYSAHEAD,
)


def define_grid():
    grid = []

    for k in range(50, 501, 50):
        for method in ["skew_gaussian"]:
            for delta_window in [8]:
                for daysahead in range(MIN_DAYSAHEAD, MAX_DAYSAHEAD + 1):
                    tag = f"{method}/k{k}/delta_window{delta_window}"
                    grid.append((k, method, delta_window, daysahead, tag))

    return grid
