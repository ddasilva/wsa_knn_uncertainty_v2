from datetime import datetime, timedelta
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from util_wsa_uncertainty import (
    KnnUncertaintyDataset,
    NOBS,
    NPRED,
    prune_inds,
)


def test_constructor():
    """Tests the KnnUncertaintyDataset constructor with valid args"""
    KnnUncertaintyDataset(input_map="AGONG", sat="ACE", real=0, daysahead=3)


def test_constructor_does_not_exist():
    """Tests the KnnUncertaintyDataset constructor with invalid args"""
    with pytest.raises(FileNotFoundError):
        KnnUncertaintyDataset(
            input_map="DOES_NOT_EXIST", sat="ACE", real=0, daysahead=3
        )


def test_lookup_neighbors():
    """Tests looking up a neighbor without making too many assumptions about the
    dataset.
    """
    k = 1000
    required_nbrs = 50

    dataset = KnnUncertaintyDataset(
        input_map="AGONG",
        sat="ACE",
        real=0,
        daysahead=3,
    )

    times = np.array(
        [datetime(2015, 1, 1) + i * timedelta(hours=6) for i in range(NPRED)],
        dtype=object,
    )

    Vp_obs = np.zeros(NOBS) + 450
    Vp_pred = np.zeros(NPRED) + 500

    neighbors = dataset.lookup_neighbors(
        times=times,
        Vp_obs=Vp_obs,
        Vp_pred=Vp_pred,
        k=k,
    )
    distances = np.array([n.distance for n in neighbors])

    assert len(neighbors) > required_nbrs
    assert len(neighbors) < k

    assert np.all(distances[:-1] <= distances[1:])  # monotonically increasing

    for nbr in neighbors:
        assert isinstance(nbr.before_times[0], datetime)
        assert isinstance(nbr.after_times[0], datetime)

        assert len(nbr.before_times) == NOBS
        assert len(nbr.before_obs) == NOBS
        assert len(nbr.before_pred) == NOBS
        assert len(nbr.after_times) == NPRED - NOBS
        assert len(nbr.after_obs) == NPRED - NOBS
        assert len(nbr.after_pred) == NPRED - NOBS


def test_prune_inds():
    """Tests for the prune inds function."""
    distances = np.arange(10)
    inds = np.array([0, 1, 2, 3, 4, 50, 51, 100, 150, 180])
    threshold = 12

    distances_pruned, inds_pruned = prune_inds(distances, inds, threshold=threshold)

    assert np.all(distances_pruned == [0, 5, 7, 8, 9])
    assert np.all(inds_pruned == [0, 50, 100, 150, 180])
