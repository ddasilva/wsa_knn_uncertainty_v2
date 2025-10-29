import bisect
from dataclasses import dataclass
from datetime import timedelta
import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import skewnorm
from scipy.optimize import minimize
from constants import (
    BIN_FREQ,
    BIN_FREQ_PER_DAY,
    DELTA_WINDOW,
    DEFAULT_K,
    DEFAULT_METHOD,
)

# Path to WSA_DATA directory
WSA_DATA_PATH = "data/WSA_DATA"

# Avoid using k-NN items within this many timesteps of the target to avoid
# duplicates
PRUNE_THRESHOLD = 12

# Avoid using k-NN items within this timedelta of the target, for validation
# and data leakage issues
VALIDATION_CLOSENESS_THROWOUT = timedelta(days=27)

# Query this many neighbors and then subset to target k after pruning
INFLATE_K_PRUNE = {
    (0, 35): 500,
    (35, 50): 1000,
    (50, 100): 2500,
}

INFLATE_K_DEFAULT = {
    (0, 250): 500,
    (250, 500): 1000,
    (500, 1000): 2000,
}


def calculate_uncertainty_gaussian(
    knn_dataset,
    times,
    Vp_pred,
    Vp_obs,
    daysahead,
    method=DEFAULT_METHOD,
    k=DEFAULT_K,
    return_neighbors=False,
    verbose=1,
):
    """Calculate uncertainty using the gaussian approach.

    Args
      times: array of times, length knn_dataset.npred
      Vp_pred: array of predictions, length knn_dataset.npred
      Vp_obs: array of observations, length knn_dataset.nobs
    Return
      forward_time
      forward_loc
      forward_scale
      forward_shape
    """
    # Checks on function parameters
    assert len(times) == knn_dataset.npred
    assert len(Vp_pred) == knn_dataset.npred
    assert len(Vp_obs) == knn_dataset.nobs

    # Use knn_dataset to query neighbors
    neighbors = knn_dataset.lookup_neighbors(
        times=times,
        Vp_obs=Vp_obs,
        Vp_pred=Vp_pred,
        k=k,
    )

    assert len(neighbors) > 0

    weights = np.array([1 / nbr.distance**2 for nbr in neighbors])

    # Calculate sigma
    errors = np.array([nbr.after_obs[-1] - nbr.after_pred[-1] for nbr in neighbors])
    forward_time = times[-1]

    if method == "gaussian":
        mask = np.isfinite(weights) & np.isfinite(errors)
        variance = np.sum(weights[mask] * np.square(errors[mask])) / weights[mask].sum()
        # forward_mean = np.sum(weights[mask] * errors[mask]) / weights[mask].sum()
        forward_loc = Vp_pred.iloc[-1]
        forward_scale = np.sqrt(variance)
        forward_shape = np.nan
    elif method == "skew_gaussian":
        values = errors + Vp_pred.iloc[-1]
        mask = np.isfinite(weights) & np.isfinite(values)
        forward_shape, forward_loc, forward_scale = weighted_skewnorm_fit(
            values[mask], weights[mask]
        )
    else:
        raise RuntimeError(f"Unknown method {method}")

    # Return
    if return_neighbors:
        return_value = (
            forward_time,
            forward_loc,
            forward_scale,
            forward_shape,
            neighbors,
        )
    else:
        return_value = (forward_time, forward_loc, forward_scale, forward_shape)

    return return_value


def weighted_skewnorm_fit(data, weights):
    # Normalize weights to sum to 1 (optional but helps)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Negative log-likelihood function
    def nll(params):
        a, loc, scale = params
        if scale <= 0:
            return np.inf
        pdf_vals = skewnorm.pdf(data, a, loc=loc, scale=scale)
        # Add small epsilon to avoid log(0)
        return -np.sum(weights * np.log(pdf_vals + 1e-12))

    # Initial guess (use unweighted fit as a starting point)
    a0, loc0, scale0 = skewnorm.fit(data)
    res = minimize(
        nll,
        [a0, loc0, scale0],
        method="L-BFGS-B",
        bounds=[(-20, 20), (None, None), (1e-6, None)],
    )
    return res.x  # returns [a, loc, scale]


def weighted_skewnorm_fit_noloc(data, weights):
    # Normalize weights to sum to 1 (optional but helps)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Negative log-likelihood function
    def nll(params):
        a, scale = params
        if scale <= 0:
            return np.inf
        pdf_vals = skewnorm.pdf(data, a, loc=0, scale=scale)
        # Add small epsilon to avoid log(0)
        return -np.sum(weights * np.log(pdf_vals + 1e-12))

    # Initial guess (use unweighted fit as a starting point)
    a0, _, scale0 = skewnorm.fit(data)
    res = minimize(
        nll,
        [a0, scale0],
        method="L-BFGS-B",
        bounds=[(-20, 20), (1e-6, None)],
    )
    return res.x  # returns [a, scale]


@dataclass
class KnnUnceratintyNeighbor:
    """Data container for KnnUncertaintyDataset.lookup_neighbors()"""

    before_times: np.ndarray
    before_obs: np.ndarray
    before_pred: np.ndarray
    after_times: np.ndarray
    after_obs: np.ndarray
    after_pred: np.ndarray

    distance: float


class KnnUncertaintyDataset:

    def __init__(self, input_map, sat, real, daysahead, delta_window=DELTA_WINDOW):
        self.daysahead = daysahead
        self.nobs = delta_window
        self.npred = delta_window + daysahead * BIN_FREQ_PER_DAY

        self.file_name = (
            f"{WSA_DATA_PATH}/UNCERTAINTY/KNN_DATASET/"
            f"{input_map.upper()}/R{real:03d}/"
            f"{input_map}_{sat}_daysahead{daysahead}_R{real:03d}_knn_dataset.csv"
        )
        if not os.path.exists(self.file_name):
            raise FileNotFoundError(
                f"Looking for file, but did not exist: {self.file_name}"
            )

        self.df_knn = pd.read_csv(self.file_name, index_col=0)
        self.df_knn.index = pd.to_datetime(self.df_knn.index)

        self.X, self.Xtime, self.y = setup_knn_variables(
            self.df_knn, self.nobs, self.npred
        )
        self.tree = KDTree(self.X)
        (
            self.before_times,
            self.before_obs,
            self.before_pred,
            self.after_times,
            self.after_obs,
            self.after_pred,
        ) = unpack_knn_variables(self.X, self.Xtime, self.y, self.nobs, self.npred)

    def lookup_neighbors(self, times, Vp_obs, Vp_pred, k=DEFAULT_K, prune=False):
        """Lookup Neighbors

        Args
          times: array of datetime objects, size self.npred
          Vp_obs: array of recent observations, size self.nobs
          Vp_pred: array of recent/future predictions, size self.npred
          k: Number of neighbors to query; subject to pruning
        Returns
          list of KnnUnceratintyNeighbor instances
        """
        assert len(Vp_obs) == self.nobs
        assert len(Vp_pred) == self.npred
        assert len(times) == len(Vp_pred)

        # Setup query target for the KDTree
        query = np.zeros(self.nobs + self.npred)
        query[: self.nobs] = Vp_obs
        query[self.nobs :] = Vp_pred

        # If pruning, need to inflate the K w query for because
        # we will have to remove some members
        inflate_k_dict = INFLATE_K_PRUNE if prune else INFLATE_K_DEFAULT
        inflate_k = list(inflate_k_dict.values())[-1]

        for (start, stop), value in inflate_k_dict.items():
            if start <= k <= stop:
                inflate_k = value
                break

        # Branch base on pruning (emove neighbors that are very close to
        # eachother in time (e.g., within a carrington of eachother).
        distances, inds = self.tree.query(query, k=inflate_k)

        if prune:
            distances_pruned, inds_pruned = prune_inds(distances, inds)
        else:
            distances_pruned, inds_pruned = distances, inds

        # Collect neighbors
        neighbors = []

        for distance, ind in zip(distances_pruned, inds_pruned):
            # Avoid using k-NN items within this timedelta of the target, for
            # validation and data leakage issues
            skip = False

            for time in [times[0], times[-1]]:
                if (
                    len(self.before_times[ind]) > 0
                    and abs(self.before_times[ind][0] - time)
                    < VALIDATION_CLOSENESS_THROWOUT
                ):
                    skip = True
                    break
                if (
                    abs(self.after_times[ind][-1] - time)
                    < VALIDATION_CLOSENESS_THROWOUT
                ):
                    skip = True
                    break

            if skip:
                continue

            # Add neighbor to list
            neighbors.append(
                KnnUnceratintyNeighbor(
                    before_times=self.before_times[ind],
                    before_obs=self.before_obs[ind],
                    before_pred=self.before_pred[ind],
                    after_times=self.after_times[ind],
                    after_obs=self.after_obs[ind],
                    after_pred=self.after_pred[ind],
                    distance=distance,
                )
            )

        assert len(neighbors) >= k, f"INFLATE_K ({inflate_k}) too small for k={k}"

        return neighbors[:k]


# def prune_inds(distances, inds, threshold=PRUNE_THRESHOLD):
#     """Remove neighbors that are very close to eachother in time (e.g., with
#     a carrington of eachother)

#     Args
#       distances: array of distances to neighbors
#       inds: array of indices to neighbors
#     Returns
#       distances_pruned: array of distances to neighbors, pruned
#       inds_pruned: array of indices to neighbors, pruned
#     See Also
#       Module variable PRUNE_THRESHOLD (this module)
#     """
#     distances_pruned = []
#     inds_pruned = []

#     for d, ind in zip(distances, inds):
#         if all(abs(ind - i) > threshold for i in inds_pruned):
#             distances_pruned.append(d)
#             inds_pruned.append(int(ind))

#     distances_pruned = np.array(distances_pruned)
#     inds_pruned = np.array(inds_pruned)

#     return distances_pruned, inds_pruned


def prune_inds(distances, inds, threshold=PRUNE_THRESHOLD):
    """Remove neighbors that are very close to eachother in time (e.g., with
    a carrington of eachother)
    """
    # Ensure input as numpy arrays for easy processing
    distances = np.asarray(distances)
    inds = np.asarray(inds)

    # Sort by inds for reproducibility, optional
    sort_order = np.argsort(inds)
    inds = inds[sort_order]
    distances = distances[sort_order]

    distances_pruned = []
    inds_pruned = []

    for d, ind in zip(distances, inds):
        # inds_pruned is always sorted, so binary search for nearest placement
        insertion_point = bisect.bisect_left(inds_pruned, ind)
        ok = True
        # Check neighbor before
        if (
            insertion_point > 0
            and abs(ind - inds_pruned[insertion_point - 1]) <= threshold
        ):
            ok = False
        # Check neighbor after
        if (
            insertion_point < len(inds_pruned)
            and ok
            and abs(ind - inds_pruned[insertion_point : insertion_point + 1][0:1])
            <= threshold
        ):
            ok = False
        if ok:
            distances_pruned.append(d)
            inds_pruned.insert(insertion_point, ind)

    # Return as numpy arrays for consistency
    return np.array(distances_pruned), np.array(inds_pruned)


def setup_knn_variables(df_knn, nobs, npred):
    """Helper function for setting up the knn variables X and y and related
    variable Xtime (dateimes).

    Args
      df_knn: DataFrame holding k-NN dataset
      nobs: size of observation window
      npred: size of prediction window in addition to observation window
    Returns
      X, Xtime, y: numpy arrays
    """
    X = []
    Xtime = []
    y = []

    for i in range(nobs):
        X.append(df_knn.Vp_obs[i : -(npred - i)].values)  # past observations
        Xtime.append(df_knn.index[i : -(npred - i)].values)  # past observation times
    for i in range(npred):
        X.append(
            df_knn.Vp_pred[i : -(npred - i)].values
        )  # predictions for past and future data
        Xtime.append(
            df_knn.index[i : -(npred - i)].values
        )  # times for past and future data
        y.append(
            df_knn.Vp_obs[i : -(npred - i)].values
        )  # these are our actual future data

    X = np.array(X).T
    Xtime = np.array(Xtime).T
    y = np.array(y).T

    # remove nulls
    mask = np.isfinite(X).all(axis=1)
    X = X[mask, :]
    Xtime = Xtime[mask, :]
    y = y[mask, nobs:]

    return X, Xtime, y


def unpack_knn_variables(X, Xtime, y, nobs, npred):
    """A simpler way to interpret the large array X and y, by breaking
    them into components and storing in named dictionaries.
    """
    before_obs = {}
    before_pred = {}
    before_times = {}
    after_obs = {}
    after_pred = {}
    after_times = {}

    targets = np.arange(X.shape[0])

    for i in targets:
        before_times[i] = np.array(
            [pd.Timestamp(x) for x in Xtime[i, :nobs]], dtype=object
        )
        before_obs[i] = X[i, :nobs]
        before_pred[i] = X[i, nobs : 2 * nobs]

        after_times[i] = np.array(
            [pd.Timestamp(x) for x in Xtime[i, 2 * nobs :]], dtype=object
        )
        after_obs[i] = y[i, :]
        after_pred[i] = X[i, 2 * nobs :]

    return (
        before_times,
        before_obs,
        before_pred,
        after_times,
        after_obs,
        after_pred,
    )
