from dataclasses import dataclass
from datetime import timedelta
import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# Path to WSA_DATA directory
WSA_DATA_PATH = "data/WSA_DATA"

# Default size of observation window (4 per day * 3 days)
NOBS = 6 * 3

# Default size of prediction window in addition to observation window
# (4 per day * 3 days)
NPRED = 6 * 6

# Avoid using k-NN items within this many timesteps of the target to avoid
# duplicates
PRUNE_THRESHOLD = 12

# Avoid using k-NN items within this timedelta of the target, for validation
# and data leakage issues
VALIDATION_CLOSENESS_THROWOUT = timedelta(days=27)


# Default K for nearest neighbor search (prior to prunining).
DEFAULT_K = 1000


def calculate_uncertainty_gaussian(
    knn_dataset,
    times,
    Vp_pred,
    Vp_obs,
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
      sigma
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

    # Calculate sigma
    weights = np.array([1 / nbr.distance for nbr in neighbors])
    errors = np.array([nbr.after_obs[0] - nbr.after_pred[0] for nbr in neighbors])
    mask = np.isfinite(weights) & np.isfinite(errors)

    variance = np.sum(weights[mask] * np.square(errors[mask])) / weights[mask].sum()
    sigma = np.sqrt(variance)

    # Print message if verbose
    if verbose:
        print(f"Found {len(neighbors)} neighbors")
        print(f"Gaussian Sigma = {sigma}")
        print()

    # Return
    if return_neighbors:
        return_value = (sigma, neighbors)
    else:
        return_value = sigma

    return return_value


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

    def __init__(self, input_map, sat, real, daysahead, nobs=NOBS, npred=NPRED):
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

        self.X, self.Xtime, self.y = setup_knn_variables(self.df_knn, nobs, npred)
        self.tree = KDTree(self.X)
        self.nobs = nobs
        self.npred = npred
        (
            self.before_times,
            self.before_obs,
            self.before_pred,
            self.after_times,
            self.after_obs,
            self.after_pred,
        ) = unpack_knn_variables(self.X, self.Xtime, self.y, self.nobs, self.npred)

    def lookup_neighbors(self, times, Vp_obs, Vp_pred, k=DEFAULT_K):
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

        # Do query of KDTree
        query = np.zeros(self.nobs + self.npred)
        query[: self.nobs] = Vp_obs
        query[self.nobs :] = Vp_pred

        distances, inds = self.tree.query(query, k=k)

        # Remove neighbors that are very close to eachother in time (e.g., with
        # carrington of eachother)
        distances_pruned, inds_pruned = prune_inds(distances, inds)

        # Collect neighbors
        neighbors = []

        for distance, ind in zip(distances_pruned, inds_pruned):
            # Avoid using k-NN items within this timedelta of the target, for
            # validation and data leakage issues
            skip = False

            for time in times:
                for before_time in self.before_times[ind]:
                    if abs(before_time - time) < VALIDATION_CLOSENESS_THROWOUT:
                        skip = True

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

        return neighbors


def prune_inds(distances, inds, threshold=PRUNE_THRESHOLD):
    """Remove neighbors that are very close to eachother in time (e.g., with
    a carrington of eachother)

    Args
      distances: array of distances to neighbors
      inds: array of indices to neighbors
    Returns
      distances_pruned: array of distances to neighbors, pruned
      inds_pruned: array of indices to neighbors, pruned
    See Also
      Module variable PRUNE_THRESHOLD (this module)
    """
    distances_pruned = []
    inds_pruned = []

    for d, ind in zip(distances, inds):
        if all(abs(ind - i) > threshold for i in inds_pruned):
            distances_pruned.append(d)
            inds_pruned.append(int(ind))

    dinstances_pruned = np.array(distances_pruned)
    inds_pruned = np.array(inds_pruned)

    return distances_pruned, inds_pruned


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
