from datetime import datetime, timedelta
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from util_wsa_uncertainty import KnnUncertaintyDataset 


def test_constructor():
    """Tests the object constructs with valid parameters."""
    KnnUncertaintyDataset(input_map='AGONG', sat='ACE', real=0, daysahead=3)
    

def test_lookup_neighbor():
    nobs = 12
    npred = 24
    k = 1000
    required_nbrs = 50
    
    dataset = KnnUncertaintyDataset(input_map='AGONG', sat='ACE', real=0, daysahead=3,
                                    nobs=nobs, npred=npred)    

    times = np.array(
        [datetime(2015, 1, 1) + i * timedelta(hours=6) for i in range(npred)],
        dtype=object
    )

    Vp_obs = np.zeros(nobs) + 450
    Vp_pred = np.zeros(nobs + nobs) + 500
    
    neighbors = dataset.lookup_neighbors(
        times=times, Vp_obs=Vp_obs, Vp_pred=Vp_pred,
        k=k,
    )

    assert len(neighbors) > required_nbrs
    assert len(neighbors) < k
    
