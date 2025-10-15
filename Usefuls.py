# -*- coding: utf-8 -*-
"""

@author: OGOUNKOLA Ebénézer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.typing import ArrayLike 

# ============================================================
#  Construct training data in (T, t, K, price) format
# ============================================================

def construct_full_dataset(expiries, tenors, K, prices):
    """
    Construct full dataset with all observations.
    Returns: X (N x 3), y (N,) where X = [T, t, K]
    """
    X_list = []
    y_list = []
    
    for i in range(len(expiries)):
        T = expiries[i]
        t = tenors[i]
        for j in range(K.shape[1]):
            strike = K.iloc[i, j]
            price = prices.iloc[i, j]
            X_list.append([T, t, strike])
            y_list.append(price)
    
    return np.array(X_list), np.array(y_list)


# ============================================================
#  Create sampling strategy 
# ============================================================

def create_training_subset(expiries, tenors, K, prices, spreads, p=0.5, seed=42):
    """
    Create training subset according to specifications:
    - Always include ATM (spread=0) for key pairs
    - Always include at least one payer OTM and one receiver OTM for key pairs
    - Key pairs: (5,5), (5,10), (10,5), (10,10)
    - Random p% of ATM swaptions
    - Random p% of OTM and ITM payer swaptions
    
    Args:
        expiries: array of expiry times
        tenors: array of tenor times
        K: strikes matrix (n_obs x n_spreads)
        prices: prices matrix (n_obs x n_spreads)
        spreads: array of spread values
        p: percentage of data to include (0 to 1)
        seed: random seed
    
    Returns:
        X_train: (N x 3) array of [T, t, K]
        y_train: (N,) array of prices
        indices: selected indices for tracking
    """
    rng = np.random.RandomState(seed)
    spreads = np.asanyarray(spreads)
    # Key pairs that must have complete coverage
    key_pairs = [(5.0, 5.0), (5.0, 10.0), (10.0, 5.0), (10.0, 10.0)]
    
    selected_indices = []
    
    # Find ATM index (spread = 0)
    atm_idx = np.where(spreads == 0)[0][0]
    
    # Payer OTM: strikes > forward, so positive spreads
    payer_otm_indices = np.where(spreads > 0)[0]
    # Receiver OTM: strikes < forward, so negative spreads  
    receiver_otm_indices = np.where(spreads < 0)[0]
    
    # Step 1: Ensure key pairs have required observations
    for exp, ten in key_pairs:
        # Find matching observation
        mask = (expiries == exp) & (tenors == ten)
        obs_idx = np.where(mask)[0]
        
        if len(obs_idx) > 0:
            obs_idx = obs_idx[0]
            
            # Always include ATM
            selected_indices.append((obs_idx, atm_idx))
            
            # Include at least one payer OTM (randomly choose one)
            payer_spread_idx = rng.choice(payer_otm_indices)
            selected_indices.append((obs_idx, payer_spread_idx))
            
            # Include at least one receiver OTM (randomly choose one)
            receiver_spread_idx = rng.choice(receiver_otm_indices)
            selected_indices.append((obs_idx, receiver_spread_idx))
    
    # Step 2: Random p% of ATM swaptions
    all_atm_indices = []
    for i in range(len(expiries)):
        all_atm_indices.append((i, atm_idx))
    
    # Remove already selected ATM from key pairs
    remaining_atm = [idx for idx in all_atm_indices if idx not in selected_indices]
    n_atm_to_add = int(p * len(remaining_atm))
    selected_atm_indices = rng.choice(len(remaining_atm), size=n_atm_to_add, replace=False)
    selected_indices.extend([remaining_atm[i] for i in selected_atm_indices])
    
    # Step 3: Random p% of payer OTM and ITM swaptions
    # Payer swaptions: all positive and negative spreads
    all_payer_indices = []
    for i in range(len(expiries)):
        for j in range(len(spreads)):
            if j != atm_idx:  # Exclude ATM already handled
                all_payer_indices.append((i, j))
    
    # Remove already selected
    remaining_payer = [idx for idx in all_payer_indices if idx not in selected_indices]
    n_payer_to_add = int(p * len(remaining_payer))
    selected_payer_indices = rng.choice(len(remaining_payer), size=n_payer_to_add, replace=False)
    selected_indices.extend([remaining_payer[i] for i in selected_payer_indices])
    
    # Convert selected indices to training data
    X_train = []
    y_train = []
    
    for obs_idx, spread_idx in selected_indices:
        T = expiries[obs_idx]
        t = tenors[obs_idx]
        strike = K[obs_idx, spread_idx]
        price = prices[obs_idx, spread_idx]
        
        X_train.append([T, t, strike])
        y_train.append(price)
    
    return np.array(X_train), np.array(y_train), selected_indices
