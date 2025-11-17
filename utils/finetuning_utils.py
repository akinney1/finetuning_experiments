# autopep8: off

import pandas as pd
import numpy as np
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.format_data_utils as format_data_utils

# autopep8: on


# ----------------------- Data helpers -----------------------

def get_finetuning_data(config: dict, train_len: int):
    """Load the full CSV from config['files']['training'] and split to train/val/test."""
    csv_path = os.path.expanduser(config['files']['training'])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

    test_yr = 2017
    val_yr = 2016
    val = data[data.Year == val_yr].reset_index(drop=True)
    test = data[data.Year >= test_yr].reset_index(drop=True)

    train_end_idx = data[data.Year == val_yr].index[0]
    train_start_idx = data[data.Year == (val_yr - train_len)].index[0]
    train = data.iloc[train_start_idx: train_end_idx - 1].reset_index(drop=True)

    return train, val, test


def get_finetuning_samples(train, val, test, scaler):
    """Format to (X, y, locs) triples for each split, then shuffle train/val."""
    X_train, y_train, locs_train = format_data_utils.format_finetuning_samples(train, scaler)
    X_val, y_val, locs_val = format_data_utils.format_finetuning_samples(val, scaler)
    X_test, y_test, locs_test = format_data_utils.format_finetuning_samples(test, scaler)

    # Shuffle train and val
    p = np.random.permutation(X_train.shape[0])
    X_train, y_train, locs_train = X_train[p], y_train[p], locs_train[p]

    p = np.random.permutation(X_val.shape[0])
    X_val, y_val, locs_val = X_val[p], y_val[p], locs_val[p]

    return X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test
