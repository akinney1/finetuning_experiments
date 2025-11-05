"""
Aedes-AI GRU finetuning script (GRU + mean temp/precip/RH)

- Updates/writes finetune_config.json
- Loads base or existing finetuned model
- Creates finetuning datasets from configured CSV
- Trains, saves history plots/CSV, and saves model (.keras)
"""

# autopep8: off
import os, sys, json, itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import gen_utils
from utils import predictions

# autopep8: on


# ----------------------- Data helpers -----------------------

def get_test_samples(data, train_len: int):
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

    test_yr = 2017
    val_yr = 2016
    val = data[data.Year == val_yr].reset_index(drop=True)
    test = data[data.Year >= test_yr].reset_index(drop=True)

    train_end_idx = data[data.Year == val_yr].index[0]
    train_start_idx = data[data.Year == (val_yr - train_len)].index[0]
    train = data.iloc[train_start_idx: train_end_idx - 1].reset_index(drop=True)

    return train, val, test


# ------------------------- TEST / PREDICT -------------------------

def test_predictions(model, scaler, data, config):
    """Generate and return predictions from a trained GRU model on the test dataset."""

    data_shape = config['data']['data_shape']

    results = predictions.gen_preds(
        model, data, data_shape, scaler, fit_scaler=False
    )
    results = pd.DataFrame(
        results,
        columns=['Location', 'Year', 'Month', 'Day', 'MoLS', 'Neural Network']
    )
    return results

# ----------------------- Analysis -----------------------


def single_plot(results, ax):
    ax.plot(results.Datetime, results.MoLS, color='tab:blue', alpha=0.8, label='MoLS')
    ax.plot(
        results.Datetime,
        results['Neural Network'],
        color='tab:orange',
        alpha=0.8,
        label='Neural Network')
    return


# ----------------------- Main -----------------------

def main():
    np.random.seed(14)
    tf.random.set_seed(14)

    fpaths_cfg = "../fpaths_config.json"
    # Expected to return (..., model_files_path, raw_mols_path)
    _, _, _, model_files_path, raw_mols_path = gen_utils.load_input_paths(fpaths_cfg)
    model_files_path = os.path.expanduser(model_files_path)
    raw_mols_path = os.path.expanduser(raw_mols_path)

    train_lens = [0, 1, 3, 5, 10, 15]
    lrs = [0.0001]
    combos = itertools.product(train_lens, lrs)

    fig, axs = plt.subplots(3, 2, figsize=(6, 4), sharex=True, sharey=True)

    for i, (train_len, lr) in enumerate(combos):
        row, col = divmod(i, 2)
        ax = axs[row, col]

        if train_len > 0:
            qualifier = f'finetuned_{train_len}yr_{lr}lr'
            model, config, scaler = gen_utils.load_finetune_model_fils(
                model_files_path, 'finetune_config.json', qualifier)
        else:
            model, config, scaler = gen_utils.load_base_model_fils(
                model_files_path, 'gru_avg_temp_config.json')

        # Load data file
        data_fil = "../data/raw_mols_predictions/San_Juan_MoLS.csv"
        data = pd.read_csv(data_fil)

        # Build datasets per config
        train, val, test = get_test_samples(data, train_len)
        test.drop(columns=['Datetime'], inplace=True)
        test.rename(columns={'Ref': 'MoLS'}, inplace=True)

        results = test_predictions(model, scaler, test, config)
        results['Datetime'] = pd.to_datetime(results[['Year', 'Month', 'Day']])
        rmse = np.sqrt(mse(results['MoLS'], results['Neural Network'])) / \
            np.average(results['MoLS'])

        single_plot(results, ax)
        if train_len > 0:
            title = f'Finetuned $Aedes$-$AI$ model,\nusing {train_len} years of observed data'
        else:
            title = 'Base $Aedes$-$AI$ model'
        ax.set_title(title)
        ax.text(
            0.75, 0.95,
            fr'Relative' '\n'
            fr'$\mathrm{{RMSE}}$: {rmse:.4f}',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize='small',
            linespacing=1.2
        )
        if col == 0:
            ax.set_ylabel('Unscaled\nabundance')
    plt.show()


if __name__ == "__main__":
    main()
