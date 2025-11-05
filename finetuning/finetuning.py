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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import gen_utils
from utils import predictions
from utils import format_data_utils
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


# ----------------------- Training -----------------------

def train_finetune_model(base_model, X_train, y_train, X_val, y_val, config):
    """Freeze all but last two layers and finetune."""
    finetuned = tf.keras.models.clone_model(base_model)
    finetuned.set_weights(base_model.get_weights())

    # freeze all except last two layers
    for layer in finetuned.layers[:-2]:
        layer.trainable = False

    finetuned.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=[predictions.r2_keras]
    )

    history = finetuned.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        **config['fit'],
        callbacks=[
            tf.keras.callbacks.TensorBoard(),
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        ]
    )
    return history, finetuned


# ----------------------- Plotting -----------------------

def plot_finetune_history(history, out_dir, data_len, lr):
    """Save loss curve and history CSV."""
    os.makedirs(out_dir, exist_ok=True)

    hist = history.history
    epochs = range(1, len(hist.get('loss', [])) + 1)

    # loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, hist['loss'], label='train')
    if 'val_loss' in hist:
        ax.plot(epochs, hist['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Finetune Loss')
    ax.legend()
    fig.tight_layout()

    loss_path = os.path.join(out_dir, f'gru_avg_temp_finetune_history_{data_len}yr_{lr}lr.png')
    fig.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # csv
    csv_path = os.path.join(out_dir, f'gru_avg_temp_finetune_history_{data_len}yr_{lr}lr.csv')
    pd.DataFrame(hist).to_csv(csv_path, index=False)

    print(f"Saved: {loss_path}")
    print(f"Saved: {csv_path}")


# ----------------------- Config update -----------------------

def write_finetune_config(base_config: dict, raw_mols_path: str, model_files_path: str) -> dict:
    """Return and persist finetune_config.json with updated training/model paths."""
    cfg = dict(base_config)  # shallow copy is fine
    cfg.setdefault('files', {})

    cfg['model'] = 'finetuned'
    cfg['files']['training'] = f"{os.path.expanduser(raw_mols_path)}/San_Juan_MoLS.csv"
    cfg['files']['model'] = f"{os.path.expanduser(model_files_path)}/gru_avg_temp_finetuned.keras"

    out_path = os.path.join(os.path.expanduser(model_files_path), 'finetune_config.json')
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(f"Finetune config written: {out_path}")
    return cfg


# ----------------------- Main -----------------------

def main():
    np.random.seed(14)
    tf.random.set_seed(14)

    fpaths_cfg = "../fpaths_config.json"
    # Expected to return (..., model_files_path, raw_mols_path)
    _, _, _, model_files_path, raw_mols_path = gen_utils.load_input_paths(fpaths_cfg)
    model_files_path = os.path.expanduser(model_files_path)
    raw_mols_path = os.path.expanduser(raw_mols_path)

    # Where to drop plots/csv
    training_output_path, _ = gen_utils.load_output_paths(fpaths_cfg)
    training_output_path = os.path.expanduser(training_output_path)
    os.makedirs(training_output_path, exist_ok=True)

    # Load base model and base config + scaler, then create finetune config
    base_model, base_config, scaler = gen_utils.load_base_model_fils(
        model_files_path, 'gru_avg_temp_config.json')
    cfg = write_finetune_config(base_config, raw_mols_path, model_files_path)
    base_or_ft_model = base_model  # start finetuning from base

    train_lens = [1, 3, 5, 10, 15]
    lrs = [0.0001, 0.00005]

    for train_len, lr in itertools.product(train_lens, lrs):
        cfg_sample = cfg.copy()
        cfg_sample['compile']['learning_rate'] = lr

        # Build datasets per config
        train, val, test = get_finetuning_data(cfg_sample, train_len)

        # Format samples
        X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test = \
            get_finetuning_samples(train, val, test, scaler)

        # Train
        history, finetuned_model = train_finetune_model(
            base_or_ft_model, X_train, y_train, X_val, y_val, cfg_sample
        )
        plot_finetune_history(history, training_output_path, train_len, lr)

        # Save model
        save_path = os.path.expanduser(cfg_sample['files']['model'])
        save_path = save_path.replace('finetuned', f'finetuned_{train_len}yr_{lr}lr')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        finetuned_model.save(save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
