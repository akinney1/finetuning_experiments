# autopep8: off


from typing import Tuple
import tensorflow as tf

import sys, os, json
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import predictions

# autopep8 on


def load_input_paths(fpath: str) -> Tuple[str, str, str, str, str]:
    """Read path names from a JSON file.

    Args:
        fpath: path to JSON file containing paths

    Returns:
        (weather_path, mols_path, nn_preds_path, model_files_path, raw_mols_path)
    """
    with open(os.path.expanduser(fpath), 'r') as f:
        paths = json.load(f)

    weather_path      = paths["input"].get("weather_data")
    trap_catch_path   = paths["input"].get("trap_catches")
    nn_preds_path     = paths["input"].get("nn_abundance_predictions")
    model_files_path  = paths["input"].get("model_files")
    raw_mols_path     = paths["input"].get("raw_mols")

    return weather_path, trap_catch_path, nn_preds_path, model_files_path, raw_mols_path


def load_output_paths(fpath: str) -> str:
    """Read training output path from JSON file.

    Args:
        fpath: path to JSON file containing paths

    Returns:
        training_output_path (str)
    """
    with open(os.path.expanduser(fpath), 'r') as f:
        paths = json.load(f)

    training_output_path = paths["output"].get('training_output')
    output_path          = paths["output"].get("output_fil_path")
    return training_output_path, output_path


def load_base_model_fils(model_files_path: str, config_name: str):
    """Load base model + config + scaler.

    Args:
        model_files_path: directory containing model files and config JSON
        config_name: config filename WITHOUT extension (e.g., 'gru_avg_temp_config')

    Returns:
        (base_model, config_dict, scaler)
    """
    config_path = f"{model_files_path}/{config_name}"
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_path = config['files'].get('model')
    # Keras 3 tip: safe_mode=False with custom objects
    base_model = tf.keras.models.load_model(
        os.path.expanduser(model_path),
        custom_objects={'r2_keras': predictions.r2_keras}
    )

    scaler_path = f"{model_files_path}/avg_scaler.pkl"
    scaler = pd.read_pickle(os.path.expanduser(scaler_path))

    return base_model, config, scaler


def load_finetune_model_fils(model_files_path: str, config_name: str, model_qualifiers: str):
    """Load finetuned model + config + scaler.

    Args:
        model_files_path: directory containing model files and config JSON
        config_name: config filename WITHOUT extension (e.g., 'finetune_config')

    Returns:
        (finetune_model, config_dict, scaler)
    """
    config_path = f"{model_files_path}/{config_name}"
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_path = config['files'].get('model')

    if model_qualifiers is not None:
        model_path = model_path.replace('finetuned', model_qualifiers)

    finetune_model = tf.keras.models.load_model(
        os.path.expanduser(model_path),
        custom_objects={'r2_keras': predictions.r2_keras}
    )

    scaler_path = f"{model_files_path}/avg_scaler.pkl"
    scaler = pd.read_pickle(os.path.expanduser(scaler_path))

    return finetune_model, config, scaler
