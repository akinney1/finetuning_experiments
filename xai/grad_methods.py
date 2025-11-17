# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils.predictions as predictions
import utils.gen_utils as gen_utils
import utils.training as training_utils

import utils.finetuning_utils as finetuning_utils
import utils.xai_utils as xai_utils

# autopep8:on


def saliency_single_locs(input_data, scaler, analyzer, metadata):
    # Compute different types of saliency maps
    vanilla_saliency = analyzer.compute_vanilla_saliency(input_data)
    smoothgrad_saliency = analyzer.compute_smoothgrad(input_data, n_samples=30)
    ig_saliency = analyzer.compute_integrated_gradients(input_data, n_steps=30)

    # Visualize results
    feature_names = ['Average Temperature', 'Precipitation', 'Relative Humidity']

    zeros_column = np.zeros((90, 1))
    sample_with_mols_col = np.hstack([input_data, zeros_column])
    original_data = scaler.inverse_transform(sample_with_mols_col)

    figname = f'../output/xai_results/{metadata}'
    # analyzer.visualize_overlay(
    #    original_data,
    #    vanilla_saliency,
    #    feature_names,
    #    f'{figname}_vanilla.png')

    analyzer.visualize_overlay(
        original_data,
        smoothgrad_saliency,
        feature_names,
        f'{figname}_smoothgrad.png')

    # analyzer.visualize_overlay(
    #    original_data,
    #    ig_saliency,
    #    feature_names,
    #    f'{figname}_integratedgrad.png')


def main():
    model_files_path = '../utils/model_files'
    # Load your trained model
    model, config, scaler = gen_utils.load_base_model_fils(
        model_files_path, 'gru_avg_temp_config.json')

    # Initialize analyzer
    analyzer = xai_utils.SaliencyAnalyzer(model)

    # Format samples
    X_test, y_test, locs_test = xai_utils.format_base_xai(config, scaler)

    demo_locs = ['Fairfield,Connecticut', 'Fortuna Foothills,Arizona']

    for i in range(0, len(X_test)):
        sample, metadata = X_test[i, :], locs_test[i]
        loc = metadata[0].astype(str)
        metadata = f'{loc}_{int(metadata[1])}_{int(metadata[2])}_{int(metadata[3])}'

        if loc in demo_locs:
            print(metadata)
            saliency_single_locs(sample, scaler, analyzer, metadata)
            plt.close('all')


# Example usage
if __name__ == "__main__":
    main()
