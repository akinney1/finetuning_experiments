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

import utils.training as training_utils
import utils.format_data_utils as format_data_utils

# autopep8:on


class SaliencyAnalyzer:
    """Compute and visualize saliency maps for time series neural networks."""

    def __init__(self, model):
        """
        Initialize the saliency analyzer.

        Args:
            model: Trained TensorFlow/Keras model
        """
        self.model = model

    def compute_vanilla_saliency(self, input_data):
        """
        Compute vanilla saliency map using gradients.

        Args:
            input_data: Input array of shape (batch_size, 90, 3) or (90, 3)

        Returns:
            Saliency map of same shape as input
        """
        # Ensure input is a tensor
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)

        # Add batch dimension if needed
        if len(input_data.shape) == 2:
            input_data = tf.expand_dims(input_data, 0)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            output = self.model(input_data, training=False)

        # Get gradients with respect to input
        gradients = tape.gradient(output, input_data)

        # Take absolute value for saliency
        saliency = tf.abs(gradients)

        return saliency.numpy()

    def compute_smoothgrad(self, input_data, n_samples=50, noise_level=0.1):
        """
        Compute SmoothGrad saliency map (averages over noisy samples).

        Args:
            input_data: Input array of shape (batch_size, 90, 3) or (90, 3)
            n_samples: Number of noisy samples to average
            noise_level: Standard deviation of Gaussian noise (relative to input std)

        Returns:
            SmoothGrad saliency map
        """
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)

        if len(input_data.shape) == 2:
            input_data = tf.expand_dims(input_data, 0)

        # Calculate noise standard deviation
        std = tf.math.reduce_std(input_data) * noise_level

        saliency_maps = []

        for _ in range(n_samples):
            # Add Gaussian noise
            noise = tf.random.normal(input_data.shape, stddev=std)
            noisy_input = input_data + noise

            with tf.GradientTape() as tape:
                tape.watch(noisy_input)
                output = self.model(noisy_input, training=False)

            gradients = tape.gradient(output, noisy_input)
            saliency_maps.append(tf.abs(gradients))

        # Average over all samples
        smoothgrad = tf.reduce_mean(tf.stack(saliency_maps), axis=0)

        return smoothgrad.numpy()

    def compute_integrated_gradients(self, input_data, baseline=None, n_steps=50):
        """
        Compute Integrated Gradients saliency map.

        Args:
            input_data: Input array of shape (batch_size, 90, 3) or (90, 3)
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps

        Returns:
            Integrated gradients saliency map
        """
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)

        if len(input_data.shape) == 2:
            input_data = tf.expand_dims(input_data, 0)

        # Create baseline (zeros by default)
        if baseline is None:
            baseline = tf.zeros_like(input_data)
        else:
            baseline = tf.constant(baseline, dtype=tf.float32)
            if len(baseline.shape) == 2:
                baseline = tf.expand_dims(baseline, 0)

        # Generate interpolation coefficients
        alphas = tf.linspace(0.0, 1.0, n_steps)

        gradients = []
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_data - baseline)

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                output = self.model(interpolated, training=False)

            gradient = tape.gradient(output, interpolated)
            gradients.append(gradient)

        # Average gradients and multiply by input difference
        avg_gradients = tf.reduce_mean(tf.stack(gradients), axis=0)
        integrated_grads = (input_data - baseline) * avg_gradients

        return tf.abs(integrated_grads).numpy()

    def visualize_saliency(self, input_data, saliency_map,
                           feature_names=None, save_path=None):
        """
        Visualize saliency map as a heatmap.

        Args:
            input_data: Original input data (90, 3) or (1, 90, 3)
            saliency_map: Saliency map (90, 3) or (1, 90, 3)
            feature_names: List of 3 feature names
            save_path: Path to save figure (optional)
        """
        if feature_names is None:
            feature_names = [f'Feature {i + 1}' for i in range(3)]

        # Remove batch dimension if present
        if saliency_map.ndim == 3:
            saliency_map = saliency_map[0]
        if isinstance(input_data, tf.Tensor):
            input_data = input_data.numpy()
        if input_data.ndim == 3:
            input_data = input_data[0]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot input data
        ax1 = axes[0]
        im1 = ax1.imshow(input_data.T, aspect='auto', cmap='viridis')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Feature')
        ax1.set_yticks(range(3))
        ax1.set_yticklabels(feature_names)
        ax1.set_title('Input Time Series')
        plt.colorbar(im1, ax=ax1, label='Value')

        # Plot saliency map
        ax2 = axes[1]
        im2 = ax2.imshow(saliency_map.T, aspect='auto', cmap='hot')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Feature')
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(feature_names)
        ax2.set_title('Saliency Map (Importance)')
        plt.colorbar(im2, ax=ax2, label='Gradient Magnitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def visualize_overlay(self, input_data, saliency_map,
                          feature_names=None, save_path=None):
        """
        Visualize input with saliency overlay for each feature.

        Args:
            input_data: Original input data (90, 3) or (1, 90, 3)
            saliency_map: Saliency map (90, 3) or (1, 90, 3)
            feature_names: List of 3 feature names
            save_path: Path to save figure (optional)
        """
        if feature_names is None:
            feature_names = [f'Feature {i + 1}' for i in range(3)]

        # Remove batch dimension if present
        if saliency_map.ndim == 3:
            saliency_map = saliency_map[0]
        if isinstance(input_data, tf.Tensor):
            input_data = input_data.numpy()
        if input_data.ndim == 3:
            input_data = input_data[0]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_steps = np.arange(90)

        for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            # Plot time series
            ax.plot(time_steps, input_data[:, i], 'b-', linewidth=2, label='Input')

            # Overlay saliency as background color
            # Normalize saliency for this feature
            sal_norm = saliency_map[:, i]
            sal_norm = (sal_norm - sal_norm.min()) / (sal_norm.max() - sal_norm.min() + 1e-8)

            # Create colored background based on saliency
            for t in range(89):
                ax.axvspan(t, t + 1, alpha=sal_norm[t] * 0.5, color='red')

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.set_title(f'{feature_name} with Saliency Overlay')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


def format_base_xai(config, scaler):
    # Build datasets per config
    test_df = pd.read_pickle(os.path.expanduser(config['files']['testing']))
    test_df.rename(columns={'MoLS': 'Ref'}, inplace=True)
    X_test, y_test, locs_test = format_data_utils.format_finetuning_samples(test_df, scaler)
    return X_test, y_test, locs_test
