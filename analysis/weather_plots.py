# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl

import utils.gen_utils as gen_utils

# autopep8: on


def plot_weather(observed, forecast, output_path):
    weather_dict = {'San Juan': observed, 'Ceiba': forecast}
    fig, axs = plt.subplots(3, figsize=(8, 6), sharex=True)
    col_dict = {
        'Avg_Temp': 'Avg. Temp. ($\\degree$C)',
        'Precip': 'Precip. (cm)',
        'Humidity': 'Humidity (%)'}
    for ax, (col, col_name) in zip(axs, col_dict.items()):
        ax.plot(observed.Datetime, observed[col], label='San Juan', color='tab:blue', alpha=0.5)
        ax.plot(forecast.Datetime, forecast[col], label='Ceiba', color='tab:orange', alpha=0.5)
        ax.set_ylabel(col_name)
    axs[0].legend(loc='upper left', ncol=2)
    fig.tight_layout()
    fig.savefig(f'{output_path}/weather_comparison.png', dpi=300, bbox_inches='tight')
    return


def plot_mols(observed, output_path):
    # CONFIRM AT HOME
    fig, axs = plt.subplots(figsize=(8, 3))
    axs.plot(observed.Datetime, observed['Ref'], color='tab:blue', alpha=0.8)
    axs.set_ylabel('Unscaled Abundance')
    axs.set_title('MoLS Unscaled Abundance Curve for San Juan, Puerto Rico')
    fig.tight_layout()
    fig.savefig(f'{output_path}/observed_mols.png', dpi=300, bbox_inches='tight')
    return


def main():
    config = '../fpaths_config.json'

    _, _, _, _, raw_mols_path = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)

    # Plot weather
    if True:
        observed_weather = pd.read_csv(f'{raw_mols_path}/San_Juan_MoLS.csv')
        observed_weather['Datetime'] = pd.to_datetime(observed_weather[['Year', 'Month', 'Day']])

        forecast_weather = pd.read_csv(f'{raw_mols_path}/Ceiba_MoLS.csv')
        forecast_weather['Datetime'] = pd.to_datetime(forecast_weather[['Year', 'Month', 'Day']])

        plot_weather(observed_weather, forecast_weather, output_path)

    # Plot MoLS
    if True:
        plot_mols(observed_weather, output_path)

    return


if __name__ == "__main__":
    main()
