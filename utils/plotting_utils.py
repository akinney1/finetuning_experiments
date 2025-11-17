# autopep8: off

import pandas as pd
import numpy as np
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.forecasting_utils as forecast_utils
import utils.analysis_utils as analysis_utils

import matplotlib.dates as mdates
import matplotlib.cm as cm

from sklearn.metrics import mean_squared_error as mse

# autopep8: on


# ---------------Reresentative Forecasts functions
def create_legend(fig, bottom_margin, bbox_to_anchor):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.legend_handler import HandlerTuple

    class HandlerLineOverWideBand(HandlerTuple):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            line, band = orig_handle
            band_height = 1.5 * height
            band_y = ydescent + (height - band_height) / 2
            band_artist = Rectangle(
                (xdescent, band_y), width, band_height,
                facecolor=band.get_facecolor(), alpha=band.get_alpha(),
                edgecolor='none', transform=trans,
            )
            line_y = band_y + band_height / 2
            line_artist = Line2D(
                [xdescent, xdescent + width], [line_y, line_y],
                color=line.get_color(), linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(), transform=trans,
            )
            return [band_artist, line_artist]

    # Reserve bottom margin for the legend
    fig.subplots_adjust(bottom=bottom_margin)

    trap_handle = Line2D([], [], marker='o', linestyle='None',
                         color='tab:orange', markerfacecolor='tab:orange',
                         markeredgewidth=0, markersize=8)

    line_handle = Line2D([], [], color='tab:blue', linewidth=2)
    band_handle = Rectangle((0, 0), 1, 1, facecolor='tab:blue', alpha=0.35, edgecolor='none')
    prob_handle = (line_handle, band_handle)

    true_handle = Line2D([], [], color='tab:green', linestyle='--', linewidth=2)

    handles = [trap_handle, prob_handle, true_handle]
    labels = ['Trap samples', 'Probabilistic forecast', 'True prediction\ninterval bounds']

    leg = fig.legend(
        handles, labels,
        handler_map={tuple: HandlerLineOverWideBand()},
        loc='lower center', bbox_to_anchor=bbox_to_anchor,
        ncol=3, fontsize='medium',
        frameon=True, fancybox=True, edgecolor='0.7',
        handlelength=2.2, columnspacing=2.0, labelspacing=1.0
    )

    # Keep legend out of layout calculations so it won't be clipped
    leg.set_in_layout(False)

    return fig


def format_forecast_single_plot(ax, sample, dist):
    trial_cols = forecast_utils.trial_names()
    trap_col, pred_col, tru_col = 'tab:orange', 'tab:blue', 'tab:green'
    s = 4

    sample.Datetime = pd.to_datetime(sample.Datetime)
    t0 = sample[sample.Location == 'Observed'].iloc[-1].Datetime
    observed = sample[sample.Location == 'Observed']
    forecast = sample[sample.Datetime >= t0]  # Include t0 week for continuity in plots

    # Add t0 line
    ax.axvline(t0, linestyle='--', color='k', alpha=0.5)

    # Plot weekly traps
    [ax.scatter(sample.Datetime, sample[col], s=s, color=trap_col, alpha=0.15)
     for col in trial_cols]

    # Plot point predictions
    ax.plot(observed.Datetime, observed.Point_predictions, alpha=0.35, color=pred_col)
    ax.plot(forecast.Datetime, forecast.Point_predictions, alpha=1, color=pred_col)

    # Plot 68% uncertainty bounds
    if dist == 'poisson':
        l_quant, u_quant = analysis_utils.poisson_quant(forecast.Point_predictions, ci=0.68)
    if dist == 'negbin':
        l_quant, u_quant = analysis_utils.negbin_quant(forecast.ns, forecast.mean_p, ci=0.68)
    ax.fill_between(forecast.Datetime, l_quant, u_quant, alpha=0.5, color=pred_col)

    # Plot true quantile bounds
    ax.plot(sample.Datetime, sample.True_l_forecast_quant, linestyle='--', color=tru_col)
    ax.plot(sample.Datetime, sample.True_u_forecast_quant, linestyle='--',
            color=tru_col, label='True prediction\ninterval bounds')

    # Set date formatting for x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 2 != 0:
            label.set_visible(False)

    # Add RMSE to top corner
    rmse = np.sqrt(
        mse(forecast.iloc[1:].Ref, forecast.iloc[1:].Point_predictions)) / np.average(forecast.iloc[1:].Ref)
    # ax.text(0.97, 0.95, fr'$\frac{{\mathrm{{RMSE}}}}{{\overline{{\mathrm{{trap}}}}}}$: {rmse:.2f}',
    #        transform=ax.transAxes, ha='right', va='top', fontsize='small')
    ax.text(
        0.75, 0.95,
        fr'Relative' '\n'
        fr'$\mathrm{{RMSE}}$: {rmse:.2f}',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize='small',
        linespacing=1.2
    )
    return ax


# ---------------Error bar plots
def format_single_bar_plot(ax, avgs, stds, labels=[]):
    if labels is None:
        labels = ['RMSE(wks. 1-4)', 'MAE(wk. 1)', 'MAE(wk. 2)', 'MAE(wk. 3)', 'MAE(wk. 4)']

    # Define colors
    # cmap = cm.get_cmap('viridis', len(labels) - 1)  # colormap for MAEs
    # colors = ['maroon'] + [cmap(i) for i in range(len(labels) - 1)]
    colors = ['#8B0000',  # deep maroon for RMSE
              '#6BAED6',  # light blue
              '#4292C6',  # medium blue
              '#2171B5',  # darker blue
              '#084594']  # navy
    ax.bar(labels, avgs, color=colors, width=0.8)
    ax.errorbar(labels, avgs, ls='none', yerr=stds, ecolor='k', elinewidth=0.8)
    ax.set_ylabel('Average Forecast\nError $(n=389)$', fontsize=10)
    ax.tick_params(axis='x', labelsize=10)

    return
