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

# --- Development rate function (Sharpe-Schoolfield model) ---


def dev_rate(RH, HA, HH, TH, T, T_min, T_max):
    K = T + 273.15
    num = RH * (K / 298.15) * np.exp(HA / 1.987 * (1 / 298.15 - 1 / K))
    denom = 1 + np.exp(HH / 1.987 * (1 / TH - 1 / K))
    rate = num / denom
    return rate if T_min <= T <= T_max else 0.0


# --- Death rates by temperature and humidity ---
def death_rates(T, RH, H_on=True):
    # SE: Eggs
    SE = 0.9 if 10 < T < 33 else 0.48 if 33 <= T <= 36 else 0.0
    SL = 1 - (0.01 + 0.9725 * np.exp(-T / 2.7035)) if 13 < T < 38 else 0.1554

    rate = dev_rate(0.15460, 33255.57, 50543.49, 301.67, T, 10, 36)
    DT = 1 / rate if rate > 0 else np.inf

    # SP: Pupa
    if 12 < T < 38 and DT != np.inf:
        SPP = 10 ** (np.log10(SL) / DT)
    else:
        SPP = 0.05

    # SL: Immatures
    if 13 < T < 36 and DT != np.inf:
        SPL = 10 ** (np.log10(SL) / DT)
    else:
        SPL = 0.05

    # SA: Adults
    SA = 0.91
    if 4 <= T <= 41 and H_on and 72 < RH < 95:
        SA = 0.98

    return SE, SPP, SPL, SA


def imap(T, RH):
    # --- Development rates ---
    dev_e = dev_rate(0.24, 10798, 100000, 14184.5, T, 10, 36)
    dev_is1 = dev_rate(0.68007, 28033.83, 72404.07, 304.33, T, 11.8, 39)
    dev_is2 = dev_rate(1.24508, 36400.55, 81383.14, 301.78, T, 11.8, 39)
    dev_is3 = dev_rate(1.06144, 41192.69, 60832.62, 301.29, T, 11.8, 39)
    dev_is4 = dev_rate(0.57065, 34455.89, 45543.49, 301.44, T, 11.8, 39)
    dev_p = dev_rate(0.74423, 19246.42, 5954.35, 302.68, T, 10.3, 39)
    dev_gc = dev_rate(0.372, 15725.23, 1756481.07, 44717, T, 18, 36)

    # --- Death rates ---
    se, spp, spl, sa = death_rates(T, RH)

    return imap

temps = np.linspace(start=-15, stop=45, num=120)
humidities = np.linspace(start=0, stop=100, num=200)



