import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import skew, kurtosis



def square(values, window):
    moving_avg = pd.Series(values).rolling(window=window).mean().dropna().values
    if len(moving_avg) > 0:
        moving_avg_diff = moving_avg[-1] - moving_avg[0]
    else:
        moving_avg_diff = 0
    return moving_avg_diff


def is_numeric(x):
    if isinstance(x, (int, float, np.number)) and not np.isnan(x):
        return x
    else:
        return 0


def calculate_autocorr(series, lag=1):
    return pd.Series(series).autocorr(lag=lag)


def calculate_trend(series):
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x, series)
    return slope


def preproces(df):
    df['values'] = df['values'].apply(lambda lst: [is_numeric(x) for x in lst])
    df['mean'] = df['values'].apply(lambda x: np.mean(x))
    df['autocorr'] = df['values'].apply(lambda x: calculate_autocorr(x, lag=1))
    df['std'] = df['values'].apply(lambda x: np.std(x))
    df['min'] = df['values'].apply(lambda x: np.min(x))
    df['max'] = df['values'].apply(lambda x: np.max(x))
    df['range'] = df['max'] - df['min']
    df['median'] = df['values'].apply(lambda x: np.median(x))
    df['delta_day'] = df['dates'].apply(lambda x: (x[-1] - x[0]).days)
    df['kurtosis'] = df['values'].apply(lambda x: kurtosis(x))
    df['moving_avg_diff5'] = df['values'].apply(lambda x: square(x, 5))
    df = df.drop(['id', 'dates', 'values'], axis=1)
    df = df.fillna(0)
    return df
