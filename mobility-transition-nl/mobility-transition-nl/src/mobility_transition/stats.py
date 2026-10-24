"""Weighted descriptive statistics."""
import numpy as np
import pandas as pd


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan


def weighted_share(condition, weights):
    return weighted_mean(pd.Series(condition).astype(float), weights)


def weighted_percentile(values, weights, quantiles):
    """Weighted percentiles using mid-point cumulative weights, linearly interpolated."""
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    positions = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(quantiles, positions, values, left=values[0], right=values[-1])


def equalise_year_weights(weights, years):
    """Rescale weights so each survey year contributes the same total weight; returns (weights, factors)."""
    totals = pd.DataFrame({"w": weights, "y": years}).dropna().groupby("y")["w"].sum()
    if len(totals) <= 1:
        return weights, {}
    factors = (totals.mean() / totals).to_dict()
    return weights * years.map(factors).fillna(1), factors
