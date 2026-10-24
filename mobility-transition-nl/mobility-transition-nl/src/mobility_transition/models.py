"""Survey-weighted logistic regression with PC4-clustered (sandwich) standard errors, plus patsy helpers."""
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

PROBABILITY_FLOOR = 1e-12


def create_design_matrix(formula, data):
    """Design matrix with all-zero dummy columns (empty categories in this sample) removed."""
    design = dmatrix(formula, data, return_type="dataframe", NA_action="raise")
    columns = [c for c in design.columns if not np.allclose(design[c].to_numpy(dtype=float), 0)]
    return design[columns].copy(), design.design_info, columns


def rebuild_design_matrix(design_info, columns_used, new_data):
    rebuilt = build_design_matrices([design_info], new_data, return_type="dataframe")[0]
    missing = [c for c in columns_used if c not in rebuilt.columns]
    if missing:
        raise ValueError(f"Prediction design matrix is missing columns:\n{missing}")
    return rebuilt[columns_used].to_numpy(dtype=float)


def _log_likelihood(y, p, w):
    return np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))


def _objective(beta, X, y, w):
    p = np.clip(expit(X @ beta), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return -_log_likelihood(y, p, w), -(X.T @ (w * (y - p)))


def fit_clustered_logit(X, y, weights, clusters, max_iter=1000, gtol=1e-8, label=None, require_both_outcomes=True):
    """
    Weighted logit by maximum likelihood (BFGS, retried with L-BFGS-B if needed).
    Standard errors: cluster-robust sandwich with the usual G/(G-1) * (N-1)/(N-K) small-sample correction.
    """
    X, y, w = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    if require_both_outcomes and np.unique(y).size != 2:
        raise ValueError(f"{label or 'Model'} does not contain both binary outcomes.")
    if label:
        print(f"\nFitting {label}: N = {len(y):,}; predictors = {X.shape[1]}")

    result = minimize(_objective, np.zeros(X.shape[1]), args=(X, y, w), jac=True, method="BFGS",
                      options={"maxiter": max_iter, "gtol": gtol, "disp": False})
    if not result.success:
        print("BFGS did not report convergence; retrying with L-BFGS-B...")
        retry = minimize(_objective, result.x, args=(X, y, w), jac=True, method="L-BFGS-B",
                         options={"maxiter": max_iter, "ftol": 1e-12, "gtol": gtol})
        if retry.fun <= result.fun:
            result = retry

    beta = result.x
    p = np.clip(expit(X @ beta), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    information = X.T @ ((w * p * (1 - p))[:, None] * X)
    bread = np.linalg.pinv(information, rcond=1e-10)

    scores = pd.DataFrame(X * (w * (y - p))[:, None])
    scores["_cluster"] = pd.Series(np.asarray(clusters), dtype="string").to_numpy()
    cluster_scores = scores.groupby("_cluster").sum().to_numpy(dtype=float)
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread

    n, k, g = X.shape[0], X.shape[1], cluster_scores.shape[0]
    if g > 1 and n > k:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = (covariance + covariance.T) / 2
    ll = _log_likelihood(y, p, w)

    return {
        "beta": beta,
        "standard_errors": np.sqrt(np.clip(np.diag(covariance), 0, np.inf)),
        "covariance": covariance,
        "probability": p,
        "log_likelihood": ll,
        "n": n,
        "n_clusters": g,
        "n_parameters": k,
        "event_share_unweighted": y.mean(),
        "aic": -2 * ll + 2 * k,
        "bic": -2 * ll + np.log(n) * k,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "gradient_infinity_norm": np.linalg.norm(X.T @ (w * (y - p)), ord=np.inf),
        "minimum_information_eigenvalue": np.linalg.eigvalsh(information).min(),
    }


CONVERGENCE_KEYS = [
    "optimizer_success", "optimizer_message", "optimizer_iterations", "gradient_infinity_norm", "minimum_information_eigenvalue",
]


def coefficient_stats(estimate, se):
    """z, two-sided p, and 95% CI bounds on the coefficient scale."""
    z = estimate / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return z, p, estimate - 1.96 * se, estimate + 1.96 * se


def predict_probability(beta, X):
    return expit(np.asarray(X, dtype=float) @ beta)
