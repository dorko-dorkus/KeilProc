
import numpy as np


def deming_fit(x, y, lambda_ratio: float = 1.0):
    """
    Deming regression (errors-in-variables) with known variance ratio lambda = sigma_y^2 / sigma_x^2.
    Returns alpha (slope), beta (intercept), and standard errors (sa, sb) using asymptotic formulas.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    xbar = np.mean(x); ybar = np.mean(y)
    Sxx = np.mean((x - xbar)**2)
    Syy = np.mean((y - ybar)**2)
    Sxy = np.mean((x - xbar)*(y - ybar))
    if Sxy == 0:
        return np.nan, np.nan, np.nan, np.nan

    term = (Syy - lambda_ratio*Sxx)
    D = term**2 + 4*lambda_ratio*(Sxy**2)
    alpha = (term + np.sqrt(D)) / (2*Sxy)
    beta = ybar - alpha * xbar

    # Large-sample variance approximations
    var_alpha = ((alpha**2 + lambda_ratio) / (n * (Sxy**2))) * (lambda_ratio*Sxx + Syy - 2*alpha*Sxy)
    var_beta  = var_alpha*(xbar**2) + (alpha**2 * (Sxx/n)) + ((lambda_ratio*Sxx + Syy - 2*alpha*Sxy) / n)
    sa = float(np.sqrt(max(var_alpha, 0.0)))
    sb = float(np.sqrt(max(var_beta, 0.0)))
    return float(alpha), float(beta), sa, sb


def jackknife_deming(x, y, lambda_ratio: float = 1.0, floor: float = 1e-8):
    """Deming regression with leave-one-out jackknife standard errors.

    Parameters
    ----------
    x, y : array-like
        Data arrays with measurement errors in both directions.
    lambda_ratio : float, optional
        Ratio of variance in y to variance in x (sigma_y^2 / sigma_x^2).
    floor : float, optional
        Minimum standard error returned to avoid zero-SE degeneracy.

    Returns
    -------
    alpha, beta, se_alpha, se_beta, cov_ab : tuple of floats
        Fitted slope/intercept along with jackknife standard errors and
        covariance between alpha and beta.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    alpha, beta, _, _ = deming_fit(x, y, lambda_ratio=lambda_ratio)

    alphas = np.empty(n)
    betas = np.empty(n)
    for i in range(n):
        x_i = np.delete(x, i)
        y_i = np.delete(y, i)
        a_i, b_i, _, _ = deming_fit(x_i, y_i, lambda_ratio=lambda_ratio)
        alphas[i] = a_i
        betas[i] = b_i

    if not (np.all(np.isfinite(alphas)) and np.all(np.isfinite(betas))):
        return float(alpha), float(beta), np.nan, np.nan, np.nan

    a_bar = np.mean(alphas)
    b_bar = np.mean(betas)
    factor = (n - 1) / n
    var_alpha = factor * np.sum((alphas - a_bar) ** 2)
    var_beta = factor * np.sum((betas - b_bar) ** 2)
    cov_ab = factor * np.sum((alphas - a_bar) * (betas - b_bar))

    se_alpha = float(np.sqrt(max(var_alpha, 0.0)))
    se_beta = float(np.sqrt(max(var_beta, 0.0)))
    se_alpha = max(se_alpha, float(floor))
    se_beta = max(se_beta, float(floor))

    return float(alpha), float(beta), se_alpha, se_beta, float(cov_ab)
