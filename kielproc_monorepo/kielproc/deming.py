
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
