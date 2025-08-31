
import numpy as np


def deming_fit(
    x,
    y,
    lambda_ratio: float = 1.0,
    *,
    bootstrap: bool = False,
    n_boot: int = 200,
    random_state: int | None = None,
):
    """Deming regression with optional bootstrap standard errors.

    Parameters
    ----------
    x, y : array_like
        Input data.  Must be finite.
    lambda_ratio : float, optional
        Known variance ratio :math:`\lambda = \sigma_y^2 / \sigma_x^2`.
    bootstrap : bool, optional
        If True, compute standard errors via bootstrap rather than the
        large-sample approximation.  Defaults to False.
    n_boot : int, optional
        Number of bootstrap resamples when ``bootstrap`` is True.
    random_state : int, optional
        Seed for bootstrap RNG.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    xbar = np.mean(x)
    ybar = np.mean(y)
    Sxx = np.mean((x - xbar) ** 2)
    Syy = np.mean((y - ybar) ** 2)
    Sxy = np.mean((x - xbar) * (y - ybar))
    if Sxy == 0:
        return np.nan, np.nan, np.nan, np.nan

    term = Syy - lambda_ratio * Sxx
    D = term**2 + 4 * lambda_ratio * (Sxy**2)
    alpha = (term + np.sqrt(D)) / (2 * Sxy)
    beta = ybar - alpha * xbar

    if bootstrap:
        rng = np.random.default_rng(random_state)
        idx = np.arange(n)
        coefs = []
        for _ in range(n_boot):
            samp = rng.choice(idx, size=n, replace=True)
            a_s, b_s, _, _ = deming_fit(
                x[samp],
                y[samp],
                lambda_ratio=lambda_ratio,
                bootstrap=False,
            )
            coefs.append((a_s, b_s))
        coefs = np.asarray(coefs)
        sa = float(np.std(coefs[:, 0], ddof=1))
        sb = float(np.std(coefs[:, 1], ddof=1))
        return float(alpha), float(beta), sa, sb

    # Large-sample variance approximations with small ridge for stability
    resid = lambda_ratio * Sxx + Syy - 2 * alpha * Sxy
    resid = max(resid, 0.0)
    eps = np.finfo(float).eps
    denom = Sxy**2
    var_alpha = ((alpha**2 + lambda_ratio) / (n * (denom + eps))) * (resid + eps)
    var_beta = var_alpha * (xbar**2) + (alpha**2 * (Sxx / n)) + ((resid + eps) / n)
    sa = float(np.sqrt(var_alpha))
    sb = float(np.sqrt(var_beta))
    return float(alpha), float(beta), sa, sb
