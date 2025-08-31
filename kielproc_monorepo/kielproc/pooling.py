
import numpy as np


def pool_alpha_beta_random_effects(alphas, alpha_vars, betas, beta_vars, cov_ab=None):
    """Random-effects pooling for correlated alpha/beta estimates.

    Parameters
    ----------
    alphas, betas : array-like
        Per-replicate estimates of slope (alpha) and intercept (beta).
    alpha_vars, beta_vars : array-like
        Corresponding variances for alpha and beta.
    cov_ab : array-like or None, optional
        Per-replicate covariances between alpha and beta.  If omitted,
        covariances are assumed to be zero.

    Returns
    -------
    tuple
        ``(alpha_pooled, alpha_se, tau2_alpha, beta_pooled, beta_se, tau2_beta,
        (Q_alpha, k_alpha), (Q_beta, k_beta), cov_pooled)`` where ``cov_pooled``
        is the 2x2 covariance matrix of the pooled estimates.
    """

    alphas = np.asarray(alphas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    Va = np.asarray(alpha_vars, dtype=float)
    Vb = np.asarray(beta_vars, dtype=float)
    if cov_ab is None:
        cov_ab = np.zeros_like(Va, dtype=float)
    else:
        cov_ab = np.asarray(cov_ab, dtype=float)

    def _pool_univariate(theta, V):
        w_FE = 1.0 / V
        theta_FE = np.sum(w_FE * theta) / np.sum(w_FE)
        Q = np.sum(w_FE * (theta - theta_FE) ** 2)
        k = theta.size
        c = np.sum(w_FE) - np.sum(w_FE ** 2) / np.sum(w_FE)
        tau2 = max(0.0, (Q - (k - 1)) / max(c, 1e-12))
        return tau2, Q, k

    tau2_a, Q_a, k_a = _pool_univariate(alphas, Va)
    tau2_b, Q_b, k_b = _pool_univariate(betas, Vb)

    # Construct covariance matrices including between-study variance
    T = np.array([[tau2_a, 0.0], [0.0, tau2_b]])
    Sum_W = np.zeros((2, 2))
    Sum_Wtheta = np.zeros(2)

    for a, b, v_a, v_b, cab in zip(alphas, betas, Va, Vb, cov_ab):
        Sigma = np.array([[v_a, cab], [cab, v_b]]) + T
        W = np.linalg.inv(Sigma)
        Sum_W += W
        Sum_Wtheta += W @ np.array([a, b])

    cov_pooled = np.linalg.inv(Sum_W)
    pooled = cov_pooled @ Sum_Wtheta
    a_hat, b_hat = pooled
    a_se, b_se = np.sqrt(np.diag(cov_pooled))

    return (
        float(a_hat),
        float(a_se),
        float(tau2_a),
        float(b_hat),
        float(b_se),
        float(tau2_b),
        (float(Q_a), int(k_a)),
        (float(Q_b), int(k_b)),
        cov_pooled,
    )
