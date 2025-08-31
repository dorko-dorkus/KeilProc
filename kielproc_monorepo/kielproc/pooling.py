
import numpy as np

def pool_alpha_beta_random_effects(alphas, alpha_vars, betas, beta_vars):
    """
    Random-effects (DerSimonian–Laird) pooling for alpha and beta separately.
    Returns: (alpha_pooled, alpha_se, tau2_alpha, beta_pooled, beta_se, tau2_beta, (Q_alpha,k_alpha), (Q_beta,k_beta))
    """
    alphas = np.asarray(alphas, dtype=float)
    betas  = np.asarray(betas, dtype=float)
    Va = np.asarray(alpha_vars, dtype=float)
    Vb = np.asarray(beta_vars, dtype=float)
    def _pool(theta, V):
        w_FE = 1.0 / V
        theta_FE = np.sum(w_FE*theta)/np.sum(w_FE)
        Q = np.sum(w_FE*(theta - theta_FE)**2)
        k = theta.size
        c = np.sum(w_FE) - np.sum(w_FE**2)/np.sum(w_FE)
        tau2 = max(0.0, (Q - (k - 1)) / max(c, 1e-12))
        w_RE = 1.0 / (V + tau2)
        theta_RE = np.sum(w_RE*theta)/np.sum(w_RE)
        se_RE = np.sqrt(1.0/np.sum(w_RE))
        return theta_RE, se_RE, tau2, Q, k
    a_hat, a_se, tau2_a, Q_a, k_a = _pool(alphas, Va)
    b_hat, b_se, tau2_b, Q_b, k_b = _pool(betas,  Vb)
    return a_hat, a_se, tau2_a, b_hat, b_se, tau2_b, (Q_a, k_a), (Q_b, k_b)
