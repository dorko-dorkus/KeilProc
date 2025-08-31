
import numpy as np
from kielproc.physics import map_qs_to_qt, venturi_dp_from_qt
from kielproc.lag import advance_series, delay_series
from kielproc.deming import deming_fit
from kielproc.pooling import pool_alpha_beta_random_effects, pool_alpha_beta_gls

def test_map_and_venturi():
    qs = np.array([10.0, 20.0, 30.0])
    qt = map_qs_to_qt(qs, r=1.1, rho_t_over_rho_s=1.0)
    assert np.allclose(qt, (1.1**2)*qs)
    dv = venturi_dp_from_qt(qt, beta=0.5)
    assert np.allclose(dv, (1-0.5**4)*qt)

def test_advance_delay_inverse():
    x = np.arange(20, dtype=float)
    y = delay_series(x, 3)
    x2 = advance_series(y, 3)
    idx = ~np.isnan(x2)
    assert np.array_equal(x[idx], x2[idx])

def test_deming_known_slope():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y_true = 2.0*x + 1.0
    x_noisy = x + rng.normal(scale=0.2, size=x.size)
    y_noisy = y_true + rng.normal(scale=0.2, size=x.size)
    a, b, sa, sb = deming_fit(x_noisy, y_noisy, lambda_ratio=1.0)
    assert 1.6 < a < 2.4
    assert sa > 0.0
    assert sb > 0.0


def test_deming_se_not_zero_with_noise():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 100)
    y_true = -1.0 + 0.5 * x
    x_noisy = x + rng.normal(scale=0.05, size=x.size)
    y_noisy = y_true + rng.normal(scale=0.05, size=x.size)
    _, _, sa, sb = deming_fit(x_noisy, y_noisy, lambda_ratio=1.0)
    assert sa > 0.0
    assert sb > 0.0

def test_pooling_workflow():
    alphas = np.array([0.9, 1.1, 1.0])
    alpha_vars = np.array([0.01, 0.01, 0.02])
    betas = np.array([2.0, 1.0, 1.5])
    beta_vars = np.array([0.5, 0.5, 0.6])
    cov_ab = np.array([0.02, 0.01, 0.015])
    a_hat, a_se, tau2_a, b_hat, b_se, tau2_b, Q_a, Q_b, cov = \
        pool_alpha_beta_random_effects(alphas, alpha_vars, betas, beta_vars, cov_ab)
    assert 0.8 < a_hat < 1.2
    assert a_se > 0
    assert cov.shape == (2, 2)
    assert cov[0, 1] > 0


def test_pooling_gls():
    alphas = np.array([0.9, 1.1, 1.0])
    betas = np.array([2.0, 1.0, 1.5])
    covs = np.array(
        [
            [[0.01, 0.02], [0.02, 0.5]],
            [[0.01, 0.01], [0.01, 0.5]],
            [[0.02, 0.015], [0.015, 0.6]],
        ]
    )

    a_hat, a_se, b_hat, b_se, cov = pool_alpha_beta_gls(alphas, betas, covs)

    Sum_W = np.zeros((2, 2))
    Sum_Wtheta = np.zeros(2)
    for a, b, C in zip(alphas, betas, covs):
        W = np.linalg.inv(C)
        Sum_W += W
        Sum_Wtheta += W @ np.array([a, b])
    cov_exp = np.linalg.inv(Sum_W)
    pooled_exp = cov_exp @ Sum_Wtheta

    assert np.allclose([a_hat, b_hat], pooled_exp)
    assert np.allclose(cov, cov_exp)
    assert a_se > 0 and b_se > 0
