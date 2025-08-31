
import numpy as np
from kielproc.physics import map_qs_to_qt, venturi_dp_from_qt
from kielproc.lag import estimate_lag_xcorr, shift_series
from kielproc.deming import deming_fit
from kielproc.pooling import pool_alpha_beta_random_effects

def test_map_and_venturi():
    qs = np.array([10.0, 20.0, 30.0])
    qt = map_qs_to_qt(qs, r=1.1, rho_t_over_rho_s=1.0)
    assert np.allclose(qt, (1.1**2)*qs)
    dv = venturi_dp_from_qt(qt, beta=0.5)
    assert np.allclose(dv, (1-0.5**4)*qt)

def test_xcorr_lag():
    x = np.sin(np.linspace(0, 10, 200))
    y = np.roll(x, 7)
    lag, _, _ = estimate_lag_xcorr(x, y, max_lag=40)
    assert lag == 7

def test_deming_known_slope():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y_true = 2.0*x + 1.0
    x_noisy = x + rng.normal(scale=0.2, size=x.size)
    y_noisy = y_true + rng.normal(scale=0.2, size=x.size)
    a, b, sa, sb = deming_fit(x_noisy, y_noisy, lambda_ratio=1.0)
    assert 1.6 < a < 2.4
    assert 0.0 < sa < 0.5

def test_pooling_workflow():
    alphas = np.array([0.9, 1.1, 1.0])
    alpha_vars = np.array([0.01, 0.01, 0.02])
    betas = np.array([2.0, 1.0, 1.5])
    beta_vars = np.array([0.5, 0.5, 0.6])
    a_hat, a_se, tau2_a, b_hat, b_se, tau2_b, *_ = pool_alpha_beta_random_effects(alphas, alpha_vars, betas, beta_vars)
    assert 0.8 < a_hat < 1.2
    assert a_se > 0
