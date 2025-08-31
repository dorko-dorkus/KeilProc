import numpy as np
from kielproc.lag import estimate_lag_xcorr


def test_estimate_lag_xcorr_defaults_to_abs():
    x = np.array([0, 1, 1, 0], dtype=float)
    y = -x
    lag, lags, c, r = estimate_lag_xcorr(x, y)
    assert lag == 0
    # ensure returned cross-correlation is signed
    assert c[lags.tolist().index(0)] < 0
    assert np.isclose(r, -1.0)


def test_estimate_lag_xcorr_signed_option():
    x = np.array([0, 1, 1, 0], dtype=float)
    y = -x
    lag, _, _, _ = estimate_lag_xcorr(x, y, use_abs=False)
    assert lag in (-2, 2)
