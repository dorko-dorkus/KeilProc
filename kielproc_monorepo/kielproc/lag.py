
import numpy as np

def estimate_lag_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = None):
    """
    Estimate integer-sample lag between signals x (reference) and y (to be shifted)
    by maximizing cross-correlation. Returns lag where y should be shifted forward
    (i.e., y_shifted[k] = y[k - lag]). Positive lag -> y lags x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n] - np.nanmean(x[:n])
    y = y[:n] - np.nanmean(y[:n])
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(max(0, min(max_lag, n - 1)))
    lags = np.arange(-max_lag, max_lag + 1)
    # compute normalized cross-correlation via FFT for speed
    # pad to next power of two
    N = 1
    while N < 2*n:
        N <<= 1
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    c = np.fft.irfft(X * np.conj(Y), N)
    # shift so that zero-lag is at center
    c = np.concatenate((c[-(n-1):], c[:n]))
    # take subset of lags
    idx_center = len(c)//2
    start = idx_center - max_lag
    end = idx_center + max_lag + 1
    c_sub = c[start:end]
    best = int(np.nanargmax(c_sub))
    lag = lags[best]
    return lag, lags, c_sub

def shift_series(y: np.ndarray, lag: int):
    """
    Shift series by integer lag. Positive lag shifts y forward (to the right),
    inserting NaN at the start.
    """
    y = np.asarray(y, dtype=float)
    if lag == 0:
        return y.copy()
    out = np.empty_like(y, dtype=float)
    out[:] = np.nan
    if lag > 0:
        out[lag:] = y[:-lag]
    else:
        out[:lag] = y[-lag:]
    return out


def first_order_lag(x: np.ndarray, tau_s: float, dt_s: float):
    """
    Apply discrete first-order lag filter with time constant tau_s, sample interval dt_s.
    y[k] = y[k-1] + (dt/tau) * (x[k] - y[k-1])
    """
    x = np.asarray(x, dtype=float)
    if tau_s <= 0 or dt_s <= 0 or len(x) == 0:
        return x.copy()
    a = float(dt_s) / float(tau_s)
    a = max(min(a, 1.0), 0.0)
    y = x.copy()
    for k in range(1, len(x)):
        y[k] = y[k-1] + a * (x[k] - y[k-1])
    return y
