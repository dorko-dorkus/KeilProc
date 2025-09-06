
from __future__ import annotations
import numpy as np
import pandas as pd
from .lag import estimate_lag_xcorr, shift_series
from .deming import deming_fit
from .pooling import pool_alpha_beta_random_effects


def compute_translation_table(
    blocks: dict,
    ref_key="mapped_ref",
    picc_key="piccolo",
    lambda_ratio=1.0,
    max_lag=300,
    sampling_hz=None,
    *,
    bootstrap: bool = False,
    n_boot: int = 200,
    random_state: int | None = None,
    beta_override: float | None = None,
):
    rows = []
    for name, df in blocks.items():
        x = df[ref_key].to_numpy(float)
        y = df[picc_key].to_numpy(float)
        if np.nanstd(y) < 1e-12 or np.all(~np.isfinite(y)):
            # skip flat/missing piccolo
            continue
        lag, _, _, r_peak = estimate_lag_xcorr(x, y, max_lag=max_lag)
        # Positive ``lag`` means piccolo lags the reference.  Align by shifting
        # piccolo forward (left) with ``shift_series(y, -lag)``.
        y_shift = shift_series(y, -lag)
        if beta_override is not None:
            y_adj = y_shift - beta_override
            m, _, sa, _ = deming_fit(
                x,
                y_adj,
                lambda_ratio=lambda_ratio,
                bootstrap=bootstrap,
                n_boot=n_boot,
                random_state=random_state,
            )
            b = float(beta_override)
            sb = float("nan")
        else:
            m, b, sa, sb = deming_fit(
                x,
                y_shift,
                lambda_ratio=lambda_ratio,
                bootstrap=bootstrap,
                n_boot=n_boot,
                random_state=random_state,
            )
        rows.append(
            dict(
                block=name,
                alpha=m,
                beta=b,
                alpha_se=sa,
                beta_se=sb,
                lag_samples=int(lag),
                r_peak=float(r_peak),
            )
        )
    tidy = pd.DataFrame(rows).set_index("block") if rows else pd.DataFrame(columns=["alpha","beta","alpha_se","beta_se","lag_samples"])
    pooled = None
    if (
        beta_override is None
        and not tidy.empty
        and tidy["alpha_se"].gt(0).all()
        and tidy["beta_se"].gt(0).all()
    ):
        Va = tidy["alpha_se"]**2
        Vb = tidy["beta_se"]**2
        cov_ab = np.zeros_like(Va)
        a_hat, a_se, tau2_a, b_hat, b_se, tau2_b, Q_a, Q_b, cov_pooled = \
            pool_alpha_beta_random_effects(tidy["alpha"], Va, tidy["beta"], Vb, cov_ab)
        pooled = dict(
            alpha=a_hat,
            alpha_se=a_se,
            tau2_alpha=tau2_a,
            beta=b_hat,
            beta_se=b_se,
            tau2_beta=tau2_b,
            cov_ab=cov_pooled[0, 1],
            Q_alpha=Q_a[0],
            k_alpha=Q_a[1],
            Q_beta=Q_b[0],
            k_beta=Q_b[1],
        )
    return tidy.reset_index(), pooled

def apply_translation(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    src_col: str = "piccolo",
    out_col: str = "piccolo_translated",
):
    """Apply a linear translation to a DataFrame column.

    Parameters
    ----------
    df:
        Input table.
    alpha, beta:
        Translation parameters.  ``alpha`` scales the source column and
        ``beta`` is added to the result.
    src_col:
        Name of the column containing the raw piccolo values.
    out_col:
        Desired name of the translated column in the output DataFrame.

    Raises
    ------
    KeyError
        If ``src_col`` is not present in ``df``.
    """

    if src_col not in df.columns:
        cols = ", ".join(df.columns)
        raise KeyError(f"Column '{src_col}' not found in input dataframe. Available columns: {cols}")

    out = df.copy()
    out[out_col] = alpha * out[src_col] + beta
    return out
