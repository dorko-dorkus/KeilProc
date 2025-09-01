from __future__ import annotations
import math

# Default QA thresholds used when gating data.  Values are fractions of
# dynamic pressure and correspond to 0.2% swirl and 1% opposing-port
# imbalance respectively.
DEFAULT_W_MAX = 0.002
DEFAULT_DELTA_OPP_MAX = 0.01


def qa_indices(pN: float, pS: float, pE: float, pW: float, q_mean: float):
    """Compute opposing-port imbalance (Δ_opp) and swirl index (W).

    QA indices used for acceptance gates (default Δ_opp ≤ 0.01·q_mean, W ≤ 0.002).

    Formulas
    --------
    Δ_opp = max(|pN - pS|, |pE - pW|) / q_mean
    W     = sqrt((pN - pS)^2 + (pE - pW)^2) / (2·q_mean)

    Worked example
    --------------
    pN=151.0 Pa, pS=149.5 Pa, pE=150.3 Pa, pW=150.2 Pa, q_mean=300.0 Pa
    Δ_opp = max(1.5, 0.1)/300 = 0.005
    W     = sqrt(1.5^2 + 0.1^2)/(2·300) ≈ 0.0025

    Parameters
    ----------
    pN,pS,pE,pW : float
        Mean wall-static pressures at the north, south, east and west ports.
    q_mean : float
        Mean dynamic pressure used for normalisation.

    Returns
    -------
    (delta_opp, W) : tuple of floats
        ``delta_opp`` is the maximum opposing-port difference divided by ``q_mean``.
        ``W`` is the swirl index based on the root-sum-square of opposing differences,
        normalised by ``2*q_mean``.
    """
    pN = float(pN); pS = float(pS); pE = float(pE); pW = float(pW); q_mean = float(q_mean)
    if q_mean == 0 or math.isnan(q_mean):
        return float("nan"), float("nan")
    d_ns = pN - pS
    d_ew = pE - pW
    delta_opp = max(abs(d_ns), abs(d_ew)) / q_mean
    W = math.sqrt(d_ns*d_ns + d_ew*d_ew) / (2.0 * q_mean)
    return delta_opp, W
