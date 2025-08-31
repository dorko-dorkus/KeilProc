from __future__ import annotations
import math

def qa_indices(pN: float, pS: float, pE: float, pW: float, q_mean: float):
    """Compute 90° wall-static quality indices.

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
