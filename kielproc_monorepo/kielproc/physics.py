
import numpy as np

R_SPECIFIC_AIR = 287.05  # J/(kg*K)

def rho_from_pT(ps_pa: np.ndarray, T_K: np.ndarray, R: float = R_SPECIFIC_AIR) -> np.ndarray:
    """
    Density from static pressure and temperature at low Mach (ideal gas).
    rho = ps / (R * T)
    """
    ps_pa = np.asarray(ps_pa, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    if np.any(T_K <= 0):
        raise ValueError("Temperature at or below 0 K encountered")
    return ps_pa / (R * T_K)

def map_qs_to_qt(qs: np.ndarray, r: float, rho_t_over_rho_s: float = 1.0) -> np.ndarray:
    """
    Map dynamic pressure measured at downstream section (area As) back to throat (At).
    qt = r^2 * (rho_t/rho_s) * qs, where r = As/At.
    """
    return (r**2) * float(rho_t_over_rho_s) * np.asarray(qs, dtype=float)

def venturi_dp_from_qt(qt: np.ndarray, beta: float) -> np.ndarray:
    """
    ISO 5167 low-Mach venturi relation: Δp_vent ≈ (1 - beta^4) * q_t
    """
    return (1.0 - beta**4) * np.asarray(qt, dtype=float)
