from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd
import numpy as np


@dataclass
class Geometry:
    # duct
    duct_width_m: float | None = None
    duct_height_m: float | None = None
    duct_area_m2: float | None = None

    # venturi throat (RECTANGULAR)
    throat_width_m: float | None = None
    throat_height_m: float | None = None
    throat_area_m2: float | None = None

    # Kiel probe ports (mapping)
    static_port_area_m2: float | None = None  # As
    total_port_area_m2: float | None = None   # At_ports (NOT the venturi throat area)

    # legacy fields we now ignore; keep for tolerant loads
    throat_diameter_m: float | None = None  # deprecated


def _area_rect(w: float | None, h: float | None) -> float | None:
    if w and h and w > 0 and h > 0:
        return w * h
    return None


def duct_area(g: Geometry) -> float | None:
    return g.duct_area_m2 or _area_rect(g.duct_width_m, g.duct_height_m)


def throat_area(g: Geometry) -> float | None:
    return g.throat_area_m2 or _area_rect(g.throat_width_m, g.throat_height_m)


def r_ratio(g: Geometry) -> float | None:
    """Section area ratio r = As / At (verification plane area to venturi throat area).

    Notes:
      - As is the duct cross-section area at the verification plane.
      - At is the venturi throat area.
      - Returns None if either area is missing or invalid.
    """
    As = duct_area(g)
    At = throat_area(g)
    if As and At and At > 0:
        return As / At
    return None


def beta_from_geometry(g: Geometry) -> float | None:
    # β ≜ Dt/D1 for circular; for noncircular use β = sqrt(At/A1) because A ∝ D^2
    A1 = duct_area(g)
    At = throat_area(g)
    if A1 and At and 0 < At < A1:
        return math.sqrt(At / A1)
    return None


@dataclass
class DiffuserGeometry:
    D1: float  # inlet diameter [m]
    D2: float  # outlet diameter (or plane-2) [m]
    L: float   # axial length from 0..L [m]
    r_As_At: float | None = None  # area ratio (verification plane to throat), optional
    dt: float | None = None       # throat diameter [m], optional

    def radius_at(self, z: np.ndarray) -> np.ndarray:
        """Linear cone by default; R(z) = 0.5*(D1 + (D2-D1)*(z/L))."""
        z = np.asarray(z, dtype=float)
        D = self.D1 + (self.D2 - self.D1) * (z / max(self.L, 1e-12))
        return 0.5 * D


def infer_geometry_from_table(df: pd.DataFrame) -> DiffuserGeometry | None:
    """
    Attempt to infer geometry from a legacy 'details' table.
    Recognized columns (case-insensitive): D1, D2, L (m or mm), r, dt, At, As.
    Returns DiffuserGeometry if enough fields present, else None.
    """
    cols = {c.lower(): c for c in df.columns}

    def get_val(*names):
        for n in names:
            if n.lower() in cols:
                return df[cols[n.lower()]].iloc[0]
        return None

    D1 = get_val('D1', 'D_inlet', 'Diameter1')
    D2 = get_val('D2', 'D_outlet', 'Diameter2')
    L = get_val('L', 'Length', 'AxialLength')

    # unit harmonization: if numbers look like mm, convert to m (heuristic: >2.0 means mm for small ducts? Let user override outside)
    def to_m(v):
        if v is None or pd.isna(v):
            return None
        v = float(v)
        # if clearly mm (>= 50) and not a huge duct, convert to m
        return v / 1000.0 if v > 50 else v

    if D1 is not None and D2 is not None and L is not None:
        geo = DiffuserGeometry(D1=to_m(D1), D2=to_m(D2), L=to_m(L))
        geo.r_As_At = get_val('r', 'As/At', 'area_ratio') or None
        dt = get_val('dt', 'D_throat', 'throat_diameter')
        geo.dt = to_m(dt) if dt is not None else None
        return geo
    return None


def planes_to_z(planes: np.ndarray, geom: DiffuserGeometry | None) -> np.ndarray:
    """Map plane identifiers to axial positions anchored to geometry."""
    z = np.asarray(planes, dtype=float)
    if geom is None or z.size == 0:
        return z
    # Heuristic: if planes are evenly spaced integers, treat them as indices
    diffs = np.diff(z)
    if z.size > 1 and np.allclose(diffs, 1.0) and np.allclose(z, np.round(z)):
        return np.linspace(0.0, geom.L, len(z))
    return z


def plane_value_to_z(pv: float, planes: np.ndarray, geom: DiffuserGeometry | None) -> float:
    """Convert a single plane identifier to axial ``z`` position (meters)."""
    z_vals = planes_to_z(planes, geom)
    planes = np.asarray(planes, dtype=float)
    return float(np.interp(float(pv), planes, z_vals))
