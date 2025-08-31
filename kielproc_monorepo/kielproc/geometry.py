
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

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
    L  = get_val('L', 'Length', 'AxialLength')
    # unit harmonization: if numbers look like mm, convert to m (heuristic: >2.0 means mm for small ducts? Let user override outside)
    def to_m(v):
        if v is None or pd.isna(v): return None
        v = float(v)
        # if clearly mm (>= 50) and not a huge duct, convert to m
        return v/1000.0 if v > 50 else v
    if D1 is not None and D2 is not None and L is not None:
        geo = DiffuserGeometry(D1=to_m(D1), D2=to_m(D2), L=to_m(L))
        geo.r_As_At = get_val('r', 'As/At', 'area_ratio') or None
        dt = get_val('dt', 'D_throat', 'throat_diameter')
        geo.dt = to_m(dt) if dt is not None else None
        return geo
    return None
