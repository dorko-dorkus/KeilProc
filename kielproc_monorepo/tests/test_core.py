import numpy as np
import pandas as pd

from kielproc.geometry import DiffuserGeometry, infer_geometry_from_table


def test_radius_at_linear_cone():
    geo = DiffuserGeometry(D1=0.1, D2=0.2, L=1.0)
    z = np.array([0.0, 0.5, 1.0])
    expected = np.array([0.05, 0.075, 0.1])
    assert np.allclose(geo.radius_at(z), expected)


def test_infer_geometry_converts_mm_to_m():
    df = pd.DataFrame({
        "D1": [100],  # mm
        "D2": [200],  # mm
        "L": [1000],  # mm
        "r": [1.5],
        "dt": [60],
    })
    geo = infer_geometry_from_table(df)
    assert geo is not None
    assert np.isclose(geo.D1, 0.1)
    assert np.isclose(geo.D2, 0.2)
    assert np.isclose(geo.L, 1.0)
    assert np.isclose(geo.dt, 0.06)
    assert np.isclose(geo.r_As_At, 1.5)

