import numpy as np
import pandas as pd
import pytest

from kielproc.geometry import (
    DiffuserGeometry,
    infer_geometry_from_table,
    planes_to_z,
    plane_value_to_z,
    Geometry,
    plane_area,
    effective_upstream_area,
    throat_area,
    r_ratio,
    beta_from_geometry,
)


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


def test_planes_to_z_maps_indices_to_length():
    geo = DiffuserGeometry(D1=0.1, D2=0.2, L=1.2)
    planes = np.array([0, 1, 2, 3])
    z = planes_to_z(planes, geo)
    assert np.allclose(z, np.linspace(0, 1.2, 4))
    # single plane value interpolation
    z_val = plane_value_to_z(2, planes, geo)
    assert np.isclose(z_val, 0.8)


def test_beta_from_geometry_defaults_to_plane_area():
    g = Geometry(duct_height_m=2.0, duct_width_m=3.0, throat_diameter_m=1.0)
    beta = beta_from_geometry(g)
    expected = np.sqrt(throat_area(g) / plane_area(g))
    assert np.isclose(beta, expected)
    assert np.isclose(r_ratio(g), plane_area(g) / throat_area(g))


def test_beta_from_geometry_with_upstream_area():
    g = Geometry(
        duct_height_m=2.0,
        duct_width_m=3.0,
        upstream_area_m2=10.0,
        throat_area_m2=2.0,
    )
    beta = beta_from_geometry(g)
    assert np.isclose(beta, np.sqrt(2.0 / 10.0))
    assert np.isclose(effective_upstream_area(g), 10.0)


def test_throat_area_requires_input():
    g = Geometry()
    with pytest.raises(ValueError):
        throat_area(g)

