import pandas as pd
import pytest
import numpy as np
from kielproc.aggregate import integrate_run, RunConfig
from kielproc.physics import rho_from_pT


def test_integrate_run_rejects_sub_absolute_zero(tmp_path):
    df = pd.DataFrame({
        "VP": [1.0],
        "Temperature": [-300.0],
        "Static": [101325.0],
    })
    csv = tmp_path / "P1.csv"
    df.to_csv(csv, index=False)
    cfg = RunConfig(height_m=1.0, width_m=1.0)
    with pytest.raises(ValueError):
        integrate_run(tmp_path, cfg)


def test_rho_from_pT_rejects_sub_zero_kelvin():
    with pytest.raises(ValueError):
        rho_from_pT(np.array([101325.0]), np.array([0.0]))
