import pandas as pd
import numpy as np

from kielproc.aggregate import integrate_run, RunConfig


def test_integrate_run_handles_mbar_and_cmh2o(tmp_path):
    df = pd.DataFrame({
        "VelPress_cmH2O": [2.0],
        "Temperature": [20.0],
        "Static_mbar": [1013.25],
    })
    csv = tmp_path / "P1.csv"
    df.to_csv(csv, index=False)
    cfg = RunConfig(height_m=1.0, width_m=1.0)
    res = integrate_run(tmp_path, cfg)
    per = res["per_port"]
    assert np.isclose(per["VP_pa_mean"].iloc[0], 2.0 * 98.0665)
    assert np.isclose(per["Static_abs_pa_mean"].iloc[0], 1013.25 * 100.0)
    meta = res["normalize_meta"][csv.name]
    assert meta["vp_unit"] == "cmH2O"
    assert meta["static_unit"] == "mbar"
