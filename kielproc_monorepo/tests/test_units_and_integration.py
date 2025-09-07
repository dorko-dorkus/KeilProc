import pandas as pd
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from kielproc.tools.legacy_parser.parser import _sanitize_headers_and_units
from kielproc.aggregate import _normalize_df, RunConfig, integrate_run


def test_header_unit_conversions():
    df = pd.DataFrame({
        "Temperature (K)": [300.0, 310.0],
        "Static_gauge (kPa)": [1.0, 1.0],
        "Baro (kPa)": [100.0, 100.0],
        "VP": [5.0, 5.0],
    })
    san = _sanitize_headers_and_units(df)
    assert "Temperature" in san.columns
    assert san["Temperature"].iloc[0] == pytest.approx(26.85, abs=1e-2)
    assert san["Static_gauge"].iloc[0] == pytest.approx(1000.0)
    assert san["Baro"].iloc[0] == pytest.approx(100000.0)
    norm, meta = _normalize_df(san, None)
    assert norm["Static_abs_Pa"].iloc[0] == pytest.approx(101000.0)


def test_integration_missing_ports(tmp_path):
    for i in range(1, 8):  # create seven ports P1..P7
        df = pd.DataFrame({
            "Time": [0, 1, 2],
            "Static": [101325.0, 101325.0, 101325.0],
            "VP": [1.0, 1.0, 1.0],
            "Temperature": [20.0, 20.0, 20.0],
        })
        df.to_csv(tmp_path / f"P{i}.csv", index=False)
    cfg = RunConfig(height_m=1.0, width_m=1.0, weights={f"P{j}": 1.0 for j in range(1, 9)})
    res = integrate_run(tmp_path, cfg, area_ratio=1.0)
    assert {f"P{i}" for i in range(1,8)} == set(res["per_port"]["Port"])
    assert res["duct"]["q_s_pa"] == pytest.approx(1.0)
