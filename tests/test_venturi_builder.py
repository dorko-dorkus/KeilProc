import json
import numpy as np
import pytest
from pathlib import Path

from kielproc.tools.venturi_builder import build_venturi_result, R_SPECIFIC_AIR


def test_build_venturi_result(tmp_path):
    out = build_venturi_result(
        tmp_path,
        beta=0.5,
        area_As_m2=1.0,
        baro_pa=101325.0,
        T_K=300.0,
        m_dot_hint_kg_s=2.0,
    )
    assert out is not None
    data = json.loads(Path(out).read_text())
    assert data["beta"] == pytest.approx(0.5)
    assert data["A1_m2"] == pytest.approx(1.0)
    assert data["At_m2"] == pytest.approx(0.25)
    expected_rho = 101325.0 / (R_SPECIFIC_AIR * 300.0)
    assert data["rho_kg_m3"] == pytest.approx(expected_rho)
    flow = np.array(data["flow_kg_s"])
    dp = np.array(data["dp_pa"])
    assert len(flow) == len(dp) == 200
    assert np.all(np.diff(flow) > 0)
    expected_dp = (1.0 - 0.5**4) * (flow**2) / (2.0 * expected_rho * (0.25**2))
    assert dp == pytest.approx(expected_dp)
