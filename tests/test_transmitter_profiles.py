import json
from pathlib import Path

from kielproc.transmitter_profiles import load_tx_profile, derive_mc_from_gain_bias


def test_derive_mc_from_gain_bias_tph():
    m, c = derive_mc_from_gain_bias(2.0, 5.0, range_mbar=50.0, full_scale_tph=100.0, bias_unit="tph")
    assert m == 2.0 * 100.0 / 50.0
    assert c == 5.0


def test_derive_mc_from_gain_bias_percent():
    m, c = derive_mc_from_gain_bias(2.0, 10.0, range_mbar=50.0, full_scale_tph=200.0, bias_unit="percent")
    assert m == 2.0 * 200.0 / 50.0
    assert c == 10.0 * 200.0 / 100.0


def test_load_tx_profile_gain_bias(tmp_path: Path):
    profile = {
        "summer": {
            "gain": 2.0,
            "bias": 5.0,
            "range_mbar": 50.0,
            "full_scale_tph": 100.0,
            "bias_unit": "tph",
        }
    }
    f = tmp_path / "mysite.json"
    f.write_text(json.dumps(profile))
    res = load_tx_profile("mysite", "summer", [tmp_path])
    assert res is not None
    m, c, rng, meta = res
    assert abs(m - (2.0 * 100.0 / 50.0)) < 1e-9
    assert c == 5.0
    assert rng == 50.0
    assert meta["source_file"] == str(f)
