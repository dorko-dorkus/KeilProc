import json
from pathlib import Path

import pandas as pd

from kielproc.cli import main as cli_main


def test_integrate_autodiscover(tmp_path, monkeypatch):
    run_dir = tmp_path
    # create geometry and weights artifacts
    (run_dir / "weights.json").write_text(json.dumps({"P1": 0.5}))
    (run_dir / "geometry.json").write_text(json.dumps({"r": 2.0, "beta": 0.7}))

    called = {}

    def fake_integrate(run_dir_path, cfg, file_glob="*.csv", baro_cli_pa=None, area_ratio=None, beta=None):
        called["weights"] = cfg.weights
        called["area_ratio"] = area_ratio
        called["beta"] = beta
        return {"per_port": pd.DataFrame(), "duct": {}, "normalize_meta": {}, "files": [], "pairs": []}

    monkeypatch.setattr("kielproc.cli.integrate_run", fake_integrate)

    cli_main([
        "integrate-ports",
        "--run-dir",
        str(run_dir),
        "--duct-height",
        "1",
        "--duct-width",
        "1",
    ])

    assert called["weights"] == {"P1": 0.5}
    assert called["area_ratio"] == 2.0
    assert called["beta"] == 0.7
