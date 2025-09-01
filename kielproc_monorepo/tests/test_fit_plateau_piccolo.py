import pandas as pd
from pathlib import Path

from kielproc.cli import main as cli_main


def test_fit_plateau_and_lag_seconds(tmp_path: Path):
    # Good block with small variations
    g = pd.DataFrame({
        "mapped_ref": [100 + 0.1 * i for i in range(10)],
        "ma": [12 + 0.01 * i for i in range(10)],
        "pN": [100] * 10,
        "pS": [100] * 10,
        "pE": [100] * 10,
        "pW": [100] * 10,
        "q_mean": [50] * 10,
    })
    g_csv = tmp_path / "g.csv"
    g.to_csv(g_csv, index=False)

    # Bad block with step causing large SD/mean
    b = pd.DataFrame({
        "mapped_ref": [100] * 5 + [200] * 5,
        "ma": [12] * 5 + [18] * 5,
        "pN": [100] * 10,
        "pS": [100] * 10,
        "pE": [100] * 10,
        "pW": [100] * 10,
        "q_mean": [50] * 10,
    })
    b_csv = tmp_path / "b.csv"
    b.to_csv(b_csv, index=False)

    outdir = tmp_path / "out"
    cli_main([
        "fit",
        "--blocks",
        f"good={g_csv}",
        f"bad={b_csv}",
        "--outdir",
        str(outdir),
        "--sampling-hz",
        "1",
        "--plateau-window-s",
        "2",
        "--plateau-sd-frac",
        "0.1",
        "--piccolo-ma-col",
        "ma",
    ])

    tbl = pd.read_csv(outdir / "alpha_beta_by_block.csv")
    assert list(tbl["block"]) == ["good"]
    assert "lag_seconds" in tbl.columns
