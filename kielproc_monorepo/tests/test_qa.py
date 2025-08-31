import pandas as pd
import numpy as np
from pathlib import Path
from kielproc.qa import qa_indices, DEFAULT_DELTA_OPP_MAX, DEFAULT_W_MAX
from kielproc.cli import main as cli_main


def test_qa_indices_basic():
    d, w = qa_indices(100, 102, 101, 99, 50)
    assert np.isclose(d, 0.04)
    assert np.isclose(w, np.sqrt(8)/(100))


def test_qa_default_thresholds():
    assert np.isclose(DEFAULT_DELTA_OPP_MAX, 0.01)
    assert np.isclose(DEFAULT_W_MAX, 0.002)


def test_fit_records_qa(tmp_path: Path):
    # create simple CSV
    df = pd.DataFrame({
        "mapped_ref": [1,2,3,4],
        "piccolo": [1.1,2.1,3.1,4.1],
        "pN": [100,100,100,100],
        "pS": [102,102,102,102],
        "pE": [101,101,101,101],
        "pW": [99,99,99,99],
        "q_mean": [50,50,50,50],
    })
    csv = tmp_path/"block.csv"; df.to_csv(csv, index=False)
    outdir = tmp_path/"out"
    cli_main(["fit","--blocks",f"b1={csv}","--outdir",str(outdir),"--qa-gate-opp","0.05","--qa-gate-w","0.05"])
    tbl = pd.read_csv(outdir/"alpha_beta_by_block.csv")
    assert "delta_opp" in tbl.columns and "W" in tbl.columns and "qa_pass" in tbl.columns
    assert bool(tbl.loc[0,"qa_pass"]) is True


def test_fit_gate_fail(tmp_path: Path):
    df = pd.DataFrame({
        "mapped_ref": [1,2,3,4],
        "piccolo": [1.1,2.1,3.1,4.1],
        "pN": [100,100,100,100],
        "pS": [105,105,105,105],
        "pE": [101,101,101,101],
        "pW": [99,99,99,99],
        "q_mean": [50,50,50,50],
    })
    csv = tmp_path/"block.csv"; df.to_csv(csv, index=False)
    outdir = tmp_path/"out"
    try:
        cli_main(["fit","--blocks",f"b1={csv}","--outdir",str(outdir),"--qa-gate-opp","0.01","--qa-gate-w","0.01"])
    except SystemExit:
        pass
    tbl = pd.read_csv(outdir/"alpha_beta_by_block.csv")
    assert bool(tbl.loc[0,"qa_pass"]) is False
