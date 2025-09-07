import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from kielproc.gui_adapter import fit_alpha_beta
from kielproc.setpoints import find_optimal_transmitter_span
from kielproc.report import plot_alignment


def test_fit_alpha_beta_with_qa_gate(tmp_path):
    df = pd.DataFrame({
        "mapped_ref": [0.0, 1.0, 2.0],
        "piccolo": [1.0, 3.0, 5.0],
        "pN": [100.0, 100.0, 100.0],
        "pS": [100.0, 100.0, 100.0],
        "pE": [100.0, 100.0, 100.0],
        "pW": [100.0, 100.0, 100.0],
        "q_mean": [100.0, 100.0, 100.0],
    })
    outdir = tmp_path / "fit"
    res = fit_alpha_beta({"blk": df}, "mapped_ref", "piccolo", 1.0, 10, outdir)
    info = res["blocks_info"][0]
    assert info["alpha"] == pytest.approx(2.0)
    assert info["beta"] == pytest.approx(1.0)

    df_bad = df.copy()
    df_bad["pN"] = [500.0, 100.0, 100.0]
    with pytest.raises(RuntimeError):
        fit_alpha_beta({"bad": df_bad}, "mapped_ref", "piccolo", 1.0, 10, tmp_path / "bad", qa_gate_opp=0.01, qa_gate_w=1.0)


def test_find_optimal_transmitter_span():
    x = [0, 1, 2, 3, 4]
    y = [1 + 2 * xi for xi in x]
    opt = find_optimal_transmitter_span(x, y, slope_sign=+1)
    assert opt.slope == pytest.approx(2.0)
    assert opt.intercept == pytest.approx(1.0)
    assert opt.x_low == pytest.approx(0.0)
    assert opt.x_high == pytest.approx(2.0)
    assert opt.setpoints["span"]["x_low"] == pytest.approx(0.0)


def test_plot_alignment_headless(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    t = np.array([0.0, 1.0, 2.0])
    ref = np.array([0.0, 1.0, 2.0])
    picc = np.array([0.1, 1.1, 2.1])
    shifted = np.array([0.1, 1.1, 2.1])
    png = plot_alignment(tmp_path, t, ref, picc, shifted, title="align", stem="test")
    assert Path(png).exists()
