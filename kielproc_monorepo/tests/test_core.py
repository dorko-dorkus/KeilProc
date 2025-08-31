import numpy as np
from duct_dp_visualizer import pearson_r, fisher_z_ci, theil_sen_subsample_ci, SliceParams, analyze_port
import pandas as pd

def test_pearson_and_fisher_ci():
    # perfect positive
    x = np.arange(100, dtype=float)
    y = 2*x + 1
    r = pearson_r(x,y)
    assert r > 0.999
    rc, lo, hi = fisher_z_ci(r, len(x))
    assert np.isnan(lo) or lo <= rc <= hi

def test_theil_sen_near_true_slope():
    rng = np.random.default_rng(0)
    x = np.linspace(0,10,200)
    y = 3.0*x + 0.5 + rng.normal(0, 0.3, size=x.size)
    m, lo, hi = theil_sen_subsample_ci(x,y,B=100,pairs_per_boot=200,seed=1)
    assert 2.5 < m < 3.5
    assert lo < m < hi

def test_analyze_port_slices():
    # Build a DF with a known sign flip: bottom slope negative, top positive
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.concatenate([ -2*x[:30] + 0.1*np.random.randn(30),
                          0.1*np.random.randn(140),
                          2*x[-30:] + 0.1*np.random.randn(30)])
    df = pd.DataFrame({"SP": x, "VP": y})
    tidy = analyze_port(df, SliceParams(frac=0.15, kmin=10))
    assert set(tidy["Slice"]) == {"bottom","middle","top"}
    # Expect bottom and top slopes of opposite sign
    b = float(tidy.loc[tidy["Slice"]=="bottom","theil_sen"])
    t = float(tidy.loc[tidy["Slice"]=="top","theil_sen"])
    assert b*t < 0