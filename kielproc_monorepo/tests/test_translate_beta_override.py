import pandas as pd
import numpy as np
from kielproc.translate import compute_translation_table


def test_compute_translation_table_beta_override():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 5.0
    df = pd.DataFrame({"mapped_ref": x, "piccolo": y})
    per, pooled = compute_translation_table({"b": df}, beta_override=5.0, max_lag=0)
    row = per.iloc[0]
    assert np.isclose(row["alpha"], 2.0)
    assert np.isclose(row["beta"], 5.0)
    assert pooled is None
