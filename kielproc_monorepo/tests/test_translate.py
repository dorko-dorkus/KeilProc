import pandas as pd
import pytest
from kielproc.translate import apply_translation

def test_apply_translation_missing_column():
    df = pd.DataFrame({'a':[1,2,3]})
    with pytest.raises(KeyError, match="Column 'piccolo' not found"):
        apply_translation(df, alpha=1.0, beta=0.0, src_col='piccolo')
