import pytest
from gui.app_gui import _float_or_zero


def test_float_or_zero_with_number():
    assert _float_or_zero("1.5") == 1.5


def test_float_or_zero_with_empty_string():
    assert _float_or_zero("") == 0.0


def test_float_or_zero_invalid_raises():
    with pytest.raises(ValueError):
        _float_or_zero("abc")
