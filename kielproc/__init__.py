"""Package redirect to monorepo implementation.

This stub allows importing ``kielproc`` when running tests from the
repository root without installing the project. It loads the actual package
located under ``kielproc_monorepo/kielproc``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent / "kielproc_monorepo" / "kielproc"
_spec = importlib.util.spec_from_file_location(
    __name__,
    _pkg_dir / "__init__.py",
    submodule_search_locations=[str(_pkg_dir)],
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)
