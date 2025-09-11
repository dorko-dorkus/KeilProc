from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

# Default pressure range in mbar if not provided in profile
default_range = 100.0


def load_tx_profile(site_name: str,
                    season: str,
                    search_paths: List[str | Path] | None = None
                   ) -> tuple[float, float, float, dict] | None:
    """Return calibration (m, c, range_mbar, meta) for a site/season.

    Profiles are stored as JSON files named ``<site_name>.json`` within any of
    the *search_paths*. Each file maps season names to calibration entries.
    An entry may provide ``m`` and ``c`` directly or it may specify ``gain``
    and ``bias`` (with optional ``full_scale_tph`` and ``bias_unit``). If
    ``gain``/``bias`` are given, they are converted to ``m``/``c`` using the
    provided ``range_mbar``.
    """
    search_paths = search_paths or []
    search: List[Path] = []
    for base in search_paths:
        p = Path(base)
        if p.is_file() and p.suffix.lower() == ".json":
            search.append(p)
        else:
            candidate = p / f"{site_name}.json"
            if candidate.exists():
                search.append(candidate)
    if not search:
        return None
    try:
        data = json.loads(search[0].read_text())
    except Exception:
        return None
    entry: Dict[str, Any] | None = (
        data.get(season)
        or data.get(season.lower())
        or data.get(season.upper())
        or data.get(season.capitalize())
    )
    if not entry:
        return None
    # Support either direct m,c or (gain,bias,FS)
    rng = entry.get("range_mbar", default_range)
    gain = entry.get("gain")
    bias = entry.get("bias")
    fs = entry.get("full_scale_tph", 100.0)
    bias_unit = (entry.get("bias_unit") or "tph").lower()  # "tph" or "percent"
    m = entry.get("m")
    c = entry.get("c")
    if (m is None or c is None):
        if (gain is None) or (bias is None) or (rng is None):
            return None
        m = float(gain) * float(fs) / float(rng)
        c = float(bias) if bias_unit == "tph" else float(bias) * float(fs) / 100.0
    if rng is None:
        return None
    return float(m), float(c), float(rng), {
        "site": site_name,
        "season": season,
        "source_file": str(search[0]) if search else "unknown",
        "gain": gain,
        "bias": bias,
        "full_scale_tph": fs,
        "bias_unit": bias_unit,
    }


def derive_mc_from_gain_bias(gain: float, bias: float, range_mbar: float,
                             full_scale_tph: float = 100.0,
                             bias_unit: str = "tph") -> Tuple[float, float]:
    """Return (m, c) from (gain, bias, range, FS).

    Parameters
    ----------
    gain, bias : float
        Calibration gain and bias.
    range_mbar : float
        Pressure range in mbar.
    full_scale_tph : float, default 100.0
        Full-scale flow in tph.
    bias_unit : {"tph", "percent"}, default "tph"
        Unit of the bias value.  ``"percent"`` indicates percentage of
        full scale.
    """
    m = float(gain) * float(full_scale_tph) / float(range_mbar)
    c = float(bias) if bias_unit.lower() == "tph" else float(bias) * float(full_scale_tph) / 100.0
    return m, c
