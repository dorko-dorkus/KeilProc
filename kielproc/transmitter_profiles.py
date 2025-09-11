from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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
    bias_unit = (entry.get("bias_unit") or "tph").lower()
    bias_mode = entry.get("bias_mode") or "current_pre_scale"
    m = entry.get("m")
    c = entry.get("c")
    if (m is None or c is None):
        if (gain is None) or (bias is None) or (rng is None):
            return None
        m, c = derive_mc_from_gain_bias(
            gain,
            bias,
            rng,
            full_scale_tph=fs,
            bias_unit=bias_unit,
            bias_mode=bias_mode,
        )
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
        "bias_mode": bias_mode,
    }


def derive_mc_from_gain_bias(
    gain: float,
    bias: float,
    range_mbar: float,
    full_scale_tph: float = 100.0,
    bias_unit: str = "tph",
    bias_mode: str = "current_pre_scale",
) -> tuple[float, float]:
    """Return (m, c) from (gain, bias, range, FS).

    Parameters
    ----------
    gain, bias : float
        Calibration gain and bias.
    range_mbar : float
        Pressure range in mbar.
    full_scale_tph : float, default 100.0
        Full-scale flow in tph.
    bias_unit : {"tph", "percent", "mA"}, default "tph"
        Unit of the bias value. ``"percent"`` indicates percentage of full scale
        and ``"mA"`` uses 4–20 mA scaling.
    bias_mode : {"current_pre_scale", "flow_output"}, default "current_pre_scale"
        Interpretation of ``bias`` when ``bias_unit`` is ``"mA"``.

    Notes
    -----
    The resulting linear map is ``Flow_820 = m*DP + c`` where ``m = gain*FS/R``.
    For ``bias_unit="mA"``:

    * ``bias_mode="current_pre_scale"`` assumes ``bias`` represents an input
      current (mA) applied before ``gain`` and DP scaling, giving
      ``c = (gain*FS/16) * bias``.
    * ``bias_mode="flow_output"`` treats ``bias`` as a current offset applied
      directly to the flow output, yielding ``c = (FS/16) * bias``.
    """
    R = float(range_mbar)
    FS = float(full_scale_tph)
    g = float(gain)
    m = g * FS / R
    u = (bias_unit or "tph").lower()
    if u == "tph":
        c = float(bias)
    elif u in ("percent", "%", "pct"):
        c = float(bias) * FS / 100.0
    elif u in ("ma", "mA", "MA"):
        bma = float(bias)
        if (bias_mode or "current_pre_scale") == "current_pre_scale":
            c = (g * FS / 16.0) * bma
        else:
            c = (FS / 16.0) * bma
    else:
        raise ValueError(f"Unsupported bias_unit '{bias_unit}'")
    return m, c
