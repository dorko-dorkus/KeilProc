from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import re, json, math
from typing import Optional, List

NUMERIC_FIELDS_CANON = ["Static", "VP", "Temperature", "Piccolo"]


@dataclass
class LegacyCube:
    """3-D workbook view: ports × samples × fields, NaN-padded per port."""
    ports: list[str]
    fields: list[str]
    data: np.ndarray  # shape (P, Nmax, F), dtype=float
    counts: np.ndarray  # per-port valid sample counts, shape (P,)
    workbook: str

    def to_dataframe(self) -> pd.DataFrame:
        P, N, F = self.data.shape
        idx = pd.MultiIndex.from_product([self.ports, range(N)], names=["Port", "Sample"])
        df = pd.DataFrame(self.data.reshape(P * N, F), index=idx, columns=self.fields)
        mask = np.concatenate([np.r_[True] * c + np.r_[False] * (N - c) for c in self.counts])
        return df[mask]

# Columns expected in every parsed sheet.  These are used to validate
# that the parser was able to populate all required fields when reading a
# legacy workbook.  The list can be extended in the future as additional
# fields become mandatory.
REQUIRED_COLUMNS = ["Time", "Static", "VP", "Temperature", "Piccolo"]

@dataclass
class SheetParseResult:
    sheet: str
    replicate: Optional[str]
    csv_path: str
    n_rows: int
    n_cols: int
    piccolo_flat: bool
    mode: str
    notes: str

def _clean_name(x):
    return str(x).strip().lower() if pd.notna(x) else ""


def _missing_required(df: pd.DataFrame) -> List[str]:
    """Return a list of required column names that are absent in ``df``."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _to_time_s(df: pd.DataFrame) -> pd.Series:
    if "Time" in df:
        t = pd.to_datetime(df["Time"], errors="coerce")
        if t.notna().any():
            t0 = t.dropna().iloc[0]
            return (t - t0).dt.total_seconds()
    if "Sample" in df:
        # Fallback: monotonic samples if present
        return pd.to_numeric(df["Sample"], errors="coerce")
    return pd.Series([np.nan] * len(df), name="Time_s")


def _frames_to_cube(workbook_name: str, frames: dict[str, pd.DataFrame]) -> LegacyCube:
    ports = list(frames.keys())
    # union of numeric fields we care about, keep deterministic order
    present = []
    for f in NUMERIC_FIELDS_CANON + ["Time_s"]:
        if any((f in df.columns) or (f == "Time_s" and "Time" in df.columns) for df in frames.values()):
            present.append(f)
    fields = present or NUMERIC_FIELDS_CANON  # never empty
    counts = np.array([len(frames[p]) for p in ports], dtype=int)
    Nmax = int(counts.max()) if len(counts) else 0
    F = len(fields)
    cube = np.full((len(ports), Nmax, F), np.nan, dtype=float)
    for i, p in enumerate(ports):
        df = frames[p].copy()
        if "Time_s" in fields and "Time_s" not in df.columns:
            df["Time_s"] = _to_time_s(df)
        for k, col in enumerate(fields):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
                n = min(len(vals), Nmax)
                cube[i, :n, k] = vals[:n]
    return LegacyCube(ports=ports, fields=fields, data=cube, counts=counts, workbook=workbook_name)

def parse_legacy_workbook(
    xlsx_path: Path,
    out_dir: Path | None = None,
    piccolo_flat_threshold: float = 1e-6,
    return_mode: str = "files",  # "files" | "frames" | "array"
):
    """
    Parse legacy workbooks.
    - files  -> writes CSVs + summary.json (backward compatible), returns summary dict
    - frames -> returns (frames: dict[port->DataFrame], summary dict)
    - array  -> returns (cube: LegacyCube, summary dict)
    """
    xlsx_path = Path(xlsx_path)
    if return_mode == "files":
        assert out_dir is not None, "out_dir is required when return_mode='files'"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    try:
        xl = pd.ExcelFile(xlsx_path)
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError("Unable to open workbook. For .xls files install 'xlrd'.") from exc
    results: List[SheetParseResult] = []
    frames: dict[str, pd.DataFrame] = {}

    for sheet in xl.sheet_names:
        try:
            raw = xl.parse(sheet, header=None)
        except Exception:
            continue
        header_row = None; unit_row = None
        for i in range(min(15, len(raw))):
            row = [_clean_name(v) for v in raw.iloc[i].tolist()]
            if "static pressure" in row and "velocity pressure" in row:
                header_row = i
                unit_row = i+1 if i+1 < len(raw) else None
                break

        piccolo_flat = False; notes = ""

        if header_row is not None:
            header_cells = [_clean_name(v) for v in raw.iloc[header_row].tolist()]
            time_positions = [j for j,v in enumerate(header_cells) if v == "time"]

            if len(time_positions) <= 1:
                # 2018-style vertical
                cols = raw.iloc[header_row].astype(str).tolist()
                units = raw.iloc[unit_row].astype(str).tolist() if unit_row is not None else [None]*len(cols)
                newcols = []
                for n,u in zip(cols, units):
                    n = str(n).strip()
                    if u not in (None, "", float("nan")):
                        newcols.append(f"{n} ({str(u).strip()})")
                    else:
                        newcols.append(n)
                df = xl.parse(sheet, header=header_row)
                df.columns = newcols[:len(df.columns)]
                if "Time" in df.columns:
                    mask = df["Time"].astype(str).map(_clean_name)
                    # Drop rows where the time column is empty, a header repeat, or an
                    # averages row.  ``_clean_name`` normalizes case/whitespace so we
                    # can simply test prefixes for robustness (e.g. "Averages:").
                    drop_time = mask.isin(['', 'nan']) | mask.str.contains(r"^(?:time|averages?)", na=False)
                    df = df[~drop_time].copy()

                # Remove any rows that are entirely header labels or contain the word
                # "averages" in any column.  This handles cases where a header row
                # leaks into the body of the sheet.
                clean_rows = df.applymap(_clean_name)
                hdr_clean = [_clean_name(c) for c in df.columns]
                bad = clean_rows.apply(
                    lambda r: r.eq('averages').any() or
                               all((v == '' or v == h) for v, h in zip(r, hdr_clean)),
                    axis=1,
                )
                df = df[~bad].copy()

                out = pd.DataFrame(index=df.index)

                def first_match(patterns):
                    for p in patterns:
                        for c in df.columns:
                            if re.search(p, str(c), flags=re.I):
                                return c
                    return None

                m_time = first_match([r"^time(?!.*unit)"])
                m_static = first_match([r"static pressure"])
                m_vp = first_match([r"velocity pressure"])
                m_temp = first_match([r"duct air temperature|temperature"])
                m_picc = first_match([r"piccolo.*current|piccolo.*dp|throat.*dp"])

                if m_time: out["Time"] = df[m_time]
                if m_static: out["Static"] = df[m_static]
                if m_vp: out["VP"] = df[m_vp]
                if m_temp: out["Temperature"] = df[m_temp]
                if m_picc: out["Piccolo"] = df[m_picc]

                key_cols = [c for c in ["Static", "VP", "Temperature", "Piccolo"] if c in out.columns]
                if key_cols:
                    # Coerce to numeric so that any leaked headers or text such as
                    # "Averages" become NaN prior to the emptiness check.
                    for c in key_cols:
                        out[c] = pd.to_numeric(out[c], errors="coerce")
                    mask_all_nan = out[key_cols].isna().all(axis=1)
                    out = out[~mask_all_nan]
                out = out.copy()
                out.insert(0, "Sample", range(1, len(out)+1))
                out["Workbook"] = xlsx_path.name
                out["Sheet"] = sheet
                m = re.search(r"(\.\d+)$", sheet)
                replicate = m.group(1) if m else None
                out["Replicate"] = replicate or ""

                if "Piccolo" in out.columns:
                    v = out["Piccolo"].to_numpy(float)
                    piccolo_flat = bool(np.nanstd(v) < piccolo_flat_threshold)

                missing = _missing_required(out)
                if missing:
                    notes = f"missing required columns: {', '.join(missing)}"

                port_key = sheet if replicate is None else f"{sheet}__{replicate}"
                frames[port_key] = out

                if return_mode == "files":
                    csv_path = Path(out_dir) / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', port_key)}.csv"
                    out.to_csv(csv_path, index=False)
                    results.append(
                        SheetParseResult(
                            sheet=sheet,
                            replicate=replicate,
                            csv_path=str(csv_path),
                            n_rows=int(len(out)),
                            n_cols=int(out.shape[1]),
                            piccolo_flat=piccolo_flat,
                            mode="vertical",
                            notes=notes,
                        )
                    )
                else:
                    results.append(
                        SheetParseResult(
                            sheet=sheet,
                            replicate=replicate,
                            csv_path=None,
                            n_rows=int(len(out)),
                            n_cols=int(out.shape[1]),
                            piccolo_flat=piccolo_flat,
                            mode="vertical",
                            notes=notes,
                        )
                    )
                continue

            else:
                # 2007-style multiport
                headers = raw.iloc[header_row].astype(str).tolist()
                group_starts = [j for j,h in enumerate(headers) if _clean_name(h) == "time"]
                group_starts.append(len(headers))
                rows_acc = []
                data_start = (unit_row or header_row) + 1

                for a,b in zip(group_starts[:-1], group_starts[1:]):
                    sub_hdr = headers[a:b]
                    port_val = None
                    for off in range(0, min(5, len(sub_hdr))):
                        cell = raw.iloc[0, a+off] if (a+off) < raw.shape[1] else None
                        try:
                            pv = float(cell)
                            if not math.isnan(pv):
                                port_val = pv; break
                        except Exception:
                            continue
                    for r in range(data_start, len(raw)):
                        vals = raw.iloc[r, a:b]
                        if pd.isna(vals).all():
                            continue
                        if any(_clean_name(v).startswith('average') for v in vals.values):
                            continue
                        rec = {"Sample": r - data_start + 1, "Workbook": xlsx_path.name, "Sheet": sheet, "Replicate": ""}
                        rec["Port"] = port_val
                        names = [str(n).strip().lower() for n in raw.iloc[header_row, a:b]]
                        for j,name in enumerate(names):
                            v = vals.iloc[j]
                            if name == "time":
                                rec["Time"] = v
                            elif "static pressure" in name:
                                rec["Static"] = v
                            elif "velocity pressure" in name:
                                rec["VP"] = v
                            elif "duct air temperature" in name:
                                rec["Temperature"] = v
                            elif "piccolo" in name:
                                rec["Piccolo"] = v
                        tval = _clean_name(rec.get('Time', ''))
                        if tval in ('', 'nan') or tval.startswith('time') or tval.startswith('average'):
                            continue
                        rows_acc.append(rec)

                if rows_acc:
                    out = pd.DataFrame(rows_acc)
                    key_cols = [c for c in ["Static", "VP", "Temperature", "Piccolo"] if c in out.columns]
                    if key_cols:
                        for c in key_cols:
                            out[c] = pd.to_numeric(out[c], errors="coerce")
                        mask_all_nan = out[key_cols].isna().all(axis=1)
                        out = out[~mask_all_nan]
                    if "Piccolo" in out.columns:
                        v = out["Piccolo"].to_numpy(float)
                        piccolo_flat = bool(np.nanstd(v) < piccolo_flat_threshold)

                    missing = _missing_required(out)
                    note = ""
                    if missing:
                        note = f"missing required columns: {', '.join(missing)}"

                    port_key = sheet
                    frames[port_key] = out

                    if return_mode == "files":
                        csv_path = Path(out_dir) / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', port_key)}.csv"
                        out.to_csv(csv_path, index=False)
                        results.append(
                            SheetParseResult(
                                sheet=sheet,
                                replicate=None,
                                csv_path=str(csv_path),
                                n_rows=int(len(out)),
                                n_cols=int(out.shape[1]),
                                piccolo_flat=piccolo_flat,
                                mode="multiport",
                                notes=note,
                            )
                        )
                    else:
                        results.append(
                            SheetParseResult(
                                sheet=sheet,
                                replicate=None,
                                csv_path=None,
                                n_rows=int(len(out)),
                                n_cols=int(out.shape[1]),
                                piccolo_flat=piccolo_flat,
                                mode="multiport",
                                notes=note,
                            )
                        )
                    continue

        # Fallback: minimal CSV
        out = pd.DataFrame({"Sample": range(1, len(raw)+1)})
        out["Workbook"] = xlsx_path.name; out["Sheet"] = sheet; out["Replicate"] = ""
        port_key = sheet
        frames[port_key] = out
        if return_mode == "files":
            csv_path = Path(out_dir) / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', port_key)}.csv"
            out.to_csv(csv_path, index=False)
            results.append(SheetParseResult(sheet=sheet, replicate=None, csv_path=str(csv_path),
                                            n_rows=int(len(out)), n_cols=int(out.shape[1]),
                                            piccolo_flat=False, mode="fallback", notes="no recognizable headers"))
        else:
            results.append(SheetParseResult(sheet=sheet, replicate=None, csv_path=None,
                                            n_rows=int(len(out)), n_cols=int(out.shape[1]),
                                            piccolo_flat=False, mode="fallback", notes="no recognizable headers"))

    summary = {"workbook": xlsx_path.name, "sheets": [asdict(r) for r in results]}

    if return_mode == "files":
        with open(Path(out_dir) / f"{xlsx_path.stem}__parse_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    if return_mode == "frames":
        return frames, summary

    if return_mode == "array":
        cube = _frames_to_cube(xlsx_path.name, frames)
        return cube, summary

    raise ValueError(f"Unknown return_mode={return_mode!r}")
