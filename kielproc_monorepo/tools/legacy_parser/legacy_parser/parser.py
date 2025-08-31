from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import re, json, math
from typing import Optional, List

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

def parse_legacy_workbook(xlsx_path: Path, out_dir: Path, piccolo_flat_threshold: float=1e-6) -> dict:
    """Parse legacy 2007/2011/2018 workbooks into standardized CSVs."""
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        xl = pd.ExcelFile(xlsx_path)
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "Unable to open workbook. For .xls files install the 'xlrd' package"
        ) from exc
    results: List[SheetParseResult] = []

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
                    mask = df["Time"].astype(str).str.strip().str.lower()
                    df = df[~mask.isin(["nan","time","averages"])].copy()

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

                key_cols = [c for c in ["Static","VP","Temperature","Piccolo"] if c in out.columns]
                if key_cols:
                    mask_all_nan = out[key_cols].isna().all(axis=1)
                    out = out[~mask_all_nan]
                out = out.copy()
                out.insert(0, "Sample", range(1, len(out)+1))
                out["Workbook"] = xlsx_path.name
                out["Sheet"] = sheet
                m = re.search(r"(\.\d+)$", sheet); out["Replicate"] = m.group(1) if m else ""

                if "Piccolo" in out.columns:
                    v = pd.to_numeric(out["Piccolo"], errors="coerce")
                    piccolo_flat = bool(np.nanstd(v) < piccolo_flat_threshold)

                missing = _missing_required(out)
                if missing:
                    notes = f"missing required columns: {', '.join(missing)}"

                csv_path = out_dir / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', sheet)}.csv"
                out.to_csv(csv_path, index=False)
                results.append(
                    SheetParseResult(
                        sheet=sheet,
                        replicate=(m.group(1) if m else None),
                        csv_path=str(csv_path),
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
                        if any(str(v).strip().lower() == 'averages' for v in vals.values):
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
                        tval = str(rec.get('Time','')).strip().lower()
                        if tval in ('', 'nan', 'time', 'averages'):
                            continue
                        rows_acc.append(rec)

                if rows_acc:
                    out = pd.DataFrame(rows_acc)
                    if "Piccolo" in out.columns:
                        v = pd.to_numeric(out["Piccolo"], errors="coerce")
                        piccolo_flat = bool(np.nanstd(v) < piccolo_flat_threshold)

                    missing = _missing_required(out)
                    note = ""
                    if missing:
                        note = f"missing required columns: {', '.join(missing)}"

                    csv_path = out_dir / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', sheet)}.csv"
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
                    continue

        # Fallback: minimal CSV
        out = pd.DataFrame({"Sample": range(1, len(raw)+1)})
        out["Workbook"] = xlsx_path.name; out["Sheet"] = sheet; out["Replicate"] = ""
        csv_path = out_dir / f"{xlsx_path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+','_', sheet)}.csv"
        out.to_csv(csv_path, index=False)
        results.append(SheetParseResult(sheet=sheet, replicate=None, csv_path=str(csv_path),
                                        n_rows=int(len(out)), n_cols=int(out.shape[1]),
                                        piccolo_flat=False, mode="fallback", notes="no recognizable headers"))

    summary = {"workbook": xlsx_path.name, "sheets": [asdict(r) for r in results]}
    with open(out_dir / f"{xlsx_path.stem}__parse_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
