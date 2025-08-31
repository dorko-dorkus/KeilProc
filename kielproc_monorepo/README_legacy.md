# Duct ΔP Model Visualizer

A small, robust tool to extract and analyze duct static/velocity pressure data, replacing manual Excel handling.
Built for legacy **Excel** logs (`P1`, `P2`, … sheets) and future **CSV** exports from the data logger.

## Features
- Load **Excel** (auto-detect headers, per-port parsing) or **CSV** (pick SP/VP/Time columns).
- Per-port slicing (bottom/middle/top = 15–70–15 by default) with a minimum segment length.
- **Pearson r** with **Fisher-z 95% CI**.
- **Theil–Sen** slope (VP ~ SP) with **bootstrap 95% CI** (fast subsampled bootstrap).
- **Sign-flip probability** (bottom vs top) using bootstrapped r.
- Tidy **CSV** outputs and **PNG** plots; optional ZIP packaging.
- **GUI** for operators (PySimpleGUI) and **CLI** for headless runs.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Note: Reading legacy `.xls` may require `xlrd<2.0`. This tool targets `.xlsx` via `openpyxl`.

## Run — GUI
```bash
python duct_dp_visualizer.py
```

## Run — CLI
```bash
# Excel legacy workbook
python duct_dp_visualizer.py --cli --excel "1C Mill Airflow Test 63 TH FU 26-9-11 (2).xlsx" --out out_dir --zip

# CSV logger export (map columns)
python duct_dp_visualizer.py --cli --csv logger.csv --sp "Static_P" --vp "Velocity_P" --time "Time" --out out_dir --zip
```

## Outputs
- `<stem>_fisher_theilsen_slices.csv` — tidy table with per-slice r + CI, Theil–Sen + CI.
- `<stem>_signflip_prob.csv` — bottom vs top sign-flip probability.
- `<stem>_r_ci_strips.png` — errorbar strips of r with 95% CI.
- `<stem>_theilsen_ci_strips.png` — errorbar strips of Theil–Sen slope with 95% CI.
- `dp_viz_outputs.zip` (optional).

## Notes & Rationale
- Fisher-z CIs stabilize correlation variance (good small-sample behavior).
- Theil–Sen is robust to outliers typical of turbulent eddies and wall effects.
- Sign-flip probability across bootstraps makes the floor/roof split obvious.

## Extending
- Sliding-window r(t)/slope(t), partial correlations, ANCOVA, and α/β continuity mapping can be added in a later module.
- If your CSV schema changes, just adjust the SP/VP column mapping in the GUI.

## License
MIT