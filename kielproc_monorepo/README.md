
# Kiel / Piccolo Processing Suite (Mono-repo)

Backend library for processing differential pressure data from Kiel probes and legacy piccolo tubes.

## Installation

```bash
python -m pip install .
```

## Usage

```bash
kielproc one-click <workbook> --site <Preset> --baro <Pa>
```

Running `kielproc one-click` executes the full pipeline:

```
Parse → Integrate → Map → Fit → Translate → Report
```

Artifacts are written to a `RUN_<STAMP>/` directory:

```
RUN_<STAMP>/
  run_context.json
  _integrated/...
  _fit/...
  _translate/translated.csv
  _report/{legacy_results.csv,json,setpoints.json}
  summary.json
```

## Notes
- Dynamics are processed in samples; seconds derived from sampling Hz.
- Throat unknown is fine; geometry is optional for scaling only.
