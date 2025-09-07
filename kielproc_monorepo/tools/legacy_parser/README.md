# Legacy Parser

The legacy parser is integrated into the `kielproc one-click` workflow.

## Installation

```bash
python -m pip install .
```

## Usage

```bash
kielproc one-click <workbook> --site <Preset> --baro-override <Pa>
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

`_translate/translated.csv` contains the parsed workbook with columns such as Sample, Time, Static, VP, Temperature, Piccolo, Port, Workbook, Sheet, and Replicate.
