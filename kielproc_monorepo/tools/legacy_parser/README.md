# Legacy Parser (standalone)

Run the parser without touching the main app.

## GUI
```
pip install -r tools/legacy_parser/requirements.txt
python tools/legacy_parser/parser_gui.py
```

Outputs: one CSV per sheet and a `<workbook>__parse_summary.json`.
Columns (when present): Sample, Time, Static, VP, Temperature, Piccolo, Port, Workbook, Sheet, Replicate.
