# Formatters Module (TODO)

This module will contain output formatters for different formats.

## Planned Components:

### `base.py`
- `BaseFormatter` - Abstract base class for all formatters

### `text_formatter.py`
- `TextFormatter` - Human-readable text output
- Tables, summaries, lists

### `json_formatter.py`
- `JsonFormatter` - JSON output for programmatic use
- Consistent structure across all outputs

### `html_formatter.py`
- `HtmlFormatter` - Rich HTML reports
- Charts, tables, styling

### `csv_formatter.py`
- `CsvFormatter` - CSV export for data analysis
- Tabular data export

## Current Status
Currently, formatting logic is distributed:
- Basic formatting in `core/utils.py`
- Module-specific formatting in each module's base class
- This will be centralized here in future refactoring