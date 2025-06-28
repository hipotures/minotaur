# Validators Module (TODO)

This module will contain input validators and data verification.

## Planned Components:

### `base.py`
- `BaseValidator` - Abstract base class for validators

### `dataset_validator.py`
- Validate dataset paths and formats
- Check required columns exist
- Verify data types

### `session_validator.py`
- Validate session IDs format
- Check session exists in database
- Verify session status

### `date_validator.py`
- Parse and validate date inputs
- Handle different date formats
- Validate date ranges

### `config_validator.py`
- Validate configuration files
- Check required settings
- Verify value ranges

## Current Status
Currently validation is done inline in services and repositories.
This module will centralize validation logic for reuse and consistency.