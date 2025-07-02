# Legacy Modules

This directory contains deprecated modules that have been replaced by newer, more flexible implementations.

## Replaced Modules

### `duckdb_data_manager.py`
- **Replaced by:** `../sqlalchemy_data_manager.py`
- **Reason:** Limited to DuckDB only, new implementation supports multiple database backends
- **Migration:** Use `SQLAlchemyDataManager` instead of `DuckDBDataManager`

## Migration Guide

When migrating from legacy modules:

1. Replace imports:
   ```python
   # Old
   from src.duckdb_data_manager import DuckDBDataManager
   
   # New
   from src.sqlalchemy_data_manager import SQLAlchemyDataManager
   ```

2. Update configuration to use new database abstraction layer:
   ```yaml
   database:
     type: duckdb  # or sqlite, postgresql
     # ... other config
   ```

3. Use database-agnostic methods instead of DuckDB-specific ones

## Removal Timeline

These modules will be removed in a future version once all migration is complete.