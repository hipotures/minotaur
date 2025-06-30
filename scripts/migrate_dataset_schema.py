#!/usr/bin/env python3
"""
Dataset Schema Migration Script

Migrates the existing dataset schema to support the new centralized
dataset management system with DuckDB import functionality.

This script adds new columns to the datasets table to support:
- File record/column counts
- File format detection
- ID column specification
- Validation file support
"""

import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import duckdb
    DB_TYPE = 'duckdb'
except ImportError:
    import duckdb
    DB_TYPE = 'duckdb'
    print("Warning: DuckDB not available, falling back to SQLite")

logger = logging.getLogger(__name__)

class DatasetSchemaMigrator:
    """Handles migration of dataset schema to new format."""
    
    def __init__(self, db_path: str, backup_path: str = None):
        """Initialize migrator with database path."""
        self.db_path = Path(db_path)
        self.backup_path = Path(backup_path) if backup_path else self.db_path.with_suffix('.backup.duckdb')
        self.db_type = DB_TYPE
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _connect(self):
        """Create database connection."""
        if self.db_type == 'duckdb':
            return duckdb.connect(str(self.db_path))
        else:
            return duckdb.connect(str(self.db_path))
    
    def create_backup(self) -> bool:
        """Create backup of existing database."""
        try:
            if not self.db_path.exists():
                logger.warning(f"Database {self.db_path} does not exist, skipping backup")
                return True
                
            logger.info(f"Creating backup: {self.db_path} -> {self.backup_path}")
            shutil.copy2(self.db_path, self.backup_path)
            logger.info("Backup created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def check_schema_version(self) -> Dict[str, bool]:
        """Check which columns exist in datasets table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                
                if self.db_type == 'duckdb':
                    cursor.execute("DESCRIBE datasets")
                    columns = [row[0] for row in cursor.fetchall()]
                else:
                    cursor.execute("PRAGMA table_info(datasets)")
                    columns = [row[1] for row in cursor.fetchall()]
                
                # Check for new columns
                new_columns = {
                    'id_column': 'id_column' in columns,
                    'train_records': 'train_records' in columns,
                    'train_columns': 'train_columns' in columns,
                    'test_records': 'test_records' in columns,
                    'test_columns': 'test_columns' in columns,
                    'submission_records': 'submission_records' in columns,
                    'submission_columns': 'submission_columns' in columns,
                    'validation_records': 'validation_records' in columns,
                    'validation_columns': 'validation_columns' in columns,
                    'train_format': 'train_format' in columns,
                    'test_format': 'test_format' in columns,
                    'submission_format': 'submission_format' in columns,
                    'validation_format': 'validation_format' in columns
                }
                
                return new_columns
                
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            return {}
    
    def migrate_schema(self) -> bool:
        """Apply schema migration to add new columns."""
        try:
            schema_status = self.check_schema_version()
            missing_columns = [col for col, exists in schema_status.items() if not exists]
            
            if not missing_columns:
                logger.info("Schema is already up to date")
                return True
            
            logger.info(f"Found {len(missing_columns)} missing columns: {missing_columns}")
            
            with self._connect() as conn:
                cursor = conn.cursor()
                
                # Column definitions
                if self.db_type == 'duckdb':
                    column_types = {
                        'id_column': 'VARCHAR',
                        'train_records': 'INTEGER',
                        'train_columns': 'INTEGER',
                        'test_records': 'INTEGER',
                        'test_columns': 'INTEGER',
                        'submission_records': 'INTEGER',
                        'submission_columns': 'INTEGER',
                        'validation_records': 'INTEGER',
                        'validation_columns': 'INTEGER',
                        'train_format': 'VARCHAR',
                        'test_format': 'VARCHAR',
                        'submission_format': 'VARCHAR',
                        'validation_format': 'VARCHAR'
                    }
                else:
                    column_types = {
                        'id_column': 'TEXT',
                        'train_records': 'INTEGER',
                        'train_columns': 'INTEGER',
                        'test_records': 'INTEGER',
                        'test_columns': 'INTEGER',
                        'submission_records': 'INTEGER',
                        'submission_columns': 'INTEGER',
                        'validation_records': 'INTEGER',
                        'validation_columns': 'INTEGER',
                        'train_format': 'TEXT',
                        'test_format': 'TEXT',
                        'submission_format': 'TEXT',
                        'validation_format': 'TEXT'
                    }
                
                # Add missing columns
                for col_name in missing_columns:
                    col_type = column_types[col_name]
                    logger.info(f"Adding column: {col_name} {col_type}")
                    cursor.execute(f"ALTER TABLE datasets ADD COLUMN {col_name} {col_type}")
                
                # Ensure target_column is not NULL
                cursor.execute("UPDATE datasets SET target_column = 'unknown' WHERE target_column IS NULL")
                
                if self.db_type == 'duckdb':
                    conn.commit()
                
                logger.info("Schema migration completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            return False
    
    def validate_migration(self) -> bool:
        """Validate that migration was successful."""
        try:
            schema_status = self.check_schema_version()
            missing_columns = [col for col, exists in schema_status.items() if not exists]
            
            if missing_columns:
                logger.error(f"Migration validation failed: missing columns {missing_columns}")
                return False
            
            # Check that existing datasets have required fields
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM datasets WHERE target_column IS NULL")
                null_targets = cursor.fetchone()[0]
                
                if null_targets > 0:
                    logger.error(f"Found {null_targets} datasets with NULL target_column")
                    return False
            
            logger.info("Migration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def rollback_migration(self) -> bool:
        """Rollback to backup if migration failed."""
        try:
            if not self.backup_path.exists():
                logger.error("No backup file found for rollback")
                return False
            
            logger.info(f"Rolling back: {self.backup_path} -> {self.db_path}")
            shutil.copy2(self.backup_path, self.db_path)
            logger.info("Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate dataset schema")
    parser.add_argument("--db-path", default="feature_discovery.db", 
                        help="Path to database file")
    parser.add_argument("--backup-path", help="Path for backup file")
    parser.add_argument("--force", action="store_true", 
                        help="Force migration even if backup fails")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate current schema, don't migrate")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback to backup")
    
    args = parser.parse_args()
    
    migrator = DatasetSchemaMigrator(args.db_path, args.backup_path)
    
    if args.rollback:
        success = migrator.rollback_migration()
        sys.exit(0 if success else 1)
    
    if args.validate_only:
        schema_status = migrator.check_schema_version()
        missing = [col for col, exists in schema_status.items() if not exists]
        
        if missing:
            print(f"❌ Schema validation failed. Missing columns: {missing}")
            sys.exit(1)
        else:
            print("✅ Schema is up to date")
            sys.exit(0)
    
    # Create backup
    if not migrator.create_backup() and not args.force:
        print("❌ Backup failed. Use --force to proceed anyway.")
        sys.exit(1)
    
    # Run migration
    if not migrator.migrate_schema():
        print("❌ Migration failed")
        if migrator.backup_path.exists():
            print("💾 Attempting rollback...")
            migrator.rollback_migration()
        sys.exit(1)
    
    # Validate migration
    if not migrator.validate_migration():
        print("❌ Migration validation failed")
        print("💾 Attempting rollback...")
        migrator.rollback_migration()
        sys.exit(1)
    
    print("✅ Dataset schema migration completed successfully")
    print(f"💾 Backup saved to: {migrator.backup_path}")


if __name__ == "__main__":
    main()