#!/usr/bin/env python3
"""
Database Cleanup Script

Clears all data from the MCTS discovery database to start fresh.
Removes all datasets, sessions, and related data.
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

class DatabaseCleaner:
    """Handles complete database cleanup."""
    
    def __init__(self, db_path: str, backup_path: str = None):
        """Initialize cleaner with database path."""
        self.db_path = Path(db_path)
        self.backup_path = Path(backup_path) if backup_path else self.db_path.with_suffix('.cleanup_backup.duckdb')
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
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables before cleanup."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                
                tables = ['datasets', 'sessions', 'exploration_history', 
                         'feature_catalog', 'feature_impact', 'operation_performance',
                         'system_performance']
                
                counts = {}
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        counts[table] = cursor.fetchone()[0]
                    except Exception:
                        counts[table] = 0  # Table doesn't exist or is empty
                
                return counts
                
        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
            return {}
    
    def cleanup_all_data(self) -> bool:
        """Remove all data from all tables."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                
                # Order matters due to foreign key constraints
                tables_to_clear = [
                    'system_performance',
                    'operation_performance', 
                    'feature_impact',
                    'feature_catalog',
                    'exploration_history',
                    'sessions',
                    'datasets'
                ]
                
                logger.info("Starting database cleanup...")
                
                for table in tables_to_clear:
                    try:
                        cursor.execute(f"DELETE FROM {table}")
                        logger.info(f"✅ Cleared table: {table}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear {table}: {e}")
                
                # Reset sequences for DuckDB
                if self.db_type == 'duckdb':
                    sequences = [
                        'exploration_history_id_seq',
                        'feature_catalog_id_seq', 
                        'feature_impact_id_seq',
                        'operation_performance_id_seq',
                        'system_performance_id_seq'
                    ]
                    
                    for seq in sequences:
                        try:
                            cursor.execute(f"ALTER SEQUENCE {seq} RESTART WITH 1")
                            logger.info(f"✅ Reset sequence: {seq}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not reset sequence {seq}: {e}")
                
                if self.db_type == 'duckdb':
                    conn.commit()
                
                logger.info("Database cleanup completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            return False
    
    def add_unique_constraints(self) -> bool:
        """Add UNIQUE constraint on dataset_name."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                
                # Check if constraint already exists
                if self.db_type == 'duckdb':
                    # DuckDB way to check constraints
                    try:
                        cursor.execute("ALTER TABLE datasets ADD CONSTRAINT unique_dataset_name UNIQUE (dataset_name)")
                        logger.info("✅ Added UNIQUE constraint on dataset_name")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "constraint" in str(e).lower():
                            logger.info("ℹ️ UNIQUE constraint on dataset_name already exists")
                        else:
                            raise e
                else:
                    # SQLite - create new table with constraint
                    cursor.execute("PRAGMA table_info(datasets)")
                    columns = cursor.fetchall()
                    
                    # Check if we can add unique constraint (no duplicate names exist)
                    cursor.execute("SELECT dataset_name, COUNT(*) FROM datasets GROUP BY dataset_name HAVING COUNT(*) > 1")
                    duplicates = cursor.fetchall()
                    
                    if duplicates:
                        logger.warning(f"Cannot add UNIQUE constraint - duplicates exist: {duplicates}")
                        return False
                    
                    # For SQLite, we would need to recreate the table, but since we just cleaned it,
                    # we can add a UNIQUE index instead
                    try:
                        cursor.execute("CREATE UNIQUE INDEX idx_unique_dataset_name ON datasets(dataset_name)")
                        logger.info("✅ Added UNIQUE index on dataset_name")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info("ℹ️ UNIQUE index on dataset_name already exists")
                        else:
                            raise e
                
                if self.db_type == 'duckdb':
                    conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to add unique constraints: {e}")
            return False
    
    def validate_cleanup(self) -> bool:
        """Validate that cleanup was successful."""
        try:
            counts = self.get_table_counts()
            
            # Check that main tables are empty
            main_tables = ['datasets', 'sessions', 'exploration_history']
            for table in main_tables:
                if counts.get(table, 0) > 0:
                    logger.error(f"Cleanup validation failed: {table} still has {counts[table]} records")
                    return False
            
            logger.info("✅ Cleanup validation passed - all main tables are empty")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup validation failed: {e}")
            return False


def main():
    """Main cleanup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean MCTS discovery database")
    parser.add_argument("--db-path", default="data/minotaur.duckdb", 
                        help="Path to database file")
    parser.add_argument("--backup-path", help="Path for backup file")
    parser.add_argument("--force", action="store_true", 
                        help="Force cleanup without confirmation")
    parser.add_argument("--skip-backup", action="store_true",
                        help="Skip creating backup")
    
    args = parser.parse_args()
    
    cleaner = DatabaseCleaner(args.db_path, args.backup_path)
    
    # Show current state
    print("📊 CURRENT DATABASE STATE")
    print("=" * 50)
    counts = cleaner.get_table_counts()
    total_records = sum(counts.values())
    
    for table, count in counts.items():
        if count > 0:
            print(f"   {table}: {count:,} records")
    
    print(f"\nTotal records: {total_records:,}")
    
    if total_records == 0:
        print("✅ Database is already clean")
        sys.exit(0)
    
    # Confirmation
    if not args.force:
        print(f"\n⚠️  WARNING: This will delete ALL data from the database!")
        print(f"   Database: {args.db_path}")
        print(f"   Backup will be saved to: {cleaner.backup_path}")
        
        response = input("\nDo you want to proceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup cancelled")
            sys.exit(0)
    
    # Create backup
    if not args.skip_backup:
        if not cleaner.create_backup():
            print("❌ Backup failed. Use --skip-backup to proceed anyway.")
            sys.exit(1)
    
    # Perform cleanup
    if not cleaner.cleanup_all_data():
        print("❌ Database cleanup failed")
        sys.exit(1)
    
    # Add unique constraints
    if not cleaner.add_unique_constraints():
        print("❌ Failed to add unique constraints")
        sys.exit(1)
    
    # Validate cleanup
    if not cleaner.validate_cleanup():
        print("❌ Cleanup validation failed")
        sys.exit(1)
    
    print("\n✅ Database cleanup completed successfully!")
    print(f"💾 Backup saved to: {cleaner.backup_path}")
    print("🚀 Ready for fresh dataset registration")


if __name__ == "__main__":
    main()