"""
Database migration runner.

This module provides version-controlled database schema management with
migration execution, rollback support, and version tracking.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from ..core.connection import DuckDBConnectionManager


class Migration:
    """
    Represents a single database migration.
    
    A migration consists of SQL statements to apply changes (up)
    and optionally SQL statements to rollback changes (down).
    """
    
    def __init__(self, version: int, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.applied_at: Optional[datetime] = None
    
    def __str__(self) -> str:
        return f"Migration {self.version:03d}: {self.name}"
    
    def __repr__(self) -> str:
        return f"Migration(version={self.version}, name='{self.name}')"


class MigrationRunner:
    """
    Database migration runner with version tracking.
    
    Manages the execution of database migrations in a controlled,
    versioned manner with rollback support.
    """
    
    def __init__(self, connection_manager: DuckDBConnectionManager):
        """
        Initialize migration runner.
        
        Args:
            connection_manager: DuckDB connection manager instance
        """
        self.conn_manager = connection_manager
        self.logger = logging.getLogger(f'db.{self.__class__.__name__.lower()}')
        
        self.logger.debug(f"🔧 Initializing MigrationRunner")
        
        # Migration directories
        self.migrations_dir = Path(__file__).parent
        self.migration_files_dir = self.migrations_dir
        
        self.logger.debug(f"Migration directory: {self.migrations_dir}")
        
        # Initialize migration tracking table
        self._init_migration_table()
        self.logger.debug("✅ MigrationRunner initialized successfully")
    
    def _init_migration_table(self) -> None:
        """Initialize the migration tracking table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR,
            execution_time_ms INTEGER
        )
        """
        
        self.conn_manager.execute_query(create_table_sql, fetch='none')
        self.logger.debug("Migration tracking table initialized")
    
    def discover_migrations(self) -> List[Migration]:
        """
        Discover migration files in the migrations directory.
        
        Returns:
            List of Migration objects sorted by version
        """
        migrations = []
        
        # Pattern to match migration files: NNN_name.sql
        migration_pattern = re.compile(r'^(\d{3})_(.+)\.sql$')
        
        for file_path in self.migration_files_dir.glob('*.sql'):
            match = migration_pattern.match(file_path.name)
            if match:
                version = int(match.group(1))
                name = match.group(2).replace('_', ' ').title()
                
                # Read migration file content
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Split up and down migrations if present
                    if '-- DOWN MIGRATION' in content:
                        parts = content.split('-- DOWN MIGRATION')
                        up_sql = parts[0].strip()
                        down_sql = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        up_sql = content.strip()
                        down_sql = ""
                    
                    migration = Migration(version, name, up_sql, down_sql)
                    migrations.append(migration)
                    
                except Exception as e:
                    self.logger.error(f"Failed to read migration file {file_path}: {e}")
        
        # Sort by version
        migrations.sort(key=lambda m: m.version)
        
        self.logger.debug(f"Discovered {len(migrations)} migration files")
        return migrations
    
    def get_applied_migrations(self) -> List[Tuple[int, str, datetime]]:
        """
        Get list of applied migrations from the database.
        
        Returns:
            List of (version, name, applied_at) tuples
        """
        try:
            query = """
            SELECT version, name, applied_at 
            FROM schema_migrations 
            ORDER BY version
            """
            
            results = self.conn_manager.execute_query(query, fetch='all')
            
            applied = []
            for row in results:
                applied_at = row[2]
                if isinstance(applied_at, str):
                    try:
                        applied_at = datetime.fromisoformat(applied_at.replace('Z', '+00:00'))
                    except ValueError:
                        applied_at = datetime.now()
                
                applied.append((row[0], row[1], applied_at))
            
            return applied
        except Exception as e:
            # Migration table doesn't exist yet, return empty list
            self.logger.debug(f"Migration table not found: {e}")
            return []
    
    def get_current_version(self) -> int:
        """
        Get the current schema version.
        
        Returns:
            Current version number, 0 if no migrations applied
        """
        try:
            query = "SELECT MAX(version) FROM schema_migrations"
            result = self.conn_manager.execute_query(query, fetch='one')
            return result[0] if result and result[0] is not None else 0
        except Exception as e:
            # Migration table doesn't exist yet, return 0
            self.logger.debug(f"Migration table not found: {e}")
            return 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        Get migrations that haven't been applied yet.
        
        Returns:
            List of pending migrations
        """
        current_version = self.get_current_version()
        all_migrations = self.discover_migrations()
        
        pending = [m for m in all_migrations if m.version > current_version]
        
        self.logger.debug(f"Found {len(pending)} pending migrations")
        return pending
    
    def run_migrations(self, target_version: Optional[int] = None) -> List[Migration]:
        """
        Run all pending migrations up to target version.
        
        Args:
            target_version: Maximum version to migrate to (None = all pending)
            
        Returns:
            List of applied migrations
        """
        self.logger.debug(f"🔄 Getting pending migrations (target_version={target_version})")
        pending_migrations = self.get_pending_migrations()
        
        self.logger.debug(f"Found {len(pending_migrations)} pending migrations: {[str(m) for m in pending_migrations]}")
        
        if target_version is not None:
            pending_migrations = [m for m in pending_migrations if m.version <= target_version]
            self.logger.debug(f"Filtered to {len(pending_migrations)} migrations for target version {target_version}")
        
        if not pending_migrations:
            self.logger.info("No pending migrations to run")
            return []
        
        applied_migrations = []
        
        for migration in pending_migrations:
            try:
                self.logger.info(f"Applying migration: {migration}")
                self.logger.debug(f"Migration SQL preview: {migration.up_sql[:200]}...")
                self._apply_migration(migration)
                applied_migrations.append(migration)
                self.logger.debug(f"✅ Successfully applied migration {migration.version}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply migration {migration}: {e}")
                self.logger.debug(f"Migration failure details: {e}", exc_info=True)
                # Stop on first failure to maintain consistency
                break
        
        if applied_migrations:
            self.logger.info(f"Successfully applied {len(applied_migrations)} migrations")
        
        return applied_migrations
    
    def _apply_migration(self, migration: Migration) -> None:
        """
        Apply a single migration.
        
        Args:
            migration: Migration to apply
        """
        start_time = datetime.now()
        
        try:
            self.logger.debug(f"🔄 Executing migration {migration.version}: {migration.name}")
            # Execute migration SQL
            if migration.up_sql:
                self.logger.debug(f"Executing SQL for migration {migration.version}")
                self.conn_manager.execute_script(migration.up_sql)
                self.logger.debug(f"✅ SQL execution completed for migration {migration.version}")
            else:
                self.logger.warning(f"Migration {migration.version} has no SQL to execute")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.debug(f"Migration {migration.version} executed in {execution_time:.2f}ms")
            
            # Calculate checksum for integrity checking
            import hashlib
            checksum = hashlib.md5(migration.up_sql.encode()).hexdigest()
            
            # Record migration as applied
            record_sql = """
            INSERT INTO schema_migrations (version, name, applied_at, checksum, execution_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """
            
            self.conn_manager.execute_query(
                record_sql,
                (migration.version, migration.name, start_time.isoformat(), 
                 checksum, int(execution_time)),
                fetch='none'
            )
            
            migration.applied_at = start_time
            self.logger.info(f"Applied migration {migration.version} in {execution_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Migration {migration.version} failed: {e}")
            raise
    
    def rollback_migration(self, target_version: int) -> List[Migration]:
        """
        Rollback migrations down to target version.
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            List of rolled back migrations
        """
        current_version = self.get_current_version()
        
        if target_version >= current_version:
            self.logger.info("Target version is not lower than current version")
            return []
        
        # Get migrations to rollback (in reverse order)
        all_migrations = self.discover_migrations()
        migrations_to_rollback = [
            m for m in all_migrations 
            if target_version < m.version <= current_version
        ]
        migrations_to_rollback.reverse()  # Rollback in reverse order
        
        rolled_back = []
        
        for migration in migrations_to_rollback:
            try:
                self.logger.info(f"Rolling back migration: {migration}")
                self._rollback_migration(migration)
                rolled_back.append(migration)
                
            except Exception as e:
                self.logger.error(f"Failed to rollback migration {migration}: {e}")
                # Stop on first failure
                break
        
        if rolled_back:
            self.logger.info(f"Successfully rolled back {len(rolled_back)} migrations")
        
        return rolled_back
    
    def _rollback_migration(self, migration: Migration) -> None:
        """
        Rollback a single migration.
        
        Args:
            migration: Migration to rollback
        """
        if not migration.down_sql:
            raise ValueError(f"Migration {migration.version} has no rollback SQL")
        
        try:
            # Execute rollback SQL
            self.conn_manager.execute_script(migration.down_sql)
            
            # Remove migration record
            delete_sql = "DELETE FROM schema_migrations WHERE version = ?"
            self.conn_manager.execute_query(delete_sql, (migration.version,), fetch='none')
            
            self.logger.info(f"Rolled back migration {migration.version}")
            
        except Exception as e:
            self.logger.error(f"Rollback of migration {migration.version} failed: {e}")
            raise
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.
        
        Returns:
            Dictionary with migration status information
        """
        current_version = self.get_current_version()
        all_migrations = self.discover_migrations()
        applied_migrations = self.get_applied_migrations()
        pending_migrations = self.get_pending_migrations()
        
        latest_available = max([m.version for m in all_migrations]) if all_migrations else 0
        
        status = {
            'current_version': current_version,
            'latest_available_version': latest_available,
            'total_migrations': len(all_migrations),
            'applied_migrations': len(applied_migrations),
            'pending_migrations': len(pending_migrations),
            'is_up_to_date': current_version == latest_available,
            'applied_migration_list': [
                {'version': v, 'name': n, 'applied_at': a.isoformat()}
                for v, n, a in applied_migrations
            ],
            'pending_migration_list': [
                {'version': m.version, 'name': m.name}
                for m in pending_migrations
            ]
        }
        
        return status
    
    def validate_migrations(self) -> List[Dict[str, Any]]:
        """
        Validate applied migrations against current files.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        applied_migrations = self.get_applied_migrations()
        available_migrations = {m.version: m for m in self.discover_migrations()}
        
        for version, name, applied_at in applied_migrations:
            if version not in available_migrations:
                issues.append({
                    'type': 'missing_file',
                    'version': version,
                    'message': f"Migration {version} is applied but file is missing"
                })
            else:
                migration = available_migrations[version]
                if migration.name != name:
                    issues.append({
                        'type': 'name_mismatch',
                        'version': version,
                        'message': f"Migration {version} name mismatch: DB='{name}' File='{migration.name}'"
                    })
        
        # Check for gaps in version numbers
        if applied_migrations:
            applied_versions = sorted([v for v, _, _ in applied_migrations])
            for i in range(1, len(applied_versions)):
                if applied_versions[i] - applied_versions[i-1] > 1:
                    issues.append({
                        'type': 'version_gap',
                        'version': applied_versions[i-1],
                        'message': f"Version gap between {applied_versions[i-1]} and {applied_versions[i]}"
                    })
        
        if issues:
            self.logger.warning(f"Found {len(issues)} migration validation issues")
        else:
            self.logger.info("All migrations validated successfully")
        
        return issues
    
    def create_migration_file(self, name: str, up_sql: str, down_sql: str = "") -> Path:
        """
        Create a new migration file.
        
        Args:
            name: Migration name (will be sanitized)
            up_sql: SQL for applying the migration
            down_sql: SQL for rolling back the migration
            
        Returns:
            Path to created migration file
        """
        # Get next version number
        all_migrations = self.discover_migrations()
        next_version = max([m.version for m in all_migrations], default=0) + 1
        
        # Sanitize name
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')
        
        # Create filename
        filename = f"{next_version:03d}_{sanitized_name}.sql"
        file_path = self.migration_files_dir / filename
        
        # Create migration content
        content = f"""-- Migration {next_version:03d}: {name}
-- Created at: {datetime.now().isoformat()}

{up_sql}"""
        
        if down_sql:
            content += f"""

-- DOWN MIGRATION
{down_sql}"""
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        
        self.logger.info(f"Created migration file: {filename}")
        return file_path