"""
Database migration tool for moving data between different database engines
"""

from sqlalchemy import inspect, MetaData, Table, create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Tool for migrating data between different database engines"""
    
    def __init__(self, source_engine: Engine, target_engine: Engine):
        """
        Initialize migrator with source and target engines
        
        Args:
            source_engine: Source database engine
            target_engine: Target database engine
        """
        self.source_engine = source_engine
        self.target_engine = target_engine
        self.source_inspector = inspect(source_engine)
        self.target_inspector = inspect(target_engine)
        
        # Metadata objects for reflection
        self.source_metadata = MetaData()
        self.target_metadata = MetaData()
    
    def get_source_tables(self) -> List[str]:
        """Get list of tables in source database"""
        return self.source_inspector.get_table_names()
    
    def get_target_tables(self) -> List[str]:
        """Get list of tables in target database"""
        return self.target_inspector.get_table_names()
    
    def table_exists_in_target(self, table_name: str) -> bool:
        """Check if table exists in target database"""
        return table_name in self.get_target_tables()
    
    def migrate_table(self, table_name: str, batch_size: int = 10000, 
                     if_exists: str = 'replace', dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate a single table from source to target database
        
        Args:
            table_name: Name of table to migrate
            batch_size: Number of rows to process at once
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            dry_run: If True, only analyze without migrating
            
        Returns:
            Migration statistics
        """
        logger.info(f"{'Analyzing' if dry_run else 'Migrating'} table: {table_name}")
        
        stats = {
            'table_name': table_name,
            'source_rows': 0,
            'migrated_rows': 0,
            'batches_processed': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Get source table structure
            source_table = Table(table_name, self.source_metadata, autoload_with=self.source_engine)
            
            # Count total rows in source
            with self.source_engine.connect() as conn:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                stats['source_rows'] = count_result.scalar()
            
            if dry_run:
                logger.info(f"Table {table_name}: {stats['source_rows']} rows, {len(source_table.columns)} columns")
                stats['success'] = True
                return stats
            
            # Create table in target database if needed
            if if_exists == 'replace' or not self.table_exists_in_target(table_name):
                # Create table structure in target
                source_table.create(self.target_engine, checkfirst=True)
                logger.info(f"Created table structure for {table_name}")
            
            # Migrate data in batches
            offset = 0
            total_migrated = 0
            
            while True:
                # Read batch from source
                query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                
                with self.source_engine.connect() as source_conn:
                    df = pd.read_sql(query, source_conn)
                
                if df.empty:
                    break
                
                # Write batch to target
                batch_if_exists = 'append' if offset > 0 or if_exists == 'append' else if_exists
                df.to_sql(table_name, self.target_engine, if_exists=batch_if_exists, index=False)
                
                total_migrated += len(df)
                stats['batches_processed'] += 1
                offset += batch_size
                
                logger.debug(f"Migrated batch {stats['batches_processed']}: {len(df)} rows "
                           f"(total: {total_migrated}/{stats['source_rows']})")
                
                # Break if we processed fewer rows than batch_size (last batch)
                if len(df) < batch_size:
                    break
            
            stats['migrated_rows'] = total_migrated
            stats['success'] = True
            
            logger.info(f"Successfully migrated {table_name}: {total_migrated} rows in {stats['batches_processed']} batches")
            
        except Exception as e:
            stats['error'] = str(e)
            logger.error(f"Failed to migrate table {table_name}: {e}")
        
        return stats
    
    def migrate_all_tables(self, batch_size: int = 10000, if_exists: str = 'replace',
                          exclude_tables: Optional[List[str]] = None,
                          include_tables: Optional[List[str]] = None,
                          dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate all tables from source to target database
        
        Args:
            batch_size: Number of rows to process at once
            if_exists: What to do if table exists
            exclude_tables: Tables to exclude from migration
            include_tables: Only migrate these tables (if specified)
            dry_run: If True, only analyze without migrating
            
        Returns:
            Overall migration statistics
        """
        source_tables = self.get_source_tables()
        exclude_tables = exclude_tables or []
        
        # Filter tables based on include/exclude lists
        if include_tables:
            tables_to_migrate = [t for t in source_tables if t in include_tables]
        else:
            tables_to_migrate = [t for t in source_tables if t not in exclude_tables]
        
        logger.info(f"{'Analyzing' if dry_run else 'Migrating'} {len(tables_to_migrate)} tables")
        
        overall_stats = {
            'total_tables': len(tables_to_migrate),
            'successful_tables': 0,
            'failed_tables': 0,
            'total_source_rows': 0,
            'total_migrated_rows': 0,
            'table_results': []
        }
        
        for table in tables_to_migrate:
            table_stats = self.migrate_table(table, batch_size, if_exists, dry_run)
            overall_stats['table_results'].append(table_stats)
            overall_stats['total_source_rows'] += table_stats['source_rows']
            
            if table_stats['success']:
                overall_stats['successful_tables'] += 1
                overall_stats['total_migrated_rows'] += table_stats['migrated_rows']
            else:
                overall_stats['failed_tables'] += 1
        
        action = "Analysis" if dry_run else "Migration"
        logger.info(f"{action} completed: {overall_stats['successful_tables']} successful, "
                   f"{overall_stats['failed_tables']} failed")
        
        return overall_stats
    
    def compare_schemas(self, table_name: str) -> Dict[str, Any]:
        """
        Compare schema between source and target for a table
        
        Args:
            table_name: Name of table to compare
            
        Returns:
            Schema comparison results
        """
        comparison = {
            'table_name': table_name,
            'source_exists': False,
            'target_exists': False,
            'source_columns': [],
            'target_columns': [],
            'missing_in_target': [],
            'missing_in_source': [],
            'type_differences': []
        }
        
        # Check source
        if table_name in self.get_source_tables():
            comparison['source_exists'] = True
            source_columns = self.source_inspector.get_columns(table_name)
            comparison['source_columns'] = [(col['name'], str(col['type'])) for col in source_columns]
        
        # Check target
        if table_name in self.get_target_tables():
            comparison['target_exists'] = True
            target_columns = self.target_inspector.get_columns(table_name)
            comparison['target_columns'] = [(col['name'], str(col['type'])) for col in target_columns]
        
        # Compare columns if both exist
        if comparison['source_exists'] and comparison['target_exists']:
            source_cols = {name: type_str for name, type_str in comparison['source_columns']}
            target_cols = {name: type_str for name, type_str in comparison['target_columns']}
            
            comparison['missing_in_target'] = [col for col in source_cols if col not in target_cols]
            comparison['missing_in_source'] = [col for col in target_cols if col not in source_cols]
            
            # Check type differences
            for col_name in set(source_cols.keys()) & set(target_cols.keys()):
                if source_cols[col_name] != target_cols[col_name]:
                    comparison['type_differences'].append({
                        'column': col_name,
                        'source_type': source_cols[col_name],
                        'target_type': target_cols[col_name]
                    })
        
        return comparison
    
    def create_migration_script(self, output_file: str, 
                               include_tables: Optional[List[str]] = None) -> None:
        """
        Create SQL migration script
        
        Args:
            output_file: Path to output SQL file
            include_tables: Tables to include (all if None)
        """
        source_tables = self.get_source_tables()
        if include_tables:
            source_tables = [t for t in source_tables if t in include_tables]
        
        script_lines = [
            "-- Database Migration Script",
            f"-- Generated for migration from {self.source_engine.url} to {self.target_engine.url}",
            "",
        ]
        
        for table_name in source_tables:
            try:
                # Get table structure
                source_table = Table(table_name, self.source_metadata, autoload_with=self.source_engine)
                
                # Generate CREATE TABLE statement
                create_sql = str(source_table.compile(self.target_engine)).replace('\n', ' ')
                script_lines.append(f"-- Table: {table_name}")
                script_lines.append(f"{create_sql};")
                script_lines.append("")
                
            except Exception as e:
                script_lines.append(f"-- ERROR creating table {table_name}: {e}")
                script_lines.append("")
        
        # Write script to file
        Path(output_file).write_text('\n'.join(script_lines))
        logger.info(f"Migration script written to {output_file}")
    
    def close(self) -> None:
        """Close database connections"""
        if hasattr(self.source_engine, 'dispose'):
            self.source_engine.dispose()
        if hasattr(self.target_engine, 'dispose'):
            self.target_engine.dispose()


def create_migrator_from_urls(source_url: str, target_url: str) -> DatabaseMigrator:
    """
    Create migrator from database URLs
    
    Args:
        source_url: Source database URL
        target_url: Target database URL
        
    Returns:
        DatabaseMigrator instance
    """
    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)
    return DatabaseMigrator(source_engine, target_engine)