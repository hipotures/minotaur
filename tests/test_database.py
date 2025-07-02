"""
Tests for database abstraction layer
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.database.engine_factory import DatabaseFactory
from src.database.config import DatabaseConfig
from src.database.base_manager import DatabaseManager
from src.database.migrations.migration_tool import DatabaseMigrator


class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_get_engine_duckdb(self):
        """Test DuckDB engine creation"""
        config = {
            'database': ':memory:',
            'engine_args': {'echo': False}
        }
        engine = DatabaseConfig.get_engine('duckdb', config)
        assert engine is not None
        assert 'duckdb' in str(engine.url)
    
    def test_get_engine_sqlite(self):
        """Test SQLite engine creation"""
        config = {
            'database': ':memory:',
            'engine_args': {'echo': False}
        }
        engine = DatabaseConfig.get_engine('sqlite', config)
        assert engine is not None
        assert 'sqlite' in str(engine.url)
    
    def test_get_engine_invalid_type(self):
        """Test invalid database type"""
        with pytest.raises(ValueError):
            DatabaseConfig.get_engine('invalid_db', {})
    
    def test_get_default_config(self):
        """Test default configuration generation"""
        config = DatabaseConfig.get_default_config('duckdb')
        assert config['db_type'] == 'duckdb'
        assert 'connection_params' in config


class TestDatabaseManager:
    """Test database manager functionality"""
    
    @pytest.fixture
    def duckdb_manager(self):
        """Create DuckDB manager for testing"""
        config = DatabaseConfig.get_default_config('duckdb')
        return DatabaseFactory.create_manager(**config)
    
    @pytest.fixture
    def sqlite_manager(self):
        """Create SQLite manager for testing"""
        config = DatabaseConfig.get_default_config('sqlite')
        return DatabaseFactory.create_manager(**config)
    
    def test_create_table_from_dataframe(self, duckdb_manager):
        """Test creating table from DataFrame"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        duckdb_manager.create_table_from_df(df, 'test_table')
        assert duckdb_manager.table_exists('test_table')
        assert 'test_table' in duckdb_manager.get_table_names()
    
    def test_bulk_insert_and_query(self, duckdb_manager):
        """Test bulk insert and querying"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Insert data
        duckdb_manager.bulk_insert_from_pandas(df, 'test_data', if_exists='replace')
        
        # Query data
        results = duckdb_manager.execute_query("SELECT * FROM test_data WHERE category = 'A'")
        assert len(results) == 3
        
        # Query as DataFrame
        df_result = duckdb_manager.execute_query_df("SELECT * FROM test_data ORDER BY id")
        assert len(df_result) == 5
        assert list(df_result['id']) == [1, 2, 3, 4, 5]
    
    def test_insert_update_delete(self, duckdb_manager):
        """Test insert, update, and delete operations"""
        # Create table
        df = pd.DataFrame({'id': [1], 'name': ['Test'], 'value': [100]})
        duckdb_manager.bulk_insert_from_pandas(df, 'crud_test', if_exists='replace')
        
        # Insert new data
        new_data = [{'id': 2, 'name': 'Test2', 'value': 200}]
        rows_inserted = duckdb_manager.insert_data('crud_test', new_data)
        # DuckDB may return -1 for rowcount, which is normal behavior
        assert rows_inserted == 1 or rows_inserted == -1
        
        # Update data
        rows_updated = duckdb_manager.update_data(
            'crud_test', 
            {'value': 150}, 
            {'id': 1}
        )
        # DuckDB may return -1 for rowcount, which is normal behavior
        assert rows_updated == 1 or rows_updated == -1
        
        # Verify update
        result = duckdb_manager.execute_query("SELECT value FROM crud_test WHERE id = 1")
        assert result[0]['value'] == 150
        
        # Delete data
        rows_deleted = duckdb_manager.delete_data('crud_test', {'id': 2})
        # DuckDB may return -1 for rowcount, which is normal behavior
        assert rows_deleted == 1 or rows_deleted == -1
        
        # Verify deletion
        result = duckdb_manager.execute_query("SELECT COUNT(*) as count FROM crud_test")
        assert result[0]['count'] == 1
    
    def test_count_rows(self, duckdb_manager):
        """Test row counting functionality"""
        df = pd.DataFrame({
            'id': range(1, 11),
            'category': ['A'] * 5 + ['B'] * 5
        })
        duckdb_manager.bulk_insert_from_pandas(df, 'count_test', if_exists='replace')
        
        # Count all rows
        total_count = duckdb_manager.count_rows('count_test')
        assert total_count == 10
        
        # Count with condition
        a_count = duckdb_manager.count_rows('count_test', "category = 'A'")
        assert a_count == 5


class TestDatabaseFactory:
    """Test database factory functionality"""
    
    @pytest.mark.parametrize("db_type", ["duckdb", "sqlite"])
    def test_create_manager(self, db_type):
        """Test creating managers for different database types"""
        config = DatabaseConfig.get_default_config(db_type)
        manager = DatabaseFactory.create_manager(**config)
        
        assert isinstance(manager, DatabaseManager)
        assert manager.db_type == db_type
        
        # Test basic functionality
        assert manager.get_table_names() == []
    
    def test_database_specific_methods(self):
        """Test that database-specific methods are added"""
        # Test DuckDB specific methods
        duckdb_config = DatabaseConfig.get_default_config('duckdb')
        duckdb_manager = DatabaseFactory.create_manager(**duckdb_config)
        
        assert hasattr(duckdb_manager, 'read_parquet')
        assert hasattr(duckdb_manager, 'sample_reservoir')
        assert hasattr(duckdb_manager, 'set_memory_limit')
        
        # Test SQLite specific methods
        sqlite_config = DatabaseConfig.get_default_config('sqlite')
        sqlite_manager = DatabaseFactory.create_manager(**sqlite_config)
        
        assert hasattr(sqlite_manager, 'vacuum_db')
        assert hasattr(sqlite_manager, 'pragma_query')
        assert hasattr(sqlite_manager, 'sample_random')
    
    def test_get_supported_databases(self):
        """Test getting supported database list"""
        supported = DatabaseFactory.get_supported_databases()
        assert 'duckdb' in supported
        assert 'sqlite' in supported
        assert 'postgresql' in supported


class TestDatabaseMigrator:
    """Test database migration functionality"""
    
    @pytest.fixture
    def source_manager(self):
        """Create source database with test data"""
        config = DatabaseConfig.get_default_config('sqlite')
        manager = DatabaseFactory.create_manager(**config)
        
        # Create test data
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
        manager.bulk_insert_from_pandas(df, 'employees', if_exists='replace')
        
        return manager
    
    @pytest.fixture
    def target_manager(self):
        """Create target database"""
        config = DatabaseConfig.get_default_config('duckdb')
        return DatabaseFactory.create_manager(**config)
    
    def test_migrate_table(self, source_manager, target_manager):
        """Test migrating a single table"""
        migrator = DatabaseMigrator(source_manager.engine, target_manager.engine)
        
        # Migrate table
        stats = migrator.migrate_table('employees', batch_size=3)
        
        assert stats['success'] is True
        assert stats['source_rows'] == 5
        assert stats['migrated_rows'] == 5
        assert stats['batches_processed'] == 2  # 3 + 2 rows
        
        # Verify data in target
        result = target_manager.execute_query("SELECT COUNT(*) as count FROM employees")
        assert result[0]['count'] == 5
    
    def test_migrate_all_tables(self, source_manager, target_manager):
        """Test migrating all tables"""
        # Add another table to source
        df2 = pd.DataFrame({
            'id': [1, 2],
            'department': ['Engineering', 'Marketing']
        })
        source_manager.bulk_insert_from_pandas(df2, 'departments', if_exists='replace')
        
        migrator = DatabaseMigrator(source_manager.engine, target_manager.engine)
        
        # Migrate all tables
        stats = migrator.migrate_all_tables(batch_size=10)
        
        assert stats['total_tables'] == 2
        assert stats['successful_tables'] == 2
        assert stats['failed_tables'] == 0
        assert stats['total_migrated_rows'] == 7  # 5 employees + 2 departments
        
        # Verify tables exist in target
        target_tables = target_manager.get_table_names()
        assert 'employees' in target_tables
        assert 'departments' in target_tables
    
    def test_dry_run_migration(self, source_manager, target_manager):
        """Test dry run migration"""
        migrator = DatabaseMigrator(source_manager.engine, target_manager.engine)
        
        # Dry run
        stats = migrator.migrate_table('employees', dry_run=True)
        
        assert stats['success'] is True
        assert stats['source_rows'] == 5
        assert stats['migrated_rows'] == 0  # No actual migration
        
        # Verify no table created in target
        assert not target_manager.table_exists('employees')
    
    def test_compare_schemas(self, source_manager, target_manager):
        """Test schema comparison"""
        # Create table in target with different schema
        df_target = pd.DataFrame({
            'id': [1],
            'name': ['Test'],
            'age': [25],
            'city': ['New York']  # Extra column
        })
        target_manager.bulk_insert_from_pandas(df_target, 'employees', if_exists='replace')
        
        migrator = DatabaseMigrator(source_manager.engine, target_manager.engine)
        comparison = migrator.compare_schemas('employees')
        
        assert comparison['source_exists'] is True
        assert comparison['target_exists'] is True
        assert 'salary' in comparison['missing_in_target']
        assert 'city' in comparison['missing_in_source']


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for the database system"""
    
    def test_full_workflow(self):
        """Test complete database workflow"""
        # Create managers
        config = DatabaseConfig.get_default_config('duckdb')
        manager = DatabaseFactory.create_manager(**config)
        
        # Create and populate table
        df = pd.DataFrame({
            'id': range(1, 101),
            'value': range(100, 200),  # Fixed range to match length
            'category': ['A', 'B'] * 50
        })
        
        manager.bulk_insert_from_pandas(df, 'test_data', if_exists='replace')
        
        # Test various operations
        assert manager.count_rows('test_data') == 100
        assert manager.count_rows('test_data', "category = 'A'") == 50
        
        # Test sampling (DuckDB specific)
        if hasattr(manager, 'sample_reservoir'):
            sample_query = manager.sample_reservoir('test_data', 10)
            sample_df = manager.execute_query_df(sample_query)
            assert len(sample_df) <= 10
        
        # Test aggregation
        agg_result = manager.execute_query_df("""
            SELECT category, COUNT(*) as count, AVG(value) as avg_value
            FROM test_data 
            GROUP BY category
        """)
        assert len(agg_result) == 2
        
        # Cleanup
        manager.drop_table('test_data')
        assert not manager.table_exists('test_data')