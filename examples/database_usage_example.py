"""
Example usage of the database abstraction layer
"""

import pandas as pd
import logging
from pathlib import Path

# Import the database components
from src.database.engine_factory import DatabaseFactory
from src.database.config import DatabaseConfig
from src.database.migrations.migration_tool import DatabaseMigrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic database operations example"""
    print("=== Basic Database Usage ===")
    
    # Configuration - easy to switch between databases
    config = {
        'db_type': 'duckdb',  # Change to 'sqlite' or 'postgresql'
        'connection_params': {
            'database': ':memory:'  # Use in-memory database for example
        }
    }
    
    # Create database manager
    db = DatabaseFactory.create_manager(**config)
    
    # Create sample data
    df = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': [20 + (i % 50) for i in range(1, 101)],
        'salary': [30000 + (i * 1000) for i in range(1, 101)],
        'department': ['Engineering', 'Marketing', 'Sales'] * 34  # Cycle through departments
    })
    
    print(f"Created sample data with {len(df)} rows")
    
    # Load data into database
    db.bulk_insert_from_pandas(df, 'employees', if_exists='replace')
    print(f"Loaded data into {config['db_type']} database")
    
    # Basic queries
    print("\n--- Basic Queries ---")
    
    # Count rows
    total_employees = db.count_rows('employees')
    print(f"Total employees: {total_employees}")
    
    # Query with conditions
    results = db.execute_query("SELECT * FROM employees WHERE age > :age", {'age': 40})
    print(f"Employees over 40: {len(results)}")
    
    # Aggregate query
    dept_stats = db.execute_query_df("""
        SELECT department, 
               COUNT(*) as employee_count,
               AVG(salary) as avg_salary,
               MAX(salary) as max_salary
        FROM employees 
        GROUP BY department
        ORDER BY avg_salary DESC
    """)
    print(f"\nDepartment statistics:")
    print(dept_stats)
    
    # Database-specific features
    print("\n--- Database-Specific Features ---")
    if hasattr(db, 'sample_reservoir'):  # DuckDB
        print("Using DuckDB-specific reservoir sampling")
        sample_query = db.sample_reservoir('employees', 10)
        sample_df = db.execute_query_df(sample_query)
        print(f"Random sample of 10 employees: {len(sample_df)} rows")
    
    elif hasattr(db, 'sample_random'):  # SQLite
        print("Using SQLite-specific random sampling")
        sample_query = db.sample_random('employees', 10)
        sample_df = db.execute_query_df(sample_query)
        print(f"Random sample of 10 employees: {len(sample_df)} rows")
    
    # Clean up
    db.close()
    print(f"\nClosed {config['db_type']} database connection")


def example_database_migration():
    """Database migration example"""
    print("\n=== Database Migration Example ===")
    
    # Create source database (SQLite)
    source_config = DatabaseConfig.get_default_config('sqlite')
    source_db = DatabaseFactory.create_manager(**source_config)
    
    # Create target database (DuckDB)
    target_config = DatabaseConfig.get_default_config('duckdb')
    target_db = DatabaseFactory.create_manager(**target_config)
    
    # Create test data in source
    employees_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing'],
        'salary': [75000, 65000, 80000, 70000, 72000]
    })
    
    departments_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Engineering', 'Marketing', 'Sales'],
        'budget': [500000, 300000, 400000]
    })
    
    source_db.bulk_insert_from_pandas(employees_df, 'employees', if_exists='replace')
    source_db.bulk_insert_from_pandas(departments_df, 'departments', if_exists='replace')
    
    print("Created source data in SQLite database")
    
    # Perform migration
    migrator = DatabaseMigrator(source_db.engine, target_db.engine)
    
    # Dry run first
    print("\n--- Dry Run Migration ---")
    dry_run_stats = migrator.migrate_all_tables(dry_run=True)
    print(f"Dry run completed: {dry_run_stats['total_tables']} tables analyzed")
    for table_result in dry_run_stats['table_results']:
        print(f"  {table_result['table_name']}: {table_result['source_rows']} rows")
    
    # Actual migration
    print("\n--- Actual Migration ---")
    migration_stats = migrator.migrate_all_tables(batch_size=3)
    print(f"Migration completed:")
    print(f"  Total tables: {migration_stats['total_tables']}")
    print(f"  Successful: {migration_stats['successful_tables']}")
    print(f"  Failed: {migration_stats['failed_tables']}")
    print(f"  Total rows migrated: {migration_stats['total_migrated_rows']}")
    
    # Verify migration
    print("\n--- Verification ---")
    target_tables = target_db.get_table_names()
    print(f"Tables in target database: {target_tables}")
    
    for table in target_tables:
        count = target_db.count_rows(table)
        print(f"  {table}: {count} rows")
    
    # Clean up
    source_db.close()
    target_db.close()
    migrator.close()


def example_database_comparison():
    """Compare different database types"""
    print("\n=== Database Type Comparison ===")
    
    # Test data
    test_df = pd.DataFrame({
        'id': range(1, 1001),
        'value': range(1000, 2001),
        'category': ['A', 'B', 'C'] * 334  # Cycle through categories
    })
    
    database_types = ['sqlite', 'duckdb']
    
    for db_type in database_types:
        print(f"\n--- Testing {db_type.upper()} ---")
        
        config = DatabaseConfig.get_default_config(db_type)
        db = DatabaseFactory.create_manager(**config)
        
        # Load data
        import time
        start_time = time.time()
        db.bulk_insert_from_pandas(test_df, 'test_data', if_exists='replace')
        load_time = time.time() - start_time
        
        # Query performance
        start_time = time.time()
        result = db.execute_query_df("""
            SELECT category, COUNT(*) as count, AVG(value) as avg_value
            FROM test_data 
            GROUP BY category
        """)
        query_time = time.time() - start_time
        
        print(f"  Load time: {load_time:.3f} seconds")
        print(f"  Query time: {query_time:.3f} seconds")
        print(f"  Result rows: {len(result)}")
        
        # Database-specific features
        if hasattr(db, 'read_parquet'):
            print(f"  Supports: Parquet files, advanced sampling")
        elif hasattr(db, 'pragma_query'):
            print(f"  Supports: Pragma queries, attach databases")
        
        db.close()


def example_advanced_features():
    """Advanced database features example"""
    print("\n=== Advanced Features ===")
    
    # Use DuckDB for advanced features
    config = DatabaseConfig.get_default_config('duckdb')
    db = DatabaseFactory.create_manager(**config)
    
    # Create test data
    large_df = pd.DataFrame({
        'id': range(1, 10001),
        'timestamp': pd.date_range('2024-01-01', periods=10000, freq='H'),
        'sensor_value': [50 + (i % 100) for i in range(10000)],
        'location': ['Location_' + str((i % 10) + 1) for i in range(10000)]
    })
    
    db.bulk_insert_from_pandas(large_df, 'sensor_data', if_exists='replace')
    print(f"Created large dataset with {len(large_df)} rows")
    
    # Advanced sampling
    if hasattr(db, 'sample_reservoir'):
        print("\n--- Advanced Sampling ---")
        
        # Reservoir sampling
        sample_query = db.sample_reservoir('sensor_data', 100, seed=42)
        sample_df = db.execute_query_df(sample_query)
        print(f"Reservoir sample: {len(sample_df)} rows")
        
        # Bernoulli sampling
        bernoulli_query = db.sample_bernoulli('sensor_data', 1.0, seed=42)  # 1%
        bernoulli_df = db.execute_query_df(bernoulli_query)
        print(f"Bernoulli sample (1%): {len(bernoulli_df)} rows")
    
    # Window functions and analytics
    print("\n--- Analytics Query ---")
    analytics_result = db.execute_query_df("""
        SELECT 
            location,
            DATE_TRUNC('day', timestamp) as day,
            AVG(sensor_value) as daily_avg,
            MAX(sensor_value) as daily_max,
            MIN(sensor_value) as daily_min,
            LAG(AVG(sensor_value)) OVER (
                PARTITION BY location 
                ORDER BY DATE_TRUNC('day', timestamp)
            ) as prev_day_avg
        FROM sensor_data
        GROUP BY location, DATE_TRUNC('day', timestamp)
        ORDER BY location, day
        LIMIT 20
    """)
    
    print(f"Analytics result: {len(analytics_result)} rows")
    print(analytics_result.head())
    
    # Performance tuning
    if hasattr(db, 'set_memory_limit'):
        print("\n--- Performance Tuning ---")
        db.set_memory_limit(2)  # 2GB memory limit
        print("Set memory limit to 2GB")
    
    db.close()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_database_migration()
    example_database_comparison()
    example_advanced_features()
    
    print("\n=== All Examples Completed ===")
    print("The database abstraction layer provides:")
    print("✓ Unified API across DuckDB, SQLite, PostgreSQL")
    print("✓ Database-specific optimizations") 
    print("✓ Easy migration between databases")
    print("✓ Advanced sampling and analytics")
    print("✓ Type-safe operations with SQLAlchemy Core")