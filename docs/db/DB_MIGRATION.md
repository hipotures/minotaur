<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02
Compatible with: SQLAlchemy refactoring implementation
Changes: Database migration tools and cross-engine data transfer procedures
-->

# Database Migration Guide

## 🎯 Migration Overview

The Minotaur database migration system enables seamless data transfer between different database engines (DuckDB, SQLite, PostgreSQL) while preserving data integrity, optimizing performance, and providing comprehensive error handling.

### Migration Use Cases

```
Development → Testing → Production
SQLite → DuckDB → PostgreSQL
  ↓         ↓         ↓
Fast dev   Analytics  Production
```

**Common Migration Scenarios:**
- **Development to Production**: SQLite → PostgreSQL for multi-user production
- **Analytics Optimization**: SQLite → DuckDB for advanced analytical features  
- **Legacy System Migration**: DuckDB → PostgreSQL for enterprise requirements
- **Testing Environment**: Any → SQLite for fast automated testing
- **Data Archival**: Any → PostgreSQL for long-term storage with ACID compliance

## 🔧 Migration Architecture

### Core Components

```
Migration Process Flow:
1. Schema Analysis     2. Type Mapping       3. Data Transfer      4. Validation
   ├── Source Tables      ├── Column Types       ├── Batch Processing   ├── Row Counts
   ├── Relationships      ├── Constraints        ├── Error Handling     ├── Data Sampling
   └── Indexes           └── Default Values     └── Progress Tracking  └── Integrity Checks
```

### DatabaseMigrator Class

The `DatabaseMigrator` class provides comprehensive migration capabilities:

```python
from src.database.migrations import DatabaseMigrator
from src.database.engine_factory import DatabaseFactory
from src.database.config import DatabaseConfig

# Create source and target databases
source_config = DatabaseConfig.get_default_config('sqlite', 'source.db')
target_config = DatabaseConfig.get_default_config('duckdb', 'target.duckdb')

source_db = DatabaseFactory.create_manager(**source_config)
target_db = DatabaseFactory.create_manager(**target_config)

# Initialize migrator
migrator = DatabaseMigrator(source_db.engine, target_db.engine)
```

## 📊 Migration Operations

### 1. Single Table Migration

```python
# Basic table migration
stats = migrator.migrate_table(
    table_name='employees',
    batch_size=10000,     # Process 10K rows at a time
    if_exists='replace'   # 'fail', 'replace', or 'append'
)

print(f"Migrated {stats['migrated_rows']} rows in {stats['batches_processed']} batches")
```

**Migration Statistics:**
```python
{
    'table_name': 'employees',
    'source_rows': 50000,
    'migrated_rows': 50000,
    'batches_processed': 5,
    'success': True,
    'error': None
}
```

### 2. Bulk Migration (All Tables)

```python
# Migrate all tables with filtering
migration_stats = migrator.migrate_all_tables(
    batch_size=5000,
    if_exists='replace',
    exclude_tables=['temp_table', 'cache_table'],  # Skip these tables
    include_tables=None,  # If set, only migrate these tables
    dry_run=False        # Set True for analysis without migration
)

print(f"Success: {migration_stats['successful_tables']}/{migration_stats['total_tables']}")
print(f"Total rows migrated: {migration_stats['total_migrated_rows']}")
```

### 3. Dry Run Analysis

```python
# Analyze migration without transferring data
dry_run_stats = migrator.migrate_all_tables(dry_run=True)

for table_result in dry_run_stats['table_results']:
    print(f"{table_result['table_name']}: {table_result['source_rows']} rows")
    if not table_result['success']:
        print(f"  Error: {table_result['error']}")
```

## 🔍 Schema Comparison & Validation

### Schema Analysis

```python
# Compare schemas between databases
comparison = migrator.compare_schemas('employees')

print(f"Source exists: {comparison['source_exists']}")
print(f"Target exists: {comparison['target_exists']}")

if comparison['missing_in_target']:
    print(f"Missing columns in target: {comparison['missing_in_target']}")

if comparison['type_differences']:
    for diff in comparison['type_differences']:
        print(f"Type mismatch - {diff['column']}: {diff['source_type']} → {diff['target_type']}")
```

**Schema Comparison Output:**
```python
{
    'table_name': 'employees',
    'source_exists': True,
    'target_exists': True,
    'source_columns': [('id', 'INTEGER'), ('name', 'VARCHAR'), ('salary', 'DECIMAL')],
    'target_columns': [('id', 'BIGINT'), ('name', 'TEXT'), ('salary', 'DOUBLE')],
    'missing_in_target': [],
    'missing_in_source': [],
    'type_differences': [
        {'column': 'id', 'source_type': 'INTEGER', 'target_type': 'BIGINT'},
        {'column': 'salary', 'source_type': 'DECIMAL', 'target_type': 'DOUBLE'}
    ]
}
```

### Migration Script Generation

```python
# Generate SQL migration script
migrator.create_migration_script(
    output_file='migration_script.sql',
    include_tables=['employees', 'departments']
)
```

Generated script example:
```sql
-- Database Migration Script
-- Generated for migration from sqlite:///source.db to duckdb:///target.duckdb

-- Table: employees
CREATE TABLE employees (id INTEGER, name VARCHAR, salary DECIMAL, department_id INTEGER);

-- Table: departments  
CREATE TABLE departments (id INTEGER, name VARCHAR, budget DECIMAL);
```

## 🚀 Performance Optimization

### Batch Size Optimization

Different database types have optimal batch sizes:

```python
def get_optimal_batch_size(source_type: str, target_type: str, table_size: int) -> int:
    """Determine optimal batch size based on database types and data size"""
    
    if source_type == 'sqlite' and target_type == 'duckdb':
        # SQLite → DuckDB: Larger batches for analytical target
        return min(50000, table_size // 10)
    
    elif source_type == 'duckdb' and target_type == 'postgresql':
        # DuckDB → PostgreSQL: Medium batches for ACID compliance
        return min(20000, table_size // 20)
    
    elif target_type == 'sqlite':
        # Any → SQLite: Smaller batches for single-threaded target
        return min(5000, table_size // 50)
    
    else:
        # Default: Conservative batch size
        return min(10000, table_size // 20)

# Use optimal batch size
table_size = migrator.get_source_table_size('large_table')
optimal_batch = get_optimal_batch_size(source_type, target_type, table_size)

stats = migrator.migrate_table('large_table', batch_size=optimal_batch)
```

### Parallel Migration (Multiple Tables)

```python
from concurrent.futures import ThreadPoolExecutor
import threading

def migrate_table_parallel(migrator, table_name, batch_size):
    """Thread-safe table migration"""
    thread_local_migrator = DatabaseMigrator(
        migrator.source_engine, 
        migrator.target_engine
    )
    return thread_local_migrator.migrate_table(table_name, batch_size)

# Migrate multiple tables in parallel
tables_to_migrate = ['table1', 'table2', 'table3', 'table4']

with ThreadPoolExecutor(max_workers=3) as executor:
    future_to_table = {
        executor.submit(migrate_table_parallel, migrator, table, 10000): table
        for table in tables_to_migrate
    }
    
    for future in concurrent.futures.as_completed(future_to_table):
        table = future_to_table[future]
        try:
            result = future.result()
            print(f"Completed {table}: {result['migrated_rows']} rows")
        except Exception as e:
            print(f"Failed {table}: {e}")
```

## 🛡️ Error Handling & Recovery

### Robust Migration with Retry Logic

```python
import time
import logging

def migrate_with_retry(migrator, table_name, max_retries=3, retry_delay=5):
    """Migrate table with exponential backoff retry"""
    
    for attempt in range(max_retries + 1):
        try:
            stats = migrator.migrate_table(table_name, batch_size=10000)
            if stats['success']:
                return stats
            else:
                raise Exception(f"Migration failed: {stats['error']}")
                
        except Exception as e:
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Migration attempt {attempt + 1} failed for {table_name}: {e}")
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Migration failed after {max_retries + 1} attempts: {e}")
                raise

# Use with error handling
try:
    result = migrate_with_retry(migrator, 'critical_table')
    print(f"Successfully migrated: {result['migrated_rows']} rows")
except Exception as e:
    print(f"Migration completely failed: {e}")
```

### Transaction-Safe Migration

```python
from sqlalchemy import text

def migrate_table_transactional(migrator, table_name, batch_size=10000):
    """Migrate table with transaction safety"""
    
    try:
        # Begin transaction in target database
        with migrator.target_engine.begin() as trans:
            # Drop target table if exists
            drop_sql = f"DROP TABLE IF EXISTS {table_name}"
            trans.execute(text(drop_sql))
            
            # Perform migration
            stats = migrator.migrate_table(table_name, batch_size)
            
            if not stats['success']:
                raise Exception(f"Migration failed: {stats['error']}")
            
            # Verify row count matches
            source_count = migrator.get_source_table_size(table_name)
            target_count = migrator.get_target_table_size(table_name)
            
            if source_count != target_count:
                raise Exception(f"Row count mismatch: {source_count} vs {target_count}")
            
            print(f"Transaction committed: {stats['migrated_rows']} rows")
            return stats
            
    except Exception as e:
        print(f"Transaction rolled back due to error: {e}")
        raise
```

## 🔄 Migration Strategies

### 1. Incremental Migration

For large datasets, migrate incrementally based on timestamps:

```python
def migrate_incremental(migrator, table_name, timestamp_column, last_migrated):
    """Migrate only new/updated records since last migration"""
    
    # Create filtered migration query
    where_clause = f"{timestamp_column} > '{last_migrated}'"
    
    # Get new records count
    count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
    new_records = migrator.source_engine.execute(text(count_query)).scalar()
    
    if new_records == 0:
        print("No new records to migrate")
        return
    
    print(f"Migrating {new_records} new records...")
    
    # Migrate with WHERE clause
    migration_query = f"""
        INSERT INTO {table_name} 
        SELECT * FROM source.{table_name} 
        WHERE {where_clause}
    """
    
    # Execute incremental migration
    with migrator.target_engine.begin() as conn:
        conn.execute(text(migration_query))
    
    print(f"Incremental migration completed: {new_records} records")
```

### 2. Blue-Green Migration

Zero-downtime migration using temporary tables:

```python
def migrate_blue_green(migrator, table_name):
    """Blue-green migration with zero downtime"""
    
    temp_table = f"{table_name}_new"
    
    try:
        # Step 1: Migrate to temporary table
        print(f"Migrating to temporary table: {temp_table}")
        stats = migrator.migrate_table(table_name, batch_size=10000)
        
        # Rename migrated table to temporary name
        with migrator.target_engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} RENAME TO {temp_table}"))
        
        # Step 2: Verify data integrity
        source_count = migrator.get_source_table_size(table_name)
        temp_count = migrator.get_target_table_size(temp_table)
        
        if source_count != temp_count:
            raise Exception(f"Data integrity check failed: {source_count} vs {temp_count}")
        
        # Step 3: Atomic switch - rename temp table to production name
        with migrator.target_engine.begin() as conn:
            # Drop old table if exists
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}_old"))
            
            # Backup current table
            if migrator.target_inspector.has_table(table_name):
                conn.execute(text(f"ALTER TABLE {table_name} RENAME TO {table_name}_old"))
            
            # Switch new table to production name
            conn.execute(text(f"ALTER TABLE {temp_table} RENAME TO {table_name}"))
        
        print(f"Blue-green migration completed successfully")
        
        # Optional: Clean up old table after verification
        # with migrator.target_engine.begin() as conn:
        #     conn.execute(text(f"DROP TABLE {table_name}_old"))
        
    except Exception as e:
        # Cleanup on failure
        print(f"Migration failed, cleaning up: {e}")
        with migrator.target_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
        raise
```

### 3. Streaming Migration

For extremely large datasets, use streaming approach:

```python
def migrate_streaming(migrator, table_name, stream_size=1000):
    """Stream large tables row by row to manage memory"""
    
    total_rows = migrator.get_source_table_size(table_name)
    migrated_rows = 0
    
    # Create target table structure
    source_table = migrator.get_source_table_structure(table_name)
    source_table.create(migrator.target_engine, checkfirst=True)
    
    # Stream data in small chunks
    with migrator.source_engine.connect() as source_conn:
        with migrator.target_engine.connect() as target_conn:
            
            offset = 0
            while offset < total_rows:
                # Fetch small batch
                query = f"SELECT * FROM {table_name} LIMIT {stream_size} OFFSET {offset}"
                batch_df = pd.read_sql(query, source_conn)
                
                if batch_df.empty:
                    break
                
                # Insert batch
                batch_df.to_sql(table_name, target_conn, if_exists='append', index=False)
                
                migrated_rows += len(batch_df)
                offset += stream_size
                
                # Progress reporting
                progress = (migrated_rows / total_rows) * 100
                print(f"Progress: {progress:.1f}% ({migrated_rows}/{total_rows} rows)")
    
    print(f"Streaming migration completed: {migrated_rows} rows")
```

## 📊 Migration Monitoring & Validation

### Comprehensive Validation

```python
def validate_migration(migrator, table_name):
    """Comprehensive migration validation"""
    
    validation_results = {
        'table_name': table_name,
        'row_count_match': False,
        'sample_data_match': False,
        'schema_compatible': False,
        'performance_acceptable': False
    }
    
    # 1. Row count validation
    source_count = migrator.get_source_table_size(table_name)
    target_count = migrator.get_target_table_size(table_name)
    validation_results['row_count_match'] = (source_count == target_count)
    
    # 2. Sample data validation (first 100 rows)
    source_sample = migrator.get_source_sample(table_name, 100)
    target_sample = migrator.get_target_sample(table_name, 100)
    
    # Compare key columns (ID, timestamps, etc.)
    key_columns = ['id'] if 'id' in source_sample.columns else source_sample.columns[:3]
    sample_match = source_sample[key_columns].equals(target_sample[key_columns])
    validation_results['sample_data_match'] = sample_match
    
    # 3. Schema validation
    schema_comparison = migrator.compare_schemas(table_name)
    validation_results['schema_compatible'] = len(schema_comparison['missing_in_target']) == 0
    
    # 4. Performance validation (query time comparison)
    test_query = f"SELECT COUNT(*) FROM {table_name}"
    
    start_time = time.time()
    migrator.source_engine.execute(text(test_query))
    source_time = time.time() - start_time
    
    start_time = time.time()
    migrator.target_engine.execute(text(test_query))
    target_time = time.time() - start_time
    
    # Target should be within 5x of source performance
    validation_results['performance_acceptable'] = (target_time <= source_time * 5)
    
    return validation_results

# Validate migration
validation = validate_migration(migrator, 'employees')
print(f"Validation results: {validation}")
```

### Migration Progress Tracking

```python
class MigrationProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.completed_tables = []
        self.failed_tables = []
        self.total_rows_migrated = 0
    
    def track_table_completion(self, table_name, stats):
        """Track completion of individual table migration"""
        if stats['success']:
            self.completed_tables.append({
                'name': table_name,
                'rows': stats['migrated_rows'],
                'batches': stats['batches_processed'],
                'completed_at': time.time()
            })
            self.total_rows_migrated += stats['migrated_rows']
        else:
            self.failed_tables.append({
                'name': table_name,
                'error': stats['error'],
                'failed_at': time.time()
            })
    
    def get_progress_report(self):
        """Generate comprehensive progress report"""
        elapsed_time = time.time() - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'completed_tables': len(self.completed_tables),
            'failed_tables': len(self.failed_tables),
            'total_rows_migrated': self.total_rows_migrated,
            'average_rows_per_second': self.total_rows_migrated / elapsed_time if elapsed_time > 0 else 0,
            'completion_details': self.completed_tables,
            'failure_details': self.failed_tables
        }

# Usage
tracker = MigrationProgressTracker()

for table in tables_to_migrate:
    stats = migrator.migrate_table(table)
    tracker.track_table_completion(table, stats)
    
    # Print progress
    report = tracker.get_progress_report()
    print(f"Progress: {report['completed_tables']}/{len(tables_to_migrate)} tables completed")
    print(f"Rate: {report['average_rows_per_second']:.0f} rows/second")
```

## 🎯 Best Practices & Recommendations

### Pre-Migration Checklist

1. **Environment Preparation**
   ```bash
   # Verify database connections
   python -c "from src.database.utils import ConnectionUtils; print(ConnectionUtils.test_connection(source_engine))"
   
   # Check available disk space
   df -h /path/to/target/database/
   
   # Verify required extensions/drivers
   pip list | grep -E "(duckdb|psycopg2|sqlalchemy)"
   ```

2. **Data Analysis**
   ```python
   # Analyze data size and complexity
   tables = migrator.get_source_tables()
   for table in tables:
       size_info = migrator.get_table_size_estimate(table)
       print(f"{table}: {size_info['row_count']} rows, {size_info['size_mb']} MB")
   ```

3. **Performance Testing**
   ```python
   # Test migration performance with small sample
   sample_stats = migrator.migrate_table('test_table', batch_size=1000, dry_run=True)
   estimated_time = (sample_stats['source_rows'] / 1000) * 0.1  # Estimate based on 1K batch
   print(f"Estimated migration time: {estimated_time:.1f} seconds")
   ```

### Post-Migration Validation

1. **Data Integrity**
   ```python
   # Run comprehensive validation
   for table in migrated_tables:
       validation = validate_migration(migrator, table)
       assert all(validation.values()), f"Validation failed for {table}: {validation}"
   ```

2. **Performance Verification**
   ```python
   # Compare query performance
   test_queries = [
       "SELECT COUNT(*) FROM employees",
       "SELECT AVG(salary) FROM employees GROUP BY department",
       "SELECT * FROM employees WHERE hire_date > '2020-01-01' LIMIT 100"
   ]
   
   for query in test_queries:
       source_time = measure_query_time(migrator.source_engine, query)
       target_time = measure_query_time(migrator.target_engine, query)
       print(f"Query performance ratio: {target_time/source_time:.2f}x")
   ```

3. **Application Testing**
   ```python
   # Test application compatibility
   from src.sqlalchemy_data_manager import SQLAlchemyDataManager
   
   # Verify the migrated database works with application
   config = {'database': {'type': target_db_type}, 'autogluon': {'dataset_name': 'test'}}
   app_manager = SQLAlchemyDataManager(config)
   
   # Test core operations
   sample_df = app_manager.sample_dataset(train_path, 100)
   assert len(sample_df) == 100, "Application integration test failed"
   ```

### Migration Strategy Selection Guide

| Scenario | Strategy | Pros | Cons |
|----------|----------|------|------|
| **Small datasets** (<1GB) | Direct migration | Simple, fast | Single point of failure |
| **Large datasets** (>10GB) | Streaming migration | Memory efficient | Slower, more complex |
| **Production systems** | Blue-green migration | Zero downtime | Requires 2x storage |
| **Ongoing sync** | Incremental migration | Minimal data transfer | Requires timestamp columns |
| **One-time migration** | Batch migration | Good performance | Downtime required |

This comprehensive migration guide ensures successful and reliable data transfers between different database engines while maintaining data integrity and optimal performance.