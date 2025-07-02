<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02
Compatible with: SQLAlchemy refactoring implementation
Changes: New SQLAlchemy-based database abstraction layer with multi-engine support
-->

# Database Abstraction Layer - Overview

## 🎯 Executive Summary

The **Minotaur Database Abstraction Layer** is a sophisticated, SQLAlchemy Core-based system that provides unified database operations across multiple database engines. The system enables seamless switching between **DuckDB**, **SQLite**, and **PostgreSQL** while maintaining optimal performance and database-specific features.

### Key Innovation: Multi-Engine Compatibility with Specialized Features

```
Traditional Approach:           Minotaur Approach:
Tightly coupled to DuckDB      Universal API → Engine-specific optimizations
    ↓                              ↓
Hard to test/migrate           Easy switching, better testing
```

## 🏗️ High-Level Architecture

### System Flow
```
1. Configuration              2. Engine Creation           3. Specialized Features
   ├── Database Type             ├── SQLAlchemy Core          ├── DuckDB: Parquet, Sampling
   ├── Connection Params         ├── Connection Pooling       ├── SQLite: Pragma, Attach
   └── Engine Arguments          └── Type Safety             └── PostgreSQL: Extensions, COPY
```

### Core Components
- **🔧 Database Factory**: Creates database managers with engine-specific functionality
- **⚡ Base Manager**: Universal SQLAlchemy Core operations for all databases
- **🗄️ Migration Tools**: Data migration between different database engines
- **📊 Query Modules**: Database-specific optimized queries and operations
- **🎯 Connection Management**: Pool management, health monitoring, and error handling

## 📋 Key Concepts & Terminology

### Database Types & Use Cases
- **DuckDB** (`db_type='duckdb'`): Analytical workloads, Parquet files, advanced sampling, OLAP operations
- **SQLite** (`db_type='sqlite'`): Development, testing, small datasets, single-user applications
- **PostgreSQL** (`db_type='postgresql'`): Production, multi-user, ACID compliance, advanced indexing

### Architecture Patterns
- **Factory Pattern**: `DatabaseFactory` creates appropriate managers for each database type
- **Strategy Pattern**: Database-specific methods added dynamically based on engine type
- **Repository Pattern**: Clean separation between business logic and data access
- **Connection Pooling**: Automatic connection management optimized per database type

### Feature Categories
- **Universal Operations**: Work across all database types (SELECT, INSERT, UPDATE, DELETE, aggregations)
- **Engine-Specific Features**: Database-optimized operations (DuckDB sampling, PostgreSQL COPY, SQLite PRAGMA)
- **Migration Operations**: Cross-database data transfer with schema mapping

## 🚀 Quick Start Guide

### Basic Usage
```python
from src.database.engine_factory import DatabaseFactory
from src.database.config import DatabaseConfig

# Easy database switching - just change db_type
config = {
    'db_type': 'duckdb',  # or 'sqlite', 'postgresql'
    'connection_params': {
        'database': './my_database.duckdb'
    }
}

# Create manager with engine-specific features
db = DatabaseFactory.create_manager(**config)

# Universal operations work on all databases
df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
db.bulk_insert_from_pandas(df, 'test_table')

results = db.execute_query("SELECT * FROM test_table WHERE value > :threshold", 
                          {'threshold': 15})

# Engine-specific features (available when using DuckDB)
if hasattr(db, 'read_parquet'):
    db.read_parquet('data.parquet', 'imported_data')
    sample_query = db.sample_reservoir('imported_data', 1000, seed=42)
    sample_df = db.execute_query_df(sample_query)
```

### Database Migration
```python
from src.database.migrations import DatabaseMigrator

# Migrate from SQLite to DuckDB
source_config = DatabaseConfig.get_default_config('sqlite', 'source.db')
target_config = DatabaseConfig.get_default_config('duckdb', 'target.duckdb')

source_db = DatabaseFactory.create_manager(**source_config)
target_db = DatabaseFactory.create_manager(**target_config)

migrator = DatabaseMigrator(source_db.engine, target_db.engine)
stats = migrator.migrate_all_tables(batch_size=10000)

print(f"Migrated {stats['total_migrated_rows']} rows across {stats['successful_tables']} tables")
```

## 📊 System Capabilities

### Performance Features
- **Connection Pooling**: Automatic per-database optimization
- **Query Optimization**: Engine-specific query patterns and indexing
- **Batch Operations**: Efficient bulk insert/update operations
- **Streaming**: Large dataset processing without memory overflow

### Reliability Features
- **Health Monitoring**: Connection pool status and performance tracking
- **Error Recovery**: Automatic retry with exponential backoff
- **Transaction Management**: ACID compliance and rollback support
- **Schema Validation**: Type mapping and constraint verification

### Developer Experience
- **Type Safety**: SQLAlchemy Core provides compile-time query validation
- **Easy Testing**: Switch to SQLite for fast unit tests
- **Database Agnostic**: Write once, run on any supported database
- **Rich Introspection**: Table schema, column types, and statistics

## 🔧 Engine-Specific Features

### DuckDB Specializations
```python
# Advanced analytics and data science features
db.read_parquet('large_dataset.parquet', 'analytics_data')
db.set_memory_limit(8)  # 8GB memory limit
db.set_threads(8)       # Multi-threading

# Advanced sampling methods
reservoir_query = db.sample_reservoir('analytics_data', 5000, seed=42)
bernoulli_query = db.sample_bernoulli('analytics_data', 1.5, seed=42)

# Export results
db.export_to_parquet('results', 'output.parquet')
```

### SQLite Specializations  
```python
# Development and testing optimizations
db.set_journal_mode('WAL')      # Write-Ahead Logging
db.set_synchronous('NORMAL')    # Performance tuning
db.set_cache_size(-64000)       # 64MB cache

# Database attachment for complex operations
db.attach_database('other.db', 'secondary')
db.execute_query("INSERT INTO main.table SELECT * FROM secondary.table")
db.detach_database('secondary')

# Maintenance operations
db.vacuum_db()
db.analyze_db()
```

### PostgreSQL Specializations
```python
# Production database features
db.create_extension('pg_stat_statements')  # Query performance monitoring
db.vacuum_analyze('large_table')           # Maintenance

# Concurrent index operations
db.create_index_concurrently('idx_performance', 'large_table', 'timestamp, user_id')

# Efficient bulk operations
db.copy_from_csv('bulk_data', '/path/to/data.csv', header=True)

# Advanced sampling
tablesample_query = db.sample_tablesample('large_table', 0.1, 'BERNOULLI')  # 0.1%
```

## 📈 Benefits & Impact

### For Data Scientists
- **Rapid Prototyping**: Switch from SQLite (development) to DuckDB (analytics) to PostgreSQL (production)
- **Consistent API**: Same code works across all database types
- **Advanced Analytics**: DuckDB's analytical features with universal interface

### For Developers  
- **Better Testing**: Fast SQLite tests, realistic DuckDB integration tests
- **Easy Deployment**: PostgreSQL for production with same codebase
- **Type Safety**: SQLAlchemy Core prevents SQL injection and type errors

### For Operations
- **Flexible Infrastructure**: Choose optimal database for each use case
- **Easy Migration**: Built-in tools for moving data between databases
- **Monitoring**: Health checks and performance metrics

### For Minotaur MCTS
- **Performance**: DuckDB's analytical engine optimized for feature discovery
- **Reliability**: PostgreSQL for production MCTS session persistence
- **Development**: SQLite for fast testing of MCTS algorithms

## 🔄 Migration from Legacy System

The new system provides a drop-in replacement for `DuckDBDataManager`:

```python
# Legacy code
from src.duckdb_data_manager import DuckDBDataManager
manager = DuckDBDataManager(config)

# New system - same interface, more flexibility
from src.sqlalchemy_data_manager import SQLAlchemyDataManager
manager = SQLAlchemyDataManager(config)

# All existing methods work the same
df = manager.sample_dataset(train_path, 1000)
manager.cache_features(hash, name, features_df)
```

## 📚 Documentation Structure

- **📋 [DB_OVERVIEW.md](DB_OVERVIEW.md)** - This overview document
- **🔧 [DB_IMPLEMENTATION.md](DB_IMPLEMENTATION.md)** - Technical implementation details
- **🔗 [DB_MIGRATION.md](DB_MIGRATION.md)** - Migration tools and procedures
- **⚙️ [DB_OPERATIONS.md](DB_OPERATIONS.md)** - Configuration and operations guide
- **⚡ [DB_PERFORMANCE.md](DB_PERFORMANCE.md)** - Performance optimization and tuning

## 🎯 Next Steps

1. **For New Projects**: Use SQLAlchemy system from the start
2. **For Existing Code**: Gradual migration using compatibility layer
3. **For Production**: Consider PostgreSQL for multi-user scenarios
4. **For Analytics**: Leverage DuckDB's advanced analytical features
5. **For Testing**: Use SQLite for fast, reliable test suites