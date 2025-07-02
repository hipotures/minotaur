<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02
Compatible with: SQLAlchemy refactoring implementation
Changes: Technical implementation details for SQLAlchemy Core abstraction layer
-->

# Database Abstraction Layer - Implementation Details

## 🏗️ Architecture Overview

The database abstraction layer is built using **SQLAlchemy Core** (not ORM) for maximum performance and control in analytical workloads. The system follows several key design patterns to ensure flexibility, maintainability, and performance.

### Core Design Patterns

```
Factory Pattern                Strategy Pattern               Repository Pattern
    ↓                              ↓                              ↓
DatabaseFactory              Database-specific methods      Clean data access
Creates managers             Added dynamically              Separates business logic
```

## 📦 Module Structure

```
src/database/
├── __init__.py                     # Package initialization
├── config.py                      # Database configuration management
├── base_manager.py                # Core SQLAlchemy operations
├── engine_factory.py              # Factory for creating database managers
├── queries/
│   ├── __init__.py
│   ├── base_queries.py            # Universal queries across all databases
│   └── specific/
│       ├── __init__.py
│       ├── duckdb_queries.py      # DuckDB-specific operations
│       ├── postgres_queries.py    # PostgreSQL-specific operations
│       └── sqlite_queries.py      # SQLite-specific operations
├── utils/
│   ├── __init__.py
│   ├── type_mapping.py            # Type conversion between databases
│   └── connection_utils.py        # Connection monitoring and utilities
└── migrations/
    ├── __init__.py
    └── migration_tool.py          # Cross-database migration tools
```

## 🔧 Core Components Implementation

### DatabaseConfig (config.py)

The configuration system provides unified database connection management:

```python
class DatabaseConfig:
    @staticmethod
    def get_engine(db_type: str, connection_params: Dict[str, Any]) -> Engine:
        """Create SQLAlchemy engine based on database type"""
        
        if db_type == 'duckdb':
            # DuckDB with duckdb-engine driver
            database = connection_params.get('database', ':memory:')
            conn_string = f"duckdb:///{database}"
            
        elif db_type == 'sqlite':
            # SQLite with built-in driver
            database = connection_params.get('database', ':memory:')
            conn_string = f"sqlite:///{database}"
            
        elif db_type == 'postgresql':
            # PostgreSQL with psycopg2 driver
            user = connection_params.get('user', 'postgres')
            password = connection_params.get('password', '')
            host = connection_params.get('host', 'localhost')
            port = connection_params.get('port', 5432)
            database = connection_params.get('database', 'postgres')
            conn_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        
        # Apply database-specific optimizations
        engine_args = connection_params.get('engine_args', {})
        return create_engine(conn_string, **engine_args)
```

**Key Features:**
- URL construction for each database type
- Engine argument optimization per database
- Security (password masking in logs)
- Connection pooling configuration

### DatabaseManager (base_manager.py)

The base manager provides universal operations using SQLAlchemy Core:

```python
class DatabaseManager:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.metadata = MetaData()
        self._tables = {}  # Table reflection cache
    
    def reflect_table(self, table_name: str) -> Table:
        """Load metadata for existing table with caching"""
        if table_name not in self._tables:
            self._tables[table_name] = Table(
                table_name, self.metadata, autoload_with=self.engine
            )
        return self._tables[table_name]
    
    def execute_query(self, query: Union[str, Any], params: Optional[Dict] = None):
        """Execute parameterized queries safely"""
        with self.engine.connect() as conn:
            if isinstance(query, str):
                result = conn.execute(text(query), params or {})
            else:
                result = conn.execute(query)
            return [dict(row._mapping) for row in result]
```

**Key Features:**
- Table reflection with caching for performance
- Parameterized queries preventing SQL injection
- Connection management with context managers
- Type-safe operations using SQLAlchemy constructs

### DatabaseFactory (engine_factory.py)

The factory implements the **Strategy Pattern** to add database-specific functionality:

```python
class DatabaseFactory:
    @staticmethod
    def create_manager(db_type: str, connection_params: Dict[str, Any]) -> DatabaseManager:
        """Create manager with database-specific features"""
        
        # Create base manager
        engine = DatabaseConfig.get_engine(db_type, connection_params)
        manager = DatabaseManager(engine)
        
        # Add database-specific methods dynamically
        if db_type == 'duckdb':
            DatabaseFactory._add_duckdb_methods(manager, engine)
        elif db_type == 'sqlite':
            DatabaseFactory._add_sqlite_methods(manager, engine)
        elif db_type == 'postgresql':
            DatabaseFactory._add_postgresql_methods(manager, engine)
        
        manager.db_type = db_type
        return manager
    
    @staticmethod
    def _add_duckdb_methods(manager: DatabaseManager, engine):
        """Add DuckDB-specific capabilities"""
        queries = DuckDBSpecificQueries()
        
        # File I/O operations
        manager.read_parquet = lambda fp, tn: queries.read_parquet(engine, fp, tn)
        manager.export_to_parquet = lambda tn, fp: queries.export_to_parquet(engine, tn, fp)
        
        # Advanced sampling
        manager.sample_reservoir = lambda tn, size, seed=None: queries.sample_reservoir(engine, tn, size, seed)
        manager.sample_bernoulli = lambda tn, pct, seed=None: queries.sample_bernoulli(engine, tn, pct, seed)
        
        # Performance tuning
        manager.set_memory_limit = lambda gb: queries.set_memory_limit(engine, gb)
        manager.set_threads = lambda count: queries.set_threads(engine, count)
```

**Key Features:**
- Dynamic method injection based on database type
- Clean separation of universal vs. specific functionality
- Lazy evaluation of database-specific queries
- Type hints preserved for IDE support

## 🎯 Database-Specific Implementations

### DuckDB Specializations (duckdb_queries.py)

DuckDB excels at analytical workloads and provides unique features:

```python
class DuckDBSpecificQueries:
    @staticmethod
    def read_parquet(engine: Engine, file_path: str, table_name: str):
        """Read Parquet files directly into tables"""
        query = text(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{file_path}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
            conn.commit()
    
    @staticmethod
    def sample_reservoir(engine: Engine, table_name: str, sample_size: int, seed=None):
        """Advanced reservoir sampling for unbiased samples"""
        if seed is not None:
            return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size}) REPEATABLE ({seed})"
        return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size})"
    
    @staticmethod
    def sample_bernoulli(engine: Engine, table_name: str, percentage: float, seed=None):
        """Bernoulli sampling for statistical analysis"""
        if seed is not None:
            return f"SELECT * FROM {table_name} USING SAMPLE {percentage}% (bernoulli, {seed})"
        return f"SELECT * FROM {table_name} USING SAMPLE {percentage}% (bernoulli)"
```

**Advanced Features:**
- **Parquet Integration**: Direct file reading without intermediate loading
- **Reservoir Sampling**: Unbiased sampling for large datasets
- **Bernoulli Sampling**: Statistical sampling with configurable percentages
- **Memory Management**: Dynamic memory limits and thread control
- **Columnar Processing**: Optimized for analytical queries

### SQLite Specializations (sqlite_queries.py)

SQLite is optimized for development, testing, and single-user scenarios:

```python
class SQLiteSpecificQueries:
    @staticmethod
    def pragma_query(engine: Engine, pragma_name: str, value=None):
        """Execute SQLite PRAGMA commands for optimization"""
        if value is not None:
            query = text(f"PRAGMA {pragma_name} = {value}")
            with engine.connect() as conn:
                conn.execute(query)
        else:
            query = text(f"PRAGMA {pragma_name}")
            with engine.connect() as conn:
                return conn.execute(query).fetchone()[0]
    
    @staticmethod
    def set_journal_mode(engine: Engine, mode: str = "WAL"):
        """Set journal mode for performance/safety balance"""
        SQLiteSpecificQueries.pragma_query(engine, "journal_mode", mode)
    
    @staticmethod
    def attach_database(engine: Engine, db_path: str, alias: str):
        """Attach additional databases for cross-database queries"""
        query = text(f"ATTACH DATABASE '{db_path}' AS {alias}")
        with engine.connect() as conn:
            conn.execute(query)
```

**Key Features:**
- **PRAGMA Optimization**: Fine-tuned performance settings
- **WAL Mode**: Write-Ahead Logging for better concurrency
- **Database Attachment**: Multi-database operations
- **VACUUM/ANALYZE**: Maintenance operations
- **Fast Development**: Minimal setup, file-based storage

### PostgreSQL Specializations (postgres_queries.py)

PostgreSQL provides enterprise-grade features for production use:

```python
class PostgreSQLSpecificQueries:
    @staticmethod
    def create_index_concurrently(engine: Engine, index_name: str, table_name: str, columns: str):
        """Create indexes without blocking operations"""
        query = text(f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} ({columns})")
        with engine.connect() as conn:
            conn.execute(query)
    
    @staticmethod
    def copy_from_csv(engine: Engine, table_name: str, file_path: str, header=True, delimiter=','):
        """High-performance bulk loading from CSV"""
        header_opt = "HEADER" if header else ""
        query = text(f"""
            COPY {table_name} 
            FROM '{file_path}' 
            WITH (FORMAT CSV, {header_opt}, DELIMITER '{delimiter}')
        """)
        with engine.connect() as conn:
            conn.execute(query)
```

**Enterprise Features:**
- **Concurrent Indexing**: Non-blocking index creation
- **COPY Command**: Fastest bulk loading method
- **Extensions**: PostGIS, pg_stat_statements, etc.
- **ACID Compliance**: Full transaction support
- **Connection Pooling**: High concurrency support

## 🔄 Migration System (migration_tool.py)

The migration system enables seamless data transfer between database types:

```python
class DatabaseMigrator:
    def __init__(self, source_engine: Engine, target_engine: Engine):
        self.source_engine = source_engine
        self.target_engine = target_engine
        self.source_inspector = inspect(source_engine)
        self.target_inspector = inspect(target_engine)
    
    def migrate_table(self, table_name: str, batch_size: int = 10000, if_exists='replace'):
        """Migrate single table with batching for large datasets"""
        
        # Reflect source table structure
        source_table = Table(table_name, MetaData(), autoload_with=self.source_engine)
        
        # Create table in target database
        source_table.create(self.target_engine, checkfirst=True)
        
        # Migrate data in batches to manage memory
        offset = 0
        while True:
            query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
            
            with self.source_engine.connect() as source_conn:
                df = pd.read_sql(query, source_conn)
            
            if df.empty:
                break
                
            # Write batch to target
            df.to_sql(table_name, self.target_engine, if_exists='append', index=False)
            offset += batch_size
```

**Migration Features:**
- **Schema Translation**: Automatic type mapping between databases
- **Batch Processing**: Memory-efficient large dataset migration
- **Error Recovery**: Rollback and retry capabilities
- **Progress Tracking**: Detailed statistics and progress reporting
- **Dry Run Mode**: Validation without actual data transfer

## 🔍 Type System (type_mapping.py)

The type mapping system ensures compatibility across different database engines:

```python
class TypeMapper:
    COMMON_TYPE_MAPPING = {
        'integer': types.Integer,
        'bigint': types.BigInteger,
        'varchar': types.String,
        'text': types.Text,
        'float': types.Float,
        'double': types.Float,
        'boolean': types.Boolean,
        'timestamp': types.DateTime,
        'json': types.JSON,
    }
    
    DB_TYPE_PREFERENCES = {
        'duckdb': {
            'string_type': types.String,
            'float_type': types.Double,
            'json_type': types.JSON,
        },
        'sqlite': {
            'string_type': types.Text,
            'json_type': types.Text,  # SQLite doesn't have native JSON
        },
        'postgresql': {
            'string_type': types.String,
            'json_type': types.JSON,
        }
    }
```

**Type Mapping Features:**
- **Universal Types**: Common types that work across all databases
- **Database Preferences**: Optimal type selection per database
- **Pandas Integration**: Automatic DataFrame type conversion
- **Schema Validation**: Type compatibility checking during migration

## ⚡ Performance Optimizations

### Connection Pooling

Each database type has optimized connection pool settings:

```python
def optimize_connection_pool(engine: Engine, db_type: str):
    if db_type == 'sqlite':
        # SQLite is single-threaded
        return {'pool_size': 1, 'max_overflow': 0}
    elif db_type == 'duckdb':
        # DuckDB benefits from small pools
        return {'pool_size': 5, 'max_overflow': 10}
    elif db_type == 'postgresql':
        # PostgreSQL handles high concurrency
        return {'pool_size': 20, 'max_overflow': 30}
```

### Query Optimization

- **Prepared Statements**: Parameterized queries for better performance
- **Batch Operations**: `executemany()` for bulk operations
- **Connection Reuse**: Pooling prevents connection overhead
- **Query Caching**: Table reflection caching reduces metadata queries

### Memory Management

- **Streaming Results**: Large result sets processed in chunks
- **Connection Cleanup**: Automatic connection disposal
- **Resource Monitoring**: Memory and connection tracking
- **Lazy Loading**: Tables reflected only when needed

## 🛡️ Security & Reliability

### SQL Injection Prevention
```python
# Safe parameterized queries
def execute_query(self, query: str, params: Optional[Dict] = None):
    with self.engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]

# Usage
results = db.execute_query(
    "SELECT * FROM users WHERE age > :min_age AND city = :city",
    {'min_age': 21, 'city': 'New York'}
)
```

### Error Handling & Recovery
```python
def execute_with_retry(engine: Engine, query: str, max_retries: int = 3):
    """Execute with exponential backoff retry logic"""
    for attempt in range(max_retries + 1):
        try:
            with engine.connect() as conn:
                return conn.execute(query)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
```

### Health Monitoring
```python
class DatabaseHealthChecker:
    def check_health(self) -> Dict[str, Any]:
        return {
            'timestamp': time.time(),
            'pool_status': 'healthy' if self._test_connection() else 'unhealthy',
            'active_connections': self.engine.pool.checkedout(),
            'total_connections': self.engine.pool.size(),
        }
```

## 🧪 Testing Strategy

### Unit Tests
- **Mock Engines**: SQLAlchemy engine mocking for isolated tests
- **In-Memory Databases**: SQLite `:memory:` for fast test execution
- **Parameter Testing**: pytest parametrization across database types

### Integration Tests
- **Real Databases**: Test against actual DuckDB, SQLite, PostgreSQL
- **Migration Tests**: Verify data integrity during cross-database transfers
- **Performance Tests**: Benchmark operations across database types

### Test Configuration
```python
@pytest.mark.parametrize("db_type", ["duckdb", "sqlite", "postgresql"])
def test_basic_operations(db_type, test_connection_params):
    config = DatabaseConfig.get_default_config(db_type)
    manager = DatabaseFactory.create_manager(**config)
    
    # Test universal operations
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    manager.bulk_insert_from_pandas(df, 'test_table')
    
    results = manager.execute_query("SELECT COUNT(*) FROM test_table")
    assert results[0]['count'] == 3
```

## 📈 Monitoring & Observability

### Performance Metrics
- Query execution times
- Connection pool utilization  
- Memory usage per operation
- Error rates and types

### Logging Integration
```python
import logging
logger = logging.getLogger(__name__)

def execute_query(self, query: str, params: Optional[Dict] = None):
    start_time = time.time()
    try:
        result = self._execute_query_impl(query, params)
        duration = time.time() - start_time
        logger.info(f"Query executed in {duration:.3f}s: {query[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Query failed: {query[:100]}... Error: {e}")
        raise
```

This implementation provides a robust, flexible, and performant database abstraction layer that maintains the power of each database engine while providing a unified interface for the Minotaur system.