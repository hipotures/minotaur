<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02
Compatible with: SQLAlchemy refactoring implementation
Changes: Operations guide for database configuration, management, and troubleshooting
-->

# Database Operations Guide

## 🎯 Configuration Management

The database abstraction layer provides flexible configuration options for different deployment scenarios and database types.

### Basic Configuration Structure

```python
# Minimal configuration (uses defaults)
config = {
    'db_type': 'duckdb',  # or 'sqlite', 'postgresql'
    'connection_params': {
        'database': './data/analytics.duckdb'
    }
}

# Advanced configuration with tuning
config = {
    'db_type': 'postgresql',
    'connection_params': {
        'user': 'minotaur_user',
        'password': os.getenv('DB_PASSWORD'),
        'host': 'db.example.com',
        'port': 5432,
        'database': 'minotaur_prod',
        'engine_args': {
            'pool_size': 20,
            'max_overflow': 30,
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'echo': False  # Set True for SQL query logging
        }
    }
}
```

### Environment-Specific Configurations

#### Development Environment
```python
# Fast local development with SQLite
development_config = {
    'db_type': 'sqlite',
    'connection_params': {
        'database': ':memory:',  # In-memory for tests
        'engine_args': {
            'echo': True,  # SQL logging for debugging
            'pool_pre_ping': True
        }
    }
}
```

#### Analytics Environment  
```python
# DuckDB for analytical workloads
analytics_config = {
    'db_type': 'duckdb',
    'connection_params': {
        'database': '/data/analytics/minotaur.duckdb',
        'engine_args': {
            'pool_size': 5,
            'max_overflow': 10,
            'echo': False
        }
    }
}
```

#### Production Environment
```python
# PostgreSQL for production with high availability
production_config = {
    'db_type': 'postgresql',
    'connection_params': {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'minotaur'),
        'engine_args': {
            'pool_size': 25,
            'max_overflow': 50,
            'pool_pre_ping': True,
            'pool_recycle': 1800,  # 30 minutes
            'pool_timeout': 30,
            'echo': False,
            'echo_pool': False,
            'connect_args': {
                'connect_timeout': 10,
                'application_name': 'minotaur-mcts'
            }
        }
    }
}
```

### Configuration Factory

```python
from src.database.config import DatabaseConfig

class ConfigurationManager:
    """Centralized configuration management"""
    
    @staticmethod
    def get_config(environment: str = None, db_type: str = None) -> Dict[str, Any]:
        """Get configuration based on environment and database type"""
        
        env = environment or os.getenv('MINOTAUR_ENV', 'development')
        
        if env == 'development':
            return ConfigurationManager._get_development_config(db_type)
        elif env == 'testing':
            return ConfigurationManager._get_testing_config(db_type)
        elif env == 'staging':
            return ConfigurationManager._get_staging_config(db_type)
        elif env == 'production':
            return ConfigurationManager._get_production_config(db_type)
        else:
            raise ValueError(f"Unknown environment: {env}")
    
    @staticmethod
    def _get_development_config(db_type: str = None) -> Dict[str, Any]:
        db_type = db_type or 'sqlite'
        if db_type == 'sqlite':
            return DatabaseConfig.get_default_config('sqlite', ':memory:')
        elif db_type == 'duckdb':
            return DatabaseConfig.get_default_config('duckdb', './dev_data.duckdb')
        else:
            return DatabaseConfig.get_default_config(db_type)
    
    @staticmethod
    def _get_production_config(db_type: str = None) -> Dict[str, Any]:
        db_type = db_type or 'postgresql'
        if db_type == 'postgresql':
            return {
                'db_type': 'postgresql',
                'connection_params': {
                    'user': os.getenv('DB_USER'),
                    'password': os.getenv('DB_PASSWORD'),
                    'host': os.getenv('DB_HOST'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'database': os.getenv('DB_NAME'),
                    'engine_args': {
                        'pool_size': 20,
                        'max_overflow': 40,
                        'pool_pre_ping': True,
                        'pool_recycle': 3600
                    }
                }
            }
        else:
            return DatabaseConfig.get_default_config(db_type)

# Usage
config = ConfigurationManager.get_config('production', 'postgresql')
db = DatabaseFactory.create_manager(**config)
```

## 🔧 Database-Specific Operations

### DuckDB Operations

#### Advanced Configuration
```python
# Create DuckDB manager with optimizations
config = DatabaseConfig.get_default_config('duckdb', './analytics.duckdb')
db = DatabaseFactory.create_manager(**config)

# Configure for large datasets
db.set_memory_limit(16)  # 16GB memory limit
db.set_threads(8)        # Use 8 CPU cores

# Enable performance optimizations
db.execute_query("SET enable_object_cache = true")
db.execute_query("SET force_compression = 'zstd'")
db.execute_query("SET enable_progress_bar = false")
```

#### File Operations
```python
# Parquet file operations
db.read_parquet('/data/large_dataset.parquet', 'analytics_table')
db.export_to_parquet('processed_results', '/output/results.parquet')

# CSV operations with auto-detection
db.read_csv_auto('/data/messy_data.csv', 'imported_data')
db.export_to_csv('clean_data', '/output/clean.csv', header=True)

# Multi-file operations
parquet_files = ['/data/part1.parquet', '/data/part2.parquet', '/data/part3.parquet']
for i, file in enumerate(parquet_files):
    table_name = f'partition_{i}'
    db.read_parquet(file, table_name)

# Union all partitions
union_query = " UNION ALL ".join([f"SELECT * FROM partition_{i}" for i in range(len(parquet_files))])
db.execute_query(f"CREATE TABLE combined AS {union_query}")
```

#### Advanced Sampling
```python
# Reservoir sampling for unbiased samples
sample_query = db.sample_reservoir('large_table', 10000, seed=42)
sample_df = db.execute_query_df(sample_query)

# Bernoulli sampling for percentage-based sampling
bernoulli_query = db.sample_bernoulli('large_table', 2.5, seed=42)  # 2.5%
bernoulli_df = db.execute_query_df(bernoulli_query)

# Stratified sampling by combining with window functions
stratified_query = """
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY RANDOM()) as rn
    FROM large_table
) WHERE rn <= 100  -- Top 100 from each category
"""
stratified_df = db.execute_query_df(stratified_query)
```

#### Database Attachment for Complex Operations
```python
# Attach external databases for cross-database operations
db.attach_database('/other/dataset.duckdb', 'external')

# Perform complex joins across databases
complex_query = """
SELECT 
    main.users.*,
    external.transactions.amount,
    external.transactions.timestamp
FROM main.users 
JOIN external.transactions ON main.users.id = external.transactions.user_id
WHERE external.transactions.timestamp > '2024-01-01'
"""
result = db.execute_query_df(complex_query)

# Clean up
db.detach_database('external')
```

### SQLite Operations

#### Performance Optimization
```python
# Create SQLite manager with performance tuning
config = DatabaseConfig.get_default_config('sqlite', './optimized.db')
db = DatabaseFactory.create_manager(**config)

# Apply performance optimizations
db.set_journal_mode('WAL')        # Write-Ahead Logging for concurrency
db.set_synchronous('NORMAL')      # Balance safety vs performance
db.set_cache_size(-64000)         # 64MB cache (negative = KB)

# Additional pragma optimizations
db.pragma_query('temp_store', 'MEMORY')          # Use memory for temp tables
db.pragma_query('mmap_size', str(256 * 1024 * 1024))  # 256MB memory mapping
db.pragma_query('page_size', '4096')             # Optimize page size
```

#### Multi-Database Operations
```python
# Attach multiple databases for complex operations
db.attach_database('./archive.db', 'archive')
db.attach_database('./staging.db', 'staging')

# Cross-database data movement
db.execute_query("""
    INSERT INTO main.processed_data 
    SELECT * FROM staging.raw_data 
    WHERE processed_at IS NULL
""")

# Archive old data
db.execute_query("""
    INSERT INTO archive.old_data 
    SELECT * FROM main.processed_data 
    WHERE created_at < date('now', '-1 year')
""")

# Clean up old data from main database
db.execute_query("""
    DELETE FROM main.processed_data 
    WHERE created_at < date('now', '-1 year')
""")

# Detach databases
db.detach_database('archive')
db.detach_database('staging')
```

#### Maintenance Operations
```python
# Regular maintenance for SQLite
def maintain_sqlite_database(db):
    """Perform SQLite maintenance operations"""
    
    # Analyze statistics for query optimization
    db.analyze_db()
    
    # Reclaim unused space and defragment
    db.vacuum_db()
    
    # Check database integrity
    integrity_result = db.pragma_query('integrity_check')
    if integrity_result != 'ok':
        logger.warning(f"Database integrity issue: {integrity_result}")
    
    # Get database statistics
    page_count = db.pragma_query('page_count')
    page_size = db.pragma_query('page_size')
    db_size_mb = (page_count * page_size) / (1024 * 1024)
    
    logger.info(f"Database size: {db_size_mb:.2f} MB ({page_count} pages)")

# Schedule maintenance
maintain_sqlite_database(db)
```

### PostgreSQL Operations

#### Production Configuration
```python
# Production PostgreSQL setup
config = {
    'db_type': 'postgresql',
    'connection_params': {
        'user': 'minotaur_app',
        'password': os.getenv('DB_PASSWORD'),
        'host': 'postgresql-primary.internal',
        'port': 5432,
        'database': 'minotaur_production',
        'engine_args': {
            'pool_size': 30,
            'max_overflow': 60,
            'pool_pre_ping': True,
            'pool_recycle': 1800,
            'pool_timeout': 30,
            'connect_args': {
                'connect_timeout': 10,
                'application_name': 'minotaur-analytics',
                'options': '-c statement_timeout=300000'  # 5 minute timeout
            }
        }
    }
}

db = DatabaseFactory.create_manager(**config)
```

#### Advanced Features
```python
# Create useful extensions
db.create_extension('pg_stat_statements')  # Query performance monitoring
db.create_extension('pg_trgm')             # Trigram similarity
db.create_extension('uuid-ossp')           # UUID generation

# High-performance bulk loading
db.copy_from_csv('bulk_import', '/data/large_file.csv', header=True, delimiter='|')

# Concurrent index creation (non-blocking)
db.create_index_concurrently('idx_performance_timestamp', 'events', 'timestamp, user_id')

# Advanced sampling with system columns
sample_query = db.sample_tablesample('large_table', 1.0, 'SYSTEM')  # 1% sample
sample_df = db.execute_query_df(sample_query)
```

#### Maintenance and Monitoring
```python
def maintain_postgresql_database(db):
    """PostgreSQL maintenance and monitoring"""
    
    # Update table statistics
    db.vacuum_analyze()  # Full database analyze
    
    # Vacuum specific large tables
    large_tables = ['events', 'transactions', 'user_activities']
    for table in large_tables:
        db.vacuum_analyze(table)
    
    # Monitor query performance
    slow_queries = db.execute_query_df("""
        SELECT query, calls, total_time, mean_time, rows
        FROM pg_stat_statements 
        WHERE mean_time > 1000  -- Queries taking > 1 second
        ORDER BY mean_time DESC 
        LIMIT 10
    """)
    
    if not slow_queries.empty:
        logger.warning(f"Found {len(slow_queries)} slow queries")
        for _, query in slow_queries.iterrows():
            logger.warning(f"Slow query ({query['mean_time']:.1f}ms): {query['query'][:100]}...")
    
    # Check database size
    db_size_result = db.execute_query(db.get_database_size())
    logger.info(f"Database size: {db_size_result[0]['pg_size_pretty']}")
    
    # Monitor connection pool
    connection_info = db.get_connection_info()
    logger.info(f"Connection pool: {connection_info}")

# Regular maintenance
maintain_postgresql_database(db)
```

## 📊 Monitoring and Health Checks

### Database Health Monitoring

```python
from src.database.utils import DatabaseHealthChecker

class DatabaseMonitoringService:
    """Comprehensive database monitoring service"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.health_checker = DatabaseHealthChecker(db_manager.engine)
        self.metrics = {
            'query_count': 0,
            'total_query_time': 0,
            'error_count': 0,
            'connection_errors': 0
        }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_info = self.health_checker.check_health()
        
        # Add application-specific checks
        health_info.update({
            'table_counts': self._check_table_counts(),
            'query_performance': self._test_query_performance(),
            'connection_pool': self._check_connection_pool(),
            'disk_space': self._check_disk_space()
        })
        
        return health_info
    
    def _check_table_counts(self) -> Dict[str, int]:
        """Verify expected tables exist and have data"""
        expected_tables = ['train_data', 'test_data', 'features_cache']
        table_counts = {}
        
        for table in expected_tables:
            try:
                if self.db_manager.table_exists(table):
                    count = self.db_manager.count_rows(table)
                    table_counts[table] = count
                else:
                    table_counts[table] = -1  # Table doesn't exist
            except Exception as e:
                logger.error(f"Error checking table {table}: {e}")
                table_counts[table] = -2  # Error checking table
        
        return table_counts
    
    def _test_query_performance(self) -> Dict[str, float]:
        """Test basic query performance"""
        test_queries = [
            "SELECT 1 AS test",
            "SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3) AS test"
        ]
        
        performance_results = {}
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                self.db_manager.execute_query(query)
                query_time = time.time() - start_time
                performance_results[f'test_query_{i+1}'] = query_time
            except Exception as e:
                logger.error(f"Test query {i+1} failed: {e}")
                performance_results[f'test_query_{i+1}'] = -1
        
        return performance_results
    
    def _check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool status"""
        try:
            pool = self.db_manager.engine.pool
            return {
                'pool_size': getattr(pool, 'size', 'N/A'),
                'checked_out': getattr(pool, 'checkedout', 'N/A'),
                'overflow': getattr(pool, 'overflow', 'N/A'),
                'checked_in': getattr(pool, 'checkedin', 'N/A')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            if hasattr(self.db_manager, 'db_path') and self.db_manager.db_path:
                db_path = Path(self.db_manager.db_path)
                if db_path.exists():
                    stat = os.statvfs(str(db_path.parent))
                    free_bytes = stat.f_bavail * stat.f_frsize
                    total_bytes = stat.f_blocks * stat.f_frsize
                    
                    return {
                        'free_gb': free_bytes / (1024**3),
                        'total_gb': total_bytes / (1024**3),
                        'usage_percent': ((total_bytes - free_bytes) / total_bytes) * 100
                    }
            return {'status': 'not_applicable'}
        except Exception as e:
            return {'error': str(e)}

# Usage
monitor = DatabaseMonitoringService(db)
health_report = monitor.perform_health_check()
print(json.dumps(health_report, indent=2))
```

### Performance Metrics Collection

```python
class PerformanceCollector:
    """Collect and analyze database performance metrics"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.metrics_history = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'database_type': getattr(self.db_manager, 'db_type', 'unknown'),
            'connection_pool': self._get_pool_metrics(),
            'query_performance': self._measure_query_performance(),
            'table_statistics': self._get_table_statistics(),
            'system_resources': self._get_system_metrics()
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def _measure_query_performance(self) -> Dict[str, float]:
        """Measure performance of common query patterns"""
        performance_tests = {
            'simple_select': "SELECT 1",
            'count_query': "SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3)",
            'aggregation': "SELECT MIN(1), MAX(1), AVG(1), SUM(1)"
        }
        
        results = {}
        for test_name, query in performance_tests.items():
            try:
                start_time = time.time()
                self.db_manager.execute_query(query)
                results[test_name] = time.time() - start_time
            except Exception as e:
                results[test_name] = -1
                logger.error(f"Performance test {test_name} failed: {e}")
        
        return results
    
    def _get_table_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tables"""
        tables = self.db_manager.get_table_names()
        stats = {}
        
        for table in tables[:10]:  # Limit to first 10 tables
            try:
                count = self.db_manager.count_rows(table)
                stats[table] = {'row_count': count}
            except Exception as e:
                stats[table] = {'error': str(e)}
        
        return stats
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {'status': 'psutil_not_available'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {'status': 'no_data', 'hours': hours}
        
        # Calculate averages
        query_times = []
        for metrics in recent_metrics:
            for test_name, duration in metrics['query_performance'].items():
                if duration > 0:  # Valid measurement
                    query_times.append(duration)
        
        return {
            'period_hours': hours,
            'measurements': len(recent_metrics),
            'avg_query_time': sum(query_times) / len(query_times) if query_times else 0,
            'max_query_time': max(query_times) if query_times else 0,
            'min_query_time': min(query_times) if query_times else 0,
            'first_measurement': recent_metrics[0]['timestamp'],
            'last_measurement': recent_metrics[-1]['timestamp']
        }

# Usage
collector = PerformanceCollector(db)

# Collect metrics periodically
for _ in range(5):
    metrics = collector.collect_metrics()
    time.sleep(10)  # Wait 10 seconds between collections

# Get summary
summary = collector.get_performance_summary(hours=1)
print(f"Performance summary: {summary}")
```

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### Connection Issues

```python
def diagnose_connection_issues(config):
    """Diagnose and resolve connection problems"""
    
    print(f"Diagnosing connection to {config['db_type']} database...")
    
    try:
        # Test basic connection
        db = DatabaseFactory.create_manager(**config)
        test_result = db.execute_query("SELECT 1 AS test")
        print("✅ Basic connection successful")
        
        # Test connection pool
        pool_info = db.get_connection_info()
        print(f"✅ Connection pool info: {pool_info}")
        
        # Test query execution
        performance_result = db.execute_query("SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2)")
        print("✅ Query execution successful")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Provide specific troubleshooting steps
        if config['db_type'] == 'postgresql':
            print("\nPostgreSQL Troubleshooting:")
            print("1. Check if PostgreSQL server is running")
            print("2. Verify connection parameters (host, port, database, user)")
            print("3. Check firewall settings")
            print("4. Verify user permissions")
            print("5. Check if database exists")
            
        elif config['db_type'] == 'duckdb':
            print("\nDuckDB Troubleshooting:")
            print("1. Check if file path is accessible")
            print("2. Verify directory permissions")
            print("3. Check available disk space")
            print("4. Ensure duckdb-engine is installed")
            
        elif config['db_type'] == 'sqlite':
            print("\nSQLite Troubleshooting:")
            print("1. Check file path permissions")
            print("2. Verify directory exists")
            print("3. Check available disk space")
            print("4. Ensure file is not locked by another process")
        
        return False

# Test different configurations
configs_to_test = [
    DatabaseConfig.get_default_config('sqlite', ':memory:'),
    DatabaseConfig.get_default_config('duckdb', './test.duckdb'),
]

for config in configs_to_test:
    diagnose_connection_issues(config)
```

#### Performance Issues

```python
def diagnose_performance_issues(db_manager):
    """Identify and resolve performance problems"""
    
    print("🔍 Diagnosing database performance issues...")
    
    # Test query performance
    slow_queries = []
    test_queries = [
        ("Simple SELECT", "SELECT 1"),
        ("COUNT query", "SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3)"),
        ("JOIN simulation", "SELECT a.val, b.val FROM (SELECT 1 as val) a JOIN (SELECT 2 as val) b ON 1=1")
    ]
    
    for name, query in test_queries:
        start_time = time.time()
        try:
            db_manager.execute_query(query)
            duration = time.time() - start_time
            
            if duration > 1.0:  # Query took more than 1 second
                slow_queries.append((name, duration))
                print(f"⚠️  Slow query detected: {name} took {duration:.3f}s")
            else:
                print(f"✅ {name}: {duration:.3f}s")
                
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Check connection pool
    try:
        pool_info = db_manager.get_connection_info()
        pool_size = pool_info.get('pool_size', 'unknown')
        checked_out = pool_info.get('pool_checked_out', 'unknown')
        
        if isinstance(checked_out, int) and isinstance(pool_size, int):
            if checked_out > pool_size * 0.8:  # More than 80% of pool used
                print(f"⚠️  High connection pool usage: {checked_out}/{pool_size}")
                print("   Consider increasing pool_size or max_overflow")
            else:
                print(f"✅ Connection pool usage normal: {checked_out}/{pool_size}")
    except Exception as e:
        print(f"❌ Could not check connection pool: {e}")
    
    # Database-specific recommendations
    db_type = getattr(db_manager, 'db_type', 'unknown')
    
    if slow_queries:
        print(f"\n🔧 Performance recommendations for {db_type}:")
        
        if db_type == 'sqlite':
            print("1. Enable WAL mode: db.set_journal_mode('WAL')")
            print("2. Increase cache size: db.set_cache_size(-64000)")
            print("3. Analyze tables: db.analyze_db()")
            print("4. Vacuum database: db.vacuum_db()")
            
        elif db_type == 'duckdb':
            print("1. Increase memory limit: db.set_memory_limit(8)")
            print("2. Use more threads: db.set_threads(4)")
            print("3. Enable object cache: SET enable_object_cache = true")
            print("4. Use column-wise operations for analytics")
            
        elif db_type == 'postgresql':
            print("1. Analyze tables: db.vacuum_analyze()")
            print("2. Create appropriate indexes")
            print("3. Increase shared_buffers in postgresql.conf")
            print("4. Monitor with pg_stat_statements")
    
    return len(slow_queries) == 0

# Run performance diagnosis
performance_ok = diagnose_performance_issues(db)
if not performance_ok:
    print("⚠️  Performance issues detected - see recommendations above")
```

#### Data Integrity Issues

```python
def diagnose_data_integrity(db_manager):
    """Check for data integrity problems"""
    
    print("🔍 Checking data integrity...")
    
    issues_found = []
    
    # Check table existence
    expected_tables = ['train_data', 'test_data', 'features_cache']
    existing_tables = db_manager.get_table_names()
    
    for table in expected_tables:
        if table not in existing_tables:
            issues_found.append(f"Missing table: {table}")
            print(f"❌ Missing table: {table}")
        else:
            print(f"✅ Table exists: {table}")
            
            # Check row counts
            try:
                count = db_manager.count_rows(table)
                if count == 0:
                    issues_found.append(f"Empty table: {table}")
                    print(f"⚠️  Empty table: {table}")
                else:
                    print(f"✅ Table {table} has {count} rows")
            except Exception as e:
                issues_found.append(f"Cannot count rows in {table}: {e}")
                print(f"❌ Cannot count rows in {table}: {e}")
    
    # Check for duplicate data
    for table in existing_tables:
        if table.endswith('_data'):  # Check main data tables
            try:
                # Simple duplicate check on ID column
                duplicate_check = db_manager.execute_query(f"""
                    SELECT COUNT(*) as total_rows, COUNT(DISTINCT id) as unique_ids
                    FROM {table}
                """)
                
                if duplicate_check:
                    total = duplicate_check[0]['total_rows']
                    unique = duplicate_check[0]['unique_ids']
                    
                    if total != unique:
                        issues_found.append(f"Duplicate IDs in {table}: {total} total, {unique} unique")
                        print(f"❌ Duplicate IDs in {table}: {total} total, {unique} unique")
                    else:
                        print(f"✅ No duplicate IDs in {table}")
                        
            except Exception as e:
                print(f"⚠️  Cannot check duplicates in {table}: {e}")
    
    # Check for data consistency
    if 'train_data' in existing_tables and 'test_data' in existing_tables:
        try:
            # Check if train and test have compatible structure
            train_sample = db_manager.execute_query("SELECT data FROM train_data LIMIT 1")
            test_sample = db_manager.execute_query("SELECT data FROM test_data LIMIT 1")
            
            if train_sample and test_sample:
                import json
                train_cols = set(json.loads(train_sample[0]['data']).keys())
                test_cols = set(json.loads(test_sample[0]['data']).keys())
                
                missing_in_test = train_cols - test_cols
                missing_in_train = test_cols - train_cols
                
                if missing_in_test:
                    issues_found.append(f"Columns in train but not test: {missing_in_test}")
                    print(f"⚠️  Columns in train but not test: {missing_in_test}")
                
                if missing_in_train:
                    issues_found.append(f"Columns in test but not train: {missing_in_train}")
                    print(f"⚠️  Columns in test but not train: {missing_in_train}")
                
                if not missing_in_test and not missing_in_train:
                    print("✅ Train and test data have compatible schemas")
                    
        except Exception as e:
            print(f"⚠️  Cannot check train/test consistency: {e}")
    
    if issues_found:
        print(f"\n🚨 Found {len(issues_found)} data integrity issues:")
        for issue in issues_found:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ No data integrity issues found")
        return True

# Run data integrity check
integrity_ok = diagnose_data_integrity(db)
```

### Automated Health Check Script

```python
#!/usr/bin/env python3
"""
Comprehensive database health check script
Usage: python health_check.py [config_file]
"""

import sys
import json
import time
from pathlib import Path

def comprehensive_health_check(config_path: str = None):
    """Run complete health check suite"""
    
    print("🔍 Starting comprehensive database health check...")
    print("=" * 60)
    
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Use default development config
        config = DatabaseConfig.get_default_config('sqlite', ':memory:')
    
    print(f"Database type: {config['db_type']}")
    print(f"Configuration: {config}")
    print("-" * 60)
    
    # Initialize components
    health_results = {
        'timestamp': time.time(),
        'database_type': config['db_type'],
        'tests': {}
    }
    
    try:
        db = DatabaseFactory.create_manager(**config)
        
        # Test 1: Connection Health
        print("1. Testing database connection...")
        connection_ok = diagnose_connection_issues(config)
        health_results['tests']['connection'] = connection_ok
        
        if not connection_ok:
            print("❌ Connection failed - aborting health check")
            return health_results
        
        # Test 2: Performance Check
        print("\n2. Testing query performance...")
        performance_ok = diagnose_performance_issues(db)
        health_results['tests']['performance'] = performance_ok
        
        # Test 3: Data Integrity
        print("\n3. Checking data integrity...")
        integrity_ok = diagnose_data_integrity(db)
        health_results['tests']['integrity'] = integrity_ok
        
        # Test 4: Resource Usage
        print("\n4. Checking resource usage...")
        monitor = DatabaseMonitoringService(db)
        health_info = monitor.perform_health_check()
        health_results['resource_info'] = health_info
        print(f"✅ Resource check completed")
        
        # Overall health status
        all_tests_passed = all(health_results['tests'].values())
        health_results['overall_status'] = 'healthy' if all_tests_passed else 'issues_detected'
        
        print("\n" + "=" * 60)
        print(f"🏥 HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {'✅ HEALTHY' if all_tests_passed else '⚠️  ISSUES DETECTED'}")
        print(f"Connection: {'✅' if health_results['tests']['connection'] else '❌'}")
        print(f"Performance: {'✅' if health_results['tests']['performance'] else '⚠️ '}")
        print(f"Data Integrity: {'✅' if health_results['tests']['integrity'] else '❌'}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        health_results['error'] = str(e)
        health_results['overall_status'] = 'error'
    
    return health_results

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    results = comprehensive_health_check(config_file)
    
    # Save results
    output_file = f"health_check_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Health check results saved to: {output_file}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'healthy':
        sys.exit(0)
    else:
        sys.exit(1)
```

This comprehensive operations guide provides the tools and procedures needed to effectively configure, monitor, and maintain the Minotaur database abstraction layer across different environments and database types.