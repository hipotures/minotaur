<!-- 
Documentation Status: CURRENT
Last Updated: 2025-07-02
Compatible with: SQLAlchemy refactoring implementation
Changes: Performance optimization guide for database abstraction layer
-->

# Database Performance Optimization Guide

## 🎯 Performance Overview

The Minotaur database abstraction layer is designed for optimal performance across different database engines, with specific optimizations for analytical workloads, MCTS operations, and large-scale feature engineering tasks.

### Performance Characteristics by Database Type

| Database | Strengths | Optimal Use Cases | Performance Focus |
|----------|-----------|-------------------|-------------------|
| **DuckDB** | Columnar analytics, Parquet I/O, Advanced sampling | Large datasets, Feature discovery, Analytics | Throughput, Memory efficiency |
| **SQLite** | Fast reads, Low overhead, Single-user | Development, Testing, Small datasets | Latency, Simplicity |
| **PostgreSQL** | ACID compliance, Concurrency, Extensions | Production, Multi-user, Enterprise | Reliability, Scalability |

## 🚀 Database-Specific Optimizations

### DuckDB Performance Tuning

#### Memory and Threading Configuration
```python
# Optimize DuckDB for analytical workloads
config = DatabaseConfig.get_default_config('duckdb', './analytics.duckdb')
db = DatabaseFactory.create_manager(**config)

# Configure memory and threads based on system resources
import os
cpu_count = os.cpu_count()
available_memory_gb = 16  # Adjust based on available RAM

# Set memory limit (leave some for OS)
db.set_memory_limit(available_memory_gb - 4)

# Use all available CPU cores for analytics
db.set_threads(cpu_count)

# Enable advanced optimizations
db.execute_query("SET enable_object_cache = true")
db.execute_query("SET force_compression = 'zstd'")
db.execute_query("SET enable_progress_bar = false")
db.execute_query("SET preserve_insertion_order = false")  # Better performance for analytics
```

#### Optimized Data Loading
```python
# High-performance Parquet loading
def load_large_dataset_optimized(db, parquet_files, table_name):
    """Load multiple Parquet files with optimal performance"""
    
    if len(parquet_files) == 1:
        # Single file - direct load
        db.read_parquet(parquet_files[0], table_name)
    else:
        # Multiple files - use UNION ALL for parallel loading
        file_selects = [f"SELECT * FROM read_parquet('{file}')" for file in parquet_files]
        union_query = " UNION ALL ".join(file_selects)
        
        db.execute_query(f"""
            CREATE TABLE {table_name} AS 
            {union_query}
        """)
    
    # Add indexes for common query patterns
    db.execute_query(f"CREATE INDEX idx_{table_name}_id ON {table_name}(id)")

# Usage
parquet_files = ['/data/part1.parquet', '/data/part2.parquet', '/data/part3.parquet']
load_large_dataset_optimized(db, parquet_files, 'large_dataset')
```

#### Advanced Sampling Optimization
```python
class OptimizedSampler:
    """High-performance sampling strategies for different scenarios"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def sample_for_mcts(self, table_name, sample_size, seed=42):
        """Optimized sampling for MCTS feature discovery"""
        
        # For large tables, use reservoir sampling
        total_rows = self.db.count_rows(table_name)
        
        if total_rows > 1000000:  # Large table
            # Use DuckDB's efficient reservoir sampling
            query = self.db.sample_reservoir(table_name, sample_size, seed)
        elif total_rows > 100000:  # Medium table
            # Use Bernoulli sampling for speed
            percentage = (sample_size / total_rows) * 100
            query = self.db.sample_bernoulli(table_name, percentage, seed)
        else:  # Small table
            # Simple random sampling
            query = f"""
                SELECT * FROM {table_name} 
                ORDER BY random() 
                LIMIT {sample_size}
            """
        
        return self.db.execute_query_df(query)
    
    def stratified_sample(self, table_name, stratify_column, samples_per_stratum):
        """Efficient stratified sampling"""
        
        query = f"""
        SELECT * FROM (
            SELECT *, 
                   ROW_NUMBER() OVER (
                       PARTITION BY {stratify_column} 
                       ORDER BY random()
                   ) as rn
            FROM {table_name}
        ) 
        WHERE rn <= {samples_per_stratum}
        """
        
        return self.db.execute_query_df(query)

# Usage
sampler = OptimizedSampler(db)
sample_df = sampler.sample_for_mcts('features_table', 5000)
```

#### Query Optimization Patterns
```python
# Efficient aggregation queries
def optimized_feature_aggregation(db, table_name):
    """Optimized patterns for feature aggregation"""
    
    # Use columnar operations for better performance
    agg_query = f"""
    SELECT 
        -- Window functions are efficient in DuckDB
        id,
        value,
        category,
        AVG(value) OVER (PARTITION BY category) as category_avg,
        value - AVG(value) OVER (PARTITION BY category) as value_deviation,
        
        -- Use efficient ranking
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as category_rank,
        
        -- Efficient quantile calculations
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) OVER (PARTITION BY category) as category_median
        
    FROM {table_name}
    """
    
    return db.execute_query_df(agg_query)

# Use efficient data types for better performance
def create_optimized_table(db, table_name):
    """Create table with optimal data types"""
    
    create_sql = f"""
    CREATE TABLE {table_name} (
        id INTEGER,
        timestamp TIMESTAMP,
        category VARCHAR,
        value DOUBLE,
        amount DECIMAL(18,2),
        flags BOOLEAN,
        metadata JSON
    )
    """
    
    db.execute_query(create_sql)
```

### SQLite Performance Tuning

#### Configuration for Development/Testing
```python
def optimize_sqlite_for_development(db):
    """Optimize SQLite for fast development and testing"""
    
    # Enable Write-Ahead Logging for better concurrency
    db.set_journal_mode('WAL')
    
    # Optimize for speed over safety (development only!)
    db.set_synchronous('NORMAL')  # or 'OFF' for maximum speed
    
    # Increase cache size significantly
    db.set_cache_size(-128000)  # 128MB cache
    
    # Store temporary tables in memory
    db.pragma_query('temp_store', 'MEMORY')
    
    # Use memory mapping for large databases
    db.pragma_query('mmap_size', str(256 * 1024 * 1024))  # 256MB
    
    # Optimize page size for SSD storage
    db.pragma_query('page_size', '4096')
    
    # Enable automatic indexing for better query performance
    db.pragma_query('automatic_index', 'ON')

# Usage
optimize_sqlite_for_development(db)
```

#### Efficient Testing Patterns
```python
class SQLiteTestOptimizer:
    """Optimizations specifically for test environments"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def setup_fast_testing(self):
        """Configure SQLite for maximum test speed"""
        
        # Use in-memory database for fastest tests
        self.db.pragma_query('journal_mode', 'MEMORY')
        self.db.pragma_query('synchronous', 'OFF')
        self.db.pragma_query('cache_size', '-200000')  # 200MB cache
        self.db.pragma_query('temp_store', 'MEMORY')
        
        # Disable fsync for speed (test environment only)
        self.db.pragma_query('locking_mode', 'EXCLUSIVE')
    
    def bulk_insert_optimized(self, table_name, data_list):
        """Optimized bulk insert for tests"""
        
        # Begin transaction for bulk operations
        df = pd.DataFrame(data_list)
        
        # Use pandas to_sql with optimized settings
        df.to_sql(
            table_name, 
            self.db.engine, 
            if_exists='append', 
            index=False,
            method='multi'  # Use executemany for better performance
        )
    
    def parallel_test_setup(self, test_data_sets):
        """Set up multiple test datasets efficiently"""
        
        # Use temporary tables for test isolation
        for test_name, data in test_data_sets.items():
            temp_table = f"temp_{test_name}"
            self.bulk_insert_optimized(temp_table, data)

# Usage in tests
test_optimizer = SQLiteTestOptimizer(db)
test_optimizer.setup_fast_testing()
```

### PostgreSQL Performance Tuning

#### Production Configuration
```python
def optimize_postgresql_for_production(db):
    """Enterprise-grade PostgreSQL optimizations"""
    
    # Create performance-critical extensions
    db.create_extension('pg_stat_statements')  # Query performance monitoring
    db.create_extension('pg_trgm')             # Faster text search
    
    # Create optimized indexes for common query patterns
    common_indexes = [
        # MCTS session queries
        "CREATE INDEX CONCURRENTLY idx_sessions_start_time ON sessions(start_time DESC)",
        "CREATE INDEX CONCURRENTLY idx_sessions_status ON sessions(status) WHERE status = 'running'",
        
        # Feature cache queries
        "CREATE INDEX CONCURRENTLY idx_features_score ON features_cache(evaluation_score DESC NULLS LAST)",
        "CREATE INDEX CONCURRENTLY idx_features_created ON features_cache(created_at DESC)",
        
        # Exploration history queries
        "CREATE INDEX CONCURRENTLY idx_exploration_session_iter ON exploration_history(session_id, iteration)",
        
        # Partial indexes for active data
        "CREATE INDEX CONCURRENTLY idx_active_sessions ON sessions(id) WHERE status IN ('running', 'paused')",
    ]
    
    for index_sql in common_indexes:
        try:
            db.execute_query(index_sql)
            logger.info(f"Created index: {index_sql}")
        except Exception as e:
            logger.warning(f"Index creation failed (may already exist): {e}")

def setup_connection_pooling(config):
    """Optimize connection pooling for production load"""
    
    # Calculate pool size based on expected concurrent users
    expected_concurrent_users = 50
    connections_per_user = 3
    
    optimal_pool_size = min(expected_concurrent_users * connections_per_user, 100)
    optimal_overflow = optimal_pool_size // 2
    
    config['connection_params']['engine_args'].update({
        'pool_size': optimal_pool_size,
        'max_overflow': optimal_overflow,
        'pool_pre_ping': True,
        'pool_recycle': 3600,  # 1 hour
        'pool_timeout': 30,
        'connect_args': {
            'connect_timeout': 10,
            'application_name': 'minotaur-mcts',
            'options': '-c statement_timeout=300000'  # 5 minute query timeout
        }
    })
    
    return config
```

#### High-Performance Bulk Operations
```python
class PostgreSQLBulkOptimizer:
    """Optimized bulk operations for PostgreSQL"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def bulk_load_from_csv(self, table_name, csv_path, batch_size=100000):
        """High-performance CSV loading using COPY command"""
        
        # COPY is much faster than INSERT for large datasets
        self.db.copy_from_csv(table_name, csv_path, header=True)
        
        # Update table statistics after bulk load
        self.db.vacuum_analyze(table_name)
    
    def bulk_feature_insertion(self, features_data, batch_size=10000):
        """Optimized feature cache insertion"""
        
        # Prepare data in batches
        for i in range(0, len(features_data), batch_size):
            batch = features_data[i:i + batch_size]
            
            # Use pandas for efficient bulk insert
            df = pd.DataFrame(batch)
            df.to_sql(
                'features_cache', 
                self.db.engine, 
                if_exists='append', 
                index=False,
                method='multi'
            )
        
        # Update statistics after bulk insertion
        self.db.vacuum_analyze('features_cache')
    
    def efficient_aggregation_queries(self, table_name):
        """Optimized aggregation patterns for PostgreSQL"""
        
        # Use PostgreSQL-specific optimizations
        query = f"""
        WITH feature_stats AS (
            SELECT 
                feature_hash,
                feature_name,
                evaluation_score,
                -- Use efficient window functions
                ROW_NUMBER() OVER (ORDER BY evaluation_score DESC) as score_rank,
                PERCENT_RANK() OVER (ORDER BY evaluation_score) as score_percentile,
                -- Efficient aggregation
                COUNT(*) OVER () as total_features
            FROM {table_name}
            WHERE evaluation_score IS NOT NULL
        )
        SELECT 
            feature_hash,
            feature_name,
            evaluation_score,
            score_rank,
            score_percentile
        FROM feature_stats 
        WHERE score_percentile >= 0.95  -- Top 5% of features
        ORDER BY evaluation_score DESC
        """
        
        return self.db.execute_query_df(query)

# Usage
pg_optimizer = PostgreSQLBulkOptimizer(db)
top_features = pg_optimizer.efficient_aggregation_queries('features_cache')
```

## ⚡ Query Optimization Strategies

### Universal Query Patterns

#### Efficient Sampling Across All Databases
```python
class UniversalQueryOptimizer:
    """Database-agnostic query optimization patterns"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.db_type = getattr(db_manager, 'db_type', 'unknown')
    
    def adaptive_sampling(self, table_name, sample_size, seed=42):
        """Use optimal sampling method based on database type"""
        
        total_rows = self.db.count_rows(table_name)
        sample_ratio = sample_size / total_rows
        
        if self.db_type == 'duckdb':
            if sample_ratio < 0.1:  # Small sample percentage
                return self.db.sample_reservoir(table_name, sample_size, seed)
            else:  # Large sample percentage
                percentage = sample_ratio * 100
                return self.db.sample_bernoulli(table_name, percentage, seed)
                
        elif self.db_type == 'postgresql':
            # Use PostgreSQL's TABLESAMPLE for large tables
            if total_rows > 100000:
                percentage = min(sample_ratio * 100, 100)
                return self.db.sample_tablesample(table_name, percentage)
            else:
                # Fall back to ORDER BY RANDOM for small tables
                return f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
                
        else:  # SQLite and others
            return f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
    
    def efficient_pagination(self, table_name, page_size, offset, order_column='id'):
        """Efficient pagination that works across databases"""
        
        if self.db_type == 'postgresql':
            # Use PostgreSQL's efficient OFFSET/LIMIT with ordering
            return f"""
                SELECT * FROM {table_name} 
                ORDER BY {order_column} 
                LIMIT {page_size} OFFSET {offset}
            """
        else:
            # Use more efficient cursor-based pagination for others
            if offset == 0:
                return f"""
                    SELECT * FROM {table_name} 
                    ORDER BY {order_column} 
                    LIMIT {page_size}
                """
            else:
                # Use WHERE clause instead of OFFSET for better performance
                return f"""
                    SELECT * FROM {table_name} 
                    WHERE {order_column} > (
                        SELECT {order_column} 
                        FROM {table_name} 
                        ORDER BY {order_column} 
                        LIMIT 1 OFFSET {offset - 1}
                    )
                    ORDER BY {order_column} 
                    LIMIT {page_size}
                """
    
    def memory_efficient_aggregation(self, table_name, group_columns, agg_columns):
        """Memory-efficient aggregation for large datasets"""
        
        # Build dynamic aggregation query
        group_clause = ", ".join(group_columns)
        agg_clauses = []
        
        for col in agg_columns:
            agg_clauses.extend([
                f"COUNT({col}) as {col}_count",
                f"AVG({col}) as {col}_avg",
                f"MIN({col}) as {col}_min",
                f"MAX({col}) as {col}_max",
                f"STDDEV({col}) as {col}_stddev"
            ])
        
        agg_clause = ", ".join(agg_clauses)
        
        if self.db_type == 'duckdb':
            # Use DuckDB's efficient columnar aggregation
            query = f"""
                SELECT {group_clause}, {agg_clause}
                FROM {table_name}
                GROUP BY {group_clause}
                ORDER BY COUNT(*) DESC
            """
        else:
            # Standard SQL aggregation with memory optimization
            query = f"""
                SELECT {group_clause}, {agg_clause}
                FROM {table_name}
                GROUP BY {group_clause}
                HAVING COUNT(*) >= 10  -- Filter small groups
                ORDER BY COUNT(*) DESC
            """
        
        return self.db.execute_query_df(query)

# Usage
optimizer = UniversalQueryOptimizer(db)
sample_query = optimizer.adaptive_sampling('large_table', 5000)
sample_df = db.execute_query_df(sample_query)
```

### MCTS-Specific Optimizations

#### Session Management Queries
```python
class MCTSQueryOptimizer:
    """Optimized queries for MCTS operations"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def get_top_features_efficiently(self, limit=100, min_score=0.0):
        """Efficiently retrieve top-performing features"""
        
        # Use covering index to avoid table lookups
        query = """
        SELECT feature_hash, feature_name, evaluation_score, node_depth
        FROM features_cache 
        WHERE evaluation_score >= :min_score
        ORDER BY evaluation_score DESC, created_at DESC
        LIMIT :limit
        """
        
        return self.db.execute_query(query, {'min_score': min_score, 'limit': limit})
    
    def get_exploration_progress(self, session_id, batch_size=1000):
        """Efficiently track exploration progress"""
        
        # Use window functions for progress calculation
        query = """
        WITH progress_stats AS (
            SELECT 
                iteration,
                total_evaluations,
                best_score,
                LAG(best_score) OVER (ORDER BY iteration) as prev_best_score,
                ROW_NUMBER() OVER (ORDER BY iteration DESC) as recency_rank
            FROM exploration_history 
            WHERE session_id = :session_id
            ORDER BY iteration DESC
            LIMIT :batch_size
        )
        SELECT 
            iteration,
            total_evaluations,
            best_score,
            CASE 
                WHEN prev_best_score IS NULL THEN 0
                ELSE best_score - prev_best_score 
            END as score_improvement
        FROM progress_stats
        ORDER BY iteration ASC
        """
        
        return self.db.execute_query_df(query, {
            'session_id': session_id, 
            'batch_size': batch_size
        })
    
    def efficient_feature_caching(self, features_batch):
        """Efficiently cache multiple features with conflict resolution"""
        
        # Use UPSERT-style operation for better performance
        if self.db.db_type == 'postgresql':
            # PostgreSQL ON CONFLICT
            base_query = """
            INSERT INTO features_cache (feature_hash, feature_name, feature_data, evaluation_score, node_depth)
            VALUES (%(feature_hash)s, %(feature_name)s, %(feature_data)s, %(evaluation_score)s, %(node_depth)s)
            ON CONFLICT (feature_hash) 
            DO UPDATE SET 
                evaluation_score = EXCLUDED.evaluation_score,
                node_depth = EXCLUDED.node_depth
            """
        else:
            # SQLite/DuckDB - use INSERT OR REPLACE
            base_query = """
            INSERT OR REPLACE INTO features_cache 
            (feature_hash, feature_name, feature_data, evaluation_score, node_depth)
            VALUES (?, ?, ?, ?, ?)
            """
        
        # Execute in batches for better performance
        with self.db.engine.begin() as conn:
            for feature in features_batch:
                conn.execute(base_query, feature)

# Usage
mcts_optimizer = MCTSQueryOptimizer(db)
top_features = mcts_optimizer.get_top_features_efficiently(limit=50, min_score=0.8)
```

## 📊 Performance Monitoring

### Real-Time Performance Tracking

```python
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    execution_time: float
    rows_affected: int
    timestamp: float
    database_type: str

class PerformanceMonitor:
    """Real-time database performance monitoring"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.metrics_history: List[QueryMetrics] = []
        self.query_counts = defaultdict(int)
        self.total_query_time = defaultdict(float)
    
    def time_query(self, query_type: str):
        """Context manager for timing queries"""
        
        class QueryTimer:
            def __init__(self, monitor, query_type):
                self.monitor = monitor
                self.query_type = query_type
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:  # Only record if no exception
                    execution_time = time.time() - self.start_time
                    
                    metrics = QueryMetrics(
                        query_type=self.query_type,
                        execution_time=execution_time,
                        rows_affected=0,  # Set by caller if available
                        timestamp=time.time(),
                        database_type=getattr(self.monitor.db, 'db_type', 'unknown')
                    )
                    
                    self.monitor._record_metrics(metrics)
        
        return QueryTimer(self, query_type)
    
    def _record_metrics(self, metrics: QueryMetrics):
        """Record query metrics"""
        self.metrics_history.append(metrics)
        self.query_counts[metrics.query_type] += 1
        self.total_query_time[metrics.query_type] += metrics.execution_time
        
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_performance_summary(self, minutes: int = 10) -> Dict:
        """Get performance summary for last N minutes"""
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'status': 'no_data', 'period_minutes': minutes}
        
        # Calculate statistics
        query_types = defaultdict(list)
        for metric in recent_metrics:
            query_types[metric.query_type].append(metric.execution_time)
        
        summary = {
            'period_minutes': minutes,
            'total_queries': len(recent_metrics),
            'query_types': len(query_types),
            'database_type': recent_metrics[0].database_type if recent_metrics else 'unknown',
            'performance_by_type': {}
        }
        
        for query_type, times in query_types.items():
            summary['performance_by_type'][query_type] = {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
                'min_time': min(times),
                'total_time': sum(times)
            }
        
        return summary
    
    def detect_performance_issues(self) -> List[Dict]:
        """Detect potential performance problems"""
        
        issues = []
        summary = self.get_performance_summary(minutes=5)
        
        if summary.get('status') == 'no_data':
            return issues
        
        # Check for slow queries
        for query_type, stats in summary['performance_by_type'].items():
            if stats['avg_time'] > 1.0:  # Queries taking more than 1 second
                issues.append({
                    'type': 'slow_queries',
                    'query_type': query_type,
                    'avg_time': stats['avg_time'],
                    'count': stats['count'],
                    'severity': 'high' if stats['avg_time'] > 5.0 else 'medium'
                })
        
        # Check for high query volume
        total_queries = summary['total_queries']
        queries_per_minute = total_queries / summary['period_minutes']
        
        if queries_per_minute > 100:  # More than 100 queries per minute
            issues.append({
                'type': 'high_query_volume',
                'queries_per_minute': queries_per_minute,
                'total_queries': total_queries,
                'severity': 'medium'
            })
        
        return issues

# Usage example
monitor = PerformanceMonitor(db)

# Time different operations
with monitor.time_query('feature_sampling'):
    sample_df = db.execute_query_df("SELECT * FROM features_table LIMIT 1000")

with monitor.time_query('feature_aggregation'):
    agg_df = db.execute_query_df("""
        SELECT category, COUNT(*), AVG(value) 
        FROM features_table 
        GROUP BY category
    """)

# Get performance report
summary = monitor.get_performance_summary(minutes=10)
issues = monitor.detect_performance_issues()

print(f"Performance Summary: {summary}")
if issues:
    print(f"Performance Issues Detected: {issues}")
```

### Automated Performance Testing

```python
class PerformanceBenchmark:
    """Automated performance testing across database types"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_database_types(self, test_data_size=10000):
        """Compare performance across different database types"""
        
        database_configs = [
            DatabaseConfig.get_default_config('sqlite', ':memory:'),
            DatabaseConfig.get_default_config('duckdb', ':memory:'),
            # Add PostgreSQL if available
        ]
        
        test_data = self._generate_test_data(test_data_size)
        
        for config in database_configs:
            db_type = config['db_type']
            print(f"Benchmarking {db_type}...")
            
            try:
                db = DatabaseFactory.create_manager(**config)
                self.results[db_type] = self._run_benchmark_suite(db, test_data)
                db.close()
            except Exception as e:
                print(f"Error benchmarking {db_type}: {e}")
                self.results[db_type] = {'error': str(e)}
        
        return self.results
    
    def _generate_test_data(self, size):
        """Generate test data for benchmarking"""
        import random
        
        return [
            {
                'id': i,
                'category': f'cat_{i % 10}',
                'value': random.uniform(0, 100),
                'timestamp': f'2024-01-{(i % 28) + 1:02d}',
                'text_field': f'text_data_{i}' * 3
            }
            for i in range(size)
        ]
    
    def _run_benchmark_suite(self, db, test_data):
        """Run comprehensive benchmark suite"""
        
        results = {}
        
        # Test 1: Bulk insert performance
        start_time = time.time()
        df = pd.DataFrame(test_data)
        db.bulk_insert_from_pandas(df, 'benchmark_table', if_exists='replace')
        results['bulk_insert_time'] = time.time() - start_time
        
        # Test 2: Simple query performance
        start_time = time.time()
        db.execute_query("SELECT COUNT(*) FROM benchmark_table")
        results['simple_query_time'] = time.time() - start_time
        
        # Test 3: Aggregation performance
        start_time = time.time()
        db.execute_query_df("""
            SELECT category, COUNT(*) as count, AVG(value) as avg_value
            FROM benchmark_table 
            GROUP BY category
        """)
        results['aggregation_time'] = time.time() - start_time
        
        # Test 4: Join performance (self-join)
        start_time = time.time()
        db.execute_query_df("""
            SELECT a.id, a.category, b.value
            FROM benchmark_table a
            JOIN benchmark_table b ON a.category = b.category
            LIMIT 1000
        """)
        results['join_time'] = time.time() - start_time
        
        # Test 5: Sampling performance (if supported)
        start_time = time.time()
        try:
            if hasattr(db, 'sample_reservoir'):
                query = db.sample_reservoir('benchmark_table', 1000)
                db.execute_query_df(query)
            else:
                db.execute_query_df("SELECT * FROM benchmark_table ORDER BY RANDOM() LIMIT 1000")
            results['sampling_time'] = time.time() - start_time
        except Exception as e:
            results['sampling_time'] = f"Error: {e}"
        
        return results
    
    def generate_report(self):
        """Generate performance comparison report"""
        
        if not self.results:
            return "No benchmark results available"
        
        report = ["Database Performance Benchmark Report", "=" * 50]
        
        # Summary table
        metrics = ['bulk_insert_time', 'simple_query_time', 'aggregation_time', 'join_time', 'sampling_time']
        
        # Header
        header = f"{'Metric':<20} " + " ".join(f"{db_type:<12}" for db_type in self.results.keys())
        report.append(header)
        report.append("-" * len(header))
        
        # Data rows
        for metric in metrics:
            row = f"{metric:<20} "
            for db_type, results in self.results.items():
                if 'error' in results:
                    value = 'ERROR'
                elif metric in results:
                    if isinstance(results[metric], (int, float)):
                        value = f"{results[metric]:.4f}s"
                    else:
                        value = str(results[metric])[:10]
                else:
                    value = 'N/A'
                row += f"{value:<12} "
            report.append(row)
        
        # Performance ranking
        report.append("\nPerformance Ranking (lower is better):")
        for metric in metrics:
            times = []
            for db_type, results in self.results.items():
                if 'error' not in results and metric in results and isinstance(results[metric], (int, float)):
                    times.append((db_type, results[metric]))
            
            if times:
                times.sort(key=lambda x: x[1])
                ranking = " > ".join([f"{db}({time:.4f}s)" for db, time in times])
                report.append(f"  {metric}: {ranking}")
        
        return "\n".join(report)

# Run benchmark
benchmark = PerformanceBenchmark()
results = benchmark.benchmark_database_types(test_data_size=50000)
report = benchmark.generate_report()
print(report)
```

## 🎯 Best Practices Summary

### General Performance Guidelines

1. **Choose the Right Database for the Task**
   - **Development/Testing**: SQLite (fast, simple, no setup)
   - **Analytics/MCTS**: DuckDB (columnar, advanced sampling, Parquet)
   - **Production/Multi-user**: PostgreSQL (ACID, concurrency, reliability)

2. **Connection Management**
   - Use appropriate pool sizes for each database type
   - Enable connection pre-ping for reliability
   - Set reasonable connection timeouts

3. **Query Optimization**
   - Use parameterized queries for security and performance
   - Leverage database-specific features (sampling, bulk operations)
   - Create appropriate indexes for common query patterns

4. **Data Loading**
   - Use bulk operations (COPY, bulk_insert_from_pandas)
   - Process large datasets in batches
   - Use appropriate data types for storage efficiency

5. **Monitoring**
   - Track query performance metrics
   - Monitor connection pool utilization
   - Set up automated health checks

### MCTS-Specific Optimizations

1. **Feature Caching**: Use UPSERT patterns for efficient feature storage
2. **Sampling**: Leverage database-specific sampling for better performance
3. **Session Management**: Use efficient queries for progress tracking
4. **Index Strategy**: Create covering indexes for common MCTS queries

### Memory Management

1. **DuckDB**: Set appropriate memory limits and thread counts
2. **SQLite**: Increase cache size and use WAL mode
3. **PostgreSQL**: Configure shared_buffers and connection pools appropriately

This comprehensive performance guide ensures optimal database performance across all supported engines while maintaining the flexibility of the abstraction layer.