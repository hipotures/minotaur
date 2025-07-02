"""
Base common queries used across all database types
"""

from sqlalchemy import text, select, func
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseQueries:
    """Common database queries that work across all supported database types"""
    
    @staticmethod
    def get_table_row_count(engine: Engine, table_name: str) -> int:
        """
        Get row count for a table (works across all databases)
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            Number of rows in table
        """
        query = text(f"SELECT COUNT(*) FROM {table_name}")
        with engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar()
    
    @staticmethod
    def get_table_column_info(engine: Engine, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table (database-specific implementations)
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            List of column information dictionaries
        """
        # Use database-specific DESCRIBE/PRAGMA queries
        db_type = engine.dialect.name
        
        if db_type == 'duckdb':
            query = text(f"DESCRIBE {table_name}")
        elif db_type == 'sqlite':
            query = text(f"PRAGMA table_info({table_name})")
        elif db_type == 'postgresql':
            query = text(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
        else:
            # Fallback - try standard SQL
            query = text(f"DESCRIBE {table_name}")
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]
    
    @staticmethod
    def check_table_exists(engine: Engine, table_name: str) -> bool:
        """
        Check if table exists (works across all databases)
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        db_type = engine.dialect.name
        
        if db_type == 'sqlite':
            query = text("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name = :table_name
            """)
        elif db_type == 'postgresql':
            query = text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = :table_name
            """)
        else:
            # DuckDB and others - use SHOW TABLES approach
            try:
                query = text("SHOW TABLES")
                with engine.connect() as conn:
                    result = conn.execute(query)
                    tables = [row[0] for row in result]
                    return table_name in tables
            except:
                # Fallback to trying to query the table
                try:
                    query = text(f"SELECT 1 FROM {table_name} LIMIT 1")
                    with engine.connect() as conn:
                        conn.execute(query)
                    return True
                except:
                    return False
        
        with engine.connect() as conn:
            result = conn.execute(query, {'table_name': table_name})
            return result.scalar() > 0
    
    @staticmethod
    def get_all_table_names(engine: Engine) -> List[str]:
        """
        Get list of all table names (database-specific)
        
        Args:
            engine: SQLAlchemy engine
            
        Returns:
            List of table names
        """
        db_type = engine.dialect.name
        
        if db_type == 'sqlite':
            query = text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
        elif db_type == 'postgresql':
            query = text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
        else:
            # DuckDB and others
            query = text("SHOW TABLES")
        
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]
    
    @staticmethod
    def get_table_size_estimate(engine: Engine, table_name: str) -> Dict[str, Any]:
        """
        Get table size estimate (varies by database)
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            
        Returns:
            Dictionary with size information
        """
        size_info = {
            'table_name': table_name,
            'row_count': 0,
            'size_bytes': None,
            'size_mb': None
        }
        
        # Get row count (universal)
        size_info['row_count'] = BaseQueries.get_table_row_count(engine, table_name)
        
        # Get size information (database-specific)
        db_type = engine.dialect.name
        
        try:
            if db_type == 'postgresql':
                query = text(f"SELECT pg_total_relation_size('{table_name}')")
                with engine.connect() as conn:
                    size_bytes = conn.execute(query).scalar()
                    size_info['size_bytes'] = size_bytes
                    size_info['size_mb'] = round(size_bytes / 1024 / 1024, 2)
            
            elif db_type == 'sqlite':
                # SQLite doesn't have built-in table size queries
                # Estimate based on page count
                query = text(f"PRAGMA page_count")
                with engine.connect() as conn:
                    page_count = conn.execute(query).scalar()
                    # Rough estimate: assume average page utilization
                    estimated_bytes = page_count * 1024 * 0.7  # 70% utilization
                    size_info['size_bytes'] = int(estimated_bytes)
                    size_info['size_mb'] = round(estimated_bytes / 1024 / 1024, 2)
            
            # DuckDB doesn't have built-in size queries, leave as None
            
        except Exception as e:
            logger.debug(f"Could not get size information for {table_name}: {e}")
        
        return size_info
    
    @staticmethod
    def sample_table_rows(engine: Engine, table_name: str, sample_size: int, 
                         seed: Optional[int] = None) -> str:
        """
        Generate sampling query (database-specific)
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table to sample
            sample_size: Number of rows to sample
            seed: Random seed (if supported)
            
        Returns:
            SQL query string for sampling
        """
        db_type = engine.dialect.name
        
        if db_type == 'duckdb':
            if seed is not None:
                return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size}) REPEATABLE ({seed})"
            else:
                return f"SELECT * FROM {table_name} TABLESAMPLE RESERVOIR({sample_size})"
        
        elif db_type == 'postgresql':
            # PostgreSQL supports TABLESAMPLE
            percentage = min(100, sample_size * 100.0 / BaseQueries.get_table_row_count(engine, table_name))
            return f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI ({percentage})"
        
        elif db_type == 'sqlite':
            # SQLite uses ORDER BY RANDOM()
            return f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
        
        else:
            # Fallback - use LIMIT with ORDER BY
            return f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
    
    @staticmethod
    def get_column_statistics(engine: Engine, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Get basic statistics for a column
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            column_name: Name of column
            
        Returns:
            Dictionary with column statistics
        """
        stats = {
            'column_name': column_name,
            'count': 0,
            'null_count': 0,
            'distinct_count': 0,
            'min_value': None,
            'max_value': None
        }
        
        try:
            # Basic count statistics
            query = text(f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT({column_name}) as non_null_count,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    MIN({column_name}) as min_value,
                    MAX({column_name}) as max_value
                FROM {table_name}
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query).fetchone()
                
                stats['count'] = result[0]
                stats['null_count'] = result[0] - result[1]
                stats['distinct_count'] = result[2]
                stats['min_value'] = result[3]
                stats['max_value'] = result[4]
        
        except Exception as e:
            logger.warning(f"Could not get statistics for {table_name}.{column_name}: {e}")
        
        return stats
    
    @staticmethod
    def create_index(engine: Engine, table_name: str, column_names: List[str], 
                    index_name: Optional[str] = None, unique: bool = False) -> str:
        """
        Create index on table columns
        
        Args:
            engine: SQLAlchemy engine
            table_name: Name of table
            column_names: List of column names for index
            index_name: Optional index name (auto-generated if None)
            unique: Whether to create unique index
            
        Returns:
            Name of created index
        """
        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        
        unique_clause = "UNIQUE " if unique else ""
        columns_clause = ", ".join(column_names)
        
        query = text(f"""
            CREATE {unique_clause}INDEX {index_name} 
            ON {table_name} ({columns_clause})
        """)
        
        with engine.begin() as conn:
            conn.execute(query)
        
        logger.info(f"Created {'unique ' if unique else ''}index {index_name} on {table_name}({columns_clause})")
        return index_name
    
    @staticmethod
    def drop_index(engine: Engine, index_name: str, if_exists: bool = True) -> None:
        """
        Drop index
        
        Args:
            engine: SQLAlchemy engine
            index_name: Name of index to drop
            if_exists: Use IF EXISTS clause if supported
        """
        db_type = engine.dialect.name
        
        if if_exists and db_type in ['sqlite', 'postgresql']:
            query = text(f"DROP INDEX IF EXISTS {index_name}")
        else:
            query = text(f"DROP INDEX {index_name}")
        
        with engine.begin() as conn:
            conn.execute(query)
        
        logger.info(f"Dropped index {index_name}")