#!/usr/bin/env python3
"""
DEPRECATED: Test script for persistent DuckDB implementation.

⚠️  This test uses legacy DuckDBDataManager. For new tests, use SQLAlchemyDataManager.

Tests:
1. Database path generation with MD5 hash
2. Database schema initialization
3. CSV data loading to database tables
4. Data sampling from database
5. Feature caching functionality
6. AutoGluon integration with DuckDB
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_path_generation():
    """Test database path generation with MD5 hash."""
    logger.info("🧪 Testing database path generation...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager
        
        # Test configuration
        config = {
            'autogluon': {
                'train_path': 'datasets/playground-series-s5e6/train.csv',
                'test_path': 'datasets/playground-series-s5e6/test.csv'
            },
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'persistent_storage': True,
                    'database_name': 'features_test.duckdb'
                }
            }
        }
        
        # Initialize manager (this should generate the database path)
        manager = DuckDBDataManager(config)
        
        # Check database path
        expected_hash_input = f"{config['autogluon']['train_path']}|{config['autogluon']['test_path']}"
        import hashlib
        expected_hash = hashlib.md5(expected_hash_input.encode()).hexdigest()
        
        logger.info(f"✅ Database path: {manager.db_path}")
        logger.info(f"✅ Expected hash: {expected_hash}")
        
        # Verify path contains hash
        if expected_hash in manager.db_path:
            logger.info("✅ Database path generation test passed")
            return True
        else:
            logger.error("❌ Database path generation test failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Database path generation test failed: {e}")
        return False

def test_database_schema():
    """Test database schema initialization."""
    logger.info("🧪 Testing database schema initialization...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager
        
        # Create temporary test configuration
        config = {
            'autogluon': {
                'train_path': '/tmp/test_train.csv',
                'test_path': '/tmp/test_test.csv'
            },
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'persistent_storage': True,
                    'database_name': 'features_test.duckdb'
                }
            }
        }
        
        # Create temporary CSV files for testing
        import pandas as pd
        
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'Nitrogen': [10, 20, 30, 40, 50],
            'Phosphorus': [5, 10, 15, 20, 25],
            'Potassium': [15, 25, 35, 45, 55],
            'pH': [6.0, 6.5, 7.0, 7.5, 8.0],
            'Rainfall': [100, 150, 200, 250, 300],
            'Temperature': [20, 25, 30, 35, 40],
            'Humidity': [60, 65, 70, 75, 80],
            'Fertilizer Name': ['A', 'B', 'C', 'D', 'E']
        })
        
        test_data.to_csv('/tmp/test_train.csv', index=False)
        test_data.to_csv('/tmp/test_test.csv', index=False)
        
        # Initialize manager
        with DuckDBDataManager(config) as manager:
            # Check if tables exist
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            try:
                # DuckDB syntax for listing tables
                tables_result = manager.connection.execute("SHOW TABLES").fetchall()
                table_names = [row[0] for row in tables_result]
                
                expected_tables = ['train_data', 'test_data', 'features_cache', 'feature_metadata']
                
                for table in expected_tables:
                    if table in table_names:
                        logger.info(f"✅ Table '{table}' exists")
                    else:
                        logger.error(f"❌ Table '{table}' missing")
                        return False
                
                logger.info("✅ Database schema test passed")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to check tables: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Database schema test failed: {e}")
        return False
    finally:
        # Cleanup temporary files
        for file in ['/tmp/test_train.csv', '/tmp/test_test.csv']:
            if os.path.exists(file):
                os.remove(file)

def test_data_loading():
    """Test CSV data loading to database."""
    logger.info("🧪 Testing CSV data loading...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'id': range(1, 101),
            'Nitrogen': [i * 0.5 for i in range(1, 101)],
            'Phosphorus': [i * 0.3 for i in range(1, 101)],
            'Potassium': [i * 0.7 for i in range(1, 101)],
            'pH': [6.0 + (i * 0.02) for i in range(1, 101)],
            'Rainfall': [100 + (i * 2) for i in range(1, 101)],
            'Temperature': [20 + (i * 0.2) for i in range(1, 101)],
            'Humidity': [60 + (i * 0.2) for i in range(1, 101)],
            'Fertilizer Name': [f'Fertilizer_{i%10}' for i in range(1, 101)]
        })
        
        test_data.to_csv('/tmp/test_train_large.csv', index=False)
        test_data.iloc[:50].to_csv('/tmp/test_test_large.csv', index=False)
        
        config = {
            'autogluon': {
                'train_path': '/tmp/test_train_large.csv',
                'test_path': '/tmp/test_test_large.csv'
            },
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'persistent_storage': True,
                    'database_name': 'features_test.duckdb'
                }
            }
        }
        
        # Initialize manager and load data
        with DuckDBDataManager(config) as manager:
            # Check data was loaded
            train_count = manager.connection.execute("SELECT COUNT(*) FROM train_data").fetchone()[0]
            test_count = manager.connection.execute("SELECT COUNT(*) FROM test_data").fetchone()[0]
            
            logger.info(f"✅ Loaded {train_count} training samples")
            logger.info(f"✅ Loaded {test_count} test samples")
            
            if train_count == 100 and test_count == 50:
                logger.info("✅ Data loading test passed")
                return True
            else:
                logger.error(f"❌ Data loading test failed: expected 100/50, got {train_count}/{test_count}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False
    finally:
        # Cleanup temporary files
        for file in ['/tmp/test_train_large.csv', '/tmp/test_test_large.csv']:
            if os.path.exists(file):
                os.remove(file)

def test_data_sampling():
    """Test data sampling from database."""
    logger.info("🧪 Testing data sampling from database...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager
        import pandas as pd
        
        # Use the data from previous test
        test_data = pd.DataFrame({
            'id': range(1, 101),
            'Nitrogen': [i * 0.5 for i in range(1, 101)],
            'Phosphorus': [i * 0.3 for i in range(1, 101)],
            'Potassium': [i * 0.7 for i in range(1, 101)],
            'pH': [6.0 + (i * 0.02) for i in range(1, 101)],
            'Rainfall': [100 + (i * 2) for i in range(1, 101)],
            'Temperature': [20 + (i * 0.2) for i in range(1, 101)],
            'Humidity': [60 + (i * 0.2) for i in range(1, 101)],
            'Fertilizer Name': [f'Fertilizer_{i%10}' for i in range(1, 101)]
        })
        
        test_data.to_csv('/tmp/test_train_sample.csv', index=False)
        
        config = {
            'autogluon': {
                'train_path': '/tmp/test_train_sample.csv',
                'test_path': None
            },
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'persistent_storage': True,
                    'database_name': 'features_test.duckdb'
                }
            }
        }
        
        # Test sampling
        with DuckDBDataManager(config) as manager:
            # Test absolute sampling
            sample_10 = manager.sample_dataset('/tmp/test_train_sample.csv', 10)
            logger.info(f"✅ Absolute sampling (10): {len(sample_10)} rows")
            
            # Test percentage sampling
            sample_20pct = manager.sample_dataset('/tmp/test_train_sample.csv', 0.2)
            logger.info(f"✅ Percentage sampling (20%): {len(sample_20pct)} rows")
            
            # Verify samples have correct structure
            expected_columns = ['id', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 
                              'Rainfall', 'Temperature', 'Humidity', 'Fertilizer Name']
            
            if list(sample_10.columns) == expected_columns and len(sample_10) == 10:
                logger.info("✅ Data sampling test passed")
                return True
            else:
                logger.error(f"❌ Sampling test failed: columns={list(sample_10.columns)}, rows={len(sample_10)}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Data sampling test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists('/tmp/test_train_sample.csv'):
            os.remove('/tmp/test_train_sample.csv')

def test_feature_caching():
    """Test feature caching functionality."""
    logger.info("🧪 Testing feature caching...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager
        import pandas as pd
        
        config = {
            'autogluon': {
                'train_path': '/tmp/dummy_train.csv',
                'test_path': None
            },
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'persistent_storage': True,
                    'database_name': 'features_test.duckdb',
                    'enable_feature_cache': True
                }
            }
        }
        
        # Create dummy CSV
        pd.DataFrame({'id': [1, 2, 3], 'value': [1, 2, 3]}).to_csv('/tmp/dummy_train.csv', index=False)
        
        with DuckDBDataManager(config) as manager:
            # Create test features
            test_features = pd.DataFrame({
                'id': [1, 2, 3],
                'feature_1': [0.1, 0.2, 0.3],
                'feature_2': [1.1, 1.2, 1.3]
            })
            
            # Cache features
            feature_hash = 'test_hash_123'
            manager.cache_features(
                feature_hash=feature_hash,
                feature_name='test_features',
                features_df=test_features,
                feature_params={'param1': 'value1'},
                evaluation_score=0.85,
                node_depth=2
            )
            
            # Retrieve cached features
            cached_features = manager.get_cached_features(feature_hash)
            
            if cached_features is not None and len(cached_features) == 3:
                logger.info("✅ Feature caching test passed")
                
                # Test cache stats
                stats = manager.get_feature_cache_stats()
                logger.info(f"✅ Cache stats: {stats}")
                
                return True
            else:
                logger.error("❌ Feature caching test failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Feature caching test failed: {e}")
        return False
    finally:
        if os.path.exists('/tmp/dummy_train.csv'):
            os.remove('/tmp/dummy_train.csv')

def main():
    """Run all persistent DuckDB tests."""
    logger.info("🚀 Starting Persistent DuckDB Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Database Path Generation", test_database_path_generation),
        ("Database Schema", test_database_schema),
        ("Data Loading", test_data_loading),
        ("Data Sampling", test_data_sampling),
        ("Feature Caching", test_feature_caching),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info("-" * 40)
        start_time = time.time()
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        elapsed = time.time() - start_time
        status = "✅ PASSED" if results[test_name] else "❌ FAILED"
        logger.info(f"{status} - {test_name} ({elapsed:.2f}s)")
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Persistent DuckDB implementation is ready.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())