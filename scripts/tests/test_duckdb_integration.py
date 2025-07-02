#!/usr/bin/env python3
"""
DEPRECATED: Test script for DuckDB integration in MCTS feature discovery system.

⚠️  This test uses legacy DuckDBDataManager. For new tests, use SQLAlchemyDataManager.

Tests:
1. DuckDB availability and version
2. DuckDB data manager initialization
3. Efficient sampling capabilities
4. Configuration loading
5. Integration with existing DataManager
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_duckdb_availability():
    """Test if DuckDB is available and working."""
    logger.info("🧪 Testing DuckDB availability...")
    
    try:
        import duckdb
        logger.info(f"✅ DuckDB available: v{duckdb.__version__}")
        
        # Test basic functionality
        conn = duckdb.connect(':memory:')
        result = conn.execute("SELECT 42 as test").fetchone()
        conn.close()
        
        if result[0] == 42:
            logger.info("✅ DuckDB basic functionality test passed")
            return True
        else:
            logger.error("❌ DuckDB basic functionality test failed")
            return False
            
    except ImportError:
        logger.error("❌ DuckDB not available - please install with: pip install duckdb")
        return False
    except Exception as e:
        logger.error(f"❌ DuckDB test failed: {e}")
        return False

def test_duckdb_data_manager():
    """Test DuckDB data manager initialization."""
    logger.info("🧪 Testing DuckDB data manager...")
    
    try:
        from src.legacy.duckdb_data_manager import DuckDBDataManager, is_duckdb_available, get_duckdb_version
        
        if not is_duckdb_available():
            logger.error("❌ DuckDB not available for data manager")
            return False
        
        logger.info(f"✅ DuckDB version: {get_duckdb_version()}")
        
        # Test configuration
        config = {
            'data': {
                'duckdb': {
                    'enable_sampling': True,
                    'max_memory_gb': 1,
                    'temp_directory': '/tmp/duckdb_test'
                }
            }
        }
        
        # Initialize manager
        with DuckDBDataManager(config) as manager:
            logger.info("✅ DuckDB data manager initialized successfully")
            
            # Test performance stats
            stats = manager.get_performance_stats()
            logger.info(f"📊 Manager stats: {stats}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ DuckDB data manager test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading with DuckDB settings."""
    logger.info("🧪 Testing configuration loading...")
    
    try:
        import yaml
        
        # Test main config
        config_path = Path('config/mcts_config.yaml')
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check DuckDB configuration
        data_config = config.get('data', {})
        duckdb_config = data_config.get('duckdb', {})
        
        logger.info(f"✅ Backend setting: {data_config.get('backend', 'not set')}")
        logger.info(f"✅ DuckDB sampling: {duckdb_config.get('enable_sampling', 'not set')}")
        logger.info(f"✅ DuckDB memory limit: {duckdb_config.get('max_memory_gb', 'not set')}GB")
        
        # Test fast test config
        fast_config_path = Path('config/mcts_config_s5e6_fast_test.yaml')
        if fast_config_path.exists():
            with open(fast_config_path) as f:
                fast_config = yaml.safe_load(f)
            
            fast_data_config = fast_config.get('data', {})
            logger.info(f"✅ Fast test backend: {fast_data_config.get('backend', 'not set')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading test failed: {e}")
        return False

def test_integrated_data_manager():
    """Test integrated DataManager with DuckDB backend."""
    logger.info("🧪 Testing integrated DataManager...")
    
    try:
        import yaml
        from src.data_utils import DataManager, DUCKDB_INTEGRATION_AVAILABLE
        
        logger.info(f"✅ DuckDB integration available: {DUCKDB_INTEGRATION_AVAILABLE}")
        
        # Load test configuration
        config_path = Path('config/mcts_config_s5e6_fast_test.yaml')
        base_config_path = Path('config/mcts_config.yaml')
        
        # Merge configs like the real system does
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        if config_path.exists():
            with open(config_path) as f:
                override_config = yaml.safe_load(f)
            
            # Simple merge (in real system this is more sophisticated)
            for key, value in override_config.items():
                if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                    base_config[key].update(value)
                else:
                    base_config[key] = value
        
        # Initialize DataManager
        with DataManager(base_config) as data_manager:
            logger.info(f"✅ DataManager backend: {data_manager.backend}")
            logger.info(f"✅ DuckDB manager available: {data_manager.duckdb_manager is not None}")
            
            # Get performance stats
            stats = data_manager.get_performance_stats()
            logger.info(f"📊 DataManager stats: {stats}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated DataManager test failed: {e}")
        return False

def test_sampling_simulation():
    """Test sampling simulation (without real data files)."""
    logger.info("🧪 Testing sampling simulation...")
    
    try:
        from src.legacy.duckdb_data_manager import estimate_sample_efficiency
        
        # Test efficiency estimates
        test_cases = [
            (1000000, 100),      # 1M dataset, 100 samples
            (100000, 5000),      # 100K dataset, 5K samples  
            (10000, 8000),       # 10K dataset, 8K samples
            (1000, 1000),        # Equal sizes
        ]
        
        for dataset_size, sample_size in test_cases:
            efficiency = estimate_sample_efficiency(dataset_size, sample_size)
            logger.info(f"📊 Dataset {dataset_size:,} → Sample {sample_size:,}: "
                       f"Efficiency gain: {efficiency['efficiency_gain']:.1f}x, "
                       f"Recommended: {efficiency['recommended_backend']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sampling simulation test failed: {e}")
        return False

def main():
    """Run all DuckDB integration tests."""
    logger.info("🚀 Starting DuckDB Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("DuckDB Availability", test_duckdb_availability),
        ("DuckDB Data Manager", test_duckdb_data_manager),
        ("Configuration Loading", test_configuration_loading),
        ("Integrated DataManager", test_integrated_data_manager),
        ("Sampling Simulation", test_sampling_simulation),
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
        logger.info("🎉 All tests passed! DuckDB integration is ready.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())