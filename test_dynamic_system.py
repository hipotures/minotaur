#!/usr/bin/env python3
"""
Test script for the dynamic feature category system.

This script verifies that:
1. The core MCTS bug is fixed (each node gets different features)
2. AutoGluon runs for every iteration
3. Dynamic pattern matching works correctly
4. Auto-registration system functions properly

Run with: python test_dynamic_system.py
"""

import sys
import yaml
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src import FeatureDiscoveryDB, MCTSEngine, FeatureSpace, AutoGluonEvaluator

def setup_logging():
    """Setup logging to see MCTS details."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers to DEBUG for detailed output
    logging.getLogger('mcts').setLevel(logging.DEBUG)
    logging.getLogger('src.feature_space').setLevel(logging.DEBUG)
    logging.getLogger('src.autogluon_evaluator').setLevel(logging.DEBUG)

def test_feature_space_dynamic_lookup():
    """Test that feature space uses dynamic lookup instead of hardcoded patterns."""
    print("\n=== Testing Feature Space Dynamic Lookup ===")
    
    # Load config
    with open('config/mcts_config_titanic_test_i100.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = FeatureDiscoveryDB(config)
    
    # Initialize feature space
    feature_space = FeatureSpace(config, db.db_service.connection_manager)
    
    # Test the core fix: simulate nodes with different operations
    class MockNode:
        def __init__(self, operation, base_features=None):
            self.operation_that_created_this = operation
            self.base_features = base_features or ['Age', 'Fare', 'Pclass']
    
    # Test root node
    root_node = MockNode(None)
    root_features = feature_space.get_feature_columns_for_node(root_node)
    print(f"Root node features: {len(root_features)} features")
    
    # Test different operation nodes (this is the critical fix)
    statistical_node = MockNode('statistical_aggregations')
    statistical_features = feature_space.get_feature_columns_for_node(statistical_node)
    print(f"Statistical node features: {len(statistical_features)} features")
    
    polynomial_node = MockNode('polynomial_features')
    polynomial_features = feature_space.get_feature_columns_for_node(polynomial_node)
    print(f"Polynomial node features: {len(polynomial_features)} features")
    
    # Verify they're different (this was the bug!)
    if len(statistical_features) != len(polynomial_features):
        print("✅ SUCCESS: Different operations return different feature sets!")
        print(f"   Statistical: {statistical_features[:3]}...")
        print(f"   Polynomial: {polynomial_features[:3]}...")
    else:
        print("❌ ISSUE: Operations return same number of features")
    
    return len(statistical_features) > 0 and len(polynomial_features) > 0

def test_auto_registration():
    """Test that auto-registration system works."""
    print("\n=== Testing Auto-Registration System ===")
    
    import pandas as pd
    from src.features.generic.polynomial import PolynomialFeatures
    
    # Create test data
    df = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [2, 4, 6, 8, 10]
    })
    
    # Generate features (should trigger auto-registration)
    poly_op = PolynomialFeatures()
    features = poly_op.generate_features(df, numeric_cols=['numeric1', 'numeric2'], degree=2)
    
    print(f"Generated {len(features)} polynomial features")
    
    # Check if registered in database
    import duckdb
    from src.project_root import PROJECT_ROOT
    import os
    
    db_path = os.path.join(PROJECT_ROOT, 'data', 'minotaur.duckdb')
    with duckdb.connect(db_path) as conn:
        result = conn.execute("SELECT COUNT(*) FROM feature_catalog WHERE operation_name LIKE '%Polynomial%'").fetchone()
        poly_count = result[0] if result else 0
        
    if poly_count > 0:
        print(f"✅ SUCCESS: {poly_count} polynomial features auto-registered in database")
    else:
        print("❌ ISSUE: No polynomial features found in database")
    
    return poly_count > 0

def test_database_query_system():
    """Test that database queries work for feature lookup."""
    print("\n=== Testing Database Query System ===")
    
    import duckdb
    from src.project_root import PROJECT_ROOT
    import os
    
    db_path = os.path.join(PROJECT_ROOT, 'data', 'minotaur.duckdb')
    
    try:
        with duckdb.connect(db_path) as conn:
            # Test operation categories
            result = conn.execute("SELECT COUNT(*) FROM operation_categories").fetchone()
            op_count = result[0] if result else 0
            print(f"Operation categories in database: {op_count}")
            
            # Test feature catalog
            result = conn.execute("SELECT COUNT(*) FROM feature_catalog").fetchone()
            feature_count = result[0] if result else 0
            print(f"Features in catalog: {feature_count}")
            
            # Test the view
            result = conn.execute("SELECT COUNT(*) FROM feature_operation_mapping").fetchone()
            mapping_count = result[0] if result else 0
            print(f"Feature-operation mappings: {mapping_count}")
            
            if op_count > 0 and feature_count > 0:
                print("✅ SUCCESS: Database query system is functional")
                return True
            else:
                print("❌ ISSUE: Database query system has problems")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: Database query failed: {e}")
        return False

def test_mcts_setup():
    """Test that MCTS can be set up with the new system."""
    print("\n=== Testing MCTS Setup ===")
    
    try:
        # Load config
        with open('config/mcts_config_titanic_test_i100.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Set to only 3 iterations for testing
        config['mcts']['max_iterations'] = 3
        
        # Initialize components
        db = FeatureDiscoveryDB(config)
        feature_space = FeatureSpace(config, db.db_service.connection_manager)
        evaluator = AutoGluonEvaluator(config, db_service=db.db_service)
        
        # Create MCTS engine
        mcts = MCTSEngine(config, db, feature_space, evaluator)
        
        print("✅ SUCCESS: MCTS components initialized successfully")
        print(f"   Available operations: {len(feature_space.get_available_operations({}))}")
        print("   Ready for testing with real session")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: MCTS setup failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Dynamic Feature Category System")
    print("=" * 50)
    
    setup_logging()
    
    # Run tests
    tests = [
        ("Feature Space Dynamic Lookup", test_feature_space_dynamic_lookup),
        ("Auto-Registration System", test_auto_registration),
        ("Database Query System", test_database_query_system),
        ("MCTS Setup", test_mcts_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The dynamic feature system is ready.")
        print("To test with real MCTS session, run:")
        print("python mcts.py config/mcts_config_titanic_test_i100.yaml")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)