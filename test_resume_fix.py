#!/usr/bin/env python3
"""Test script to verify MCTS resume fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_session_validation():
    """Test that invalid session ID throws error."""
    from src.db_service import DatabaseService
    
    config = {
        'database': {'path': 'data/minotaur.duckdb'},
        'autogluon': {'dataset_name': 'test'}
    }
    
    db = DatabaseService(config, read_only=True)
    
    try:
        # This should throw ValueError
        result = db._resume_session("non-existent-session-id")
        print("❌ FAILED: Should have thrown ValueError for non-existent session")
        return False
    except ValueError as e:
        print(f"✅ PASSED: Got expected error: {e}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Got unexpected error: {e}")
        return False
    finally:
        db.close()

def test_tree_rebuild():
    """Test that tree can be rebuilt from database."""
    from src.mcts_engine import MCTSEngine
    from src.db_service import DatabaseService
    
    config = {
        'database': {'path': 'data/minotaur.duckdb'},
        'autogluon': {'dataset_name': 'test'},
        'mcts': {
            'exploration_weight': 1.4,
            'max_iterations': 10,
            'expansion_budget': 3,
            'max_children_per_node': 5,
            'max_tree_depth': 10
        },
        'session': {
            'max_iterations': 10,
            'max_runtime_hours': 1.0
        }
    }
    
    db = DatabaseService(config, read_only=True)
    engine = MCTSEngine(config)
    
    # Find a session with tree nodes
    query = """
    SELECT DISTINCT session_id 
    FROM mcts_tree_nodes 
    ORDER BY session_id DESC 
    LIMIT 1
    """
    
    try:
        with db.connection_manager.get_connection() as conn:
            result = conn.execute(query).fetchone()
            
        if not result:
            print("⚠️  No sessions with tree nodes found to test")
            return True
            
        session_id = result[0]
        print(f"Testing tree rebuild for session: {session_id}")
        
        # Try to rebuild tree
        base_features = {'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'cabin', 'ticket'}
        success = engine.rebuild_tree_from_database(db, session_id, base_features)
        
        if success:
            print(f"✅ PASSED: Successfully rebuilt tree")
            print(f"   Root node ID: {engine.root.node_id}")
            print(f"   Root visits: {engine.root.visit_count}")
            print(f"   Root score: {engine.root.evaluation_score}")
            print(f"   Children count: {len(engine.root.children)}")
            return True
        else:
            print("❌ FAILED: Could not rebuild tree")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Error during tree rebuild: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def main():
    """Run all tests."""
    print("Testing MCTS Resume Fixes...")
    print("=" * 60)
    
    tests = [
        ("Session Validation", test_session_validation),
        ("Tree Rebuild", test_tree_rebuild)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 40)
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)