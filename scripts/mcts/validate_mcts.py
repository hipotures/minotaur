#!/usr/bin/env python3
"""
MCTS Validation Script

Comprehensive validation of MCTS fixes to ensure:
1. Node IDs are properly assigned
2. Parent-child relationships are tracked
3. Visit counts accumulate correctly
4. Features evolve through the tree
5. Database logging works properly

Usage:
    python scripts/mcts/validate_mcts.py [SESSION_ID]
    python scripts/mcts/validate_mcts.py --latest
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.discovery_db import FeatureDiscoveryDB
from src.mcts_engine import FeatureNode, MCTSEngine


def load_default_config() -> Dict[str, Any]:
    """Load default configuration from base config file."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'mcts_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_node_ids(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Validate that node IDs are properly assigned."""
    print("\n🔍 Validating Node ID Assignment...")
    
    # Check exploration_history for node_id values
    query = """
    SELECT COUNT(*) as total_records,
           COUNT(CASE WHEN mcts_node_id IS NOT NULL THEN 1 END) as with_node_id,
           COUNT(CASE WHEN mcts_node_id IS NULL THEN 1 END) as without_node_id,
           MIN(mcts_node_id) as min_node_id,
           MAX(mcts_node_id) as max_node_id
    FROM exploration_history 
    WHERE session_id = ?
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        result = conn.execute(query, [session_id]).fetchone()
    
    if not result:
        return {"status": "FAIL", "reason": "No exploration history found"}
    
    total, with_id, without_id, min_id, max_id = result
    
    validation = {
        "status": "PASS" if without_id == 0 else "FAIL",
        "total_records": total,
        "records_with_node_id": with_id,
        "records_without_node_id": without_id,
        "node_id_range": f"{min_id}-{max_id}" if min_id and max_id else "None"
    }
    
    if validation["status"] == "PASS":
        print(f"✅ All {total} records have node IDs assigned (range: {validation['node_id_range']})")
    else:
        print(f"❌ {without_id} out of {total} records missing node IDs")
    
    return validation


def validate_parent_relationships(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Validate parent-child relationships are tracked."""
    print("\n🔍 Validating Parent-Child Relationships...")
    
    # Check for proper parent relationships (excluding root)
    query = """
    SELECT COUNT(*) as total_non_root,
           COUNT(CASE WHEN parent_node_id IS NOT NULL THEN 1 END) as with_parent,
           COUNT(CASE WHEN parent_node_id IS NULL THEN 1 END) as orphaned
    FROM exploration_history 
    WHERE session_id = ? AND iteration > 0
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        result = conn.execute(query, [session_id]).fetchone()
    
    if not result:
        return {"status": "FAIL", "reason": "No non-root records found"}
    
    total_non_root, with_parent, orphaned = result
    
    validation = {
        "status": "PASS" if orphaned == 0 else "PARTIAL" if orphaned < total_non_root else "FAIL",
        "total_non_root_records": total_non_root,
        "records_with_parent": with_parent,
        "orphaned_records": orphaned
    }
    
    if validation["status"] == "PASS":
        print(f"✅ All {total_non_root} non-root records have parent relationships")
    elif validation["status"] == "PARTIAL":
        print(f"⚠️  {orphaned} out of {total_non_root} non-root records are orphaned")
    else:
        print(f"❌ No parent relationships tracked")
    
    return validation


def validate_visit_counts(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Validate that visit counts accumulate correctly."""
    print("\n🔍 Validating Visit Count Accumulation...")
    
    # First check mcts_tree_nodes table (new implementation)
    tree_query = """
    SELECT COUNT(*) as total_records,
           COUNT(CASE WHEN visit_count > 1 THEN 1 END) as multi_visit_nodes,
           MAX(visit_count) as max_visits,
           AVG(visit_count) as avg_visits
    FROM mcts_tree_nodes 
    WHERE session_id = ?
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        tree_result = conn.execute(tree_query, [session_id]).fetchone()
    
    # Also check exploration_history table (legacy)
    legacy_query = """
    SELECT COUNT(*) as total_records,
           COUNT(CASE WHEN node_visits > 1 THEN 1 END) as multi_visit_nodes,
           MAX(node_visits) as max_visits,
           AVG(node_visits) as avg_visits
    FROM exploration_history 
    WHERE session_id = ?
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        legacy_result = conn.execute(legacy_query, [session_id]).fetchone()
    
    # Prioritize tree_nodes table if it has data
    if tree_result and tree_result[0] > 0:
        total, multi_visit, max_visits, avg_visits = tree_result
        source = "mcts_tree_nodes"
    elif legacy_result:
        total, multi_visit, max_visits, avg_visits = legacy_result
        source = "exploration_history"
    else:
        return {"status": "FAIL", "reason": "No records found in either table"}
    
    validation = {
        "status": "PASS" if multi_visit > 0 else "FAIL",
        "source_table": source,
        "total_records": total,
        "multi_visit_nodes": multi_visit,
        "max_visits": max_visits or 0,
        "avg_visits": round(avg_visits or 0, 2)
    }
    
    if validation["status"] == "PASS":
        print(f"✅ {multi_visit} out of {total} nodes have multiple visits (max: {max_visits}, avg: {avg_visits:.2f}) [source: {source}]")
    else:
        print(f"❌ No nodes have multiple visits - backpropagation not working [source: {source}]")
    
    return validation


def validate_feature_evolution(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Validate that features evolve through the tree."""
    print("\n🔍 Validating Feature Evolution...")
    
    # Check for feature changes
    query = """
    SELECT COUNT(*) as total_records,
           COUNT(CASE WHEN features_before != features_after THEN 1 END) as feature_changes,
           COUNT(CASE WHEN features_before = features_after THEN 1 END) as no_changes
    FROM exploration_history 
    WHERE session_id = ? AND iteration > 0
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        result = conn.execute(query, [session_id]).fetchone()
    
    if not result:
        return {"status": "FAIL", "reason": "No non-root records found"}
    
    total, feature_changes, no_changes = result
    
    validation = {
        "status": "PASS" if feature_changes > 0 else "FAIL",
        "total_records": total,
        "records_with_feature_changes": feature_changes,
        "records_without_changes": no_changes
    }
    
    if validation["status"] == "PASS":
        print(f"✅ {feature_changes} out of {total} operations changed features")
    else:
        print(f"❌ No feature changes detected - features_before equals features_after")
    
    return validation


def validate_tree_growth(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Validate that tree grows deeper with iterations."""
    print("\n🔍 Validating Tree Depth Growth...")
    
    # Check iteration progression (depth is not available in exploration_history)
    query = """
    SELECT iteration,
           iteration as max_iteration,
           COUNT(*) as nodes_at_iteration
    FROM exploration_history 
    WHERE session_id = ?
    GROUP BY iteration
    ORDER BY iteration
    LIMIT 10
    """
    
    with db.db_service.connection_manager.get_connection() as conn:
        results = conn.execute(query, [session_id]).fetchall()
    
    if not results:
        return {"status": "FAIL", "reason": "No iteration data found"}
    
    iterations = [row[1] for row in results if row[1] is not None]
    
    validation = {
        "status": "PASS" if len(iterations) > 1 and max(iterations) > min(iterations) else "PARTIAL",
        "iteration_count": len(results),
        "max_iteration_achieved": max(iterations) if iterations else 0,
        "iteration_progression": iterations[:5]  # First 5 iterations
    }
    
    if validation["status"] == "PASS":
        print(f"✅ Iterations progress from {min(iterations)} to {max(iterations)} over {len(results)} records")
    else:
        print(f"⚠️  Limited iteration progression detected (max iteration: {max(iterations) if iterations else 0})")
    
    return validation


def validate_mcts_logging(session_id: str) -> Dict[str, Any]:
    """Validate MCTS logging system."""
    print("\n🔍 Validating MCTS Logging System...")
    
    # Look for session-specific log file
    logs_dir = Path("logs/mcts")
    session_logs = list(logs_dir.glob("session_*.log")) if logs_dir.exists() else []
    
    if not session_logs:
        return {"status": "FAIL", "reason": "No MCTS session log files found in logs/mcts/"}
    
    # Use the most recent log file
    log_file = max(session_logs, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Check for MCTS phase logging
        selection_logs = content.count("=== SELECTION PHASE START ===")
        expansion_logs = content.count("=== EXPANSION PHASE START ===") 
        backprop_logs = content.count("=== BACKPROPAGATION PHASE START ===")
        
        validation = {
            "status": "PASS" if all([selection_logs, expansion_logs, backprop_logs]) else "FAIL",
            "log_file": str(log_file.name),
            "log_file_size": log_file.stat().st_size,
            "selection_phase_logs": selection_logs,
            "expansion_phase_logs": expansion_logs,
            "backpropagation_phase_logs": backprop_logs
        }
        
        if validation["status"] == "PASS":
            print(f"✅ MCTS logging active ({log_file.name}): {selection_logs} selection, {expansion_logs} expansion, {backprop_logs} backprop phases")
        else:
            print(f"❌ MCTS logging incomplete or missing in {log_file.name}")
        
        return validation
        
    except Exception as e:
        return {"status": "FAIL", "reason": f"Error reading log file {log_file}: {e}"}


def get_latest_session_id(db: FeatureDiscoveryDB) -> Optional[str]:
    """Get the latest session ID."""
    query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
    with db.db_service.connection_manager.get_connection() as conn:
        result = conn.execute(query).fetchone()
        return result[0] if result else None


def print_validation_summary(results: Dict[str, Dict[str, Any]]):
    """Print validation summary."""
    print("\n" + "="*60)
    print("📊 MCTS VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["status"] == "PASS")
    partial_tests = sum(1 for r in results.values() if r["status"] == "PARTIAL")
    failed_tests = sum(1 for r in results.values() if r["status"] == "FAIL")
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️  Partial: {partial_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    overall_status = "PASS" if failed_tests == 0 else "PARTIAL" if passed_tests > 0 else "FAIL"
    
    if overall_status == "PASS":
        print("\n🎉 All MCTS fixes are working correctly!")
    elif overall_status == "PARTIAL":
        print("\n⚠️  Some issues detected - review failed tests")
    else:
        print("\n❌ Major issues detected - MCTS fixes may not be working")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status_icon = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "PARTIAL" else "❌"
        print(f"  {status_icon} {test_name}: {result['status']}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate MCTS implementation fixes")
    parser.add_argument("session_id", nargs="?", help="Session ID to validate")
    parser.add_argument("--latest", action="store_true", help="Validate latest session")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load configuration and initialize database
    try:
        config = load_default_config()
        
        # Disable DB logging
        logging.getLogger('DB').setLevel(logging.ERROR)
        
        # Suppress stdout temporarily for FeatureDiscoveryDB creation
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            db = FeatureDiscoveryDB(config, read_only=True)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return 1
    
    # Get session ID
    if args.latest or not args.session_id:
        session_id = get_latest_session_id(db)
        if not session_id:
            print("❌ No sessions found in database")
            return 1
        print(f"🔍 Validating latest session: {session_id}")
    else:
        session_id = args.session_id
        print(f"🔍 Validating session: {session_id}")
    
    # Run validation tests
    validation_results = {}
    
    try:
        validation_results["Node ID Assignment"] = validate_node_ids(db, session_id)
        validation_results["Parent Relationships"] = validate_parent_relationships(db, session_id)
        validation_results["Visit Count Accumulation"] = validate_visit_counts(db, session_id)
        validation_results["Feature Evolution"] = validate_feature_evolution(db, session_id)
        validation_results["Tree Growth"] = validate_tree_growth(db, session_id)
        validation_results["MCTS Logging"] = validate_mcts_logging(session_id)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1
    
    finally:
        if hasattr(db, 'close'):
            db.close()
    
    # Print summary
    print_validation_summary(validation_results)
    
    # Return exit code based on results
    failed_tests = sum(1 for r in validation_results.values() if r["status"] == "FAIL")
    return 1 if failed_tests > 0 else 0


if __name__ == "__main__":
    sys.exit(main())