#!/usr/bin/env python3
"""
MCTS Session Analysis Script

Deep analysis of MCTS session performance, convergence, and behavior.
Provides detailed metrics, operation statistics, and performance insights.

Usage:
    python scripts/mcts/analyze_session.py [SESSION_ID] [--detailed] [--export]
    python scripts/mcts/analyze_session.py --latest --detailed
"""

import sys
import json
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.discovery_db import FeatureDiscoveryDB


def calculate_tree_depth(db: FeatureDiscoveryDB, session_id: str) -> int:
    """Calculate actual tree depth from parent-child relationships."""
    query = """
    WITH RECURSIVE tree_depth AS (
        -- Base case: root nodes (no parent)
        SELECT mcts_node_id, 0 as depth
        FROM exploration_history 
        WHERE session_id = ? AND parent_node_id IS NULL
        
        UNION ALL
        
        -- Recursive case: children nodes
        SELECT eh.mcts_node_id, td.depth + 1
        FROM exploration_history eh
        JOIN tree_depth td ON eh.parent_node_id = td.mcts_node_id
        WHERE eh.session_id = ?
    )
    SELECT MAX(depth) as max_depth FROM tree_depth
    """
    result = db.db_service.connection_manager.execute_query(query, params=(session_id, session_id), fetch='one')
    return result[0] if result and result[0] is not None else 0


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


def analyze_session_overview(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Get session overview and basic statistics."""
    
    # Session info
    session_query = """
    SELECT session_id, start_time, end_time, total_iterations, best_score,
           status, config_snapshot
    FROM sessions WHERE session_id = ?
    """
    session_info = db.db_service.connection_manager.execute_query(session_query, params=(session_id,), fetch='one')
    
    if not session_info:
        return {"error": "Session not found"}
    
    # Basic exploration statistics
    exploration_query = """
    SELECT COUNT(*) as total_records,
           COUNT(DISTINCT mcts_node_id) as unique_nodes,
           MIN(iteration) as first_iteration,
           MAX(iteration) as last_iteration,
           AVG(evaluation_score) as avg_score,
           MAX(evaluation_score) as best_score_seen,
           SUM(evaluation_time) as total_eval_time
    FROM exploration_history WHERE session_id = ?
    """
    exploration_stats = db.db_service.connection_manager.execute_query(exploration_query, params=(session_id,), fetch='one')
    
    # Calculate actual tree depth
    max_tree_depth = calculate_tree_depth(db, session_id)
    
    overview = {
        "session_id": session_info[0],
        "start_time": session_info[1],
        "end_time": session_info[2],
        "total_iterations": session_info[3],
        "reported_best_score": session_info[4],
        "status": session_info[5],
        "exploration_records": exploration_stats[0] if exploration_stats else 0,
        "unique_nodes": exploration_stats[1] if exploration_stats else 0,
        "iteration_range": f"{exploration_stats[2]}-{exploration_stats[3]}" if exploration_stats and exploration_stats[2] else "N/A",
        "max_tree_depth": max_tree_depth,
        "avg_score": round(exploration_stats[4], 5) if exploration_stats and exploration_stats[4] else 0,
        "best_score_observed": round(exploration_stats[5], 5) if exploration_stats and exploration_stats[5] else 0,
        "total_evaluation_time": round(exploration_stats[6], 2) if exploration_stats and exploration_stats[6] else 0
    }
    
    return overview


def draw_tree_structure(db: FeatureDiscoveryDB, session_id: str) -> str:
    """Draw ASCII tree structure showing MCTS exploration."""
    # Get tree data with visit counts from mcts_tree_nodes if available
    tree_query = """
    SELECT t.node_id, e.operation_applied, t.parent_node_id,
           t.evaluation_score, t.visit_count, e.mcts_ucb1_score
    FROM mcts_tree_nodes t
    JOIN exploration_history e ON t.session_id = e.session_id AND t.node_id = e.mcts_node_id
    WHERE t.session_id = ?
    ORDER BY t.node_id
    """
    
    tree_records = db.db_service.connection_manager.execute_query(tree_query, params=(session_id,), fetch='all')
    
    if not tree_records:
        # Fallback to exploration_history only
        query = """
        SELECT mcts_node_id, operation_applied, parent_node_id, 
               evaluation_score, node_visits, mcts_ucb1_score
        FROM exploration_history 
        WHERE session_id = ?
        ORDER BY iteration
        """
        records = db.db_service.connection_manager.execute_query(query, params=(session_id,), fetch='all')
    else:
        records = tree_records
    
    if not records:
        return "❌ No tree data found"
    
    # Build node structure
    nodes = {}
    children = {}
    root_id = None
    
    for record in records:
        node_id, operation, parent_id, score, visits, ucb1 = record
        if node_id is None:
            continue
            
        nodes[node_id] = {
            'operation': operation or 'root',
            'parent': parent_id,
            'score': score or 0,
            'visits': visits or 0,
            'ucb1': ucb1 or 0
        }
        
        if parent_id is None:
            root_id = node_id
        else:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(node_id)
    
    if root_id is None:
        return "❌ No root node found"
    
    # Recursive function to draw tree
    def draw_node(node_id, prefix="", is_last=True):
        node = nodes[node_id]
        connector = "└── " if is_last else "├── "
        
        # Node info with score and visits
        node_info = f"{node['operation']} (ID:{node_id})"
        stats = f"[score:{node['score']:.5f}, visits:{node['visits']}]"
        
        lines = [f"{prefix}{connector}{node_info} {stats}"]
        
        # Draw children
        node_children = children.get(node_id, [])
        for i, child_id in enumerate(node_children):
            is_child_last = (i == len(node_children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(draw_node(child_id, child_prefix, is_child_last))
        
        return lines
    
    tree_lines = draw_node(root_id)
    return "\n".join(tree_lines)


def analyze_convergence_pattern(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Analyze convergence and improvement patterns."""
    
    query = """
    SELECT iteration, evaluation_score, evaluation_time
    FROM exploration_history 
    WHERE session_id = ? AND evaluation_score IS NOT NULL
    ORDER BY iteration
    """
    
    records = db.db_service.connection_manager.execute_query(query, params=(session_id,), fetch='all')
    
    if not records:
        return {"error": "No score data found"}
    
    scores = [r[1] for r in records]
    iterations = [r[0] for r in records]
    eval_times = [r[2] for r in records]
    
    # Track best score over time
    best_scores = []
    current_best = 0
    improvements = []
    
    for i, score in enumerate(scores):
        if score > current_best:
            current_best = score
            improvements.append(iterations[i])
        best_scores.append(current_best)
    
    # Calculate convergence metrics
    convergence = {
        "total_evaluations": len(scores),
        "score_range": {
            "min": round(min(scores), 5),
            "max": round(max(scores), 5),
            "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 5)
        },
        "improvement_points": len(improvements),
        "last_improvement_iteration": max(improvements) if improvements else 0,
        "iterations_since_improvement": max(iterations) - max(improvements) if improvements else 0,
        "depth_progression": {
            "max_depth": 0,  # Depth info not available in current schema
            "avg_depth": 0   # Depth info not available in current schema
        },
        "evaluation_times": {
            "avg_time": round(statistics.mean(eval_times) if eval_times else 0, 2),
            "total_time": round(sum(eval_times) if eval_times else 0, 2)
        }
    }
    
    return convergence


def analyze_operation_performance(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Analyze performance of different operations."""
    
    query = """
    SELECT operation_applied, COUNT(*) as usage_count,
           AVG(evaluation_score) as avg_score,
           MAX(evaluation_score) as best_score,
           AVG(evaluation_time) as avg_eval_time
    FROM exploration_history 
    WHERE session_id = ? AND operation_applied != 'root'
    GROUP BY operation_applied
    ORDER BY avg_score DESC
    """
    
    records = db.db_service.connection_manager.execute_query(query, params=(session_id,), fetch='all')
    
    operations = {}
    for record in records:
        op_name, count, avg_score, best_score, avg_time = record
        operations[op_name] = {
            "usage_count": count,
            "avg_score": round(avg_score, 5) if avg_score else 0,
            "best_score": round(best_score, 5) if best_score else 0,
            "avg_eval_time": round(avg_time, 2) if avg_time else 0,
            "efficiency": round(avg_score / avg_time, 5) if avg_time and avg_score else 0
        }
    
    return operations


def analyze_mcts_behavior(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Analyze MCTS-specific behavior (exploration vs exploitation)."""
    
    # Visit distribution - check both mcts_tree_nodes (new) and exploration_history (legacy)
    tree_visit_query = """
    SELECT node_id, visit_count
    FROM mcts_tree_nodes 
    WHERE session_id = ?
    """
    
    tree_visit_records = db.db_service.connection_manager.execute_query(tree_visit_query, params=(session_id,), fetch='all')
    
    if tree_visit_records:
        # Use new mcts_tree_nodes table with accurate backpropagation data
        visit_records = tree_visit_records
        visit_source = "mcts_tree_nodes"
    else:
        # Fallback to legacy exploration_history table
        visit_query = """
        SELECT mcts_node_id, MAX(node_visits) as max_visits
        FROM exploration_history 
        WHERE session_id = ? AND mcts_node_id IS NOT NULL
        GROUP BY mcts_node_id
        """
        visit_records = db.db_service.connection_manager.execute_query(visit_query, params=(session_id,), fetch='all')
        visit_source = "exploration_history"
    
    if not visit_records:
        return {"error": "No node visit data found"}
    
    visit_counts = [r[1] for r in visit_records]
    
    # UCB1 score analysis
    ucb1_query = """
    SELECT mcts_ucb1_score, evaluation_score
    FROM exploration_history 
    WHERE session_id = ? AND mcts_ucb1_score IS NOT NULL
    ORDER BY iteration
    """
    
    ucb1_records = db.db_service.connection_manager.execute_query(ucb1_query, params=(session_id,), fetch='all')
    
    behavior = {
        "node_statistics": {
            "visit_data_source": visit_source,
            "total_unique_nodes": len(visit_counts),
            "single_visit_nodes": sum(1 for v in visit_counts if v == 1),
            "multi_visit_nodes": sum(1 for v in visit_counts if v > 1),
            "max_visits": max(visit_counts) if visit_counts else 0,
            "avg_visits": round(statistics.mean(visit_counts) if visit_counts else 0, 2)
        },
        "exploration_ratio": round(sum(1 for v in visit_counts if v == 1) / len(visit_counts) if visit_counts else 0, 3),
        "exploitation_ratio": round(sum(1 for v in visit_counts if v > 1) / len(visit_counts) if visit_counts else 0, 3)
    }
    
    if ucb1_records:
        ucb1_scores = [r[0] for r in ucb1_records if r[0] is not None]
        if ucb1_scores:
            behavior["ucb1_statistics"] = {
                "avg_ucb1": round(statistics.mean(ucb1_scores), 4),
                "max_ucb1": round(max(ucb1_scores), 4),
                "ucb1_variance": round(statistics.variance(ucb1_scores) if len(ucb1_scores) > 1 else 0, 4)
            }
    
    return behavior


def analyze_feature_impact(db: FeatureDiscoveryDB, session_id: str) -> Dict[str, Any]:
    """Analyze impact of different features and operations."""
    
    # Feature count changes
    feature_query = """
    SELECT operation_applied, 
           AVG(LENGTH(features_after) - LENGTH(features_before)) as avg_feature_change,
           COUNT(*) as operation_count
    FROM exploration_history 
    WHERE session_id = ? AND operation_applied != 'root'
    GROUP BY operation_applied
    ORDER BY avg_feature_change DESC
    """
    
    feature_records = db.db_service.connection_manager.execute_query(feature_query, params=(session_id,), fetch='all')
    
    # Score improvements by operation
    improvement_query = """
    SELECT eh1.operation_applied,
           COUNT(*) as instances,
           AVG(eh1.evaluation_score - eh2.evaluation_score) as avg_improvement
    FROM exploration_history eh1
    JOIN exploration_history eh2 ON eh1.parent_node_id = eh2.mcts_node_id
    WHERE eh1.session_id = ? AND eh2.session_id = ?
    AND eh1.operation_applied != 'root'
    GROUP BY eh1.operation_applied
    ORDER BY avg_improvement DESC
    """
    
    improvement_records = db.db_service.connection_manager.execute_query(improvement_query, params=(session_id, session_id), fetch='all')
    
    feature_impact = {
        "feature_generation": {},
        "score_improvements": {}
    }
    
    for record in feature_records:
        op_name, avg_change, count = record
        feature_impact["feature_generation"][op_name] = {
            "avg_feature_change": round(avg_change, 2) if avg_change else 0,
            "operation_count": count
        }
    
    for record in improvement_records:
        op_name, instances, avg_improvement = record
        feature_impact["score_improvements"][op_name] = {
            "instances": instances,
            "avg_improvement": round(avg_improvement, 5) if avg_improvement else 0
        }
    
    return feature_impact


def get_session_timeline(db: FeatureDiscoveryDB, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get timeline of key events in the session."""
    
    query = """
    SELECT iteration, operation_applied, evaluation_score, evaluation_time, mcts_node_id
    FROM exploration_history 
    WHERE session_id = ?
    ORDER BY iteration
    LIMIT ?
    """
    
    records = db.db_service.connection_manager.execute_query(query, params=(session_id, limit), fetch='all')
    
    timeline = []
    for record in records:
        iteration, operation, score, eval_time, node_id = record
        timeline.append({
            "iteration": iteration,
            "operation": operation or "root",
            "score": round(score, 5) if score else 0,
            "eval_time": round(eval_time, 2) if eval_time else 0,
            "node_id": node_id
        })
    
    return timeline


def get_latest_session_id(db: FeatureDiscoveryDB) -> Optional[str]:
    """Get the latest session ID."""
    query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
    result = db.db_service.connection_manager.execute_query(query, fetch='one')
    return result[0] if result else None


def export_analysis(analysis: Dict[str, Any], session_id: str, output_dir: str = "outputs"):
    """Export analysis to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = output_path / f"mcts_analysis_{session_id}.json"
    
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"📁 Analysis exported to: {filename}")


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print formatted analysis summary."""
    
    print("\n" + "="*60)
    print("📊 MCTS SESSION ANALYSIS")
    print("="*60)
    
    # Session Overview
    overview = analysis.get("overview", {})
    print(f"\n🔍 Session Overview:")
    print(f"  Session ID: {overview.get('session_id', 'N/A')}")
    print(f"  Status: {overview.get('status', 'N/A')}")
    print(f"  Iterations: {overview.get('total_iterations', 0)}")
    print(f"  Unique Nodes: {overview.get('unique_nodes', 0)}")
    print(f"  Max Depth: {overview.get('max_tree_depth', 0)}")
    print(f"  Best Score: {overview.get('best_score_observed', 0)}")
    print(f"  Total Eval Time: {overview.get('total_evaluation_time', 0)}s")
    
    # Convergence
    convergence = analysis.get("convergence", {})
    if convergence and "error" not in convergence:
        print(f"\n📈 Convergence Pattern:")
        print(f"  Total Evaluations: {convergence.get('total_evaluations', 0)}")
        print(f"  Improvement Points: {convergence.get('improvement_points', 0)}")
        print(f"  Last Improvement: Iteration {convergence.get('last_improvement_iteration', 0)}")
        print(f"  Score Range: {convergence.get('score_range', {}).get('min', 0)} - {convergence.get('score_range', {}).get('max', 0)}")
        print(f"  Avg Eval Time: {convergence.get('evaluation_times', {}).get('avg_time', 0)}s")
    
    # Top Operations
    operations = analysis.get("operations", {})
    if operations:
        print(f"\n🎯 Top Performing Operations:")
        for i, (op_name, stats) in enumerate(list(operations.items())[:5]):
            print(f"  {i+1}. {op_name}: {stats.get('avg_score', 0)} avg score ({stats.get('usage_count', 0)} uses)")
    
    # MCTS Behavior
    behavior = analysis.get("mcts_behavior", {})
    if behavior and "error" not in behavior:
        node_stats = behavior.get("node_statistics", {})
        print(f"\n🌳 MCTS Behavior:")
        print(f"  Exploration Ratio: {behavior.get('exploration_ratio', 0)}")
        print(f"  Max Node Visits: {node_stats.get('max_visits', 0)}")
        print(f"  Multi-visit Nodes: {node_stats.get('multi_visit_nodes', 0)}")
        if 'visit_data_source' in node_stats:
            print(f"  Visit Data Source: {node_stats['visit_data_source']}")
    
    # Timeline Sample
    timeline = analysis.get("timeline", [])
    if timeline:
        print(f"\n⏱️  Session Timeline (first 5 iterations):")
        for event in timeline[:5]:
            print(f"  Iter {event.get('iteration', 0)}: {event.get('operation', 'N/A')} "
                  f"(score: {event.get('score', 0)})")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze MCTS session performance")
    parser.add_argument("session_id", nargs="?", help="Session ID to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze latest session")
    parser.add_argument("--detailed", action="store_true", help="Include detailed analysis")
    parser.add_argument("--export", action="store_true", help="Export analysis to JSON")
    parser.add_argument("--tree", action="store_true", help="Show tree structure in console")
    parser.add_argument("--timeline-limit", type=int, default=20, help="Timeline events to show")
    
    args = parser.parse_args()
    
    # Load configuration and initialize database
    try:
        config = load_default_config()
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
        print(f"📊 Analyzing latest session: {session_id}")
    else:
        session_id = args.session_id
        print(f"📊 Analyzing session: {session_id}")
    
    try:
        # Perform analysis
        analysis = {
            "session_id": session_id,
            "overview": analyze_session_overview(db, session_id),
            "convergence": analyze_convergence_pattern(db, session_id),
            "operations": analyze_operation_performance(db, session_id),
            "mcts_behavior": analyze_mcts_behavior(db, session_id),
            "timeline": get_session_timeline(db, session_id, args.timeline_limit)
        }
        
        if args.detailed:
            analysis["feature_impact"] = analyze_feature_impact(db, session_id)
        
        # Check for errors
        if analysis["overview"].get("error"):
            print(f"❌ {analysis['overview']['error']}")
            return 1
        
        # Print summary
        print_analysis_summary(analysis)
        
        # Show tree structure if requested
        if args.tree:
            print("\n" + "="*60)
            print("🌳 MCTS TREE STRUCTURE")
            print("="*60)
            tree_display = draw_tree_structure(db, session_id)
            print(tree_display)
        
        # Export if requested
        if args.export:
            export_analysis(analysis, session_id)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    finally:
        if hasattr(db, 'close'):
            db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())