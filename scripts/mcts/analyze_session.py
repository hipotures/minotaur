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
import os
import json
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.discovery_db import FeatureDiscoveryDB
from src.utils import (
    create_session_resolver, SessionResolutionError, 
    add_universal_optional_session_args, validate_optional_session_args,
    get_formatter
)


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


def analyze_session_overview(db: FeatureDiscoveryDB, session_id: str, resumed: bool = False) -> Dict[str, Any]:
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
    
    # Detect if session was resumed (can be passed as parameter or detected from data)
    first_iteration = exploration_stats[2] if exploration_stats else 0
    last_iteration = exploration_stats[3] if exploration_stats else 0
    
    # Detection logic: look for multiple distinct iteration ranges or gaps
    gap_query = """
    SELECT COUNT(DISTINCT iteration) as iteration_count,
           MAX(iteration) - MIN(iteration) + 1 as expected_count
    FROM exploration_history WHERE session_id = ?
    """
    gap_stats = db.db_service.connection_manager.execute_query(gap_query, params=(session_id,), fetch='one')
    
    # If there are fewer iterations than expected range, there might be gaps (indicating resume)
    has_gaps = gap_stats and gap_stats[0] < gap_stats[1]
    session_resumed = resumed or (first_iteration is not None and first_iteration > 0) or has_gaps
    
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
        "total_evaluation_time": round(exploration_stats[6], 2) if exploration_stats and exploration_stats[6] else 0,
        "session_resumed": session_resumed,
        "first_iteration": first_iteration
    }
    
    return overview


def draw_tree_structure(db: FeatureDiscoveryDB, session_id: str, max_lines: int = 50) -> str:
    """Draw ASCII tree structure showing MCTS exploration (limited to max_lines for readability)."""
    
    # Always get base data from exploration_history for complete tree structure
    base_query = """
    SELECT mcts_node_id, operation_applied, parent_node_id, 
           evaluation_score, node_visits, mcts_ucb1_score
    FROM exploration_history 
    WHERE session_id = ?
    ORDER BY iteration
    """
    records = db.db_service.connection_manager.execute_query(base_query, params=(session_id,), fetch='all')
    
    # Get enhanced visit counts from mcts_tree_nodes if available
    visit_query = """
    SELECT node_id, visit_count
    FROM mcts_tree_nodes 
    WHERE session_id = ?
    """
    visit_records = db.db_service.connection_manager.execute_query(visit_query, params=(session_id,), fetch='all')
    
    # Create visit count lookup
    enhanced_visits = {node_id: visit_count for node_id, visit_count in visit_records} if visit_records else {}
    
    # Update records with enhanced visit counts where available
    updated_records = []
    for record in records:
        node_id, operation, parent_id, score, visits, ucb1 = record
        # Use enhanced visit count if available, otherwise keep original
        actual_visits = enhanced_visits.get(node_id, visits)
        updated_records.append((node_id, operation, parent_id, score, actual_visits, ucb1))
    
    records = updated_records
    
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
    
    # Counter for lines to respect max_lines limit
    line_counter = [0]  # Using list to allow mutation in nested function
    truncated = [False]
    
    # Recursive function to draw tree
    def draw_node(node_id, prefix="", is_last=True, depth=0):
        # Stop if we've reached max lines
        if line_counter[0] >= max_lines:
            if not truncated[0]:
                truncated[0] = True
                return [f"{prefix}... (tree truncated at {max_lines} lines, showing depth {depth})"]
            return []
            
        node = nodes[node_id]
        connector = "└── " if is_last else "├── "
        
        # Node info with score and visits
        node_info = f"{node['operation']} (ID:{node_id})"
        stats = f"[score:{node['score']:.5f}, visits:{node['visits']}]"
        
        lines = [f"{prefix}{connector}{node_info} {stats}"]
        line_counter[0] += 1
        
        # Draw children (no limit on depth, only on total lines)
        node_children = children.get(node_id, [])
        for i, child_id in enumerate(node_children):
            if line_counter[0] >= max_lines:
                break
            is_child_last = (i == len(node_children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(draw_node(child_id, child_prefix, is_child_last, depth + 1))
        
        return lines
    
    tree_lines = draw_node(root_id)
    
    # Add summary if truncated
    if truncated[0]:
        total_nodes = len(nodes)
        tree_lines.append(f"\n📊 Tree summary: {total_nodes} total nodes, showing first {max_lines} lines")
    
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


def print_analysis_summary(analysis: Dict[str, Any], use_plain: bool = False):
    """Print formatted analysis summary using universal formatter."""
    formatter = get_formatter(force_plain=use_plain)
    
    # Check if this is a resumed session by looking for continuation data
    overview = analysis.get("overview", {})
    is_resumed_session = overview.get('session_resumed', False)
    
    # Main header
    header_title = f"{formatter.emoji('📊', '[MCTS]')} ANALIZA SESJI MCTS"
    formatter.print(formatter.header(header_title, "bold blue"))
    
    if is_resumed_session:
        # For resumed sessions: show global stats first, then continuation stats
        section_title = f"{formatter.emoji('📈', '[GLOBAL]')} STATYSTYKI CAŁEJ SESJI (WSZYSTKIE URUCHOMIENIA)"
        formatter.print(formatter.section_header(section_title, "bold green"))
        _print_session_overview(formatter, overview, "global")
        _print_convergence_pattern(formatter, analysis.get("convergence", {}), "global")
        _print_top_operations(formatter, analysis.get("operations", {}), "global")
        _print_mcts_behavior(formatter, analysis.get("mcts_behavior", {}), "global")
        
        continuation_title = f"{formatter.emoji('🔄', '[CONTINUATION]')} STATYSTYKI BIEŻĄCEJ KONTYNUACJI"
        formatter.print(formatter.section_header(continuation_title, "bold yellow"))
        formatter.print(formatter._format_text("Statystyki kontynuacji będą dostępne po implementacji", "italic"))
    else:
        # For new sessions: show normal stats
        _print_session_overview(formatter, overview, "normal")
        _print_convergence_pattern(formatter, analysis.get("convergence", {}), "normal")
        _print_top_operations(formatter, analysis.get("operations", {}), "normal")
        _print_mcts_behavior(formatter, analysis.get("mcts_behavior", {}), "normal")
    
    # Timeline (always shown at the end)
    _print_timeline(formatter, analysis.get("timeline", []))


def _print_session_overview(formatter, overview: Dict[str, Any], mode: str):
    """Print session overview section."""
    data = {
        "Session ID": overview.get('session_id', 'N/A'),
        "Status": overview.get('status', 'N/A'),
        "Iterations": str(overview.get('total_iterations', 0)),
        "Unique Nodes": str(overview.get('unique_nodes', 0)),
        "Max Depth": str(overview.get('max_tree_depth', 0)),
        "Best Score": f"{overview.get('best_score_observed', 0):.5f}",
        "Total Eval Time": f"{overview.get('total_evaluation_time', 0):.1f}s"
    }
    
    title = f"{formatter.emoji('🔍', '[OVERVIEW]')} Przegląd Sesji"
    formatter.print(formatter.key_value_pairs(data, title, "blue"))
    if formatter.plain_mode:
        print()  # Add extra newline in plain mode


def _print_convergence_pattern(formatter, convergence: Dict[str, Any], mode: str):
    """Print convergence pattern section."""
    if not convergence or "error" in convergence:
        return
    
    data = {
        "Total Evaluations": str(convergence.get('total_evaluations', 0)),
        "Improvement Points": str(convergence.get('improvement_points', 0)),
        "Last Improvement": f"Iteration {convergence.get('last_improvement_iteration', 0)}"
    }
    
    score_range = convergence.get('score_range', {})
    if score_range:
        data["Score Range"] = f"{score_range.get('min', 0):.5f} - {score_range.get('max', 0):.5f}"
    
    eval_times = convergence.get('evaluation_times', {})
    if eval_times:
        data["Avg Eval Time"] = f"{eval_times.get('avg_time', 0):.2f}s"
    
    title = f"{formatter.emoji('📈', '[CONVERGENCE]')} Wzorzec Konwergencji"
    formatter.print(formatter.key_value_pairs(data, title, "green"))
    if formatter.plain_mode:
        print()  # Add extra newline in plain mode


def _print_top_operations(formatter, operations: Dict[str, Any], mode: str):
    """Print top performing operations section."""
    if not operations:
        return
    
    table_data = []
    for i, (op_name, stats) in enumerate(list(operations.items())[:5]):
        table_data.append({
            "Rank": f"{i+1}.",
            "Operation": op_name,
            "Avg Score": f"{stats.get('avg_score', 0):.5f}",
            "Uses": str(stats.get('usage_count', 0))
        })
    
    title = f"{formatter.emoji('🎯', '[OPERATIONS]')} Najlepsze Operacje"
    headers = ["Rank", "Operation", "Avg Score", "Uses"]
    
    table_content = formatter.table(table_data, title, headers)
    
    if formatter.plain_mode:
        formatter.print(table_content)
    else:
        # For Rich mode, create Rich table for better styling
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(show_header=True, box=None)
        table.add_column("Rank", style="bold cyan", width=4)
        table.add_column("Operation", style="white")
        table.add_column("Avg Score", style="green", justify="right")
        table.add_column("Uses", style="yellow", justify="right")
        
        for row in table_data:
            table.add_row(row["Rank"], row["Operation"], row["Avg Score"], row["Uses"])
        
        formatter.print(Panel(table, title="🎯 Najlepsze Operacje", border_style="yellow"))
    
    if formatter.plain_mode:
        print()  # Add extra newline in plain mode


def _print_mcts_behavior(formatter, behavior: Dict[str, Any], mode: str):
    """Print MCTS behavior section."""
    if not behavior or "error" in behavior:
        return
    
    node_stats = behavior.get("node_statistics", {})
    
    data = {
        "Exploration Ratio": f"{behavior.get('exploration_ratio', 0):.3f}",
        "Max Node Visits": str(node_stats.get('max_visits', 0)),
        "Multi-visit Nodes": str(node_stats.get('multi_visit_nodes', 0))
    }
    
    if 'visit_data_source' in node_stats:
        data["Visit Data Source"] = node_stats['visit_data_source']
    
    title = f"{formatter.emoji('🌳', '[MCTS]')} Zachowanie MCTS"
    formatter.print(formatter.key_value_pairs(data, title, "magenta"))
    if formatter.plain_mode:
        print()  # Add extra newline in plain mode


def _print_timeline(formatter, timeline: List[Dict[str, Any]]):
    """Print session timeline section."""
    if not timeline:
        return
    
    table_data = []
    for event in timeline[:5]:
        table_data.append({
            "Iteration": f"Iter {event.get('iteration', 0)}",
            "Operation": event.get('operation', 'N/A'),
            "Score": f"{event.get('score', 0):.5f}"
        })
    
    title = f"{formatter.emoji('⏱️', '[TIMELINE]')} Timeline Sesji (pierwsze 5 iteracji)"
    headers = ["Iteration", "Operation", "Score"]
    
    table_content = formatter.table(table_data, title, headers)
    
    if formatter.plain_mode:
        formatter.print(table_content)
    else:
        # For Rich mode, create Rich table for better styling
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(show_header=True, box=None)
        table.add_column("Iteration", style="bold cyan", width=8)
        table.add_column("Operation", style="white")
        table.add_column("Score", style="green", justify="right")
        
        for row in table_data:
            table.add_row(row["Iteration"], row["Operation"], row["Score"])
        
        formatter.print(Panel(table, title="⏱️ Timeline Sesji (pierwsze 5 iteracji)", border_style="blue"))
    
    if formatter.plain_mode:
        print()  # Add extra newline in plain mode


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze MCTS session performance")
    
    # Add universal optional session argument
    add_universal_optional_session_args(parser, "Session analysis")
    
    parser.add_argument("--detailed", action="store_true", help="Include detailed analysis")
    parser.add_argument("--export", action="store_true", help="Export analysis to JSON")
    parser.add_argument("--tree", action="store_true", help="Show tree structure in console")
    parser.add_argument("--timeline-limit", type=int, default=20, help="Timeline events to show")
    parser.add_argument("--resumed", action="store_true", help="Indicate this session was resumed")
    parser.add_argument("--plain", action="store_true", help="Plain text output (no Rich formatting or emoji)")
    
    args = parser.parse_args()
    
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
    
    # Resolve session using universal resolver (with optional fallback)
    try:
        session_identifier = validate_optional_session_args(args)
        resolver = create_session_resolver(config)
        session_info = resolver.resolve_session(session_identifier)
        session_id = session_info.session_id
        
        # Use formatter for session info display (respect environment variable if --plain not explicitly set)
        use_plain = args.plain or os.environ.get('MINOTAUR_PLAIN_OUTPUT', '').lower() in ('1', 'true', 'yes', 'on')
        formatter = get_formatter(force_plain=use_plain)
        
        session_header = f"{formatter.emoji('📊', '[ANALYSIS]')} Analyzing session: {session_id[:8]}... ({session_info.session_name})"
        print(session_header)
        print(f"   Started: {session_info.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Iterations: {session_info.total_iterations}")
        if session_info.best_score is not None:
            print(f"   Best score: {session_info.best_score:.5f}")
        print()
        
    except SessionResolutionError as e:
        print(f"❌ Session resolution failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error resolving session: {e}")
        return 1
    
    try:
        # Perform analysis
        analysis = {
            "session_id": session_id,
            "overview": analyze_session_overview(db, session_id, args.resumed),
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
        print_analysis_summary(analysis, use_plain=use_plain)
        
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