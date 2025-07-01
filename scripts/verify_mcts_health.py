#!/usr/bin/env python3
"""
MCTS Health Verification Script

Comprehensive health analysis for MCTS exploration sessions.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.core.connection import DuckDBConnectionManager

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

if RICH_AVAILABLE:
    console = Console()

class HealthStatus(Enum):
    """Health check status."""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    SKIP = "⏭️  SKIP"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Any] = None
    severity: str = "info"  # info, warning, error

class MCTSHealthChecker:
    """Comprehensive MCTS health analysis."""
    
    def __init__(self, db_path: str):
        """Initialize with database connection."""
        config = {
            'database': {
                'path': db_path,
                'pool_size': 1
            }
        }
        self.conn_manager = DuckDBConnectionManager(
            main_config=config,
            read_only=True
        )
    
    def get_session_info(self, session_input: Optional[str]) -> Optional[Tuple[str, str]]:
        """Get session ID and name from input. Returns (session_id, session_name) or None."""
        with self.conn_manager.get_connection() as conn:
            # If no session provided, get the latest one
            if session_input is None:
                query = """
                    SELECT session_id, session_name FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """
                result = conn.execute(query).fetchone()
                return (result[0], result[1]) if result else None
            
            # Try direct lookup first
            query = "SELECT session_id, session_name FROM sessions WHERE session_id = ?"
            result = conn.execute(query, [session_input]).fetchone()
            if result:
                return (result[0], result[1])
            
            # Try pattern matching for session_* names
            query = """
                SELECT session_id, session_name FROM sessions 
                WHERE session_id LIKE ? OR session_name LIKE ?
                ORDER BY start_time DESC 
                LIMIT 1
            """
            result = conn.execute(query, [f"{session_input}%", f"{session_input}%"]).fetchone()
            if result:
                return (result[0], result[1])
                
            return None
    
    def check_session_exists(self, session_id: str, session_name: str) -> HealthCheckResult:
        """Check if session exists in database."""
        with self.conn_manager.get_connection() as conn:
            query = "SELECT COUNT(*) FROM sessions WHERE session_id = ?"
            count = conn.execute(query, [session_id]).fetchone()[0]
            
            if count > 0:
                return HealthCheckResult(
                    name="Session Existence",
                    status=HealthStatus.PASS,
                    message=f"{session_name} ({session_id[:8]}...) found in database"
                )
            else:
                return HealthCheckResult(
                    name="Session Existence",
                    status=HealthStatus.FAIL,
                    message=f"{session_name} ({session_id[:8]}...) not found in database",
                    severity="error"
                )
    
    def check_tree_structure_integrity(self, session_id: str) -> HealthCheckResult:
        """Check if MCTS tree structure is valid."""
        with self.conn_manager.get_connection() as conn:
            # Check for orphaned nodes (nodes with parent_id that doesn't exist)
            query = """
                SELECT COUNT(*) as orphaned_count
                FROM exploration_history e1
                WHERE e1.session_id = ? 
                  AND e1.parent_node_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM exploration_history e2 
                      WHERE e2.session_id = ? 
                        AND e2.mcts_node_id = e1.parent_node_id
                  )
            """
            orphaned_count = conn.execute(query, [session_id, session_id]).fetchone()[0]
            
            # Check for multiple roots
            query = """
                SELECT COUNT(*) as root_count
                FROM exploration_history
                WHERE session_id = ? AND parent_node_id IS NULL
            """
            root_count = conn.execute(query, [session_id]).fetchone()[0]
            
            issues = []
            if orphaned_count > 0:
                issues.append(f"{orphaned_count} orphaned nodes")
            if root_count != 1:
                issues.append(f"{root_count} root nodes (should be 1)")
            
            if issues:
                return HealthCheckResult(
                    name="Tree Structure Integrity",
                    status=HealthStatus.FAIL,
                    message=f"Tree structure issues: {', '.join(issues)}",
                    details={"orphaned_nodes": orphaned_count, "root_count": root_count},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="Tree Structure Integrity",
                    status=HealthStatus.PASS,
                    message="Tree structure is valid"
                )
    
    def check_feature_diversity(self, session_id: str) -> HealthCheckResult:
        """Check for feature diversity across tree depths."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    iteration,
                    COUNT(DISTINCT mcts_node_id) as node_count,
                    COUNT(DISTINCT json_array_length(features_after)) as feature_diversity,
                    MIN(json_array_length(features_after)) as min_features,
                    MAX(json_array_length(features_after)) as max_features
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                HAVING node_count > 1
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            low_diversity_depths = []
            for row in results:
                iteration, node_count, feature_diversity, min_features, max_features = row
                if feature_diversity == 1:  # All nodes have same feature count
                    low_diversity_depths.append(iteration)
            
            if low_diversity_depths:
                return HealthCheckResult(
                    name="Feature Diversity",
                    status=HealthStatus.FAIL,
                    message=f"Low feature diversity at depths: {low_diversity_depths}",
                    details={"affected_depths": low_diversity_depths},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="Feature Diversity",
                    status=HealthStatus.PASS,
                    message="Good feature diversity across depths"
                )
    
    def check_feature_accumulation_bug(self, session_id: str) -> HealthCheckResult:
        """Check for the specific feature accumulation bug from Point A."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    iteration,
                    MAX(json_array_length(features_after)) as max_features
                FROM exploration_history
                WHERE session_id = ? AND iteration <= 5
                GROUP BY iteration
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            suspicious_depths = []
            for iteration, max_features in results:
                if max_features > 30:  # Suspicious threshold
                    suspicious_depths.append((iteration, max_features))
            
            if suspicious_depths:
                return HealthCheckResult(
                    name="Feature Accumulation Bug Detection",
                    status=HealthStatus.FAIL,
                    message=f"Detected possible feature accumulation bug: {len(suspicious_depths)} depths affected",
                    details={"suspicious_depths": suspicious_depths},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="Feature Accumulation Bug Detection",
                    status=HealthStatus.PASS,
                    message="No feature accumulation bug detected"
                )
    
    def check_operation_diversity(self, session_id: str) -> HealthCheckResult:
        """Check for operation diversity across depths."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    iteration,
                    COUNT(DISTINCT mcts_node_id) as node_count,
                    COUNT(DISTINCT operation_applied) as unique_operations,
                    COUNT(*) as total_operations
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                HAVING node_count > 1
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            low_diversity_depths = []
            for row in results:
                iteration, node_count, unique_ops, total_ops = row
                diversity_ratio = unique_ops / total_ops if total_ops > 0 else 0
                if diversity_ratio < 0.5:  # Less than 50% unique operations
                    low_diversity_depths.append((iteration, diversity_ratio))
            
            if low_diversity_depths:
                return HealthCheckResult(
                    name="Operation Diversity",
                    status=HealthStatus.WARN,
                    message=f"Low operation diversity at {len(low_diversity_depths)} depths",
                    details={"affected_depths": low_diversity_depths},
                    severity="warning"
                )
            else:
                return HealthCheckResult(
                    name="Operation Diversity",
                    status=HealthStatus.PASS,
                    message="Good operation diversity"
                )
    
    def check_evaluation_consistency(self, session_id: str) -> HealthCheckResult:
        """Check for evaluation score consistency."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    COUNT(*) as total_evaluations,
                    COUNT(CASE WHEN evaluation_score IS NULL THEN 1 END) as null_scores,
                    COUNT(CASE WHEN evaluation_score < 0 OR evaluation_score > 1 THEN 1 END) as invalid_scores
                FROM exploration_history
                WHERE session_id = ?
            """
            
            result = conn.execute(query, [session_id]).fetchone()
            total_evaluations, null_scores, invalid_scores = result
            
            issues = []
            if null_scores > 0:
                issues.append(f"{null_scores} null scores")
            if invalid_scores > 0:
                issues.append(f"{invalid_scores} invalid scores")
            
            if issues:
                return HealthCheckResult(
                    name="Evaluation Consistency",
                    status=HealthStatus.FAIL,
                    message=f"Evaluation issues: {', '.join(issues)}",
                    details={"null_scores": null_scores, "invalid_scores": invalid_scores},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="Evaluation Consistency",
                    status=HealthStatus.PASS,
                    message=f"All {total_evaluations} evaluations are consistent"
                )
    
    def check_convergence_patterns(self, session_id: str) -> HealthCheckResult:
        """Check for healthy convergence patterns."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    iteration,
                    MAX(evaluation_score) as best_score,
                    AVG(evaluation_score) as avg_score
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            if len(results) < 5:
                return HealthCheckResult(
                    name="Convergence Patterns",
                    status=HealthStatus.SKIP,
                    message="Insufficient data for convergence analysis"
                )
            
            # Check if best score is improving
            best_scores = [row[1] for row in results]
            is_improving = any(best_scores[i] > best_scores[i-1] for i in range(1, len(best_scores)))
            
            if is_improving:
                return HealthCheckResult(
                    name="Convergence Patterns",
                    status=HealthStatus.PASS,
                    message="Healthy convergence - scores improving over time"
                )
            else:
                return HealthCheckResult(
                    name="Convergence Patterns",
                    status=HealthStatus.WARN,
                    message="No score improvement detected",
                    severity="warning"
                )
    
    def check_exploration_exploitation_balance(self, session_id: str) -> HealthCheckResult:
        """Check if MCTS is balancing exploration and exploitation properly."""
        with self.conn_manager.get_connection() as conn:
            # Get visit distribution across tree
            query = """
                SELECT 
                    iteration,
                    COUNT(DISTINCT mcts_node_id) as unique_nodes_visited,
                    MAX(node_visits) as max_visits,
                    AVG(node_visits) as avg_visits,
                    STDDEV(node_visits) as visit_stddev
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                HAVING iteration > 10  -- After initial exploration
                ORDER BY iteration DESC
                LIMIT 10
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            if not results:
                return HealthCheckResult(
                    name="Exploration-Exploitation Balance",
                    status=HealthStatus.SKIP,
                    message="Insufficient data for balance analysis"
                )
            
            # Calculate metrics
            avg_unique_nodes = sum(r[1] for r in results) / len(results)
            max_visit_concentration = max(r[2] for r in results)
            stddev_values = [r[4] for r in results if r[4] is not None]
            avg_visit_stddev = sum(stddev_values) / len(stddev_values) if stddev_values else None
            
            issues = []
            if avg_unique_nodes < 2:  # Too few nodes being explored
                issues.append("Over-exploitation: exploring too few nodes")
            if max_visit_concentration > 50:  # Too many visits to single node
                issues.append(f"Visit concentration too high: {max_visit_concentration}")
            if avg_visit_stddev is not None and avg_visit_stddev < 1:
                issues.append("Low visit variance: possible premature convergence")
            
            if issues:
                return HealthCheckResult(
                    name="Exploration-Exploitation Balance",
                    status=HealthStatus.WARN,
                    message=f"Balance issues: {'; '.join(issues)}",
                    details={"avg_unique_nodes": avg_unique_nodes, "max_visits": max_visit_concentration},
                    severity="warning"
                )
            else:
                return HealthCheckResult(
                    name="Exploration-Exploitation Balance",
                    status=HealthStatus.PASS,
                    message="Good exploration-exploitation balance"
                )
    
    def check_ucb1_score_sanity(self, session_id: str) -> HealthCheckResult:
        """Check for UCB1 score calculation issues."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN mcts_ucb1_score IS NULL THEN 1 END) as null_scores,
                    COUNT(CASE WHEN mcts_ucb1_score = 'inf' THEN 1 END) as inf_scores,
                    COUNT(CASE WHEN mcts_ucb1_score < 0 THEN 1 END) as negative_scores,
                    MIN(mcts_ucb1_score) as min_score,
                    MAX(CASE WHEN mcts_ucb1_score != 'inf' THEN mcts_ucb1_score END) as max_finite_score
                FROM exploration_history
                WHERE session_id = ? AND iteration > 0
            """
            
            result = conn.execute(query, [session_id]).fetchone()
            total, null_scores, inf_scores, negative_scores, min_score, max_finite = result
            
            issues = []
            if negative_scores > 0:
                issues.append(f"{negative_scores} negative UCB1 scores")
            if null_scores > total * 0.1:  # More than 10% null scores
                issues.append(f"High null score rate: {null_scores}/{total}")
            if inf_scores == 0 and total > 10:  # No unvisited nodes explored
                issues.append("No infinite UCB1 scores - may indicate expansion issues")
            if max_finite is not None and max_finite > 10:  # Unusually high UCB1 scores
                issues.append(f"Unusually high UCB1 scores: max={max_finite:.2f}")
            
            if issues:
                return HealthCheckResult(
                    name="UCB1 Score Sanity",
                    status=HealthStatus.FAIL,
                    message=f"UCB1 calculation issues: {'; '.join(issues)}",
                    details={"null_scores": null_scores, "negative_scores": negative_scores},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="UCB1 Score Sanity",
                    status=HealthStatus.PASS,
                    message="UCB1 scores are within expected ranges"
                )
    
    def check_tree_balance(self, session_id: str) -> HealthCheckResult:
        """Check if the MCTS tree is well-balanced."""
        with self.conn_manager.get_connection() as conn:
            # Get tree shape metrics
            query = """
                WITH tree_stats AS (
                    SELECT 
                        iteration as depth,
                        COUNT(DISTINCT mcts_node_id) as nodes_at_depth,
                        COUNT(DISTINCT parent_node_id) as parents_at_depth
                    FROM exploration_history
                    WHERE session_id = ?
                    GROUP BY iteration
                )
                SELECT 
                    MAX(depth) as max_depth,
                    AVG(nodes_at_depth) as avg_width,
                    MAX(nodes_at_depth) as max_width,
                    COUNT(*) as depth_count,
                    SUM(nodes_at_depth) as total_nodes
                FROM tree_stats
            """
            
            result = conn.execute(query, [session_id]).fetchone()
            max_depth, avg_width, max_width, depth_count, total_nodes = result
            
            if not max_depth:
                return HealthCheckResult(
                    name="Tree Balance",
                    status=HealthStatus.SKIP,
                    message="No tree data available"
                )
            
            issues = []
            
            # Check for too deep tree (inefficient)
            if max_depth > 50:
                issues.append(f"Tree too deep: {max_depth} levels")
            
            # Check for too wide tree (memory issues)
            if max_width > 100:
                issues.append(f"Tree too wide: {max_width} nodes at single level")
            
            # Check for degenerate tree (linear chain)
            if avg_width < 2 and max_depth > 10:
                issues.append("Degenerate tree: mostly linear exploration")
            
            # Check branching factor
            avg_branching = total_nodes / max_depth if max_depth > 0 else 0
            if avg_branching < 1.5:
                issues.append(f"Low branching factor: {avg_branching:.2f}")
            
            if issues:
                return HealthCheckResult(
                    name="Tree Balance",
                    status=HealthStatus.WARN,
                    message=f"Tree structure issues: {'; '.join(issues)}",
                    details={"max_depth": max_depth, "avg_width": avg_width, "max_width": max_width},
                    severity="warning"
                )
            else:
                return HealthCheckResult(
                    name="Tree Balance",
                    status=HealthStatus.PASS,
                    message=f"Well-balanced tree: depth={max_depth}, avg_width={avg_width:.1f}"
                )
    
    def check_trap_states(self, session_id: str) -> HealthCheckResult:
        """Check for potential trap states - nodes that look good initially but lead to worse outcomes."""
        with self.conn_manager.get_connection() as conn:
            query = """
                WITH node_outcomes AS (
                    SELECT 
                        parent_node_id,
                        mcts_node_id,
                        evaluation_score,
                        node_visits,
                        iteration
                    FROM exploration_history
                    WHERE session_id = ? AND parent_node_id IS NOT NULL
                ),
                parent_child_comparison AS (
                    SELECT 
                        p.mcts_node_id as parent_id,
                        p.evaluation_score as parent_score,
                        AVG(c.evaluation_score) as avg_child_score,
                        COUNT(c.mcts_node_id) as child_count,
                        p.node_visits as parent_visits
                    FROM exploration_history p
                    JOIN node_outcomes c ON p.mcts_node_id = c.parent_node_id
                    WHERE p.session_id = ? 
                        AND p.evaluation_score > 0.7  -- High scoring parents
                        AND p.node_visits > 3  -- Well-explored
                    GROUP BY p.mcts_node_id, p.evaluation_score, p.node_visits
                    HAVING COUNT(c.mcts_node_id) >= 2  -- Has multiple children
                )
                SELECT 
                    COUNT(*) as trap_candidates
                FROM parent_child_comparison
                WHERE avg_child_score < parent_score * 0.8  -- Children much worse than parent
            """
            
            result = conn.execute(query, [session_id, session_id]).fetchone()
            trap_candidates = result[0]
            
            if trap_candidates > 5:
                return HealthCheckResult(
                    name="Trap State Detection",
                    status=HealthStatus.WARN,
                    message=f"Found {trap_candidates} potential trap states",
                    details={"trap_candidates": trap_candidates},
                    severity="warning"
                )
            else:
                return HealthCheckResult(
                    name="Trap State Detection",
                    status=HealthStatus.PASS,
                    message="No significant trap states detected"
                )
    
    def check_value_propagation(self, session_id: str) -> HealthCheckResult:
        """Check if values are properly propagated up the tree."""
        with self.conn_manager.get_connection() as conn:
            # Check if parent nodes have reasonable aggregate scores
            query = """
                WITH node_hierarchy AS (
                    SELECT 
                        mcts_node_id,
                        parent_node_id,
                        evaluation_score,
                        node_visits
                    FROM exploration_history
                    WHERE session_id = ?
                ),
                parent_child_stats AS (
                    SELECT 
                        p.mcts_node_id as parent_id,
                        p.evaluation_score as parent_score,
                        p.node_visits as parent_visits,
                        AVG(c.evaluation_score) as avg_child_score,
                        SUM(c.node_visits) as total_child_visits
                    FROM node_hierarchy p
                    JOIN node_hierarchy c ON p.mcts_node_id = c.parent_node_id
                    WHERE p.node_visits > 0
                    GROUP BY p.mcts_node_id, p.evaluation_score, p.node_visits
                )
                SELECT 
                    COUNT(*) as total_parents,
                    COUNT(CASE WHEN total_child_visits > parent_visits * 1.5 THEN 1 END) as visit_propagation_errors,
                    COUNT(CASE WHEN ABS(parent_score - avg_child_score) > 0.3 THEN 1 END) as score_propagation_errors
                FROM parent_child_stats
            """
            
            result = conn.execute(query, [session_id]).fetchone()
            total_parents, visit_errors, score_errors = result
            
            if not total_parents:
                return HealthCheckResult(
                    name="Value Propagation",
                    status=HealthStatus.SKIP,
                    message="Insufficient hierarchy data"
                )
            
            issues = []
            if visit_errors > total_parents * 0.1:
                issues.append(f"Visit count propagation errors: {visit_errors}/{total_parents}")
            if score_errors > total_parents * 0.2:
                issues.append(f"Score propagation inconsistencies: {score_errors}/{total_parents}")
            
            if issues:
                return HealthCheckResult(
                    name="Value Propagation",
                    status=HealthStatus.FAIL,
                    message=f"Value propagation issues: {'; '.join(issues)}",
                    details={"visit_errors": visit_errors, "score_errors": score_errors},
                    severity="error"
                )
            else:
                return HealthCheckResult(
                    name="Value Propagation",
                    status=HealthStatus.PASS,
                    message="Values properly propagated through tree"
                )
    
    def check_memory_usage_patterns(self, session_id: str) -> HealthCheckResult:
        """Check for memory usage issues in MCTS exploration."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    iteration,
                    COUNT(*) as nodes_at_iteration,
                    AVG(memory_usage_mb) as avg_memory,
                    MAX(memory_usage_mb) as max_memory,
                    SUM(CASE WHEN memory_usage_mb IS NULL THEN 1 ELSE 0 END) as missing_memory_data
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                ORDER BY iteration
            """
            
            results = conn.execute(query, [session_id]).fetchall()
            
            if not results:
                return HealthCheckResult(
                    name="Memory Usage Patterns",
                    status=HealthStatus.SKIP,
                    message="No memory usage data available"
                )
            
            # Calculate metrics
            total_nodes = sum(r[1] for r in results)
            max_memory_usage = max(r[3] for r in results if r[3] is not None) if any(r[3] for r in results) else None
            missing_data_count = sum(r[4] for r in results)
            
            # Check for memory growth pattern
            memory_values = [r[2] for r in results if r[2] is not None]
            if len(memory_values) > 10:
                # Check if memory is growing linearly (potential leak)
                first_half_avg = sum(memory_values[:len(memory_values)//2]) / (len(memory_values)//2)
                second_half_avg = sum(memory_values[len(memory_values)//2:]) / (len(memory_values) - len(memory_values)//2)
                memory_growth_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
            else:
                memory_growth_ratio = 1
            
            issues = []
            
            if missing_data_count == total_nodes:
                return HealthCheckResult(
                    name="Memory Usage Patterns",
                    status=HealthStatus.SKIP,
                    message="Memory tracking not enabled"
                )
            
            if max_memory_usage is not None and max_memory_usage > 1000:  # > 1GB
                issues.append(f"High memory usage: {max_memory_usage:.1f}MB")
            
            if memory_growth_ratio > 2:  # Memory doubled
                issues.append(f"Potential memory leak: {memory_growth_ratio:.1f}x growth")
            
            if total_nodes > 10000:
                issues.append(f"Large tree size: {total_nodes} nodes")
            
            if issues:
                return HealthCheckResult(
                    name="Memory Usage Patterns",
                    status=HealthStatus.WARN,
                    message=f"Memory concerns: {'; '.join(issues)}",
                    details={"max_memory_mb": max_memory_usage, "total_nodes": total_nodes},
                    severity="warning"
                )
            else:
                return HealthCheckResult(
                    name="Memory Usage Patterns",
                    status=HealthStatus.PASS,
                    message=f"Memory usage stable (max: {max_memory_usage:.1f}MB)" if max_memory_usage else "Memory usage stable"
                )
    
    def get_detailed_analysis(self, session_id: str) -> Dict:
        """Get detailed analysis data for --details flag."""
        with self.conn_manager.get_connection() as conn:
            # Depth statistics
            query = """
                SELECT 
                    iteration,
                    COUNT(DISTINCT mcts_node_id) as node_count,
                    COUNT(DISTINCT operation_applied) as unique_operations,
                    MIN(json_array_length(features_after)) as min_features,
                    MAX(json_array_length(features_after)) as max_features,
                    AVG(json_array_length(features_after)) as avg_features,
                    COUNT(DISTINCT json_array_length(features_after)) as feature_diversity,
                    AVG(evaluation_score) as avg_score
                FROM exploration_history
                WHERE session_id = ?
                GROUP BY iteration
                ORDER BY iteration
            """
            
            depth_stats = conn.execute(query, [session_id]).fetchall()
            
            # Session summary
            query = """
                SELECT 
                    COUNT(*) as total_nodes,
                    MAX(iteration) as max_depth,
                    MIN(evaluation_score) as min_score,
                    MAX(evaluation_score) as max_score,
                    AVG(evaluation_score) as avg_score,
                    COUNT(DISTINCT operation_applied) as total_unique_operations
                FROM exploration_history
                WHERE session_id = ?
            """
            
            summary = conn.execute(query, [session_id]).fetchone()
            
            return {
                'depth_stats': depth_stats,
                'summary': summary
            }
    
    def run_health_checks(self, session_id: str, session_name: str) -> List[HealthCheckResult]:
        """Run all health checks for a session."""
        checks = [
            (self.check_session_exists, True),  # Needs session_name
            (self.check_tree_structure_integrity, False),
            (self.check_feature_diversity, False),
            (self.check_feature_accumulation_bug, False),
            (self.check_operation_diversity, False),
            (self.check_evaluation_consistency, False),
            (self.check_convergence_patterns, False),
            (self.check_exploration_exploitation_balance, False),
            (self.check_ucb1_score_sanity, False),
            (self.check_tree_balance, False),
            (self.check_trap_states, False),
            (self.check_value_propagation, False),
            (self.check_memory_usage_patterns, False)
        ]
        
        results = []
        for check, needs_name in checks:
            try:
                if needs_name:
                    result = check(session_id, session_name)
                else:
                    result = check(session_id)
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    name=check.__name__.replace('check_', '').replace('_', ' ').title(),
                    status=HealthStatus.FAIL,
                    message=f"Check failed with error: {str(e)}",
                    severity="error"
                ))
        
        return results
    
    def print_summary_report(self, session_id: str, session_name: str, results: List[HealthCheckResult]):
        """Print summary report using rich if available."""
        if RICH_AVAILABLE:
            self._print_rich_summary(session_id, session_name, results)
        else:
            self._print_plain_summary(session_id, session_name, results)
    
    def _print_rich_summary(self, session_id: str, session_name: str, results: List[HealthCheckResult]):
        """Print rich formatted summary."""
        console.print()
        console.print(Panel.fit(
            f"[bold]MCTS Health Check Report[/bold]\n"
            f"Session: [cyan]{session_name}[/cyan] ([dim]{session_id[:8]}...[/dim])",
            style="blue"
        ))
        console.print()
        
        # Summary table
        table = Table(title="Health Check Summary", box=box.ROUNDED)
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Message", style="dim")
        
        overall_status = HealthStatus.PASS
        for result in results:
            # Determine row style based on status
            if result.status == HealthStatus.FAIL:
                status_style = "red"
                overall_status = HealthStatus.FAIL
            elif result.status == HealthStatus.WARN:
                status_style = "yellow"
                if overall_status == HealthStatus.PASS:
                    overall_status = HealthStatus.WARN
            elif result.status == HealthStatus.SKIP:
                status_style = "dim"
            else:
                status_style = "green"
            
            table.add_row(
                result.name,
                f"[{status_style}]{result.status.value}[/{status_style}]",
                result.message
            )
        
        console.print(table)
        console.print()
        
        # Overall status
        if overall_status == HealthStatus.FAIL:
            console.print(Panel(
                "[red]❌ CRITICAL ISSUES DETECTED[/red]\n"
                "MCTS may not be functioning correctly. Review failed checks above.",
                style="red"
            ))
        elif overall_status == HealthStatus.WARN:
            console.print(Panel(
                "[yellow]⚠️  WARNINGS DETECTED[/yellow]\n"
                "MCTS is functioning but may have performance issues.",
                style="yellow"
            ))
        else:
            console.print(Panel(
                "[green]✅ ALL CHECKS PASSED[/green]\n"
                "MCTS is functioning correctly.",
                style="green"
            ))
        console.print()
    
    def _print_plain_summary(self, session_id: str, session_name: str, results: List[HealthCheckResult]):
        """Print plain text summary."""
        print("=" * 80)
        print("MCTS HEALTH CHECK REPORT")
        print(f"Session: {session_name} ({session_id[:8]}...)")
        print("=" * 80)
        print()
        
        for result in results:
            print(f"{result.status.value:12} {result.name}: {result.message}")
        
        print()
        print("=" * 80)
        
        # Determine overall status
        has_fails = any(r.status == HealthStatus.FAIL for r in results)
        has_warns = any(r.status == HealthStatus.WARN for r in results)
        
        if has_fails:
            print("❌ CRITICAL ISSUES DETECTED")
            print("MCTS may not be functioning correctly.")
        elif has_warns:
            print("⚠️  WARNINGS DETECTED")
            print("MCTS is functioning but may have performance issues.")
        else:
            print("✅ ALL CHECKS PASSED")
            print("MCTS is functioning correctly.")
        
        print("=" * 80)
        print()
    
    def print_detailed_report(self, session_id: str, session_name: str):
        """Print detailed analysis report."""
        analysis = self.get_detailed_analysis(session_id)
        
        if RICH_AVAILABLE:
            self._print_rich_details(session_id, session_name, analysis)
        else:
            self._print_plain_details(session_id, session_name, analysis)
    
    def _print_rich_details(self, session_id: str, session_name: str, analysis: Dict):
        """Print rich formatted detailed report."""
        console.print()
        console.print(Panel.fit(
            f"[bold]Detailed Analysis[/bold]\n"
            f"Session: [cyan]{session_name}[/cyan] ([dim]{session_id[:8]}...[/dim])",
            style="blue"
        ))
        console.print()
        
        # Summary stats
        summary = analysis['summary']
        summary_table = Table(title="Session Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")
        
        summary_table.add_row("Total Nodes", str(summary[0]))
        summary_table.add_row("Max Depth", str(summary[1]))
        summary_table.add_row("Score Range", f"{summary[2]:.4f} - {summary[3]:.4f}")
        summary_table.add_row("Average Score", f"{summary[4]:.4f}")
        summary_table.add_row("Unique Operations", str(summary[5]))
        
        console.print(summary_table)
        console.print()
        
        # Depth statistics
        depth_table = Table(title="Depth Statistics", box=box.ROUNDED)
        depth_table.add_column("Depth", justify="right")
        depth_table.add_column("Nodes", justify="right")
        depth_table.add_column("Ops", justify="right")
        depth_table.add_column("Min Feat", justify="right")
        depth_table.add_column("Max Feat", justify="right")
        depth_table.add_column("Avg Feat", justify="right")
        depth_table.add_column("Diversity", justify="right")
        depth_table.add_column("Avg Score", justify="right")
        
        for row in analysis['depth_stats']:
            depth_table.add_row(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                f"{row[5]:.1f}",
                str(row[6]),
                f"{row[7]:.4f}"
            )
        
        console.print(depth_table)
        console.print()
    
    def _print_plain_details(self, session_id: str, session_name: str, analysis: Dict):
        """Print plain text detailed report."""
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print(f"Session: {session_name} ({session_id[:8]}...)")
        print("=" * 80)
        
        # Summary
        summary = analysis['summary']
        print(f"\nSession Summary:")
        print(f"  Total Nodes: {summary[0]}")
        print(f"  Max Depth: {summary[1]}")
        print(f"  Score Range: {summary[2]:.4f} - {summary[3]:.4f}")
        print(f"  Average Score: {summary[4]:.4f}")
        print(f"  Unique Operations: {summary[5]}")
        
        # Depth statistics
        print(f"\nDepth Statistics:")
        print(f"{'Depth':>6} | {'Nodes':>6} | {'Ops':>6} | {'Min Feat':>9} | {'Max Feat':>9} | {'Avg Feat':>9} | {'Diversity':>10} | {'Avg Score':>10}")
        print("-" * 85)
        
        for row in analysis['depth_stats']:
            print(f"{row[0]:>6} | {row[1]:>6} | {row[2]:>6} | {row[3]:>9} | {row[4]:>9} | {row[5]:>9.1f} | {row[6]:>10} | {row[7]:>10.4f}")
        
        print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive MCTS health verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                 # Check latest session
  %(prog)s session_12345                   # Check session by partial ID
  %(prog)s f09084df-01c1-446f-9996-4b59    # Check by full session ID
  %(prog)s session_12345 --details         # Show detailed analysis
        """
    )
    
    parser.add_argument('session', type=str, nargs='?',
                        help='Session ID (full hash) or session name (session_*). If not provided, uses latest session.')
    parser.add_argument('--details', action='store_true',
                        help='Show detailed analysis tables')
    parser.add_argument('--db-path', type=str, default='data/minotaur.duckdb',
                        help='Path to database')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = MCTSHealthChecker(args.db_path)
    
    # Resolve session ID and name
    session_info = checker.get_session_info(args.session)
    if not session_info:
        if args.session is None:
            error_msg = "❌ No sessions found in database"
        else:
            error_msg = f"❌ Session '{args.session}' not found"
        
        if RICH_AVAILABLE:
            console.print(f"[red]{error_msg}[/red]")
        else:
            print(error_msg)
        sys.exit(1)
    
    session_id, session_name = session_info
    
    # Run health checks
    results = checker.run_health_checks(session_id, session_name)
    
    # Print summary report
    checker.print_summary_report(session_id, session_name, results)
    
    # Print detailed analysis if requested
    if args.details:
        checker.print_detailed_report(session_id, session_name)
    
    # Exit with appropriate code
    has_failures = any(r.status == HealthStatus.FAIL for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()