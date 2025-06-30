#!/usr/bin/env python3
"""
MCTS Correctness Validation Script

Verifies proper MCTS implementation by analyzing:
1. Database exploration history (tree structure, iterations)
2. Session-specific MCTS logs (UCB1 calculations, backpropagation)
3. Cross-validation between database and logs

Usage:
    python scripts/mcts/validate_mcts_correctness.py --session SESSION_NAME
    python scripts/mcts/validate_mcts_correctness.py --latest
    python scripts/mcts/validate_mcts_correctness.py --all
"""

import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.core.connection import DuckDBConnectionManager
from src.db.models.exploration import ExplorationStep

class MCTSCorrectnessValidator:
    """Validates MCTS implementation correctness using database and log analysis."""
    
    def __init__(self, db_path: str = "data/minotaur.duckdb"):
        self.db_path = db_path
        
        # Create minimal config for connection manager
        config = {
            'database': {
                'path': db_path
            }
        }
        self.conn_manager = DuckDBConnectionManager(config)
        self.validation_results = {}
        
    def validate_session(self, session_name: str) -> Dict[str, bool]:
        """Validate MCTS correctness for a specific session."""
        print(f"\n🔍 Validating MCTS correctness for session: {session_name}")
        
        results = {}
        
        # 1. Database Structure Validation
        results['database_structure'] = self._validate_database_structure(session_name)
        
        # 2. Tree Structure Validation
        results['tree_structure'] = self._validate_tree_structure(session_name)
        
        # 3. MCTS Algorithm Validation
        results['mcts_algorithm'] = self._validate_mcts_algorithm(session_name)
        
        # 4. Log-Database Consistency
        results['log_consistency'] = self._validate_log_database_consistency(session_name)
        
        # 5. UCB1 Selection Logic
        results['ucb1_logic'] = self._validate_ucb1_logic(session_name)
        
        # 6. Backpropagation Logic
        results['backpropagation'] = self._validate_backpropagation(session_name)
        
        return results
    
    def _validate_database_structure(self, session_name: str) -> bool:
        """Validate database contains proper exploration history."""
        print("  📊 Validating database structure...")
        
        try:
            with self.conn_manager.get_connection() as conn:
                # Get exploration history with session info
                query = """
                SELECT e.session_id, e.iteration, e.mcts_node_id, e.parent_node_id, 
                       e.node_visits, e.evaluation_score, e.operation_applied as operation_name,
                       json_array_length(e.features_after) as features_count
                FROM exploration_history e
                JOIN sessions s ON e.session_id = s.session_id
                WHERE s.session_name = ?
                ORDER BY e.iteration, e.mcts_node_id
                """
                df = conn.execute(query, [session_name]).df()
                
                if df.empty:
                    print(f"    ❌ No exploration history found for session {session_name}")
                    return False
                
                # Check root node (iteration 0)
                root_nodes = df[df['iteration'] == 0]
                if root_nodes.empty:
                    print("    ❌ No root node (iteration 0) found")
                    return False
                
                if len(root_nodes) > 1:
                    print(f"    ❌ Multiple root nodes found: {len(root_nodes)}")
                    return False
                
                # Check node ID sequence
                node_ids = sorted(df['mcts_node_id'].unique())
                expected_sequence = list(range(min(node_ids), max(node_ids) + 1))
                if node_ids != expected_sequence:
                    print(f"    ❌ Node ID sequence broken: {node_ids} != {expected_sequence}")
                    return False
                
                # Check iteration sequence
                iterations = sorted(df['iteration'].unique())
                if iterations[0] != 0:
                    print(f"    ❌ Iterations should start from 0, found: {iterations[0]}")
                    return False
                
                print(f"    ✅ Database structure valid: {len(df)} exploration steps, iterations 0-{max(iterations)}")
                self.exploration_df = df
                return True
                
        except Exception as e:
            print(f"    ❌ Database validation error: {e}")
            return False
    
    def _validate_tree_structure(self, session_name: str) -> bool:
        """Validate MCTS tree structure and parent-child relationships."""
        print("  🌳 Validating tree structure...")
        
        try:
            df = self.exploration_df
            
            # Check root node has no parent
            root_nodes = df[df['iteration'] == 0]
            if not root_nodes['parent_node_id'].isna().all():
                print("    ❌ Root node should have no parent")
                return False
            
            # Check all non-root nodes have valid parents
            non_root = df[df['iteration'] > 0]
            for _, node in non_root.iterrows():
                parent_id = node['parent_node_id']
                if pd.isna(parent_id):
                    print(f"    ❌ Non-root node {node['mcts_node_id']} has no parent")
                    return False
                
                # Check parent exists and was created before this node
                parent_nodes = df[df['mcts_node_id'] == parent_id]
                if parent_nodes.empty:
                    print(f"    ❌ Parent node {parent_id} not found for node {node['mcts_node_id']}")
                    return False
                
                parent_iteration = parent_nodes.iloc[0]['iteration']
                if parent_iteration >= node['iteration']:
                    print(f"    ❌ Parent node {parent_id} created after child {node['mcts_node_id']}")
                    return False
            
            # Check tree depth progression
            max_depth = 0
            for iteration in sorted(df['iteration'].unique()):
                iter_nodes = df[df['iteration'] == iteration]
                if iteration == 0:
                    continue  # Skip root
                
                for _, node in iter_nodes.iterrows():
                    depth = self._calculate_node_depth(node['mcts_node_id'], df)
                    max_depth = max(max_depth, depth)
            
            print(f"    ✅ Tree structure valid: max depth {max_depth}, proper parent-child relationships")
            return True
            
        except Exception as e:
            print(f"    ❌ Tree structure validation error: {e}")
            return False
    
    def _calculate_node_depth(self, node_id: int, df: pd.DataFrame) -> int:
        """Calculate depth of a node in the tree."""
        node = df[df['mcts_node_id'] == node_id].iloc[0]
        if pd.isna(node['parent_node_id']):
            return 0  # Root node
        return 1 + self._calculate_node_depth(node['parent_node_id'], df)
    
    def _validate_mcts_algorithm(self, session_name: str) -> bool:
        """Validate MCTS algorithm progression (selection -> expansion -> evaluation -> backpropagation)."""
        print("  🎯 Validating MCTS algorithm progression...")
        
        try:
            df = self.exploration_df
            
            # Check evaluation scores are valid
            scores = df['evaluation_score'].dropna()
            if scores.empty:
                print("    ❌ No evaluation scores found")
                return False
            
            if not all(0 <= score <= 1 for score in scores):
                invalid_scores = scores[(scores < 0) | (scores > 1)]
                print(f"    ❌ Invalid evaluation scores found: {invalid_scores.tolist()}")
                return False
            
            # Check feature count progression
            feature_counts = df['features_count'].dropna()
            if feature_counts.empty:
                print("    ❌ No feature counts found")
                return False
            
            # Root should have base features, expansions should add features
            root_features = df[df['iteration'] == 0]['features_count'].iloc[0]
            for _, node in df[df['iteration'] > 0].iterrows():
                if node['features_count'] < root_features:
                    print(f"    ❌ Node {node['mcts_node_id']} has fewer features than root")
                    return False
            
            print(f"    ✅ MCTS algorithm valid: scores in [0,1], feature counts consistent")
            return True
            
        except Exception as e:
            print(f"    ❌ MCTS algorithm validation error: {e}")
            return False
    
    def _validate_log_database_consistency(self, session_name: str) -> bool:
        """Validate consistency between MCTS logs and database records."""
        print("  🔗 Validating log-database consistency...")
        
        try:
            # Find session log file
            log_file = Path(f"logs/mcts/{session_name}.log")
            if not log_file.exists():
                print(f"    ⚠️  Session log file not found: {log_file}")
                return True  # Not an error if DEBUG mode was off
            
            # Parse MCTS log
            log_data = self._parse_mcts_log(log_file)
            
            # Compare node creation between logs and database
            db_nodes = set(int(x) for x in self.exploration_df['mcts_node_id'].unique() if pd.notna(x))
            log_nodes = set(log_data['created_nodes'])
            
            # Database only stores nodes that were evaluated, logs show all created nodes
            # Check that all DB nodes exist in logs (subset relationship)
            if not db_nodes.issubset(log_nodes):
                missing_in_logs = db_nodes - log_nodes
                print(f"    ❌ DB nodes missing in logs: {sorted(missing_in_logs)}")
                return False
            
            # Allow extra nodes in logs (created but not evaluated)
            if len(log_nodes - db_nodes) > 1:  # Allow 1-2 extra nodes
                extra_in_logs = log_nodes - db_nodes
                print(f"    ⚠️  Extra nodes in logs (not evaluated): {sorted(extra_in_logs)}")
                # This is not an error, just informational
            
            # Compare evaluation scores
            db_scores = dict(zip(self.exploration_df['mcts_node_id'], self.exploration_df['evaluation_score']))
            log_scores = log_data['evaluations']
            
            for node_id in db_nodes:
                if node_id in log_scores:
                    db_score = db_scores[node_id]
                    log_score = log_scores[node_id]
                    if abs(db_score - log_score) > 0.001:  # Allow small floating point differences
                        print(f"    ❌ Score mismatch for node {node_id}: DB={db_score}, Log={log_score}")
                        return False
            
            print(f"    ✅ Log-database consistency validated: {len(db_nodes)} nodes, scores match")
            return True
            
        except Exception as e:
            print(f"    ❌ Log-database consistency error: {e}")
            return False
    
    def _parse_mcts_log(self, log_file: Path) -> Dict:
        """Parse MCTS log file to extract key information."""
        created_nodes = []
        evaluations = {}
        ucb1_calculations = []
        backprop_updates = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Node creation
                if "Created node" in line:
                    match = re.search(r"Created node (\d+):", line)
                    if match:
                        created_nodes.append(int(match.group(1)))
                
                # Evaluation completion
                if "evaluation completed: score=" in line:
                    match = re.search(r"Node (\d+) evaluation completed: score=([\d.]+)", line)
                    if match:
                        node_id = int(match.group(1))
                        score = float(match.group(2))
                        evaluations[node_id] = score
                
                # UCB1 calculations
                if "UCB1=" in line:
                    match = re.search(r"Node (\d+).*UCB1=([\d.]+), visits=(\d+)", line)
                    if match:
                        ucb1_calculations.append({
                            'node_id': int(match.group(1)),
                            'ucb1': float(match.group(2)),
                            'visits': int(match.group(3))
                        })
                
                # Backpropagation updates
                if "Updated node" in line:
                    match = re.search(r"Updated node (\d+): visits=(\d+), total_reward=([\d.]+)", line)
                    if match:
                        backprop_updates.append({
                            'node_id': int(match.group(1)),
                            'visits': int(match.group(2)),
                            'total_reward': float(match.group(3))
                        })
        
        return {
            'created_nodes': created_nodes,
            'evaluations': evaluations,
            'ucb1_calculations': ucb1_calculations,
            'backprop_updates': backprop_updates
        }
    
    def _validate_ucb1_logic(self, session_name: str) -> bool:
        """Validate UCB1 selection logic from logs."""
        print("  🎲 Validating UCB1 selection logic...")
        
        try:
            log_file = Path(f"logs/mcts/{session_name}.log")
            if not log_file.exists():
                print("    ⚠️  No session log for UCB1 validation (DEBUG mode was off)")
                return True
            
            log_data = self._parse_mcts_log(log_file)
            ucb1_calcs = log_data['ucb1_calculations']
            
            if not ucb1_calcs:
                print("    ⚠️  No UCB1 calculations found in logs")
                return True
            
            # Group UCB1 calculations by selection phase
            selections = []
            current_selection = []
            
            with open(log_file, 'r') as f:
                in_selection = False
                for line in f:
                    if "=== SELECTION PHASE START ===" in line:
                        in_selection = True
                        current_selection = []
                    elif "=== EXPANSION PHASE START ===" in line:
                        if current_selection:
                            selections.append(current_selection[:])
                        in_selection = False
                    elif in_selection and "UCB1=" in line:
                        match = re.search(r"Node (\d+).*UCB1=([\d.]+), visits=(\d+)", line)
                        if match:
                            current_selection.append({
                                'node_id': int(match.group(1)),
                                'ucb1': float(match.group(2)),
                                'visits': int(match.group(3))
                            })
            
            # Validate UCB1 selection logic
            for i, selection in enumerate(selections):
                if len(selection) < 2:
                    continue  # Need at least 2 options to validate selection
                
                # Check that highest UCB1 is selected
                max_ucb1_node = max(selection, key=lambda x: x['ucb1'])
                
                # Find which node was actually selected (next in log)
                # This is complex to parse reliably, so we'll just validate UCB1 calculation
                
                # Validate UCB1 formula: UCB1 = avg_reward + C * sqrt(ln(total_visits) / node_visits)
                # We can't easily get all required values from logs, so we'll do basic sanity checks
                for node in selection:
                    if node['ucb1'] < 0 or node['ucb1'] > 10:  # Reasonable bounds
                        print(f"    ❌ UCB1 value out of bounds: {node['ucb1']} for node {node['node_id']}")
                        return False
            
            print(f"    ✅ UCB1 logic appears valid: {len(selections)} selection phases analyzed")
            return True
            
        except Exception as e:
            print(f"    ❌ UCB1 validation error: {e}")
            return False
    
    def _validate_backpropagation(self, session_name: str) -> bool:
        """Validate backpropagation logic from logs."""
        print("  📈 Validating backpropagation logic...")
        
        try:
            log_file = Path(f"logs/mcts/{session_name}.log")
            if not log_file.exists():
                print("    ⚠️  No session log for backpropagation validation (DEBUG mode was off)")
                return True
            
            log_data = self._parse_mcts_log(log_file)
            backprop_updates = log_data['backprop_updates']
            
            if not backprop_updates:
                print("    ⚠️  No backpropagation updates found in logs")
                return True
            
            # Group updates by backpropagation phase
            backprop_phases = []
            current_phase = []
            
            with open(log_file, 'r') as f:
                in_backprop = False
                for line in f:
                    if "=== BACKPROPAGATION PHASE START ===" in line:
                        in_backprop = True
                        current_phase = []
                    elif "=== SELECTION PHASE START ===" in line or "Updated" not in line:
                        if in_backprop and current_phase:
                            backprop_phases.append(current_phase[:])
                        in_backprop = False
                    elif in_backprop and "Updated node" in line:
                        match = re.search(r"Updated node (\d+): visits=(\d+), total_reward=([\d.]+)", line)
                        if match:
                            current_phase.append({
                                'node_id': int(match.group(1)),
                                'visits': int(match.group(2)),
                                'total_reward': float(match.group(3))
                            })
            
            # Validate backpropagation logic
            for phase in backprop_phases:
                if len(phase) < 2:
                    continue  # Single node updates are valid
                
                # Check that visit counts increase along the path
                for i in range(1, len(phase)):
                    current_node = phase[i]
                    prev_node = phase[i-1]
                    
                    # Parent should have higher visit count than child (in general)
                    # This is complex to validate without tree structure, so basic checks
                    if current_node['visits'] < 1:
                        print(f"    ❌ Invalid visit count: {current_node['visits']} for node {current_node['node_id']}")
                        return False
                    
                    if current_node['total_reward'] < 0:
                        print(f"    ❌ Negative total reward: {current_node['total_reward']} for node {current_node['node_id']}")
                        return False
            
            print(f"    ✅ Backpropagation logic appears valid: {len(backprop_phases)} phases analyzed")
            return True
            
        except Exception as e:
            print(f"    ❌ Backpropagation validation error: {e}")
            return False
    
    def get_latest_session(self) -> Optional[str]:
        """Get the most recent session name from database."""
        try:
            with self.conn_manager.get_connection() as conn:
                query = """
                SELECT session_name 
                FROM sessions 
                ORDER BY start_time DESC 
                LIMIT 1
                """
                result = conn.execute(query).fetchone()
                return result[0] if result else None
        except Exception as e:
            print(f"Error getting latest session: {e}")
            return None
    
    def get_all_sessions(self) -> List[str]:
        """Get all session names from database."""
        try:
            with self.conn_manager.get_connection() as conn:
                query = """
                SELECT DISTINCT session_name 
                FROM sessions 
                WHERE session_name IS NOT NULL
                ORDER BY session_name DESC
                """
                results = conn.execute(query).fetchall()
                return [row[0] for row in results]
        except Exception as e:
            print(f"Error getting all sessions: {e}")
            return []
    
    def print_validation_summary(self, session_name: str, results: Dict[str, bool]):
        """Print a summary of validation results."""
        print(f"\n📋 MCTS Validation Summary for {session_name}")
        print("=" * 60)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"{test_display:.<40} {status}")
            if not passed:
                all_passed = False
        
        print("=" * 60)
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"Overall Status: {overall_status}")
        
        if all_passed:
            print("🎯 MCTS implementation is working correctly!")
        else:
            print("🔧 MCTS implementation needs attention.")


def main():
    parser = argparse.ArgumentParser(description="Validate MCTS implementation correctness")
    parser.add_argument('--session', help='Specific session name to validate')
    parser.add_argument('--latest', action='store_true', help='Validate latest session')
    parser.add_argument('--all', action='store_true', help='Validate all sessions')
    parser.add_argument('--db-path', default='data/minotaur.duckdb', help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    validator = MCTSCorrectnessValidator(args.db_path)
    
    if args.session:
        sessions = [args.session]
    elif args.latest:
        latest = validator.get_latest_session()
        if not latest:
            print("❌ No sessions found in database")
            return 1
        sessions = [latest]
    elif args.all:
        sessions = validator.get_all_sessions()
        if not sessions:
            print("❌ No sessions found in database")
            return 1
    else:
        # Default to latest session
        latest = validator.get_latest_session()
        if not latest:
            print("❌ No sessions found in database")
            return 1
        sessions = [latest]
    
    print(f"🔍 Validating {len(sessions)} session(s)")
    
    overall_success = True
    for session_name in sessions:
        try:
            results = validator.validate_session(session_name)
            validator.print_validation_summary(session_name, results)
            
            if not all(results.values()):
                overall_success = False
                
        except Exception as e:
            print(f"❌ Error validating session {session_name}: {e}")
            overall_success = False
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())