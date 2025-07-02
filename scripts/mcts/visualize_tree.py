#!/usr/bin/env python3
"""
MCTS Tree Visualization Script

Displays MCTS tree structure with nodes, visits, scores, and relationships.
Shows the exploration pattern and helps debug tree growth issues.

Usage:
    python scripts/mcts/visualize_tree.py [SESSION_ID] [--depth N] [--format text|json]
    python scripts/mcts/visualize_tree.py --latest --depth 3
"""

import sys
import json
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.discovery_db import FeatureDiscoveryDB
from src.utils import create_session_resolver, SessionResolutionError, add_universal_optional_session_args, validate_optional_session_args


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


class MCTSTreeNode:
    """Represents a node in the MCTS tree for visualization."""
    
    def __init__(self, node_id: int, operation: str, parent_id: Optional[int] = None):
        self.node_id = node_id
        self.operation = operation
        self.parent_id = parent_id
        self.children: List['MCTSTreeNode'] = []
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.ucb1_score = 0.0
        self.depth = 0
        
        # Feature information
        self.features_before = []
        self.features_after = []
        self.feature_count_change = 0
        
        # Evaluation info
        self.evaluation_score = 0.0
        self.evaluation_time = 0.0
        
    def add_child(self, child: 'MCTSTreeNode'):
        """Add a child node."""
        child.depth = self.depth + 1
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "visits": self.visits,
            "avg_reward": self.avg_reward,
            "ucb1_score": self.ucb1_score,
            "evaluation_score": self.evaluation_score,
            "feature_count_change": self.feature_count_change,
            "children": [child.to_dict() for child in self.children]
        }


def load_tree_from_database(db: FeatureDiscoveryDB, session_id: str, max_depth: Optional[int] = None) -> Optional[MCTSTreeNode]:
    """Load MCTS tree structure from database."""
    
    # Get all exploration records for the session
    query = """
    SELECT mcts_node_id, operation_applied, parent_node_id,
           node_visits, mcts_ucb1_score, evaluation_score,
           features_before, features_after, evaluation_time
    FROM exploration_history 
    WHERE session_id = ?
    ORDER BY iteration
    """
    
    records = db.db_service.connection_manager.execute_query(query, params=(session_id,), fetch='all')
    
    if not records:
        return None
    
    # Build node lookup
    nodes = {}
    root = None
    
    for record in records:
        (node_id, operation, parent_id, visits, ucb1_score, eval_score, 
         features_before, features_after, eval_time) = record
        
        if node_id is None:
            continue  # Skip records without node IDs
        
        # Create node if not exists
        if node_id not in nodes:
            nodes[node_id] = MCTSTreeNode(node_id, operation or 'root', parent_id)
        
        node = nodes[node_id]
        
        # Update node statistics (keep the latest values)
        node.visits = visits or 0
        node.total_reward = 0.0  # Not available in current schema
        node.avg_reward = 0.0    # Not available in current schema
        node.ucb1_score = ucb1_score or 0.0
        node.depth = 0  # Not available in current schema
        node.evaluation_score = eval_score or 0.0
        node.evaluation_time = eval_time or 0.0
        
        # Feature information
        node.features_before = features_before or []
        node.features_after = features_after or []
        node.feature_count_change = len(node.features_after) - len(node.features_before)
        
        # Identify root
        if parent_id is None or operation == 'root':
            root = node
    
    # Build parent-child relationships
    for node in nodes.values():
        if node.parent_id and node.parent_id in nodes:
            parent = nodes[node.parent_id]
            if node not in parent.children:  # Avoid duplicates
                parent.add_child(node)
    
    # Apply depth filter
    if max_depth is not None:
        root = _filter_tree_by_depth(root, max_depth)
    
    return root


def _filter_tree_by_depth(node: MCTSTreeNode, max_depth: int) -> MCTSTreeNode:
    """Filter tree to only include nodes up to max_depth."""
    if node.depth >= max_depth:
        # Create a copy without children
        filtered = MCTSTreeNode(node.node_id, node.operation, node.parent_id)
        _copy_node_data(node, filtered)
        return filtered
    
    # Create copy with filtered children
    filtered = MCTSTreeNode(node.node_id, node.operation, node.parent_id)
    _copy_node_data(node, filtered)
    
    for child in node.children:
        if child.depth <= max_depth:
            filtered_child = _filter_tree_by_depth(child, max_depth)
            filtered.add_child(filtered_child)
    
    return filtered


def _copy_node_data(source: MCTSTreeNode, target: MCTSTreeNode):
    """Copy node data from source to target."""
    target.visits = source.visits
    target.total_reward = source.total_reward
    target.avg_reward = source.avg_reward
    target.ucb1_score = source.ucb1_score
    target.depth = source.depth
    target.features_before = source.features_before
    target.features_after = source.features_after
    target.feature_count_change = source.feature_count_change
    target.evaluation_score = source.evaluation_score
    target.evaluation_time = source.evaluation_time


def print_tree_text(node: MCTSTreeNode, prefix: str = "", is_last: bool = True, show_features: bool = False):
    """Print tree in text format."""
    
    # Node connector
    connector = "└── " if is_last else "├── "
    
    # Node information
    node_info = f"{node.operation} (ID:{node.node_id})"
    
    # Statistics
    stats = f"[visits:{node.visits}, avg:{node.avg_reward:.4f}, ucb1:{node.ucb1_score:.4f}]"
    
    # Feature change info
    feature_info = ""
    if show_features and node.feature_count_change != 0:
        feature_info = f" +{node.feature_count_change}feat"
    
    print(f"{prefix}{connector}{node_info} {stats}{feature_info}")
    
    # Print children
    for i, child in enumerate(node.children):
        is_child_last = (i == len(node.children) - 1)
        child_prefix = prefix + ("    " if is_last else "│   ")
        print_tree_text(child, child_prefix, is_child_last, show_features)


def get_tree_statistics(node: MCTSTreeNode) -> Dict[str, Any]:
    """Get comprehensive tree statistics."""
    
    def collect_stats(n: MCTSTreeNode, stats: Dict[str, Any]):
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], n.depth)
        stats['total_visits'] += n.visits
        stats['total_rewards'].append(n.avg_reward)
        
        if n.visits > 1:
            stats['multi_visit_nodes'] += 1
        
        if n.feature_count_change > 0:
            stats['nodes_adding_features'] += 1
        
        for child in n.children:
            collect_stats(child, stats)
    
    stats = {
        'total_nodes': 0,
        'max_depth': 0,
        'total_visits': 0,
        'total_rewards': [],
        'multi_visit_nodes': 0,
        'nodes_adding_features': 0
    }
    
    collect_stats(node, stats)
    
    # Calculate derived statistics
    stats['avg_visits_per_node'] = stats['total_visits'] / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
    stats['avg_reward'] = sum(stats['total_rewards']) / len(stats['total_rewards']) if stats['total_rewards'] else 0
    stats['max_reward'] = max(stats['total_rewards']) if stats['total_rewards'] else 0
    stats['exploration_ratio'] = stats['multi_visit_nodes'] / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
    
    # Remove the list for cleaner output
    del stats['total_rewards']
    
    return stats


def find_best_path(node: MCTSTreeNode) -> List[MCTSTreeNode]:
    """Find the path to the highest reward leaf node."""
    if not node.children:
        return [node]
    
    best_child = max(node.children, key=lambda c: c.avg_reward)
    return [node] + find_best_path(best_child)


def get_latest_session_id(db: FeatureDiscoveryDB) -> Optional[str]:
    """Get the latest session ID."""
    query = "SELECT session_id FROM sessions ORDER BY start_time DESC LIMIT 1"
    result = db.db_service.connection_manager.execute_query(query, fetch='one')
    return result[0] if result else None


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize MCTS tree structure")
    
    # Add universal optional session argument
    add_universal_optional_session_args(parser, "Tree visualization")
    
    parser.add_argument("--depth", type=int, help="Maximum depth to display")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--features", action="store_true", help="Show feature count changes")
    parser.add_argument("--stats", action="store_true", help="Show tree statistics")
    parser.add_argument("--best-path", action="store_true", help="Highlight best path")
    
    args = parser.parse_args()
    
    # Load configuration and initialize database
    try:
        config = load_default_config()
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
        
        print(f"🌳 Visualizing session: {session_id[:8]}... ({session_info.session_name})")
        print(f"   Started: {session_info.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Iterations: {session_info.total_iterations}")
        print()
        
        # Close resolver connection to avoid conflicts
        resolver.connection_manager.close()
        
    except SessionResolutionError as e:
        print(f"❌ Session resolution failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error resolving session: {e}")
        return 1
    
    try:
        # Load tree
        root = load_tree_from_database(db, session_id, args.depth)
        
        if not root:
            print("❌ No tree data found for session")
            return 1
        
        # Output based on format
        if args.format == "json":
            print(json.dumps(root.to_dict(), indent=2))
        else:
            print("\n" + "="*60)
            print(f"MCTS TREE VISUALIZATION - Session {session_id}")
            print("="*60)
            
            # Show tree statistics
            if args.stats:
                stats = get_tree_statistics(root)
                print("\n📊 Tree Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
            
            # Show best path
            if args.best_path:
                best_path = find_best_path(root)
                print("\n🎯 Best Path (highest rewards):")
                for i, node in enumerate(best_path):
                    print(f"  {i+1}. {node.operation} (reward: {node.avg_reward:.4f})")
                print()
            
            # Print tree structure
            print("🌳 Tree Structure:")
            print_tree_text(root, show_features=args.features)
            
            print(f"\nDepth filter: {args.depth if args.depth else 'None'}")
            print(f"Total nodes shown: {get_tree_statistics(root)['total_nodes']}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return 1
    
    finally:
        if hasattr(db, 'close'):
            db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())