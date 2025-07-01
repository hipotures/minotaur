#!/usr/bin/env python3
"""
MCTS Tree Visualization Script

Creates visual representation of MCTS exploration tree for debugging.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.core.connection import DuckDBConnectionManager

try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.text import Text
    from rich import box
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

if RICH_AVAILABLE:
    console = Console()


class MCTSTreeVisualizer:
    """Visualize MCTS exploration tree."""
    
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
    
    def get_tree_data(self, session_id: str, max_depth: int = 10) -> Dict:
        """Get tree structure from database."""
        with self.conn_manager.get_connection() as conn:
            query = """
                SELECT 
                    mcts_node_id,
                    parent_node_id,
                    operation_applied,
                    evaluation_score,
                    node_visits,
                    json_array_length(features_after) as feature_count,
                    mcts_ucb1_score,
                    iteration
                FROM exploration_history
                WHERE session_id = ? AND iteration <= ?
                ORDER BY iteration, mcts_node_id
            """
            
            results = conn.execute(query, [session_id, max_depth]).fetchall()
            
            # Build tree structure
            nodes = {}
            root_id = None
            
            for row in results:
                node_id, parent_id, operation, score, visits, features, ucb1, depth = row
                
                nodes[node_id] = {
                    'id': node_id,
                    'parent': parent_id,
                    'operation': operation,
                    'score': score,
                    'visits': visits,
                    'features': features,
                    'ucb1': ucb1,
                    'depth': depth,
                    'children': []
                }
                
                if parent_id is None:
                    root_id = node_id
            
            # Link children to parents
            for node_id, node in nodes.items():
                if node['parent'] is not None and node['parent'] in nodes:
                    nodes[node['parent']]['children'].append(node_id)
            
            return {'nodes': nodes, 'root': root_id}
    
    def create_node_label(self, node: Dict) -> str:
        """Create formatted label for a node."""
        if RICH_AVAILABLE:
            # Color based on score
            if node['score'] > 0.75:
                color = "green"
            elif node['score'] > 0.65:
                color = "yellow"
            else:
                color = "red"
            
            # Format UCB1 score
            ucb1_str = f"{node['ucb1']:.3f}" if node['ucb1'] and node['ucb1'] != float('inf') else "∞"
            
            return (f"[bold]{node['operation']}[/bold] "
                   f"([{color}]score: {node['score']:.3f}[/{color}], "
                   f"visits: {node['visits']}, "
                   f"features: {node['features']}, "
                   f"UCB1: {ucb1_str})")
        else:
            ucb1_str = f"{node['ucb1']:.3f}" if node['ucb1'] and node['ucb1'] != float('inf') else "inf"
            return f"{node['operation']} (score: {node['score']:.3f}, visits: {node['visits']}, features: {node['features']}, UCB1: {ucb1_str})"
    
    def build_tree_visual(self, tree_data: Dict, node_id: int = None, tree_obj: Tree = None) -> Tree:
        """Build rich tree visualization."""
        if not RICH_AVAILABLE:
            return None
        
        nodes = tree_data['nodes']
        
        if node_id is None:
            node_id = tree_data['root']
            node = nodes[node_id]
            tree_obj = Tree(self.create_node_label(node))
        
        node = nodes[node_id]
        
        # Sort children by UCB1 score (exploration priority)
        children = sorted(
            node['children'], 
            key=lambda c: nodes[c]['ucb1'] if nodes[c]['ucb1'] else 0,
            reverse=True
        )
        
        for child_id in children:
            child = nodes[child_id]
            child_branch = tree_obj.add(self.create_node_label(child))
            
            if child['children']:
                self.build_tree_visual(tree_data, child_id, child_branch)
        
        return tree_obj
    
    def print_tree_text(self, tree_data: Dict, node_id: int = None, prefix: str = "", is_last: bool = True):
        """Print tree in text format for non-rich environments."""
        nodes = tree_data['nodes']
        
        if node_id is None:
            node_id = tree_data['root']
        
        node = nodes[node_id]
        
        # Print current node
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.create_node_label(node))
        
        # Prepare prefix for children
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension
        
        # Sort children by UCB1 score
        children = sorted(
            node['children'], 
            key=lambda c: nodes[c]['ucb1'] if nodes[c]['ucb1'] else 0,
            reverse=True
        )
        
        # Print children
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            self.print_tree_text(tree_data, child_id, child_prefix, is_last_child)
    
    def print_summary_stats(self, tree_data: Dict):
        """Print summary statistics about the tree."""
        nodes = tree_data['nodes']
        
        if RICH_AVAILABLE:
            # Create summary table
            table = Table(title="Tree Statistics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            # Calculate stats
            total_nodes = len(nodes)
            max_depth = max(n['depth'] for n in nodes.values()) if nodes else 0
            avg_score = sum(n['score'] for n in nodes.values()) / len(nodes) if nodes else 0
            max_score = max(n['score'] for n in nodes.values()) if nodes else 0
            total_visits = sum(n['visits'] for n in nodes.values())
            avg_features = sum(n['features'] for n in nodes.values()) / len(nodes) if nodes else 0
            
            table.add_row("Total Nodes", str(total_nodes))
            table.add_row("Max Depth", str(max_depth))
            table.add_row("Average Score", f"{avg_score:.4f}")
            table.add_row("Best Score", f"{max_score:.4f}")
            table.add_row("Total Visits", str(total_visits))
            table.add_row("Avg Features/Node", f"{avg_features:.1f}")
            
            console.print(table)
        else:
            # Plain text stats
            print("\n" + "="*40)
            print("Tree Statistics")
            print("="*40)
            total_nodes = len(nodes)
            max_depth = max(n['depth'] for n in nodes.values()) if nodes else 0
            avg_score = sum(n['score'] for n in nodes.values()) / len(nodes) if nodes else 0
            max_score = max(n['score'] for n in nodes.values()) if nodes else 0
            total_visits = sum(n['visits'] for n in nodes.values())
            avg_features = sum(n['features'] for n in nodes.values()) / len(nodes) if nodes else 0
            
            print(f"Total Nodes: {total_nodes}")
            print(f"Max Depth: {max_depth}")
            print(f"Average Score: {avg_score:.4f}")
            print(f"Best Score: {max_score:.4f}")
            print(f"Total Visits: {total_visits}")
            print(f"Avg Features/Node: {avg_features:.1f}")
            print("="*40 + "\n")
    
    def get_session_info(self, session_input: Optional[str]) -> Optional[Tuple[str, str]]:
        """Get session ID and name from input. Returns (session_id, session_name) or None."""
        with self.conn_manager.get_connection() as conn:
            if session_input is None:
                query = """
                    SELECT session_id, session_name FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """
                result = conn.execute(query).fetchone()
                return (result[0], result[1]) if result else None
            
            # Try direct lookup
            query = "SELECT session_id, session_name FROM sessions WHERE session_id = ?"
            result = conn.execute(query, [session_input]).fetchone()
            if result:
                return (result[0], result[1])
            
            # Try pattern matching
            query = """
                SELECT session_id, session_name FROM sessions 
                WHERE session_id LIKE ? OR session_name LIKE ?
                ORDER BY start_time DESC 
                LIMIT 1
            """
            result = conn.execute(query, [f"{session_input}%", f"{session_input}%"]).fetchone()
            return (result[0], result[1]) if result else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize MCTS exploration tree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Visualize latest session
  %(prog)s session_123              # Visualize specific session
  %(prog)s --max-depth 5            # Limit tree depth
  %(prog)s --no-stats               # Skip statistics
        """
    )
    
    parser.add_argument('session', type=str, nargs='?',
                        help='Session ID or name. If not provided, uses latest.')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum tree depth to display (default: 10)')
    parser.add_argument('--db-path', type=str, default='data/minotaur.duckdb',
                        help='Path to database')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip summary statistics')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MCTSTreeVisualizer(args.db_path)
    
    # Get session ID and name
    session_info = visualizer.get_session_info(args.session)
    if not session_info:
        error_msg = "❌ No sessions found" if args.session is None else f"❌ Session '{args.session}' not found"
        if RICH_AVAILABLE:
            console.print(f"[red]{error_msg}[/red]")
        else:
            print(error_msg)
        sys.exit(1)
    
    session_id, session_name = session_info
    
    # Get tree data
    tree_data = visualizer.get_tree_data(session_id, args.max_depth)
    
    if not tree_data['nodes']:
        if RICH_AVAILABLE:
            console.print("[red]No tree data found for session[/red]")
        else:
            print("No tree data found for session")
        sys.exit(1)
    
    # Print header
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            f"[bold]MCTS Tree Visualization[/bold]\n"
            f"Session: [cyan]{session_name}[/cyan] ([dim]{session_id[:8]}...[/dim])\n"
            f"Max Depth: {args.max_depth}",
            style="blue"
        ))
        console.print()
    else:
        print("="*50)
        print(f"MCTS Tree Visualization")
        print(f"Session: {session_name} ({session_id[:8]}...)")
        print(f"Max Depth: {args.max_depth}")
        print("="*50 + "\n")
    
    # Print statistics
    if not args.no_stats:
        visualizer.print_summary_stats(tree_data)
        if RICH_AVAILABLE:
            console.print()
    
    # Visualize tree
    if RICH_AVAILABLE:
        tree_visual = visualizer.build_tree_visual(tree_data)
        console.print(tree_visual)
    else:
        visualizer.print_tree_text(tree_data)
    
    if RICH_AVAILABLE:
        console.print()


if __name__ == "__main__":
    main()