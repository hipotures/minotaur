#!/usr/bin/env python3
"""
Diagnostic script to demonstrate the feature accumulation bug in get_available_operations().

This script shows how using 'applied_operations' instead of 'operation_that_created_this'
causes incorrect feature accumulation across sibling nodes.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_space import FeatureSpace
from src.mcts_engine import FeatureNode

def demonstrate_bug():
    """Show how the bug causes incorrect feature accumulation."""
    
    print("=" * 80)
    print("FEATURE ACCUMULATION BUG DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create mock feature space
    feature_space = Mock(spec=FeatureSpace)
    
    # Mock the method that returns operation features
    def mock_get_operation_output_columns(op_name):
        """Mock feature outputs for different operations."""
        feature_map = {
            'cabin_features': ['cabin_deck', 'cabin_num', 'has_cabin'],
            'fare_features': ['fare_binned', 'fare_log', 'fare_per_person'],
            'age_features': ['age_binned', 'is_child', 'is_adult']
        }
        return feature_map.get(op_name, [])
    
    feature_space._get_operation_output_columns = mock_get_operation_output_columns
    
    # Create a root node
    root = FeatureNode(
        base_features=['Age', 'Fare', 'Pclass', 'Sex'],
        applied_operations=[],
        operation_that_created_this='root'
    )
    
    print("ROOT NODE:")
    print(f"  Base features: {root.base_features}")
    print(f"  Applied operations: {root.applied_operations}")
    print()
    
    # Create three child nodes with different operations
    child1 = FeatureNode(
        parent=root,
        base_features=root.base_features,
        applied_operations=['cabin_features'],  # This is cumulative
        operation_that_created_this='cabin_features'  # This is just this node
    )
    
    child2 = FeatureNode(
        parent=root,
        base_features=root.base_features,
        applied_operations=['fare_features'],
        operation_that_created_this='fare_features'
    )
    
    child3 = FeatureNode(
        parent=root,
        base_features=root.base_features,
        applied_operations=['age_features'],
        operation_that_created_this='age_features'
    )
    
    print("CHILD NODES (siblings, all children of root):")
    for i, child in enumerate([child1, child2, child3], 1):
        print(f"\nChild {i} (Node ID: {child.node_id}):")
        print(f"  Operation that created this: {child.operation_that_created_this}")
        print(f"  Applied operations: {child.applied_operations}")
    
    print("\n" + "-" * 80)
    print("SIMULATING get_available_operations() BEHAVIOR:")
    print("-" * 80)
    
    # Simulate the buggy behavior in get_available_operations
    for i, node in enumerate([child1, child2, child3], 1):
        print(f"\nFor Child {i}:")
        
        # Start with base features
        current_features = set(node.base_features)
        print(f"  Starting features: {current_features}")
        
        # BUG: Uses applied_operations (should use operation_that_created_this)
        for op_name in node.applied_operations:
            op_features = mock_get_operation_output_columns(op_name)
            print(f"  Adding features from '{op_name}': {op_features}")
            current_features.update(op_features)
        
        print(f"  Final features (BUGGY): {sorted(current_features)}")
        
        # Show what it SHOULD be
        correct_features = set(node.base_features)
        if node.operation_that_created_this and node.operation_that_created_this != 'root':
            correct_features.update(mock_get_operation_output_columns(node.operation_that_created_this))
        print(f"  Final features (CORRECT): {sorted(correct_features)}")
    
    print("\n" + "=" * 80)
    print("IMPACT OF THE BUG:")
    print("=" * 80)
    print("""
The bug doesn't create identical features across siblings, but it incorrectly
limits available operations based on accumulated features instead of the actual
features at each node.

Example: If we create deeper nodes:
- Child1 → GrandChild1 with 'statistical_features'
- Child1 → GrandChild2 with 'polynomial_features'

With the bug:
- GrandChild2 would accumulate: cabin_features + statistical_features + polynomial_features
- This would limit what operations are available, as it thinks it has more features than it does

Without the bug:
- GrandChild2 would only have: cabin_features + polynomial_features
- More operations would be available for exploration
""")

if __name__ == "__main__":
    demonstrate_bug()