"""Unit tests for MCTS Engine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime

from src.mcts_engine import MCTSEngine, FeatureNode
from src.feature_space import FeatureOperation


class TestFeatureNode:
    """Test FeatureNode class functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        operation = FeatureOperation(
            name="test_op",
            operation_type="transform",
            operation_subtype="numeric",
            parameters={},
            new_features=["feature_1"],
            description="Test operation"
        )
        
        node = FeatureNode(
            state_id="test_state",
            parent=None,
            operation_that_created_this=operation,
            features_before=["base_feature"],
            features_after=["base_feature", "feature_1"]
        )
        
        assert node.state_id == "test_state"
        assert node.parent is None
        assert node.operation_that_created_this == operation
        assert len(node.features_after) == 2
        assert node.visit_count == 0
        assert node.total_score == 0.0
    
    def test_node_hierarchy(self):
        """Test parent-child relationships."""
        parent = FeatureNode("parent", None, None, [], [])
        child = FeatureNode("child", parent, None, [], [])
        
        assert child.parent == parent
        assert child in parent.children
        assert parent.depth == 0
        assert child.depth == 1
    
    def test_ucb1_score_calculation(self):
        """Test UCB1 score calculation."""
        parent = FeatureNode("parent", None, None, [], [])
        parent.visit_count = 10
        
        child = FeatureNode("child", parent, None, [], [])
        child.visit_count = 3
        child.total_score = 2.1
        
        # Calculate UCB1 with exploration weight of 1.4
        ucb1 = child.ucb1_score(exploration_weight=1.4)
        
        # Expected: average_score + exploration_weight * sqrt(ln(parent_visits) / child_visits)
        expected_avg = 2.1 / 3
        expected_exploration = 1.4 * np.sqrt(np.log(10) / 3)
        expected = expected_avg + expected_exploration
        
        assert abs(ucb1 - expected) < 0.0001
    
    def test_is_fully_expanded(self):
        """Test fully expanded check."""
        node = FeatureNode("test", None, None, [], [])
        
        # No possible operations means fully expanded
        assert node.is_fully_expanded([])
        
        # With possible operations, not fully expanded initially
        operations = [
            FeatureOperation("op1", "type1", "subtype1", {}, ["f1"], ""),
            FeatureOperation("op2", "type2", "subtype2", {}, ["f2"], "")
        ]
        assert not node.is_fully_expanded(operations)
        
        # After adding children for all operations
        for op in operations:
            child = FeatureNode("child", node, op, [], [])
        assert node.is_fully_expanded(operations)


class TestMCTSEngine:
    """Test MCTS Engine functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'mcts': {
                'exploration_weight': 1.4,
                'max_tree_depth': 5,
                'min_visits_for_expansion': 1,
                'selection_strategy': 'ucb1',
                'enable_pruning': True,
                'pruning_threshold': 0.1
            },
            'session': {
                'max_iterations': 10,
                'save_interval': 5
            },
            'testing': {
                'use_mock_evaluator': True
            },
            'feature_space': {
                'max_features_per_operation': 3
            }
        }
    
    @pytest.fixture
    def mock_feature_space(self):
        """Create mock feature space."""
        mock = Mock()
        mock.get_possible_operations.return_value = [
            FeatureOperation("op1", "type1", "subtype1", {}, ["f1"], ""),
            FeatureOperation("op2", "type2", "subtype2", {}, ["f2"], "")
        ]
        mock.apply_operation.return_value = pd.DataFrame({
            'base_feature': [1, 2, 3],
            'new_feature': [4, 5, 6]
        })
        return mock
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        mock = Mock()
        mock.evaluate_features.return_value = 0.75
        return mock
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        mock = Mock()
        mock.log_exploration.return_value = None
        mock.update_best_score.return_value = None
        mock.get_session_analytics.return_value = {
            'total_explorations': 10,
            'best_score': 0.8,
            'unique_operations': 5
        }
        return mock
    
    def test_engine_initialization(self, mock_config, mock_feature_space, 
                                 mock_evaluator, mock_db):
        """Test MCTS engine initialization."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        assert engine.exploration_weight == 1.4
        assert engine.max_tree_depth == 5
        assert engine.root is not None
        assert engine.best_score == 0.0
        assert engine.iteration_count == 0
    
    def test_selection_phase(self, mock_config, mock_feature_space, 
                           mock_evaluator, mock_db):
        """Test MCTS selection phase."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        # Create a simple tree structure
        child1 = FeatureNode("child1", engine.root, None, [], [])
        child1.visit_count = 5
        child1.total_score = 3.5
        
        child2 = FeatureNode("child2", engine.root, None, [], [])
        child2.visit_count = 3
        child2.total_score = 2.4
        
        engine.root.visit_count = 10
        
        # Selection should choose based on UCB1
        selected = engine.selection()
        assert selected in [child1, child2]
    
    def test_expansion_phase(self, mock_config, mock_feature_space, 
                           mock_evaluator, mock_db):
        """Test MCTS expansion phase."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        # Expand from root
        expanded = engine.expansion(engine.root)
        
        assert expanded is not None
        assert expanded.parent == engine.root
        assert expanded in engine.root.children
        mock_feature_space.get_possible_operations.assert_called()
    
    @patch('pandas.DataFrame')
    def test_simulation_phase(self, mock_df, mock_config, mock_feature_space, 
                            mock_evaluator, mock_db):
        """Test MCTS simulation phase."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        node = FeatureNode("test", engine.root, None, ["f1"], ["f1", "f2"])
        
        # Mock the feature generation
        mock_features_df = Mock()
        mock_feature_space.generate_features.return_value = mock_features_df
        
        score = engine.simulation(node)
        
        assert score == 0.75
        assert node.evaluation_score == 0.75
        mock_evaluator.evaluate_features.assert_called_once()
    
    def test_backpropagation_phase(self, mock_config, mock_feature_space, 
                                  mock_evaluator, mock_db):
        """Test MCTS backpropagation phase."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        # Create a path from leaf to root
        child = FeatureNode("child", engine.root, None, [], [])
        grandchild = FeatureNode("grandchild", child, None, [], [])
        
        # Backpropagate score
        engine.backpropagation(grandchild, 0.8)
        
        assert grandchild.visit_count == 1
        assert grandchild.total_score == 0.8
        assert child.visit_count == 1
        assert child.total_score == 0.8
        assert engine.root.visit_count == 1
        assert engine.root.total_score == 0.8
    
    def test_pruning(self, mock_config, mock_feature_space, 
                    mock_evaluator, mock_db):
        """Test tree pruning functionality."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        # Create nodes with different scores
        good_child = FeatureNode("good", engine.root, None, [], [])
        good_child.visit_count = 10
        good_child.total_score = 8.0  # avg: 0.8
        
        bad_child = FeatureNode("bad", engine.root, None, [], [])
        bad_child.visit_count = 10
        bad_child.total_score = 1.0  # avg: 0.1
        
        engine.best_score = 0.9
        
        # Prune with threshold 0.1
        engine.prune_tree()
        
        # Bad child should be pruned (0.1 < 0.9 - 0.1)
        assert good_child in engine.root.children
        assert good_child.is_pruned == False
        assert bad_child.is_pruned == True
    
    @patch('src.mcts_engine.generate_tree_visualization')
    def test_run_single_iteration(self, mock_viz, mock_config, mock_feature_space, 
                                mock_evaluator, mock_db):
        """Test running a single MCTS iteration."""
        engine = MCTSEngine(
            config=mock_config,
            feature_space=mock_feature_space,
            evaluator=mock_evaluator,
            database=mock_db
        )
        
        # Run one iteration
        engine.run()
        
        assert engine.iteration_count == 1
        assert engine.best_score > 0
        mock_db.log_exploration.assert_called()
        
        # Check that best score was updated if improved
        if mock_evaluator.evaluate_features.return_value > 0:
            mock_db.update_best_score.assert_called()


class TestFeatureOperation:
    """Test FeatureOperation dataclass."""
    
    def test_operation_creation(self):
        """Test creating a feature operation."""
        op = FeatureOperation(
            name="polynomial_features",
            operation_type="transformation",
            operation_subtype="polynomial",
            parameters={"degree": 2},
            new_features=["x_squared", "y_squared"],
            description="Create polynomial features of degree 2"
        )
        
        assert op.name == "polynomial_features"
        assert op.parameters["degree"] == 2
        assert len(op.new_features) == 2
    
    def test_operation_equality(self):
        """Test operation equality comparison."""
        op1 = FeatureOperation(
            name="test", 
            operation_type="type",
            operation_subtype="subtype",
            parameters={"a": 1},
            new_features=["f1"],
            description="desc"
        )
        
        op2 = FeatureOperation(
            name="test",
            operation_type="type", 
            operation_subtype="subtype",
            parameters={"a": 1},
            new_features=["f1"],
            description="desc"
        )
        
        assert op1 == op2
        
        # Different parameters should not be equal
        op3 = FeatureOperation(
            name="test",
            operation_type="type",
            operation_subtype="subtype", 
            parameters={"a": 2},
            new_features=["f1"],
            description="desc"
        )
        
        assert op1 != op3