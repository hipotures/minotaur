"""
MCTS Engine for Automated Feature Discovery

Core Monte Carlo Tree Search implementation for exploring feature space.
Includes UCB1 selection, node expansion, and backpropagation algorithms.
"""

import math
import random
import time
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import gc
import psutil
import os

from .timing import timed, timing_context, get_timing_collector, record_timing

logger = logging.getLogger(__name__)

@dataclass
class FeatureNode:
    """Node in the MCTS tree representing a feature state."""
    
    # Core MCTS attributes
    visit_count: int = 0
    total_reward: float = 0.0
    children: List['FeatureNode'] = field(default_factory=list)
    parent: Optional['FeatureNode'] = None
    
    # Feature engineering attributes
    base_features: Set[str] = field(default_factory=set)
    applied_operations: List[str] = field(default_factory=list)
    operation_that_created_this: Optional[str] = None
    
    # Evaluation results
    evaluation_score: Optional[float] = None
    evaluation_time: float = 0.0
    evaluation_count: int = 0
    
    # MCTS-specific
    is_fully_expanded: bool = False
    depth: int = 0
    node_id: Optional[int] = None
    
    # Performance tracking
    memory_usage_mb: Optional[float] = None
    feature_generation_time: float = 0.0
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        if self.parent:
            self.depth = self.parent.depth + 1
    
    @property
    def average_reward(self) -> float:
        """Average reward (evaluation score) for this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count
    
    @property
    def current_features(self) -> Set[str]:
        """Current set of features at this node (base + generated)."""
        # This will be computed lazily when needed
        return self.base_features.copy()
    
    def ucb1_score(self, exploration_weight: float = 1.4, parent_visits: int = None) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if parent_visits is None:
            parent_visits = self.parent.visit_count if self.parent else self.visit_count
        
        if parent_visits <= 0:
            return self.average_reward
        
        exploration_term = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        
        return self.average_reward + exploration_term
    
    def add_child(self, operation: str, features: Set[str] = None) -> 'FeatureNode':
        """Add a child node representing the application of an operation."""
        child = FeatureNode(
            parent=self,
            base_features=features or self.base_features.copy(),
            applied_operations=self.applied_operations + [operation],
            operation_that_created_this=operation,
            depth=self.depth + 1
        )
        
        self.children.append(child)
        return child
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def can_expand(self) -> bool:
        """Check if this node can be expanded (has unexplored operations)."""
        return not self.is_fully_expanded
    
    def select_best_child(self, exploration_weight: float = 1.4) -> Optional['FeatureNode']:
        """Select best child using UCB1 criterion."""
        if not self.children:
            return None
        
        return max(
            self.children, 
            key=lambda child: child.ucb1_score(exploration_weight, self.visit_count)
        )
    
    def update_reward(self, reward: float, evaluation_time: float = 0.0) -> None:
        """Update node statistics with new reward."""
        self.visit_count += 1
        self.total_reward += reward
        self.evaluation_time += evaluation_time
        self.evaluation_count += 1
        
        # Track memory usage if available
        try:
            process = psutil.Process(os.getpid())
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except:
            pass
    
    def get_path_from_root(self) -> List[str]:
        """Get the sequence of operations from root to this node."""
        path = []
        current = self
        while current.parent is not None:
            if current.operation_that_created_this:
                path.append(current.operation_that_created_this)
            current = current.parent
        return list(reversed(path))
    
    def get_best_path(self) -> List['FeatureNode']:
        """Get path from root to best leaf node in subtree."""
        if self.is_leaf():
            return [self]
        
        best_child = max(self.children, key=lambda c: c.average_reward)
        return [self] + best_child.get_best_path()
    
    def __str__(self) -> str:
        """String representation of the node."""
        return (f"FeatureNode(depth={self.depth}, visits={self.visit_count}, "
                f"avg_reward={self.average_reward:.4f}, children={len(self.children)}, "
                f"operation={self.operation_that_created_this})")
    
    def __repr__(self) -> str:
        return self.__str__()


class MCTSEngine:
    """
    Monte Carlo Tree Search engine for automated feature discovery.
    
    Explores the space of feature engineering operations using MCTS algorithm
    with UCB1 selection and random rollouts evaluated by AutoGluon.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MCTS engine with configuration."""
        self.config = config
        self.mcts_config = config['mcts']
        self.resource_config = config['resources']
        
        # MCTS parameters
        self.exploration_weight = self.mcts_config['exploration_weight']
        self.max_tree_depth = self.mcts_config['max_tree_depth']
        self.expansion_threshold = self.mcts_config['expansion_threshold']
        self.max_children_per_node = self.mcts_config['max_children_per_node']
        self.expansion_budget = self.mcts_config['expansion_budget']
        
        # Tree management
        self.root: Optional[FeatureNode] = None
        self.current_iteration = 0
        self.max_iterations = config['session']['max_iterations']
        self.max_runtime_hours = config['session']['max_runtime_hours']
        
        # Performance tracking
        self.start_time = time.time()
        self.best_score = 0.0
        self.best_node: Optional[FeatureNode] = None
        self.total_evaluations = 0
        
        # Memory management
        self.max_nodes_in_memory = self.mcts_config['max_nodes_in_memory']
        self.prune_threshold = self.mcts_config['prune_threshold']
        self.last_gc_iteration = 0
        
        logger.info(f"Initialized MCTSEngine with exploration_weight={self.exploration_weight}")
    
    def initialize_tree(self, base_features: Set[str]) -> FeatureNode:
        """Initialize the MCTS tree with base features."""
        self.root = FeatureNode(
            base_features=base_features,
            applied_operations=[],
            depth=0
        )
        
        logger.info(f"Initialized MCTS tree with {len(base_features)} base features")
        return self.root
    
    @timed("mcts.selection")
    def selection(self) -> FeatureNode:
        """
        Selection phase: Navigate from root to a leaf using UCB1.
        
        Returns:
            FeatureNode: Selected leaf node for expansion/evaluation
        """
        current = self.root
        path = [current]
        
        # Navigate down the tree using UCB1 until we reach a leaf or unexpandable node
        while not current.is_leaf() and current.visit_count >= self.expansion_threshold:
            current = current.select_best_child(self.exploration_weight)
            if current is None:
                break
            path.append(current)
        
        logger.debug(f"Selected node at depth {current.depth} with {current.visit_count} visits")
        return current
    
    @timed("mcts.expansion", include_memory=True)
    def expansion(self, node: FeatureNode, available_operations: List[str]) -> List[FeatureNode]:
        """
        Expansion phase: Add new child nodes for unexplored operations.
        
        Args:
            node: Node to expand
            available_operations: List of operations that can be applied
            
        Returns:
            List[FeatureNode]: Newly created child nodes
        """
        if node.depth >= self.max_tree_depth:
            node.is_fully_expanded = True
            return []
        
        if not available_operations:
            node.is_fully_expanded = True
            return []
        
        # Limit expansion to budget and max children
        existing_operations = {child.operation_that_created_this for child in node.children}
        new_operations = [op for op in available_operations if op not in existing_operations]
        
        # Select operations to expand (up to budget)
        expansion_count = min(
            len(new_operations),
            self.expansion_budget,
            self.max_children_per_node - len(node.children)
        )
        
        if expansion_count <= 0:
            node.is_fully_expanded = True
            return []
        
        # Randomly sample operations to expand (or use all if under budget)
        if len(new_operations) > expansion_count:
            operations_to_expand = random.sample(new_operations, expansion_count)
        else:
            operations_to_expand = new_operations
        
        # Create child nodes
        new_children = []
        for operation in operations_to_expand:
            child = node.add_child(operation)
            new_children.append(child)
        
        # Mark as fully expanded if we've exhausted all operations
        if len(node.children) >= len(available_operations):
            node.is_fully_expanded = True
        
        logger.debug(f"Expanded node with {len(new_children)} new children")
        return new_children
    
    @timed("mcts.simulation", include_memory=True)
    def simulation(self, node: FeatureNode, evaluator, feature_space) -> Tuple[float, float]:
        """
        Simulation phase: Evaluate the node using AutoGluon.
        
        This is a simplified simulation - we directly evaluate the node
        rather than doing random rollouts, since AutoGluon provides
        the evaluation we need.
        
        Args:
            node: Node to evaluate
            evaluator: AutoGluon evaluator instance
            feature_space: Feature space manager
            
        Returns:
            Tuple[float, float]: (evaluation_score, evaluation_time)
        """
        start_time = time.time()
        
        try:
            # Generate features for this node (lazy loading)
            features = feature_space.generate_features_for_node(node)
            
            # Evaluate using AutoGluon
            score = evaluator.evaluate_features(features, node.depth, self.current_iteration)
            
            evaluation_time = time.time() - start_time
            self.total_evaluations += 1
            
            # Track best score
            if score > self.best_score:
                self.best_score = score
                self.best_node = node
                logger.info(f"New best score: {score:.5f} at iteration {self.current_iteration}")
            
            logger.debug(f"Evaluated node: score={score:.5f}, time={evaluation_time:.2f}s")
            
            return score, evaluation_time
            
        except Exception as e:
            logger.error(f"Evaluation failed for node: {e}")
            return 0.0, time.time() - start_time
    
    def backpropagation(self, node: FeatureNode, reward: float, evaluation_time: float) -> None:
        """
        Backpropagation phase: Update all ancestors with the reward.
        
        Args:
            node: Starting node (usually evaluated leaf)
            reward: Reward value to propagate
            evaluation_time: Time taken for evaluation
        """
        current = node
        nodes_updated = 0
        
        while current is not None:
            current.update_reward(reward, evaluation_time)
            nodes_updated += 1
            current = current.parent
        
        logger.debug(f"Backpropagated reward {reward:.5f} through {nodes_updated} nodes")
    
    def mcts_iteration(self, evaluator, feature_space, db) -> Dict[str, Any]:
        """
        Execute one complete MCTS iteration.
        
        Args:
            evaluator: AutoGluon evaluator
            feature_space: Feature space manager  
            db: Database interface
            
        Returns:
            Dict: Iteration statistics
        """
        iteration_start = time.time()
        
        # 1. SELECTION
        selected_node = self.selection()
        
        # 2. EXPANSION  
        available_operations = feature_space.get_available_operations(selected_node)
        expanded_children = self.expansion(selected_node, available_operations)
        
        # 3. SIMULATION & EVALUATION
        # Evaluate the selected node if it hasn't been evaluated yet
        if selected_node.evaluation_score is None:
            nodes_to_evaluate = [selected_node]
        else:
            nodes_to_evaluate = expanded_children
        
        evaluation_results = []
        for node in nodes_to_evaluate:
            score, eval_time = self.simulation(node, evaluator, feature_space)
            node.evaluation_score = score
            evaluation_results.append((node, score, eval_time))
            
            # 4. BACKPROPAGATION
            self.backpropagation(node, score, eval_time)
            
            # Log to database
            if db:
                try:
                    db.log_exploration_step(
                        iteration=self.current_iteration,
                        operation=node.operation_that_created_this or 'root',
                        features_before=list(node.parent.current_features) if node.parent else [],
                        features_after=list(node.current_features),
                        score=score,
                        eval_time=eval_time,
                        autogluon_config=evaluator.get_current_config(),
                        ucb1_score=node.ucb1_score(self.exploration_weight),
                        parent_node_id=node.parent.node_id if node.parent else None,
                        memory_usage_mb=node.memory_usage_mb
                    )
                    # Node ID is returned by log_exploration_step method
                except Exception as e:
                    logger.error(f"Failed to log to database: {e}")
        
        iteration_time = time.time() - iteration_start
        
        # Memory management
        if self.current_iteration % 10 == 0:
            self._memory_management()
        
        return {
            'iteration': self.current_iteration,
            'selected_node_depth': selected_node.depth,
            'expanded_children': len(expanded_children),
            'evaluations': len(evaluation_results),
            'best_score': max(result[1] for result in evaluation_results) if evaluation_results else 0.0,
            'iteration_time': iteration_time,
            'total_nodes': self._count_total_nodes(),
            'memory_usage_mb': self._get_memory_usage()
        }
    
    @timed("mcts.run_search", include_memory=True)
    def run_search(self, evaluator, feature_space, db, initial_features: Set[str]) -> Dict[str, Any]:
        """
        Run the complete MCTS search process.
        
        Args:
            evaluator: AutoGluon evaluator
            feature_space: Feature space manager
            db: Database interface
            initial_features: Starting feature set
            
        Returns:
            Dict: Final search results
        """
        logger.info("Starting MCTS feature discovery search")
        
        # Initialize tree
        self.initialize_tree(initial_features)
        
        # Evaluate root node first
        root_score, root_time = self.simulation(self.root, evaluator, feature_space)
        self.root.evaluation_score = root_score
        self.root.update_reward(root_score, root_time)
        
        logger.info(f"Root evaluation: {root_score:.5f}")
        
        # Main MCTS loop
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            
            # Check termination conditions
            if self._should_terminate():
                logger.info(f"Terminating search at iteration {iteration}")
                break
            
            # Execute MCTS iteration
            try:
                iteration_stats = self.mcts_iteration(evaluator, feature_space, db)
                
                # Progress reporting
                if iteration % self.config['logging']['progress_interval'] == 0:
                    self._report_progress(iteration_stats)
                    
            except KeyboardInterrupt:
                logger.info("Search interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                continue
        
        # Final results
        results = {
            'total_iterations': self.current_iteration,
            'total_evaluations': self.total_evaluations,
            'best_score': self.best_score,
            'best_node': self.best_node,
            'total_runtime': time.time() - self.start_time,
            'tree_size': self._count_total_nodes(),
            'final_memory_usage': self._get_memory_usage()
        }
        
        logger.info(f"MCTS search completed: {results['total_iterations']} iterations, "
                   f"best score: {results['best_score']:.5f}")
        
        return results
    
    def get_best_path(self) -> List[FeatureNode]:
        """Get the path to the best discovered feature combination."""
        if self.best_node:
            return self.best_node.get_path_from_root()
        return []
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tree statistics."""
        if not self.root:
            return {}
        
        total_nodes = self._count_total_nodes()
        depths = self._get_node_depths()
        rewards = self._get_all_rewards()
        
        return {
            'total_nodes': total_nodes,
            'max_depth': max(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'best_reward': max(rewards) if rewards else 0,
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
            'exploration_coverage': self._calculate_exploration_coverage()
        }
    
    def _should_terminate(self) -> bool:
        """Check if search should terminate based on configured conditions."""
        # Time limit
        runtime_hours = (time.time() - self.start_time) / 3600
        if runtime_hours >= self.max_runtime_hours:
            return True
        
        # Memory limit
        memory_gb = self._get_memory_usage() / 1024
        max_memory_gb = self.resource_config.get('max_memory_gb', 16)
        if memory_gb >= max_memory_gb:
            logger.warning(f"Memory limit reached: {memory_gb:.1f}GB")
            return True
        
        return False
    
    def _memory_management(self) -> None:
        """Perform memory management and cleanup."""
        current_nodes = self._count_total_nodes()
        
        if current_nodes > self.max_nodes_in_memory:
            logger.info(f"Memory cleanup: {current_nodes} nodes, pruning low-value nodes")
            self._prune_tree()
        
        # Force garbage collection periodically
        gc_interval = self.resource_config.get('force_gc_interval', 50)
        if self.current_iteration - self.last_gc_iteration >= gc_interval:
            gc.collect()
            self.last_gc_iteration = self.current_iteration
    
    def _prune_tree(self) -> None:
        """Remove low-value nodes to free memory."""
        # This is a simplified pruning strategy
        # In practice, you might want more sophisticated pruning
        pass
    
    def _count_total_nodes(self) -> int:
        """Count total nodes in the tree."""
        if not self.root:
            return 0
        
        def count_recursive(node):
            count = 1
            for child in node.children:
                count += count_recursive(child)
            return count
        
        return count_recursive(self.root)
    
    def _get_node_depths(self) -> List[int]:
        """Get depths of all nodes in the tree."""
        if not self.root:
            return []
        
        depths = []
        
        def collect_depths(node):
            depths.append(node.depth)
            for child in node.children:
                collect_depths(child)
        
        collect_depths(self.root)
        return depths
    
    def _get_all_rewards(self) -> List[float]:
        """Get rewards of all evaluated nodes."""
        if not self.root:
            return []
        
        rewards = []
        
        def collect_rewards(node):
            if node.evaluation_score is not None:
                rewards.append(node.evaluation_score)
            for child in node.children:
                collect_rewards(child)
        
        collect_rewards(self.root)
        return rewards
    
    def _calculate_exploration_coverage(self) -> float:
        """Calculate how much of the feature space has been explored."""
        # This is a simplified metric - in practice you'd want a more sophisticated measure
        if not self.root:
            return 0.0
        
        total_possible_operations = 100  # This should come from feature_space
        explored_operations = len(set(self._get_all_operations()))
        
        return explored_operations / total_possible_operations
    
    def _get_all_operations(self) -> List[str]:
        """Get all operations applied in the tree."""
        if not self.root:
            return []
        
        operations = []
        
        def collect_operations(node):
            if node.operation_that_created_this:
                operations.append(node.operation_that_created_this)
            for child in node.children:
                collect_operations(child)
        
        collect_operations(self.root)
        return operations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _report_progress(self, iteration_stats: Dict[str, Any]) -> None:
        """Report progress to logger."""
        stats = iteration_stats
        progress = (self.current_iteration / self.max_iterations) * 100
        
        logger.info(
            f"Progress: {progress:.1f}% ({self.current_iteration}/{self.max_iterations}) | "
            f"Best: {self.best_score:.5f} | "
            f"Nodes: {stats['total_nodes']} | "
            f"Memory: {stats['memory_usage_mb']:.1f}MB | "
            f"Eval/iter: {stats['evaluations']}"
        )