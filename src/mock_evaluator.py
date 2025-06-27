"""
Mock AutoGluon Evaluator for Testing

Simulates AutoGluon evaluation for testing MCTS system without requiring
full AutoGluon installation. Generates realistic but random scores.
"""

import time
import random
import logging
from typing import Dict, List, Set, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .timing import timed, timing_context, record_timing

logger = logging.getLogger(__name__)

class MockAutoGluonEvaluator:
    """Mock evaluator that simulates AutoGluon behavior for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock evaluator."""
        self.config = config
        self.autogluon_config = config['autogluon']
        
        # Simulation parameters from config
        testing_config = config.get('testing', {})
        self.base_score = testing_config.get('mock_base_score', 0.30)
        self.score_variance = testing_config.get('mock_score_variance', 0.05) 
        self.improvement_rate = 0.001  # How much features can improve scores
        
        # Tracking
        self.evaluation_count = 0
        self.best_score = 0.0
        self.current_phase = 'exploration'
        
        # Feature impact simulation
        self.feature_impacts = {
            'npk_basic_ratios': 0.01,
            'npk_advanced_interactions': 0.015,
            'stress_indicators': 0.008,
            'optimal_conditions': 0.012,
            'crop_nutrient_deficits': 0.020,
            'soil_adjustments': 0.018,
            'soil_groupby_stats': 0.025,
            'crop_groupby_stats': 0.022,
            'numerical_binning': 0.005,
            'polynomial_features': 0.003,
            'correlation_filter': -0.002,  # May hurt performance
        }
        
        logger.info("Initialized MockAutoGluonEvaluator")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current evaluation configuration."""
        if self.current_phase == 'exploration':
            return self.autogluon_config['fast_eval'].copy()
        else:
            return self.autogluon_config['thorough_eval'].copy()
    
    @timed("mock.evaluate_features")
    def evaluate_features(self, 
                         features_df: pd.DataFrame, 
                         node_depth: int = 0, 
                         iteration: int = 0) -> float:
        """Mock feature evaluation."""
        self.evaluation_count += 1
        
        # Update phase
        self._update_evaluation_phase(iteration)
        
        # Get current config for timing simulation
        eval_config = self.get_current_config()
        
        # Simulate evaluation time based on config
        testing_config = self.config.get('testing', {})
        if testing_config.get('fast_test_mode', False):
            # Ultra-fast for testing
            actual_time = random.uniform(0.1, 0.5)
        else:
            # Realistic simulation
            base_time = eval_config.get('time_limit', 30)
            actual_time = random.uniform(base_time * 0.05, base_time * 0.15)  # Much faster than real
        
        time.sleep(min(actual_time, 2.0))  # Cap at 2 seconds for testing
        
        # Calculate simulated score
        score = self._calculate_simulated_score(features_df, node_depth)
        
        # Update best score
        if score > self.best_score:
            self.best_score = score
        
        logger.debug(f"Mock evaluation: {score:.5f} for {len(features_df.columns)} features")
        return score
    
    def _calculate_simulated_score(self, features_df: pd.DataFrame, depth: int) -> float:
        """Calculate realistic simulated score based on features."""
        
        # Start with base score
        score = self.base_score
        
        # Add improvements based on feature types
        feature_cols = [col for col in features_df.columns 
                       if col not in ['id', 'Fertilizer Name']]
        
        # Simulate feature impact
        for feature_col in feature_cols:
            # Check if this feature matches known patterns
            for pattern, impact in self.feature_impacts.items():
                if pattern.lower() in feature_col.lower():
                    # Add impact with some randomness
                    actual_impact = impact * random.uniform(0.5, 1.5)
                    score += actual_impact
                    break
            else:
                # Unknown feature - small random impact
                score += random.uniform(-0.002, 0.005)
        
        # Depth penalty (deeper trees might overfit)
        depth_penalty = depth * 0.001
        score -= depth_penalty
        
        # Feature count impact (too many features may hurt)
        if len(feature_cols) > 100:
            feature_penalty = (len(feature_cols) - 100) * 0.0001
            score -= feature_penalty
        
        # Add random variance
        score += random.gauss(0, self.score_variance)
        
        # Ensure reasonable bounds
        score = max(0.15, min(0.50, score))
        
        return score
    
    def _update_evaluation_phase(self, iteration: int) -> None:
        """Update evaluation phase."""
        max_iterations = self.config['session']['max_iterations']
        threshold = self.autogluon_config['thorough_eval_threshold']
        
        switch_iteration = int(max_iterations * threshold)
        
        if iteration >= switch_iteration and self.current_phase == 'exploration':
            self.current_phase = 'exploitation'
            logger.info(f"Mock: Switched to exploitation phase at iteration {iteration}")
    
    def evaluate_final_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock final evaluation."""
        logger.info("Mock: Performing final evaluation...")
        
        # Simulate longer evaluation time
        time.sleep(3.0)
        
        score = self._calculate_simulated_score(features_df, 0)
        
        # Add some bonus for final evaluation
        score += 0.01
        
        return {
            'evaluation_metrics': {'MAP@3': score},
            'feature_importance': pd.DataFrame({
                'feature': ['Nitrogen', 'Phosphorous', 'npk_harmony'],
                'importance': [0.15, 0.12, 0.08]
            }),
            'leaderboard': pd.DataFrame({
                'model': ['LightGBM', 'CatBoost', 'XGBoost'],
                'score_val': [score-0.01, score-0.005, score]
            }),
            'evaluation_time': 180.0,
            'num_features': len(features_df.columns) - 2,  # Exclude id and target
            'training_samples': len(features_df)
        }
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get mock evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'total_eval_time': self.evaluation_count * 15.0,  # Simulated
            'avg_eval_time': 15.0,
            'best_score': self.best_score,
            'current_phase': self.current_phase,
            'cache_size': 0,
            'cache_hit_rate': 0.0
        }
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        logger.info("Mock evaluator cleanup completed")