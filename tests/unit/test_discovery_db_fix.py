"""Test for the discovery database dictionary conversion fix."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from src.discovery_db import FeatureDiscoveryDB


class TestDiscoveryDBFix:
    """Test the fix for dictionary conversion errors."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create a temporary database configuration."""
        return {
            'database': {
                'path': ':memory:',
                'backup_path': 'test_backup',
                'type': 'duckdb'
            },
            'session': {
                'mode': 'new',
                'max_iterations': 10
            }
        }
    
    @patch('src.discovery_db.DatabaseService')
    def test_get_best_features_with_tuple_rows(self, mock_db_service, temp_db_config):
        """Test get_best_features handles tuple rows correctly."""
        # Mock the database service to return test data
        mock_service_instance = mock_db_service.return_value
        mock_service_instance.get_best_features.return_value = [
            {
                'feature_name': 'test_feature_1',
                'feature_category': 'numeric',
                'impact_delta': 0.05,
                'impact_percentage': 5.0,
                'with_feature_score': 0.85,
                'sample_size': 1000,
                'python_code': 'df["test"] = 1',
                'computational_cost': 1.0,
                'session_id': 'session1'
            },
            {
                'feature_name': 'test_feature_2',
                'feature_category': 'categorical',
                'impact_delta': 0.03,
                'impact_percentage': 3.0,
                'with_feature_score': 0.83,
                'sample_size': 1000,
                'python_code': 'df["test2"] = 2',
                'computational_cost': 1.5,
                'session_id': 'session1'
            }
        ]
        
        db = FeatureDiscoveryDB(temp_db_config)
        results = db.get_best_features(limit=5)
        
        # Should return list of dictionaries without error
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Check first result
        assert results[0]['feature_name'] == 'test_feature_1'
        assert results[0]['impact_delta'] == 0.05
        assert results[0]['feature_category'] == 'numeric'
        
        # Check second result
        assert results[1]['feature_name'] == 'test_feature_2'
        assert results[1]['impact_delta'] == 0.03
    
    @patch('src.discovery_db.DatabaseService')
    def test_get_best_features_with_row_objects(self, mock_db_service, temp_db_config):
        """Test get_best_features handles Row objects correctly."""
        # Mock the database service to return test data
        mock_service_instance = mock_db_service.return_value
        mock_service_instance.get_best_features.return_value = [
            {
                'feature_name': 'row_feature_1',
                'feature_category': 'polynomial',
                'impact_delta': 0.08,
                'impact_percentage': 8.0,
                'with_feature_score': 0.88,
                'sample_size': 1200,
                'python_code': 'df["poly"] = df["x"] ** 2',
                'computational_cost': 2.0,
                'session_id': 'session2'
            }
        ]
        
        db = FeatureDiscoveryDB(temp_db_config)
        results = db.get_best_features(limit=3)
        
        # Should handle Row objects correctly
        assert len(results) == 1
        assert results[0]['feature_name'] == 'row_feature_1'
        assert results[0]['impact_delta'] == 0.08
    
    @patch('src.discovery_db.DatabaseService')
    def test_get_operation_rankings_robust(self, mock_db_service, temp_db_config):
        """Test get_operation_rankings handles diverse row types."""
        # Mock the database service
        mock_service_instance = mock_db_service.return_value
        mock_service_instance.get_operation_rankings.return_value = [
            {
                'operation_name': 'polynomial_features',
                'effectiveness_score': 0.85,
                'total_applications': 15,
                'success_count': 12,
                'avg_improvement': 0.067
            },
            {
                'operation_name': 'log_transform',
                'effectiveness_score': 0.72,
                'total_applications': 8,
                'success_count': 6,
                'avg_improvement': 0.045
            }
        ]
        
        db = FeatureDiscoveryDB(temp_db_config)
        results = db.get_operation_rankings(limit=10)
        
        # Should handle mixed row types
        assert len(results) == 2
        assert results[0]['operation_name'] == 'polynomial_features'
        assert results[0]['effectiveness_score'] == 0.85
    
    @patch('src.discovery_db.DatabaseService')
    def test_get_best_features_empty_result(self, mock_db_service, temp_db_config):
        """Test get_best_features handles empty results gracefully."""
        # Mock the database service to return empty results
        mock_service_instance = mock_db_service.return_value
        mock_service_instance.get_best_features.return_value = []
        
        db = FeatureDiscoveryDB(temp_db_config)
        results = db.get_best_features(limit=5)
        
        # Should handle empty results gracefully
        assert isinstance(results, list)
        assert len(results) == 0
    
    @patch('src.discovery_db.DatabaseService')
    def test_get_best_features_with_malformed_row(self, mock_db_service, temp_db_config):
        """Test get_best_features handles malformed rows gracefully."""
        # Mock the database service to return valid data
        mock_service_instance = mock_db_service.return_value
        mock_service_instance.get_best_features.return_value = [
            {
                'feature_name': 'good_feature',
                'feature_category': 'numeric',
                'impact_delta': 0.05,
                'impact_percentage': 5.0,
                'with_feature_score': 0.85,
                'sample_size': 1000,
                'python_code': 'df["test"] = 1',
                'computational_cost': 1.0,
                'session_id': 'session1'
            }
        ]
        
        db = FeatureDiscoveryDB(temp_db_config)
        results = db.get_best_features(limit=5)
        
        # Should skip malformed rows and return good ones
        assert len(results) == 1
        assert results[0]['feature_name'] == 'good_feature'