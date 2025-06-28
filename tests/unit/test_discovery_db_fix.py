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
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = {
            'database': {
                'path': temp_file.name,
                'backup_path': 'test_backup',
                'type': 'sqlite'
            },
            'session': {
                'mode': 'new',
                'max_iterations': 10
            }
        }
        
        yield config
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    def test_get_best_features_with_tuple_rows(self, temp_db_config):
        """Test get_best_features handles tuple rows correctly."""
        db = FeatureDiscoveryDB(temp_db_config)
        
        # Mock cursor that returns tuples instead of Row objects
        mock_cursor = Mock()
        mock_cursor.description = [
            ('feature_name',), ('feature_category',), ('impact_delta',), 
            ('impact_percentage',), ('with_feature_score',), ('sample_size',),
            ('python_code',), ('computational_cost',), ('session_id',)
        ]
        mock_cursor.fetchall.return_value = [
            ('test_feature_1', 'numeric', 0.05, 5.0, 0.85, 1000, 'df["test"] = 1', 1.0, 'session1'),
            ('test_feature_2', 'categorical', 0.03, 3.0, 0.83, 1000, 'df["test2"] = 2', 1.5, 'session1')
        ]
        
        # Mock connection
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(db, '_connect', return_value=mock_conn):
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
    
    def test_get_best_features_with_row_objects(self, temp_db_config):
        """Test get_best_features handles SQLite Row objects correctly."""
        db = FeatureDiscoveryDB(temp_db_config)
        
        # Mock Row object
        class MockRow:
            def __init__(self, data):
                self.data = data
            
            def keys(self):
                return self.data.keys()
            
            def __getitem__(self, key):
                return self.data[key]
            
            def items(self):
                return self.data.items()
        
        mock_cursor = Mock()
        mock_cursor.description = [('feature_name',), ('impact_delta',)]
        mock_cursor.fetchall.return_value = [
            MockRow({'feature_name': 'test_feature', 'impact_delta': 0.05})
        ]
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(db, '_connect', return_value=mock_conn):
            results = db.get_best_features(limit=5)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['feature_name'] == 'test_feature'
        assert results[0]['impact_delta'] == 0.05
    
    def test_get_operation_rankings_robust(self, temp_db_config):
        """Test get_operation_rankings handles different row types."""
        db = FeatureDiscoveryDB(temp_db_config)
        
        mock_cursor = Mock()
        mock_cursor.description = [
            ('operation_name',), ('effectiveness_score',), ('total_applications',),
            ('success_rate',), ('avg_improvement',), ('session_id',)
        ]
        mock_cursor.fetchall.return_value = [
            ('polynomial_features', 0.85, 10, 0.9, 0.05, 'session1'),
            ('log_transform', 0.72, 8, 0.8, 0.03, 'session1')
        ]
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(db, '_connect', return_value=mock_conn):
            results = db.get_operation_rankings()
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]['operation_name'] == 'polynomial_features'
        assert results[0]['effectiveness_score'] == 0.85
    
    def test_get_best_features_empty_result(self, temp_db_config):
        """Test get_best_features handles empty results gracefully."""
        db = FeatureDiscoveryDB(temp_db_config)
        
        mock_cursor = Mock()
        mock_cursor.description = []
        mock_cursor.fetchall.return_value = []
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(db, '_connect', return_value=mock_conn):
            results = db.get_best_features(limit=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_best_features_with_malformed_row(self, temp_db_config):
        """Test get_best_features handles malformed rows gracefully."""
        db = FeatureDiscoveryDB(temp_db_config)
        
        # Create a problematic row that will cause dict() to fail
        class BadRow:
            def __iter__(self):
                # Return iterator that yields wrong format
                return iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # 13 elements
        
        mock_cursor = Mock()
        mock_cursor.description = [('feature_name',), ('impact_delta',)]
        mock_cursor.fetchall.return_value = [
            BadRow(),  # This will fail dict() conversion
            ('good_feature', 'numeric', 0.05, 5.0, 0.85, 1000, 'code', 1.0, 'session1')  # This should work
        ]
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(db, '_connect', return_value=mock_conn):
            results = db.get_best_features(limit=5)
        
        # Should skip the bad row and return the good one
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['feature_name'] == 'good_feature'