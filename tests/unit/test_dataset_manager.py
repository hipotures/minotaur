import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd

from src.dataset_manager import DatasetManager

class TestDatasetManager:

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for DatasetManager."""
        return {
            'autogluon': {
                'dataset_name': 'test_dataset',
                'target_column': 'target_col',
                'id_column': 'id_col'
            },
            'database': {
                'path': 'data/test.duckdb'
            }
        }

    @pytest.fixture
    def mock_registered_dataset_info(self):
        """Mock data for a registered dataset from the database."""
        mock_result = Mock()
        mock_result.dataset_id = '123'
        mock_result.dataset_name = 'test_dataset'
        mock_result.train_path = '/path/to/train.csv'
        mock_result.test_path = '/path/to/test.csv'
        mock_result.target_column = 'target_col'
        mock_result.id_column = 'id_col'
        mock_result.train_records = 100
        mock_result.train_columns = 10
        mock_result.test_records = 50
        mock_result.test_columns = 9
        mock_result.competition_name = 'test_comp'
        mock_result.description = 'test_desc'
        mock_result.is_active = True
        return mock_result

    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_get_dataset_from_config_registered(self, MockFeatureDiscoveryDB, mock_config, mock_registered_dataset_info):
        """Test getting dataset info when dataset_name is specified and registered."""
        # Mock the database lookup
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_registered_dataset_info

        manager = DatasetManager(mock_config)
        dataset_info = manager.get_dataset_from_config()

        assert dataset_info['dataset_name'] == 'test_dataset'
        assert dataset_info['is_registered'] is True
        assert dataset_info['train_path'] == '/path/to/train.csv'
        assert dataset_info['target_column'] == 'target_col'
        mock_db_instance.db_service.dataset_repo.get_by_name.assert_called_once_with('test_dataset')

    def test_get_dataset_from_config_no_dataset_name(self, mock_config):
        """Test getting dataset info when dataset_name is missing."""
        mock_config['autogluon']['dataset_name'] = None
        manager = DatasetManager(mock_config)

        with pytest.raises(ValueError, match="Configuration error: 'autogluon.dataset_name' must be specified."):
            manager.get_dataset_from_config()

    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_get_dataset_from_config_not_found(self, MockFeatureDiscoveryDB, mock_config):
        """Test getting dataset info when dataset_name is specified but not found in DB."""
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = None

        manager = DatasetManager(mock_config)
        with pytest.raises(ValueError, match="Dataset 'test_dataset' not found or inactive"):
            manager.get_dataset_from_config()

    @patch('src.dataset_manager.Path')
    @patch('pandas.read_csv')
    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_load_dataset_files(self, MockFeatureDiscoveryDB, mock_read_csv, MockPath, mock_config, mock_registered_dataset_info):
        """Test loading dataset files into pandas DataFrames."""
        # Mock registered dataset info
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_registered_dataset_info

        # Mock Path.exists()
        MockPath.return_value.exists.return_value = True

        # Mock pandas.read_csv
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2], 'target_col': [0, 1]}), # train_df
            pd.DataFrame({'col1': [3, 4]}) # test_df
        ]

        manager = DatasetManager(mock_config)
        train_df, test_df = manager.load_dataset_files('test_dataset')

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) == 2
        assert len(test_df) == 2
        mock_read_csv.assert_any_call('/path/to/train.csv')
        mock_read_csv.assert_any_call('/path/to/test.csv')

    @patch('src.dataset_manager.Path')
    @patch('pandas.read_csv')
    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_load_dataset_files_sampling(self, MockFeatureDiscoveryDB, mock_read_csv, MockPath, mock_config, mock_registered_dataset_info):
        """Test loading dataset files with sampling."""
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_registered_dataset_info

        MockPath.return_value.exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'col': range(200), 'target_col': [0]*200}) # Large DataFrame

        manager = DatasetManager(mock_config)
        train_df, _ = manager.load_dataset_files('test_dataset', sample_size=10)

        assert len(train_df) == 10
        assert mock_read_csv.call_count == 2  # Both train and test are loaded

    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_validate_dataset_registration(self, MockFeatureDiscoveryDB, mock_config, mock_registered_dataset_info):
        """Test dataset registration validation."""
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_registered_dataset_info

        # Mock Path.exists() for train, test, and duckdb_path
        with patch('src.dataset_manager.Path') as MockPath:
            MockPath.return_value.exists.return_value = True # All paths exist

            manager = DatasetManager(mock_config)
            is_valid = manager.validate_dataset_registration('test_dataset')
            assert is_valid is True

            # Test missing train file
            MockPath.return_value.exists.side_effect = [False, True, True] # train_path, test_path, duckdb_path
            is_valid = manager.validate_dataset_registration('test_dataset')
            assert is_valid is False

            # Reset to all paths existing for remaining tests
            MockPath.return_value.exists.return_value = True
            is_valid = manager.validate_dataset_registration('test_dataset')
            assert is_valid is True

    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_get_target_column(self, MockFeatureDiscoveryDB, mock_config, mock_registered_dataset_info):
        """Test getting target column name."""
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_registered_dataset_info

        manager = DatasetManager(mock_config)
        target_col = manager.get_target_column('test_dataset')
        assert target_col == 'target_col'

        # Test missing target column by clearing cache and mocking None
        manager._dataset_cache.clear()  # Clear cache to force re-fetch
        mock_result_none = Mock()
        mock_result_none.target_column = None
        mock_result_none.dataset_name = 'test_dataset'
        mock_result_none.train_path = '/path/to/train.csv'
        mock_result_none.test_path = '/path/to/test.csv'
        mock_result_none.dataset_id = '123'
        mock_result_none.id_column = 'id_col'
        mock_result_none.train_records = 100
        mock_result_none.train_columns = 10
        mock_result_none.test_records = 50
        mock_result_none.test_columns = 9
        mock_result_none.competition_name = 'test_comp'
        mock_result_none.description = 'test_desc'
        mock_result_none.is_active = True
        mock_db_instance.db_service.dataset_repo.get_by_name.return_value = mock_result_none
        
        with pytest.raises(ValueError, match="No target column specified for dataset 'test_dataset'"):
            manager.get_target_column('test_dataset')

    @patch('src.discovery_db.FeatureDiscoveryDB')
    def test_update_dataset_usage(self, MockFeatureDiscoveryDB, mock_config, mock_registered_dataset_info):
        """Test updating dataset usage timestamp."""
        mock_db_instance = MockFeatureDiscoveryDB.return_value
        mock_db_instance.db_service.dataset_repo.find_by_name.return_value = mock_registered_dataset_info

        manager = DatasetManager(mock_config)
        manager.update_dataset_usage('test_dataset')
        mock_db_instance.db_service.dataset_repo.mark_dataset_used.assert_called_once_with('123')

        # Test dataset not found
        mock_db_instance.db_service.dataset_repo.find_by_name.return_value = None
        manager.update_dataset_usage('non_existent_dataset')
        # Should not raise error, but log warning (checked by logging mock if needed)
        mock_db_instance.db_service.dataset_repo.mark_dataset_used.assert_called_once() # Still called once from previous test
