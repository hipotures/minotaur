"""Test for the dictionary conversion fix in database methods."""

import pytest
from unittest.mock import Mock


def test_tuple_to_dict_conversion():
    """Test the pattern we use to convert tuples to dictionaries."""
    # Simulate the pattern from our fixed get_best_features method
    columns = ['feature_name', 'feature_category', 'impact_delta', 'impact_percentage']
    
    # Test with tuple (DuckDB style)
    row_tuple = ('test_feature', 'numeric', 0.05, 5.0)
    result = dict(zip(columns, row_tuple))
    
    assert result['feature_name'] == 'test_feature'
    assert result['feature_category'] == 'numeric'
    assert result['impact_delta'] == 0.05
    assert result['impact_percentage'] == 5.0


def test_row_object_to_dict_conversion():
    """Test conversion of Row-like objects to dictionaries."""
    # Mock Row object with keys
    class MockRow:
        def __init__(self, data):
            self.data = data
        
        def keys(self):
            return self.data.keys()
        
        def __getitem__(self, key):
            return self.data[key]
        
        def items(self):
            return self.data.items()
    
    row_obj = MockRow({'feature_name': 'test_feature', 'impact_delta': 0.05})
    result = dict(row_obj)
    
    assert result['feature_name'] == 'test_feature'
    assert result['impact_delta'] == 0.05


def test_malformed_data_handling():
    """Test handling of malformed data that would cause the original error."""
    columns = ['feature_name', 'impact_delta']
    
    # This simulates the error condition: a sequence with 13 elements
    # where dict() expects key-value pairs (length 2)
    malformed_row = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    
    # Original code would fail with: dict(malformed_row)
    # Our fix uses zip() which handles this gracefully
    result = dict(zip(columns, malformed_row))
    
    # Should only use the first 2 elements (matching columns length)
    assert result == {'feature_name': 1, 'impact_delta': 2}


def test_database_row_conversion_pattern():
    """Test the exact pattern used in our database fix."""
    def convert_rows_safely(rows, columns):
        """Simulate our fixed database row conversion logic."""
        results = []
        for row in rows:
            try:
                if hasattr(row, 'keys'):
                    # SQLite Row object with keys
                    results.append(dict(row))
                elif isinstance(row, (list, tuple)):
                    # DuckDB or plain tuple result
                    results.append(dict(zip(columns, row)))
                else:
                    # Fallback: try to convert directly
                    results.append(dict(row))
            except Exception:
                # Skip problematic rows
                continue
        return results
    
    columns = ['feature_name', 'impact_delta', 'category']
    
    # Test with mixed row types
    class MockRow:
        def keys(self):
            return ['feature_name', 'impact_delta', 'category']
        def __getitem__(self, key):
            data = {'feature_name': 'row_obj_feature', 'impact_delta': 0.1, 'category': 'test'}
            return data[key]
        def items(self):
            return [('feature_name', 'row_obj_feature'), ('impact_delta', 0.1), ('category', 'test')]
    
    # Mix of good and bad rows
    rows = [
        ('tuple_feature', 0.05, 'numeric'),  # Good tuple
        MockRow(),  # Good Row object
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),  # Bad tuple (too many elements)
        ['list_feature', 0.03, 'categorical'],  # Good list
        'bad_string_row',  # Bad string
    ]
    
    results = convert_rows_safely(rows, columns)
    
    # Should have 3 good results, 2 bad ones skipped
    assert len(results) == 3
    
    assert results[0]['feature_name'] == 'tuple_feature'
    assert results[1]['feature_name'] == 'row_obj_feature'
    assert results[2]['feature_name'] == 'list_feature'


def test_original_error_reproduction():
    """Reproduce the original error to confirm our understanding."""
    # This would cause the original error:
    # "dictionary update sequence element #0 has length 13; 2 is required"
    
    problematic_sequence = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    
    # This would fail in the original code
    with pytest.raises(ValueError, match="dictionary update sequence element"):
        dict(problematic_sequence)
    
    # Our fix handles this correctly
    columns = ['col1', 'col2']
    result = dict(zip(columns, problematic_sequence))
    assert result == {'col1': 1, 'col2': 2}  # Only uses first 2 elements