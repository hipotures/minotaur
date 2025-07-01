"""
DateTime Parser - Examples and Integration

Shows how to use the datetime parser with real-world messy data
"""

import pandas as pd
import numpy as np
from datetime_parser import DateTimeParser, DateTimeValidator
from datetime_parser import parse_mixed_datetime_column, detect_and_convert_timestamps

# Example 1: Mixed date formats in one column
print("=== Example 1: Mixed Date Formats ===")
mixed_dates = pd.Series([
    '2024-01-15',           # ISO format
    '15/01/2024',           # DD/MM/YYYY
    '01/15/2024',           # MM/DD/YYYY
    '2024-01',              # Partial date (year-month)
    '2024',                 # Year only
    '15-Jan-2024',          # Text month
    '2024-01-15 14:30:00',  # With time
    '2024-01-15T14:30:00Z', # ISO with timezone
    np.nan,                 # Missing value
    '15.01.2024',           # Dot separator
    'Jan 15, 2024',         # US text format
])

# Detect format
parser = DateTimeParser()
format_info = parser.detect_datetime_format(mixed_dates)
print("Detected formats:", format_info['formats_found'])
print("Primary format:", format_info['primary_format'])
print("Has mixed formats:", format_info['is_mixed_format'])

# Parse intelligently
parsed = parser.parse_datetime_intelligent(mixed_dates)
print("\nParsed dates:")
for original, parsed_date in zip(mixed_dates, parsed):
    print(f"{str(original):25} -> {parsed_date}")

# Example 2: Ambiguous dates (DD/MM vs MM/DD)
print("\n=== Example 2: Ambiguous Dates ===")
ambiguous_dates = pd.Series([
    '01/02/2024',  # Could be Jan 2 or Feb 1
    '12/13/2024',  # Clearly Dec 13 (month can't be 13)
    '25/12/2024',  # Clearly Dec 25 (day can't be 25 in month)
    '10/10/2024',  # Could be either
])

# Parse with different assumptions
parsed_dmy = parser.parse_datetime_intelligent(ambiguous_dates, dayfirst=True)
parsed_mdy = parser.parse_datetime_intelligent(ambiguous_dates, dayfirst=False)

print("Date         | DMY Format    | MDY Format")
print("-" * 50)
for orig, dmy, mdy in zip(ambiguous_dates, parsed_dmy, parsed_mdy):
    print(f"{orig} | {dmy.strftime('%b %d, %Y')} | {mdy.strftime('%b %d, %Y')}")

# Example 3: Unix timestamps
print("\n=== Example 3: Unix Timestamps ===")
timestamps = pd.Series([
    1704067200,      # Unix seconds (2024-01-01)
    1704067200000,   # Unix milliseconds
    '1704067200',    # String timestamp
    1234567890,      # Another timestamp
    np.nan,
    'not_a_timestamp'
])

# Detect and convert
converted = detect_and_convert_timestamps(timestamps)
print("Original -> Converted:")
for orig, conv in zip(timestamps, converted):
    print(f"{orig} -> {conv}")

# Example 4: Time formats (12h vs 24h)
print("\n=== Example 4: Time Formats ===")
times = pd.Series([
    '14:30:00',      # 24h with seconds
    '14:30',         # 24h without seconds
    '2:30 PM',       # 12h with PM
    '2:30:00 PM',    # 12h with seconds
    '02:30 AM',      # 12h with AM
    '23:59:59',      # 24h end of day
    '12:00 AM',      # Midnight
    '12:00 PM',      # Noon
])

parsed_times = parser.parse_datetime_intelligent(times)
print("Time Format  -> Parsed")
for orig, parsed in zip(times, parsed_times):
    print(f"{orig:12} -> {parsed}")

# Example 5: Validation
print("\n=== Example 5: Date Validation ===")
problematic_dates = pd.Series([
    '2024-01-15',
    '2025-12-31',    # Future date
    '1899-01-01',    # Very old date
    '32/13/2024',    # Invalid date
    'not a date',    # Invalid format
    '2024-02-30',    # Invalid day for February
])

validator = DateTimeValidator()
validation_results = validator.validate_datetime_column(problematic_dates)

print(f"Total values: {validation_results['total_values']}")
print(f"Parsing errors: {len(validation_results['parsing_errors'])}")
print(f"Future dates: {len(validation_results['future_dates'])}")
print(f"Very old dates: {len(validation_results['very_old_dates'])}")
print("\nRecommendations:")
for rec in validation_results['recommendations']:
    print(f"- {rec}")

# Example 6: Feature extraction from partial dates
print("\n=== Example 6: Features from Partial Dates ===")
partial_dates = pd.Series([
    '2024',          # Year only
    '2024-03',       # Year-month
    '2024-03-15',    # Full date
    '2024-03-15 14:30:00',  # Full datetime
])

features = parser.extract_datetime_components(partial_dates)
feature_df = pd.DataFrame(features)
print("\nExtracted features:")
print(feature_df[['year', 'month', 'day', 'date_precision', 'is_partial_date']])

# Example 7: Integration with main feature engineering pipeline
print("\n=== Example 7: Integration with Feature Engineering ===")

def enhanced_datetime_feature_engineering(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """
    Enhanced datetime feature engineering that handles messy data.
    """
    all_features = {}
    
    for col in datetime_columns:
        if col not in df.columns:
            continue
        
        # First validate the column
        validator = DateTimeValidator()
        validation = validator.validate_datetime_column(df[col])
        
        # Log any issues
        if validation['parsing_errors']:
            print(f"Warning: {len(validation['parsing_errors'])} parsing errors in column '{col}'")
        
        # Detect if it's a timestamp column
        if pd.api.types.is_numeric_dtype(df[col]):
            # Try to detect Unix timestamps
            converted = detect_and_convert_timestamps(df[col])
            if not converted.equals(df[col]):  # Was converted
                df[col] = converted
                print(f"Detected Unix timestamps in column '{col}'")
        
        # Parse the datetime column
        parser = DateTimeParser()
        parsed = parser.parse_datetime_intelligent(df[col])
        
        # Extract all features
        datetime_features = parser.create_datetime_features(
            pd.DataFrame({col: df[col]}), 
            [col], 
            include_cyclical=True
        )
        
        all_features.update(datetime_features)
    
    # Create feature dataframe
    feature_df = pd.DataFrame(all_features, index=df.index)
    
    # Combine with original
    return pd.concat([df, feature_df], axis=1)

# Test the integrated function
test_df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'created_date': ['2024-01-15', '15/01/2024', '2024-01', '2024', '01/15/2024 2:30 PM'],
    'timestamp': [1704067200, 1704067200000, np.nan, 1234567890, 1700000000],
    'birth_date': ['1990-05-15', '15/05/1990', '1985', '1985-03', np.nan]
})

print("\nOriginal DataFrame:")
print(test_df)

# Apply enhanced feature engineering
result_df = enhanced_datetime_feature_engineering(
    test_df, 
    ['created_date', 'timestamp', 'birth_date']
)

print("\nColumns after feature engineering:")
print(result_df.columns.tolist())

# Show sample of created features
feature_cols = [col for col in result_df.columns if any(
    x in col for x in ['_year', '_month', '_day', '_age', '_precision', '_sin', '_cos']
)]
print("\nSample of created features:")
print(result_df[['id'] + feature_cols[:10]].head())

# Example 8: Handling different separators and formats
print("\n=== Example 8: Various Separators and Formats ===")
various_formats = pd.Series([
    # Different separators
    '2024-01-15',
    '2024/01/15',
    '2024.01.15',
    
    # With time in different formats
    '2024-01-15 14:30',
    '2024-01-15 14:30:45',
    '2024-01-15T14:30:45',
    '2024-01-15T14:30:45Z',
    '2024-01-15T14:30:45+00:00',
    
    # Text formats
    'January 15, 2024',
    '15 Jan 2024',
    '15-Jan-24',
    
    # Partial formats
    'Q1 2024',      # Quarter notation
    '2024W03',      # Week notation
    'FY2024',       # Fiscal year
])

# Parse and show results
parsed_various = parser.parse_datetime_intelligent(various_formats)
print("Format                        | Parsed")
print("-" * 60)
for orig, parsed in zip(various_formats, parsed_various):
    if pd.notna(parsed):
        print(f"{str(orig):30} | {parsed}")
    else:
        print(f"{str(orig):30} | Failed to parse")

# Example 9: Performance with large datasets
print("\n=== Example 9: Performance Test ===")
# Generate large dataset with mixed formats
n_rows = 10000
date_formats = [
    lambda: pd.Timestamp.now().strftime('%Y-%m-%d'),
    lambda: pd.Timestamp.now().strftime('%d/%m/%Y'),
    lambda: pd.Timestamp.now().strftime('%m/%d/%Y'),
    lambda: pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    lambda: str(int(pd.Timestamp.now().timestamp())),
]

large_dates = pd.Series([
    np.random.choice(date_formats)() if np.random.random() > 0.1 else np.nan
    for _ in range(n_rows)
])

import time
start_time = time.time()
parsed_large = parser.parse_datetime_intelligent(large_dates)
end_time = time.time()

print(f"Parsed {n_rows} mixed format dates in {end_time - start_time:.2f} seconds")
print(f"Success rate: {parsed_large.notna().sum() / large_dates.notna().sum() * 100:.1f}%")

# Example 10: Custom format detection and standardization
print("\n=== Example 10: Standardization ===")
messy_dates = pd.DataFrame({
    'order_date': ['2024-01-15', '15/01/2024', '01/15/2024', '2024-01'],
    'ship_date': ['2024-01-16 14:30', '16/01/2024 2:30 PM', '01/16/2024', '2024-01-16'],
})

print("Before standardization:")
print(messy_dates)

# Standardize all date columns to ISO format
from datetime_parser import standardize_datetime_format
standardized = standardize_datetime_format(
    messy_dates, 
    ['order_date', 'ship_date'], 
    target_format='ISO'
)

print("\nAfter standardization (ISO format):")
print(standardized)

# Standardize to US format
standardized_us = standardize_datetime_format(
    messy_dates, 
    ['order_date', 'ship_date'], 
    target_format='US'
)

print("\nAfter standardization (US format):")
print(standardized_us)