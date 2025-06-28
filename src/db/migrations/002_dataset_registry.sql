-- Migration 002: Dataset Registry
-- Created at: 2025-06-28T12:05:00

-- Dataset registry for centralized dataset management
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR PRIMARY KEY,
    dataset_name VARCHAR NOT NULL,
    train_path VARCHAR NOT NULL,
    test_path VARCHAR,
    submission_path VARCHAR,
    validation_path VARCHAR,
    target_column VARCHAR NOT NULL,
    id_column VARCHAR,
    competition_name VARCHAR,
    description VARCHAR,
    -- File counts and formats
    train_records INTEGER,
    train_columns INTEGER,
    test_records INTEGER,
    test_columns INTEGER,
    submission_records INTEGER,
    submission_columns INTEGER,
    validation_records INTEGER,
    validation_columns INTEGER,
    train_format VARCHAR,
    test_format VARCHAR,
    submission_format VARCHAR,
    validation_format VARCHAR,
    -- Legacy columns for backward compatibility
    column_count INTEGER,
    train_row_count INTEGER,
    test_row_count INTEGER,
    data_size_mb DOUBLE,
    feature_types JSON,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- DOWN MIGRATION
DROP TABLE IF EXISTS datasets;