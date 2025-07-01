-- Migration 001: Initial Schema
-- Created at: 2025-06-28T12:00:00

-- Create sequences for DuckDB autoincrement
CREATE SEQUENCE IF NOT EXISTS exploration_history_id_seq;
CREATE SEQUENCE IF NOT EXISTS feature_catalog_id_seq;
CREATE SEQUENCE IF NOT EXISTS feature_impact_id_seq;
CREATE SEQUENCE IF NOT EXISTS operation_performance_id_seq;
CREATE SEQUENCE IF NOT EXISTS system_performance_id_seq;

-- Main exploration history table
CREATE TABLE IF NOT EXISTS exploration_history (
    id BIGINT PRIMARY KEY DEFAULT nextval('exploration_history_id_seq'),
    session_id VARCHAR NOT NULL,
    iteration INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_node_id BIGINT,
    operation_applied VARCHAR NOT NULL,
    features_before JSON NOT NULL,
    features_after JSON NOT NULL,
    evaluation_score DOUBLE NOT NULL,
    target_metric VARCHAR NOT NULL,
    evaluation_time DOUBLE NOT NULL,
    autogluon_config JSON,
    mcts_ucb1_score DOUBLE,
    node_visits INTEGER DEFAULT 1,
    is_best_so_far BOOLEAN DEFAULT FALSE,
    memory_usage_mb DOUBLE,
    notes VARCHAR
);

-- Feature catalog with Python code
CREATE TABLE IF NOT EXISTS feature_catalog (
    id BIGINT PRIMARY KEY DEFAULT nextval('feature_catalog_id_seq'),
    feature_name VARCHAR UNIQUE NOT NULL,
    feature_category VARCHAR NOT NULL,
    python_code VARCHAR NOT NULL,
    dependencies JSON,
    description VARCHAR,
    created_by VARCHAR DEFAULT 'mcts',
    creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    computational_cost DOUBLE DEFAULT 1.0,
    data_type VARCHAR DEFAULT 'float64',
    operation_name VARCHAR,
    origin VARCHAR
);

-- Indexes for feature_catalog
CREATE INDEX IF NOT EXISTS idx_feature_catalog_operation ON feature_catalog(operation_name);
CREATE INDEX IF NOT EXISTS idx_feature_catalog_origin ON feature_catalog(origin);

-- Feature impact analysis
CREATE TABLE IF NOT EXISTS feature_impact (
    id BIGINT PRIMARY KEY DEFAULT nextval('feature_impact_id_seq'),
    feature_name VARCHAR NOT NULL,
    baseline_score DOUBLE NOT NULL,
    with_feature_score DOUBLE NOT NULL,
    impact_delta DOUBLE NOT NULL,
    impact_percentage DOUBLE NOT NULL,
    evaluation_context JSON,
    sample_size INTEGER DEFAULT 1,
    confidence_interval JSON,
    statistical_significance DOUBLE,
    first_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_evaluated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR NOT NULL
);

-- Operation performance tracking
CREATE TABLE IF NOT EXISTS operation_performance (
    id BIGINT PRIMARY KEY DEFAULT nextval('operation_performance_id_seq'),
    operation_name VARCHAR NOT NULL,
    operation_category VARCHAR,
    total_applications INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_improvement DOUBLE DEFAULT 0.0,
    best_improvement DOUBLE DEFAULT 0.0,
    worst_result DOUBLE DEFAULT 0.0,
    avg_execution_time DOUBLE DEFAULT 0.0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    effectiveness_score DOUBLE DEFAULT 0.0,
    session_id VARCHAR NOT NULL
);

-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR PRIMARY KEY,
    session_name VARCHAR,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_iterations INTEGER DEFAULT 0,
    best_score DOUBLE DEFAULT 0.0,
    config_snapshot JSON,
    status VARCHAR DEFAULT 'active',
    strategy VARCHAR DEFAULT 'default',
    is_test_mode BOOLEAN DEFAULT FALSE,
    notes VARCHAR,
    dataset_hash VARCHAR
);

-- System performance logs
CREATE TABLE IF NOT EXISTS system_performance (
    id BIGINT PRIMARY KEY DEFAULT nextval('system_performance_id_seq'),
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_usage_mb DOUBLE,
    cpu_usage_percent DOUBLE,
    disk_usage_mb DOUBLE,
    gpu_memory_mb DOUBLE,
    active_nodes INTEGER,
    evaluation_queue_size INTEGER
);

-- DOWN MIGRATION
DROP TABLE IF EXISTS system_performance;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS operation_performance;
DROP TABLE IF EXISTS feature_impact;
DROP TABLE IF EXISTS feature_catalog;
DROP TABLE IF EXISTS exploration_history;

DROP SEQUENCE IF EXISTS system_performance_id_seq;
DROP SEQUENCE IF EXISTS operation_performance_id_seq;
DROP SEQUENCE IF EXISTS feature_impact_id_seq;
DROP SEQUENCE IF EXISTS feature_catalog_id_seq;
DROP SEQUENCE IF EXISTS exploration_history_id_seq;