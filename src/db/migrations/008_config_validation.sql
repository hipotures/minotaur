-- Migration 008: Configuration Validation Support
-- Created at: 2025-07-02T11:45:00

-- Add configuration validation columns to sessions table
ALTER TABLE sessions ADD COLUMN config_hash VARCHAR;
ALTER TABLE sessions ADD COLUMN validation_errors JSON;

-- Create index on config_hash for fast lookups during compatibility checking
CREATE INDEX IF NOT EXISTS idx_sessions_config_hash ON sessions(config_hash);

-- Create view for configuration compatibility checking
CREATE OR REPLACE VIEW session_compatibility_info AS
SELECT 
    session_id,
    session_name,
    start_time,
    config_hash,
    config_snapshot,
    validation_errors,
    status,
    dataset_hash
FROM sessions
WHERE config_hash IS NOT NULL
ORDER BY start_time DESC;

-- DOWN MIGRATION
-- DROP VIEW session_compatibility_info;
-- DROP INDEX IF EXISTS idx_sessions_config_hash;
-- ALTER TABLE sessions DROP COLUMN validation_errors;
-- ALTER TABLE sessions DROP COLUMN config_hash;