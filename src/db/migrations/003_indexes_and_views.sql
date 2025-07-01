-- Migration 003: Indexes And Views
-- Created at: 2025-06-28T12:10:00

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_exploration_session ON exploration_history(session_id);
CREATE INDEX IF NOT EXISTS idx_exploration_score ON exploration_history(evaluation_score DESC);
CREATE INDEX IF NOT EXISTS idx_exploration_timestamp ON exploration_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_exploration_iteration ON exploration_history(session_id, iteration);
CREATE INDEX IF NOT EXISTS idx_impact_delta ON feature_impact(impact_delta DESC);
CREATE INDEX IF NOT EXISTS idx_impact_session ON feature_impact(session_id);
CREATE INDEX IF NOT EXISTS idx_operation_session ON operation_performance(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_dataset ON sessions(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_datasets_last_used ON datasets(last_used DESC);
CREATE INDEX IF NOT EXISTS idx_datasets_active ON datasets(is_active);


-- Session summary statistics
CREATE VIEW IF NOT EXISTS session_summary AS
SELECT 
    eh.session_id,
    ANY_VALUE(s.session_name) as session_name,
    ANY_VALUE(s.start_time) as start_time,
    ANY_VALUE(s.end_time) as end_time,
    COUNT(*) as total_iterations,
    MIN(eh.evaluation_score) as min_score,
    MAX(eh.evaluation_score) as max_score,
    MAX(eh.evaluation_score) - MIN(eh.evaluation_score) as improvement,
    AVG(eh.evaluation_time) as avg_eval_time,
    SUM(eh.evaluation_time) as total_eval_time,
    ANY_VALUE(s.status) as status,
    ANY_VALUE(eh.target_metric) as target_metric
FROM exploration_history eh
JOIN sessions s ON eh.session_id = s.session_id
GROUP BY eh.session_id;

-- Feature discovery timeline
CREATE VIEW IF NOT EXISTS discovery_timeline AS
SELECT 
    DATE(eh.timestamp) as discovery_date,
    eh.session_id,
    eh.operation_applied,
    eh.evaluation_score,
    eh.target_metric,
    fi.impact_delta,
    ROW_NUMBER() OVER (PARTITION BY eh.session_id ORDER BY eh.timestamp) as discovery_order
FROM exploration_history eh
LEFT JOIN feature_impact fi ON eh.operation_applied = fi.feature_name
WHERE eh.is_best_so_far = TRUE
ORDER BY eh.timestamp;

-- Operation effectiveness ranking
CREATE VIEW IF NOT EXISTS operation_ranking AS
SELECT 
    operation_name,
    ANY_VALUE(operation_category) as operation_category,
    SUM(total_applications) as total_applications,
    SUM(success_count) as success_count,
    ROUND(100.0 * SUM(success_count) / SUM(total_applications), 2) as success_rate,
    AVG(avg_improvement) as avg_improvement,
    MAX(best_improvement) as best_improvement,
    AVG(effectiveness_score) as effectiveness_score,
    COUNT(DISTINCT session_id) as used_in_sessions
FROM operation_performance
WHERE total_applications > 0
GROUP BY operation_name
ORDER BY effectiveness_score DESC;

-- DOWN MIGRATION
DROP VIEW IF EXISTS operation_ranking;
DROP VIEW IF EXISTS discovery_timeline;
DROP VIEW IF EXISTS session_summary;

DROP INDEX IF EXISTS idx_datasets_active;
DROP INDEX IF EXISTS idx_datasets_last_used;
DROP INDEX IF EXISTS idx_datasets_name;
DROP INDEX IF EXISTS idx_sessions_dataset;
DROP INDEX IF EXISTS idx_sessions_start;
DROP INDEX IF EXISTS idx_operation_session;
DROP INDEX IF EXISTS idx_impact_session;
DROP INDEX IF EXISTS idx_impact_delta;
DROP INDEX IF EXISTS idx_exploration_iteration;
DROP INDEX IF EXISTS idx_exploration_timestamp;
DROP INDEX IF EXISTS idx_exploration_score;
DROP INDEX IF EXISTS idx_exploration_session;