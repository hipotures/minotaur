-- Migration 007: Session Resume Parameters View
-- Creates view for aggregating session parameters needed for MCTS resume

-- Drop view if exists (for rerunning migration)
DROP VIEW IF EXISTS session_resume_params;

-- Create view that aggregates all parameters needed to resume an MCTS session
CREATE VIEW session_resume_params AS
SELECT 
    s.session_id,
    s.session_name,
    s.start_time,
    s.status,
    s.total_iterations AS session_total_iterations,
    s.best_score AS session_best_score,
    
    -- Exploration history aggregates
    COALESCE(eh.max_iteration, -1) AS last_iteration,
    COALESCE(eh.total_evaluations, 0) AS total_evaluations,
    COALESCE(eh.best_observed_score, 0.0) AS best_observed_score,
    COALESCE(eh.total_eval_time, 0.0) AS total_evaluation_time,
    COALESCE(eh.unique_nodes, 0) AS unique_nodes_count,
    COALESCE(eh.root_score, 0.0) AS root_evaluation_score,
    
    -- Next iteration to start from (for continuous numbering)
    COALESCE(eh.max_iteration, -1) + 1 AS next_iteration,
    
    -- Resume metadata
    CURRENT_TIMESTAMP AS resume_prepared_at,
    
    -- Flags for resume validation
    CASE 
        WHEN eh.max_iteration IS NOT NULL THEN true 
        ELSE false 
    END AS has_exploration_history,
    
    CASE 
        WHEN s.status = 'active' THEN true 
        ELSE false 
    END AS is_resumable

FROM sessions s
LEFT JOIN (
    SELECT 
        session_id,
        MAX(iteration) AS max_iteration,
        COUNT(*) AS total_evaluations,
        MAX(evaluation_score) AS best_observed_score,
        SUM(evaluation_time) AS total_eval_time,
        COUNT(DISTINCT mcts_node_id) AS unique_nodes,
        MAX(CASE WHEN operation_applied = 'root' THEN evaluation_score END) AS root_score
    FROM exploration_history 
    GROUP BY session_id
) eh ON s.session_id = eh.session_id;

-- Create index on session_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_session_resume_params_session_id 
ON sessions(session_id);

-- Add comment explaining the view purpose
COMMENT ON VIEW session_resume_params IS 
'Aggregated view of all parameters needed to resume an MCTS session with continuous iteration numbering. 
Includes session metadata, exploration history aggregates, and calculated next iteration number.';