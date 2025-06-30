-- Migration: MCTS Tree Persistence
-- Purpose: Store MCTS tree structure for session resumption and analysis

-- Create sequence for primary key
CREATE SEQUENCE IF NOT EXISTS mcts_tree_nodes_id_seq;

-- MCTS tree nodes table - stores the actual tree structure
CREATE TABLE IF NOT EXISTS mcts_tree_nodes (
    id BIGINT PRIMARY KEY DEFAULT NEXTVAL('mcts_tree_nodes_id_seq'),
    session_id VARCHAR NOT NULL REFERENCES sessions(session_id),
    node_id INTEGER NOT NULL,  -- MCTS internal node ID
    parent_node_id INTEGER,    -- Parent's MCTS node ID (NULL for root)
    depth INTEGER NOT NULL DEFAULT 0,
    
    -- Node properties
    operation_applied VARCHAR,  -- NULL for root
    state_id VARCHAR,
    
    -- MCTS statistics
    visit_count INTEGER NOT NULL DEFAULT 0,
    total_reward DOUBLE NOT NULL DEFAULT 0.0,
    evaluation_score DOUBLE,
    evaluation_time DOUBLE DEFAULT 0.0,
    
    -- Feature tracking
    base_features JSON NOT NULL,
    features_before JSON NOT NULL,
    features_after JSON NOT NULL,
    applied_operations JSON NOT NULL DEFAULT '[]',
    
    -- Tree structure
    is_leaf BOOLEAN NOT NULL DEFAULT TRUE,
    is_fully_expanded BOOLEAN NOT NULL DEFAULT FALSE,
    is_pruned BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_usage_mb DOUBLE,
    
    -- Constraints
    UNIQUE(session_id, node_id)
);

-- Indexes for efficient tree traversal
CREATE INDEX idx_mcts_tree_parent ON mcts_tree_nodes(session_id, parent_node_id);
CREATE INDEX idx_mcts_tree_depth ON mcts_tree_nodes(session_id, depth);
CREATE INDEX idx_mcts_tree_visit_count ON mcts_tree_nodes(session_id, visit_count DESC);

-- View for tree analysis
CREATE OR REPLACE VIEW mcts_tree_analysis AS
SELECT 
    mtn.session_id,
    s.session_name,
    mtn.node_id,
    mtn.parent_node_id,
    mtn.depth,
    mtn.operation_applied,
    mtn.visit_count,
    mtn.total_reward,
    CASE 
        WHEN mtn.visit_count > 0 THEN mtn.total_reward / mtn.visit_count 
        ELSE 0 
    END as average_reward,
    mtn.evaluation_score,
    mtn.is_leaf,
    mtn.is_fully_expanded,
    JSON_ARRAY_LENGTH(mtn.features_after) - JSON_ARRAY_LENGTH(mtn.features_before) as features_added,
    mtn.created_at,
    mtn.updated_at
FROM mcts_tree_nodes mtn
JOIN sessions s ON mtn.session_id = s.session_id;

-- Function to get UCB1 score (for analysis)
-- Note: DuckDB doesn't support stored procedures, so we'll use the calculation inline in views

-- View for UCB1 analysis
CREATE OR REPLACE VIEW mcts_ucb1_analysis AS
WITH parent_visits AS (
    SELECT 
        session_id,
        node_id,
        visit_count as parent_visit_count
    FROM mcts_tree_nodes
)
SELECT 
    c.session_id,
    c.node_id,
    c.parent_node_id,
    c.operation_applied,
    c.visit_count,
    c.total_reward,
    p.parent_visit_count,
    CASE 
        WHEN c.visit_count = 0 THEN 1e10
        WHEN p.parent_visit_count IS NULL OR p.parent_visit_count <= 0 THEN c.total_reward / c.visit_count
        ELSE (c.total_reward / c.visit_count) + 1.4 * SQRT(LN(p.parent_visit_count) / c.visit_count)
    END as ucb1_score
FROM mcts_tree_nodes c
LEFT JOIN parent_visits p ON c.session_id = p.session_id AND c.parent_node_id = p.node_id
WHERE c.operation_applied IS NOT NULL;

-- Update exploration_history to link with MCTS nodes
ALTER TABLE exploration_history 
ADD COLUMN IF NOT EXISTS mcts_node_id INTEGER;

-- Add comment explaining the column
COMMENT ON COLUMN exploration_history.mcts_node_id IS 'References the MCTS internal node ID (not the database ID)';

-- Migration metadata handled by MigrationRunner