-- Migration: Add feature metadata tracking
-- Purpose: Track feature generation metadata for analysis and optimization

-- Feature metadata table
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_id INTEGER PRIMARY KEY,
    feature_name VARCHAR NOT NULL,
    dataset_name VARCHAR NOT NULL,
    feature_type VARCHAR NOT NULL CHECK (feature_type IN ('original', 'custom', 'generic', 'derived')),
    category VARCHAR NOT NULL,
    generation_time REAL NOT NULL,
    has_signal BOOLEAN NOT NULL,
    source_columns JSON,  -- List of source column names
    operation VARCHAR,    -- Operation that created this feature
    parameters JSON,      -- Parameters used in generation
    statistics JSON,      -- Feature statistics (mean, std, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint on feature name per dataset
    UNIQUE(feature_name, dataset_name)
);

-- Index for fast lookups
CREATE INDEX idx_feature_metadata_dataset ON feature_metadata(dataset_name);
CREATE INDEX idx_feature_metadata_type ON feature_metadata(feature_type);
CREATE INDEX idx_feature_metadata_signal ON feature_metadata(has_signal);
CREATE INDEX idx_feature_metadata_category ON feature_metadata(category);

-- Feature usage tracking (how often features are used in MCTS)
CREATE TABLE IF NOT EXISTS feature_usage (
    usage_id INTEGER PRIMARY KEY,
    feature_name VARCHAR NOT NULL,
    dataset_name VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    iteration INTEGER NOT NULL,
    node_depth INTEGER NOT NULL,
    evaluation_score REAL,
    improvement REAL,  -- Score improvement when this feature was added
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (feature_name, dataset_name) REFERENCES feature_metadata(feature_name, dataset_name)
);

-- Index for usage analysis
CREATE INDEX idx_feature_usage_feature ON feature_usage(feature_name, dataset_name);
CREATE INDEX idx_feature_usage_session ON feature_usage(session_id);
CREATE INDEX idx_feature_usage_score ON feature_usage(evaluation_score);

-- Feature dependencies table (tracks which features depend on others)
CREATE TABLE IF NOT EXISTS feature_dependencies (
    dependency_id INTEGER PRIMARY KEY,
    feature_name VARCHAR NOT NULL,
    dataset_name VARCHAR NOT NULL,
    depends_on_feature VARCHAR NOT NULL,
    dependency_type VARCHAR DEFAULT 'direct',  -- 'direct' or 'transitive'
    
    FOREIGN KEY (feature_name, dataset_name) REFERENCES feature_metadata(feature_name, dataset_name),
    UNIQUE(feature_name, dataset_name, depends_on_feature)
);

-- Index for dependency queries
CREATE INDEX idx_feature_deps_feature ON feature_dependencies(feature_name, dataset_name);
CREATE INDEX idx_feature_deps_depends_on ON feature_dependencies(depends_on_feature);

-- View for feature performance analysis
CREATE VIEW IF NOT EXISTS feature_performance AS
SELECT 
    fm.feature_name,
    fm.dataset_name,
    fm.feature_type,
    fm.category,
    fm.generation_time,
    fm.has_signal,
    COUNT(fu.usage_id) as usage_count,
    AVG(fu.improvement) as avg_improvement,
    MAX(fu.improvement) as max_improvement,
    AVG(fu.evaluation_score) as avg_score
FROM feature_metadata fm
LEFT JOIN feature_usage fu ON fm.feature_name = fu.feature_name 
    AND fm.dataset_name = fu.dataset_name
GROUP BY fm.feature_name, fm.dataset_name, fm.feature_type, 
         fm.category, fm.generation_time, fm.has_signal;

-- View for feature dependency graph
CREATE VIEW IF NOT EXISTS feature_dependency_graph AS
WITH RECURSIVE dependency_tree AS (
    -- Base case: direct dependencies
    SELECT 
        feature_name,
        dataset_name,
        depends_on_feature,
        1 as depth
    FROM feature_dependencies
    WHERE dependency_type = 'direct'
    
    UNION ALL
    
    -- Recursive case: transitive dependencies
    SELECT 
        dt.feature_name,
        dt.dataset_name,
        fd.depends_on_feature,
        dt.depth + 1 as depth
    FROM dependency_tree dt
    JOIN feature_dependencies fd ON dt.depends_on_feature = fd.feature_name 
        AND dt.dataset_name = fd.dataset_name
    WHERE fd.dependency_type = 'direct' AND dt.depth < 10  -- Prevent infinite recursion
)
SELECT DISTINCT
    feature_name,
    dataset_name,
    depends_on_feature,
    MIN(depth) as min_depth
FROM dependency_tree
GROUP BY feature_name, dataset_name, depends_on_feature;