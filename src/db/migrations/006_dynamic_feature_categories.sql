-- NOTE: This migration uses hardcoded categories and should be reviewed
-- Consider using dynamic categories from OPERATION_METADATA
-- Migration 006: Dynamic Feature Categories System
-- Adds support for dynamic feature-operation mappings
-- Replaces hardcoded patterns with database-driven categorization

-- Create operation_categories table for mapping operations to categories
CREATE TABLE IF NOT EXISTS operation_categories (
    operation_name VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    description VARCHAR,
    dataset_name VARCHAR,
    is_generic BOOLEAN DEFAULT false,
    output_patterns VARCHAR[], -- Array of patterns this operation generates
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_operation_categories_category ON operation_categories(category);
CREATE INDEX IF NOT EXISTS idx_operation_categories_dataset ON operation_categories(dataset_name);
CREATE INDEX IF NOT EXISTS idx_operation_categories_generic ON operation_categories(is_generic);

-- Create view for easy feature-operation mapping queries
CREATE OR REPLACE VIEW feature_operation_mapping AS
SELECT 
    fc.feature_name,
    fc.operation_name,
    oc.category,
    oc.dataset_name,
    oc.is_generic,
    oc.output_patterns
FROM feature_catalog fc
LEFT JOIN operation_categories oc ON fc.operation_name = oc.operation_name;

-- Insert initial generic operation categories
INSERT INTO operation_categories (operation_name, category, description, is_generic, output_patterns) VALUES
    ('statistical_aggregations', 'statistical', 'Statistical aggregations by categorical features', true, 
     ['_mean_by_', '_std_by_', '_dev_from_', '_count_by_', '_min_by_', '_max_by_', '_norm_by_']),
    ('polynomial_features', 'polynomial', 'Polynomial feature transformations', true,
     ['_squared', '_cubed', '_sqrt', '_log', '_exp']),
    ('binning_features', 'binning', 'Quantile-based binning transformations', true,
     ['_bin_', '_quantile_']),
    ('ranking_features', 'ranking', 'Rank-based transformations', true,
     ['_rank', '_percentile_']),
    ('interaction_features', 'interaction', 'Feature interaction and combinations', true,
     ['_interaction_', '_cross_', '_ratio_']),
    ('feature_selection', 'selection', 'Feature selection and filtering', true,
     ['_selected_', '_filtered_'])
ON CONFLICT (operation_name) DO UPDATE SET
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    is_generic = EXCLUDED.is_generic,
    output_patterns = EXCLUDED.output_patterns,
    updated_at = NOW();