-- Migration 006: Dynamic Feature Categories System
-- Adds support for dynamic feature-operation mappings
-- Replaces hardcoded patterns with database-driven categorization

-- Add operation_name column to feature_catalog
ALTER TABLE feature_catalog ADD COLUMN IF NOT EXISTS operation_name VARCHAR;

-- Create index for efficient operation_name lookups
CREATE INDEX IF NOT EXISTS idx_feature_catalog_operation ON feature_catalog(operation_name);

-- Create operation_categories table for mapping operations to categories
CREATE TABLE IF NOT EXISTS operation_categories (
    operation_name VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    description VARCHAR,
    dataset_name VARCHAR,
    is_generic BOOLEAN DEFAULT false,
    output_patterns TEXT[], -- Array of patterns this operation generates
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
     ARRAY['_mean_by_', '_std_by_', '_dev_from_', '_count_by_', '_min_by_', '_max_by_', '_norm_by_']),
    ('polynomial_features', 'polynomial', 'Polynomial feature transformations', true,
     ARRAY['_squared', '_cubed', '_sqrt', '_log', '_exp']),
    ('binning_features', 'binning', 'Quantile-based binning transformations', true,
     ARRAY['_bin_', '_quantile_']),
    ('ranking_features', 'ranking', 'Rank-based transformations', true,
     ARRAY['_rank', '_percentile_']),
    ('interaction_features', 'interaction', 'Feature interaction and combinations', true,
     ARRAY['_interaction_', '_cross_', '_ratio_']),
    ('feature_selection', 'selection', 'Feature selection and filtering', true,
     ARRAY['_selected_', '_filtered_'])
ON CONFLICT (operation_name) DO UPDATE SET
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    is_generic = EXCLUDED.is_generic,
    output_patterns = EXCLUDED.output_patterns,
    updated_at = CURRENT_TIMESTAMP;

-- Create function to automatically detect operation from feature name
CREATE OR REPLACE FUNCTION detect_operation_from_feature(feature_name VARCHAR)
RETURNS VARCHAR AS $$
DECLARE
    operation VARCHAR;
    pattern VARCHAR;
BEGIN
    -- Check each operation's patterns
    FOR operation, pattern IN 
        SELECT oc.operation_name, unnest(oc.output_patterns)
        FROM operation_categories oc
    LOOP
        IF feature_name ILIKE '%' || pattern || '%' THEN
            RETURN operation;
        END IF;
    END LOOP;
    
    -- Return null if no pattern matches
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-populate operation_name when inserting features
CREATE OR REPLACE FUNCTION auto_populate_operation_name()
RETURNS TRIGGER AS $$
BEGIN
    -- Only auto-populate if operation_name is null
    IF NEW.operation_name IS NULL THEN
        NEW.operation_name := detect_operation_from_feature(NEW.feature_name);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_auto_populate_operation_name ON feature_catalog;
CREATE TRIGGER trigger_auto_populate_operation_name
    BEFORE INSERT OR UPDATE ON feature_catalog
    FOR EACH ROW
    EXECUTE FUNCTION auto_populate_operation_name();

-- Update existing feature_catalog entries to populate operation_name
UPDATE feature_catalog 
SET operation_name = detect_operation_from_feature(feature_name)
WHERE operation_name IS NULL;

-- Create function to get features for a specific operation
CREATE OR REPLACE FUNCTION get_features_for_operation(op_name VARCHAR, ds_name VARCHAR DEFAULT NULL)
RETURNS TABLE(feature_name VARCHAR) AS $$
BEGIN
    IF ds_name IS NULL THEN
        RETURN QUERY
        SELECT fc.feature_name
        FROM feature_catalog fc
        WHERE fc.operation_name = op_name;
    ELSE
        RETURN QUERY
        SELECT fc.feature_name
        FROM feature_catalog fc
        WHERE fc.operation_name = op_name 
        AND fc.dataset_name = ds_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to get all available operations for a dataset
CREATE OR REPLACE FUNCTION get_available_operations(ds_name VARCHAR)
RETURNS TABLE(operation_name VARCHAR, category VARCHAR, feature_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fc.operation_name,
        COALESCE(oc.category, 'unknown') as category,
        COUNT(fc.feature_name) as feature_count
    FROM feature_catalog fc
    LEFT JOIN operation_categories oc ON fc.operation_name = oc.operation_name
    WHERE fc.dataset_name = ds_name
    AND fc.operation_name IS NOT NULL
    GROUP BY fc.operation_name, oc.category
    ORDER BY feature_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Create cleanup function to remove unused operation mappings
CREATE OR REPLACE FUNCTION cleanup_unused_operations()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete operation categories that have no corresponding features
    DELETE FROM operation_categories oc
    WHERE NOT EXISTS (
        SELECT 1 FROM feature_catalog fc 
        WHERE fc.operation_name = oc.operation_name
    )
    AND oc.is_generic = false; -- Keep generic operations
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;