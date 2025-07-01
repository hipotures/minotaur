-- Migration 007: Add Origin Field to Feature Catalog
-- Adds support for distinguishing between train/generic/custom features
-- Enables complete feature catalog including original dataset columns

-- Update existing features to have 'generic' origin (column already exists from migration 001)
UPDATE feature_catalog 
SET origin = 'generic' 
WHERE origin IS NULL;

-- Create view for easy feature filtering by origin
CREATE OR REPLACE VIEW features_by_origin AS
SELECT 
    origin,
    feature_category,
    operation_name,
    COUNT(*) as feature_count
FROM feature_catalog
GROUP BY origin, feature_category, operation_name
ORDER BY origin, feature_count DESC;