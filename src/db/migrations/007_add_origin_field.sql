-- Migration 007: Add Origin Field to Feature Catalog
-- Adds support for distinguishing between train/generic/custom features
-- Enables complete feature catalog including original dataset columns

-- Add origin column to feature_catalog table
ALTER TABLE feature_catalog ADD COLUMN origin VARCHAR;

-- Update existing features to have 'generic' origin (since they are generated features)
UPDATE feature_catalog 
SET origin = 'generic' 
WHERE origin IS NULL;

-- Create index for efficient origin filtering
CREATE INDEX IF NOT EXISTS idx_feature_catalog_origin ON feature_catalog(origin);

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