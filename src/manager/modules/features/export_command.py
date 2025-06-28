"""
Export Command - Export feature data to various formats.

Provides feature data export capabilities including:
- CSV format for spreadsheet analysis
- JSON format for programmatic access
- Filtered exports based on criteria
- Custom output file naming
"""

from typing import Dict, Any, List
import json
import csv
from pathlib import Path
from .base import BaseFeaturesCommand


class ExportCommand(BaseFeaturesCommand):
    """Handle --export command for features."""
    
    def execute(self, args) -> None:
        """Execute the feature export command."""
        try:
            export_format = args.export
            if export_format not in ['csv', 'json']:
                self.print_error("Export format must be 'csv' or 'json'")
                return
            
            # Build filters
            filters = self.build_feature_filters(args)
            
            # Get features to export
            features = self._get_export_data(filters)
            
            if not features:
                self.print_info("No features found matching the criteria.")
                return
            
            # Generate filename if not provided
            output_file = getattr(args, 'output_file', None)
            if not output_file:
                output_file = f"features_export.{export_format}"
            
            # Perform export
            if export_format == 'csv':
                self._export_csv(features, output_file)
            else:
                self._export_json(features, output_file, filters)
                
        except Exception as e:
            self.print_error(f"Failed to export features: {e}")
    
    def _get_export_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get feature data for export."""
        try:
            query = """
            SELECT 
                fc.feature_name,
                fc.category,
                fc.operation,
                fc.dependencies,
                fc.description,
                COUNT(fi.feature_name) as usage_count,
                AVG(fi.impact) as avg_impact,
                MAX(fi.impact) as max_impact,
                MIN(fi.impact) as min_impact,
                STDDEV(fi.impact) as impact_stddev,
                SUM(CASE WHEN fi.impact > 0 THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN fi.impact < 0 THEN 1 ELSE 0 END) as negative_count,
                SUM(fi.success_count) as total_successes,
                SUM(fi.failure_count) as total_failures
            FROM feature_catalog fc
            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
            """
            
            conditions = []
            params = []
            
            # Apply filters
            if 'session_id' in filters:
                conditions.append("fi.session_id = ?")
                params.append(filters['session_id'])
            
            if 'category' in filters:
                conditions.append("fc.category = ?")
                params.append(filters['category'])
            
            if 'dataset_hash' in filters:
                conditions.append("fi.dataset_hash = ?")
                params.append(filters['dataset_hash'])
            
            if 'min_impact' in filters:
                conditions.append("AVG(fi.impact) >= ?")
                params.append(filters['min_impact'])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += """
            GROUP BY fc.feature_name, fc.category, fc.operation, fc.dependencies, fc.description
            ORDER BY fc.category, fc.feature_name
            """
            
            if 'limit' in filters:
                query += f" LIMIT {filters['limit']}"
            
            return self.feature_service.repository.fetch_all(query, tuple(params))
            
        except Exception as e:
            self.print_error(f"Database query failed: {e}")
            return []
    
    def _export_csv(self, features: List[Dict[str, Any]], filename: str) -> None:
        """Export features to CSV format."""
        try:
            # Prepare CSV headers
            headers = [
                'feature_name', 'category', 'operation', 'dependencies',
                'description', 'usage_count', 'avg_impact', 'max_impact',
                'min_impact', 'impact_stddev', 'positive_count', 'negative_count',
                'total_successes', 'total_failures', 'success_rate'
            ]
            
            # Write CSV file
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for feature in features:
                    # Calculate success rate
                    usage_count = feature.get('usage_count', 0)
                    positive_count = feature.get('positive_count', 0)
                    success_rate = (positive_count / usage_count * 100) if usage_count > 0 else 0
                    
                    # Prepare row data
                    row = {
                        'feature_name': feature.get('feature_name', ''),
                        'category': feature.get('category', ''),
                        'operation': feature.get('operation', ''),
                        'dependencies': feature.get('dependencies', ''),
                        'description': feature.get('description', ''),
                        'usage_count': feature.get('usage_count', 0),
                        'avg_impact': feature.get('avg_impact', 0),
                        'max_impact': feature.get('max_impact', 0),
                        'min_impact': feature.get('min_impact', 0),
                        'impact_stddev': feature.get('impact_stddev', 0),
                        'positive_count': feature.get('positive_count', 0),
                        'negative_count': feature.get('negative_count', 0),
                        'total_successes': feature.get('total_successes', 0),
                        'total_failures': feature.get('total_failures', 0),
                        'success_rate': success_rate
                    }
                    
                    writer.writerow(row)
            
            self.print_success(f"Exported {len(features)} features to {filename}")
            
        except Exception as e:
            self.print_error(f"Failed to export CSV: {e}")
    
    def _export_json(self, features: List[Dict[str, Any]], filename: str, filters: Dict[str, Any]) -> None:
        """Export features to JSON format."""
        try:
            # Calculate additional statistics for each feature
            for feature in features:
                usage_count = feature.get('usage_count', 0)
                positive_count = feature.get('positive_count', 0)
                feature['success_rate'] = (positive_count / usage_count * 100) if usage_count > 0 else 0
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'export_timestamp': str(self._get_current_timestamp()),
                    'feature_count': len(features),
                    'filters_applied': filters,
                    'format': 'json'
                },
                'summary': {
                    'total_features': len(features),
                    'categories': list(set(f.get('category', '') for f in features)),
                    'total_usages': sum(f.get('usage_count', 0) for f in features),
                    'avg_impact': sum(f.get('avg_impact', 0) for f in features) / len(features) if features else 0
                },
                'features': features
            }
            
            # Write JSON file
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, default=str)
            
            self.print_success(f"Exported {len(features)} features to {filename}")
            
        except Exception as e:
            self.print_error(f"Failed to export JSON: {e}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for export metadata."""
        from datetime import datetime
        return datetime.now().isoformat()