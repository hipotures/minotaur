#!/usr/bin/env python3
"""
Features Module - Feature analysis and management

Provides commands for analyzing feature performance, impact analysis, and feature catalog management.
"""

import argparse
import json
from typing import Dict, List, Any, Optional
from . import ModuleInterface

class FeaturesModule(ModuleInterface):
    """Module for analyzing and managing features."""
    
    @property
    def name(self) -> str:
        return "features"
    
    @property
    def description(self) -> str:
        return "Analyze feature performance and manage feature catalog"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--list": "List features with performance metrics",
            "--top": "Show top performing features",
            "--impact": "Analyze feature impact on model performance", 
            "--catalog": "Show feature catalog summary",
            "--export": "Export feature analysis to CSV/JSON",
            "--search": "Search features by name or category",
            "--help": "Show detailed help for features module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add feature-specific arguments."""
        features_group = parser.add_argument_group('Features Module')
        features_group.add_argument('--list', action='store_true',
                                  help='List all features with metrics')
        features_group.add_argument('--top', type=int, metavar='N', default=10,
                                  help='Show top N performing features')
        features_group.add_argument('--impact', type=str, metavar='FEATURE_NAME',
                                  help='Show impact analysis for specific feature')
        features_group.add_argument('--catalog', action='store_true',
                                  help='Show feature catalog summary')
        features_group.add_argument('--export', type=str, metavar='FORMAT',
                                  choices=['csv', 'json'],
                                  help='Export feature data')
        features_group.add_argument('--search', type=str, metavar='QUERY',
                                  help='Search features by name/category')
        features_group.add_argument('--session', type=str, metavar='SESSION_ID',
                                  help='Filter features by session')
        features_group.add_argument('--category', type=str,
                                  help='Filter features by category')
        features_group.add_argument('--dataset', type=str, metavar='HASH_OR_NAME',
                                  help='Filter features by dataset hash (e.g., 6c7ccd95) or name (e.g., Titanic)')
        features_group.add_argument('--dataset-name', type=str, metavar='NAME',
                                  help='Filter features by dataset name (e.g., Titanic, Fertilizer S5E6)')
        features_group.add_argument('--min-impact', type=float, default=0.0,
                                  help='Minimum impact threshold')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute features module commands."""
        
        if getattr(args, 'list', False):
            self._list_features(args, manager)
        elif getattr(args, 'export', None):
            self._export_features(args.export, args, manager)
        elif args.impact:
            self._show_feature_impact(args.impact, args, manager)
        elif args.catalog:
            self._show_feature_catalog(args, manager)
        elif args.search:
            self._search_features(args.search, args, manager)
        elif hasattr(args, 'top'):
            # Default action if no specific command
            self._show_top_features(args, manager)
        else:
            print("❌ No features command specified. Use --help for options.")
    
    def _list_features(self, args: argparse.Namespace, manager) -> None:
        """List all features with performance metrics."""
        print("🧪 FEATURE PERFORMANCE LIST")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                # Build dynamic query based on filters
                where_clauses = []
                params = []
                
                if args.session:
                    where_clauses.append("fi.session_id = ?")
                    params.append(args.session)
                
                if args.category:
                    where_clauses.append("fc.feature_category = ?")
                    params.append(args.category)
                
                if args.min_impact > 0:
                    where_clauses.append("fi.impact_delta >= ?")
                    params.append(args.min_impact)
                
                if getattr(args, 'dataset', None) or getattr(args, 'dataset_name', None):
                    dataset_filter = getattr(args, 'dataset', None) or getattr(args, 'dataset_name', None)
                    # Check if it's a name or hash by trying to find matching dataset
                    dataset_hash = self._resolve_dataset_identifier(dataset_filter, manager)
                    if dataset_hash:
                        where_clauses.append("s.dataset_hash = ?")
                        params.append(dataset_hash)
                    else:
                        # Fallback to partial hash matching
                        where_clauses.append("s.dataset_hash LIKE ?")
                        params.append(f"{dataset_filter}%")
                
                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                query = f"""
                    SELECT 
                        fc.feature_name,
                        fc.feature_category,
                        fi.impact_delta,
                        fi.impact_percentage,
                        fi.with_feature_score,
                        fi.baseline_score,
                        fi.sample_size,
                        fc.computational_cost,
                        fi.session_id
                    FROM feature_catalog fc
                    LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                    LEFT JOIN sessions s ON fi.session_id = s.session_id
                    {where_clause}
                    ORDER BY fi.impact_delta DESC NULLS LAST
                """
                
                result = conn.execute(query, params).fetchall()
                
                if not result:
                    print("No features found matching criteria.")
                    return
                
                # Display results
                print(f"{'Feature Name':<30} {'Category':<15} {'Impact':<10} {'Score':<10} {'Cost':<6} {'Session':<10}")
                print("-" * 85)
                
                for row in result:
                    name, category, impact, impact_pct, score, baseline, samples, cost, session_id = row
                    
                    name_short = name[:29] if name else "Unknown"
                    category_short = (category or "N/A")[:14]
                    impact_display = f"{impact:.5f}" if impact else "N/A"
                    score_display = f"{score:.5f}" if score else "N/A"
                    cost_display = f"{cost:.1f}" if cost else "N/A"
                    session_short = session_id[:8] if session_id else "N/A"
                    
                    print(f"{name_short:<30} {category_short:<15} {impact_display:<10} {score_display:<10} {cost_display:<6} {session_short:<10}")
                
                print(f"\nTotal features: {len(result)}")
                
                # Summary statistics
                valid_impacts = [row[2] for row in result if row[2] is not None]
                if valid_impacts:
                    print(f"Impact range: {min(valid_impacts):.5f} - {max(valid_impacts):.5f}")
                    print(f"Average impact: {sum(valid_impacts)/len(valid_impacts):.5f}")
                
        except Exception as e:
            print(f"❌ Error listing features: {e}")
    
    def _show_top_features(self, args: argparse.Namespace, manager) -> None:
        """Show top performing features."""
        limit = getattr(args, 'top', 10)
        print(f"🏆 TOP {limit} PERFORMING FEATURES")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                # Build dynamic query with dataset filtering
                where_clauses = ["fi.impact_delta > 0"]
                params = []
                
                if getattr(args, 'dataset', None) or getattr(args, 'dataset_name', None):
                    dataset_filter = getattr(args, 'dataset', None) or getattr(args, 'dataset_name', None)
                    # Check if it's a name or hash by trying to find matching dataset
                    dataset_hash = self._resolve_dataset_identifier(dataset_filter, manager)
                    if dataset_hash:
                        where_clauses.append("s.dataset_hash = ?")
                        params.append(dataset_hash)
                    else:
                        # Fallback to partial hash matching
                        where_clauses.append("s.dataset_hash LIKE ?")
                        params.append(f"{dataset_filter}%")
                
                where_clause = "WHERE " + " AND ".join(where_clauses)
                params.append(limit)  # Add limit parameter at the end
                
                # Get top features by impact with optional dataset filtering
                query = f"""
                    SELECT 
                        fc.feature_name,
                        fc.feature_category,
                        fi.impact_delta,
                        fi.impact_percentage,
                        fi.with_feature_score,
                        fi.baseline_score,
                        fi.sample_size,
                        fc.computational_cost,
                        fc.python_code,
                        fi.session_id
                    FROM feature_catalog fc
                    JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                    LEFT JOIN sessions s ON fi.session_id = s.session_id
                    {where_clause}
                    ORDER BY fi.impact_delta DESC
                    LIMIT ?
                """
                
                result = conn.execute(query, params).fetchall()
                
                if not result:
                    print("No features with positive impact found.")
                    return
                
                for i, row in enumerate(result, 1):
                    name, category, impact, impact_pct, score, baseline, samples, cost, code, session_id = row
                    
                    print(f"{i:2}. {name}")
                    print(f"    Category: {category}")
                    print(f"    Impact: {impact:.5f} ({impact_pct:.2f}%)")
                    print(f"    Score: {baseline:.5f} → {score:.5f}")
                    print(f"    Samples: {samples}, Cost: {cost:.1f}")
                    print(f"    Session: {session_id[:8]}...")
                    
                    # Show code snippet
                    if code:
                        code_preview = code[:100] + "..." if len(code) > 100 else code
                        print(f"    Code: {code_preview}")
                    print()
                
        except Exception as e:
            print(f"❌ Error showing top features: {e}")
    
    def _show_feature_impact(self, feature_name: str, args: argparse.Namespace, manager) -> None:
        """Show detailed impact analysis for a specific feature."""
        print(f"📊 FEATURE IMPACT ANALYSIS: {feature_name}")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                # Get feature details
                feature_info = conn.execute("""
                    SELECT 
                        fc.feature_name,
                        fc.feature_category,
                        fc.description,
                        fc.created_by,
                        fc.creation_timestamp,
                        fc.computational_cost,
                        fc.python_code
                    FROM feature_catalog fc
                    WHERE fc.feature_name = ?
                """, [feature_name]).fetchone()
                
                if not feature_info:
                    print(f"❌ Feature not found in catalog: {feature_name}")
                    return
                
                name, category, description, created_by, created_at, cost, code = feature_info
                
                print("📋 FEATURE INFORMATION:")
                print(f"   Name: {name}")
                print(f"   Category: {category}")
                print(f"   Description: {description or 'No description'}")
                print(f"   Created by: {created_by}")
                print(f"   Created: {created_at}")
                print(f"   Computational Cost: {cost}")
                print()
                
                # Get impact data across sessions
                impact_data = conn.execute("""
                    SELECT 
                        fi.session_id,
                        fi.baseline_score,
                        fi.with_feature_score,
                        fi.impact_delta,
                        fi.impact_percentage,
                        fi.sample_size,
                        fi.first_discovered,
                        fi.last_evaluated,
                        s.session_name
                    FROM feature_impact fi
                    LEFT JOIN sessions s ON fi.session_id = s.session_id
                    WHERE fi.feature_name = ?
                    ORDER BY fi.impact_delta DESC
                """, [feature_name]).fetchall()
                
                if impact_data:
                    print("📈 IMPACT ACROSS SESSIONS:")
                    total_impact = sum(row[3] for row in impact_data if row[3])
                    avg_impact = total_impact / len(impact_data) if impact_data else 0
                    
                    print(f"   Sessions tested: {len(impact_data)}")
                    print(f"   Average impact: {avg_impact:.5f}")
                    print(f"   Total improvement: {total_impact:.5f}")
                    print()
                    
                    print(f"{'Session':<10} {'Name':<15} {'Baseline':<10} {'With Feature':<12} {'Impact':<10} {'%':<8}")
                    print("-" * 70)
                    
                    for row in impact_data:
                        session_id, baseline, with_feat, impact, impact_pct, samples, first, last, sess_name = row
                        
                        session_short = session_id[:8]
                        name_short = (sess_name or "Unnamed")[:14]
                        baseline_str = f"{baseline:.5f}" if baseline else "N/A"
                        with_feat_str = f"{with_feat:.5f}" if with_feat else "N/A"
                        impact_str = f"{impact:.5f}" if impact else "N/A"
                        impact_pct_str = f"{impact_pct:.2f}%" if impact_pct else "N/A"
                        
                        print(f"{session_short:<10} {name_short:<15} {baseline_str:<10} {with_feat_str:<12} {impact_str:<10} {impact_pct_str:<8}")
                    
                    print()
                else:
                    print("⚠️  No impact data found for this feature.")
                
                # Show code
                if code:
                    print("💻 FEATURE CODE:")
                    print(code)
                    print()
                
        except Exception as e:
            print(f"❌ Error showing feature impact: {e}")
    
    def _show_feature_catalog(self, args: argparse.Namespace, manager) -> None:
        """Show feature catalog summary."""
        print("📚 FEATURE CATALOG SUMMARY")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Get catalog statistics
                catalog_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_features,
                        COUNT(DISTINCT feature_category) as categories,
                        AVG(computational_cost) as avg_cost,
                        COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_features
                    FROM feature_catalog
                """).fetchone()
                
                if catalog_stats:
                    total, categories, avg_cost, active = catalog_stats
                    print(f"📊 STATISTICS:")
                    print(f"   Total Features: {total}")
                    print(f"   Active Features: {active}")
                    print(f"   Categories: {categories}")
                    print(f"   Average Cost: {avg_cost:.2f}" if avg_cost is not None else "   Average Cost: N/A")
                    print()
                
                # Get features by category
                category_breakdown = conn.execute("""
                    SELECT 
                        feature_category,
                        COUNT(*) as count,
                        AVG(computational_cost) as avg_cost,
                        COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_count
                    FROM feature_catalog
                    GROUP BY feature_category
                    ORDER BY count DESC
                """).fetchall()
                
                if category_breakdown:
                    print("📁 BY CATEGORY:")
                    print(f"{'Category':<20} {'Total':<8} {'Active':<8} {'Avg Cost':<10}")
                    print("-" * 50)
                    
                    for category, count, avg_cost, active_count in category_breakdown:
                        category_name = category or "Uncategorized"
                        cost_str = f"{avg_cost:.2f}" if avg_cost else "N/A"
                        print(f"{category_name:<20} {count:<8} {active_count:<8} {cost_str:<10}")
                    print()
                
                # Show recent additions
                recent_features = conn.execute("""
                    SELECT feature_name, feature_category, created_by, creation_timestamp
                    FROM feature_catalog
                    ORDER BY creation_timestamp DESC
                    LIMIT 5
                """).fetchall()
                
                if recent_features:
                    print("🆕 RECENT ADDITIONS:")
                    for name, category, created_by, created_at in recent_features:
                        print(f"   • {name} ({category}) by {created_by} on {created_at}")
                    print()
                
        except Exception as e:
            print(f"❌ Error showing feature catalog: {e}")
    
    def _export_features(self, format_type: str, args: argparse.Namespace, manager) -> None:
        """Export feature data to file."""
        print(f"📦 EXPORTING FEATURES TO {format_type.upper()}")
        print("=" * 50)
        
        try:
            from pathlib import Path
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type == 'csv':
                # Use dedicated DuckDB export directory
                export_config = manager.get_export_config()
                exports_dir = manager.project_root / export_config['export_dir']
                exports_dir.mkdir(parents=True, exist_ok=True)
                output_file = exports_dir / f"features_export_{timestamp}.csv"
                
                with manager._connect() as conn:
                    # Use string formatting for COPY TO since parameter binding doesn't work
                    query = f"""
                        COPY (
                            SELECT 
                                fc.feature_name,
                                fc.feature_category,
                                fi.impact_delta,
                                fi.impact_percentage,
                                fi.with_feature_score,
                                fi.baseline_score,
                                fi.sample_size,
                                fc.computational_cost,
                                fc.created_by,
                                fc.creation_timestamp,
                                fi.session_id
                            FROM feature_catalog fc
                            LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                            ORDER BY fi.impact_delta DESC NULLS LAST
                        ) TO '{str(output_file)}' (HEADER, DELIMITER ',')
                    """
                    conn.execute(query)
                    
                print(f"✅ Exported to: {output_file}")
                
            elif format_type == 'json':
                import json
                # Use dedicated DuckDB export directory
                export_config = manager.get_export_config()
                exports_dir = manager.project_root / export_config['export_dir']
                exports_dir.mkdir(parents=True, exist_ok=True)
                output_file = exports_dir / f"features_export_{timestamp}.json"
                
                with manager._connect() as conn:
                    features = conn.execute("""
                        SELECT 
                            fc.feature_name,
                            fc.feature_category,
                            fc.description,
                            fc.python_code,
                            fc.dependencies,
                            fc.computational_cost,
                            fc.created_by,
                            fc.creation_timestamp,
                            fc.is_active,
                            fi.impact_delta,
                            fi.impact_percentage,
                            fi.with_feature_score,
                            fi.baseline_score,
                            fi.sample_size,
                            fi.session_id
                        FROM feature_catalog fc
                        LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                        ORDER BY fi.impact_delta DESC NULLS LAST
                    """).fetchall()
                    
                    features_data = []
                    for feature in features:
                        feature_dict = {
                            'name': feature[0],
                            'category': feature[1],
                            'description': feature[2],
                            'python_code': feature[3],
                            'dependencies': json.loads(feature[4]) if feature[4] else [],
                            'computational_cost': feature[5],
                            'created_by': feature[6],
                            'creation_timestamp': feature[7],
                            'is_active': feature[8],
                            'impact': {
                                'delta': feature[9],
                                'percentage': feature[10],
                                'with_feature_score': feature[11],
                                'baseline_score': feature[12],
                                'sample_size': feature[13],
                                'session_id': feature[14]
                            } if feature[9] is not None else None
                        }
                        features_data.append(feature_dict)
                    
                    with open(output_file, 'w') as f:
                        json.dump(features_data, f, indent=2, default=str)
                    
                print(f"✅ Exported {len(features_data)} features to: {output_file}")
                
        except Exception as e:
            print(f"❌ Error exporting features: {e}")
    
    def _search_features(self, query: str, args: argparse.Namespace, manager) -> None:
        """Search features by name or category."""
        print(f"🔍 SEARCHING FEATURES: '{query}'")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Search in name, category, and description
                result = conn.execute("""
                    SELECT 
                        fc.feature_name,
                        fc.feature_category,
                        fc.description,
                        fi.impact_delta,
                        fc.computational_cost,
                        fc.creation_timestamp
                    FROM feature_catalog fc
                    LEFT JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                    WHERE fc.feature_name ILIKE ?
                       OR fc.feature_category ILIKE ?
                       OR fc.description ILIKE ?
                    ORDER BY fi.impact_delta DESC NULLS LAST
                """, [f"%{query}%", f"%{query}%", f"%{query}%"]).fetchall()
                
                if not result:
                    print(f"No features found matching '{query}'")
                    return
                
                print(f"Found {len(result)} features:")
                print()
                
                for i, row in enumerate(result, 1):
                    name, category, description, impact, cost, created_at = row
                    
                    print(f"{i:2}. {name}")
                    print(f"    Category: {category}")
                    if description:
                        desc_preview = description[:80] + "..." if len(description) > 80 else description
                        print(f"    Description: {desc_preview}")
                    print(f"    Impact: {impact:.5f}" if impact else "    Impact: No data")
                    print(f"    Cost: {cost:.2f}, Created: {created_at}")
                    print()
                
        except Exception as e:
            print(f"❌ Error searching features: {e}")
    
    def _resolve_dataset_identifier(self, identifier: str, manager) -> Optional[str]:
        """Resolve dataset name or partial hash to full dataset ID."""
        try:
            with manager._connect() as conn:
                # Try exact name match first
                result = conn.execute("""
                    SELECT dataset_id FROM datasets 
                    WHERE dataset_name = ?
                    LIMIT 1
                """, [identifier]).fetchone()
                
                if result:
                    return result[0]
                
                # Try case-insensitive name match
                result = conn.execute("""
                    SELECT dataset_id FROM datasets 
                    WHERE dataset_name ILIKE ?
                    LIMIT 1
                """, [identifier]).fetchone()
                
                if result:
                    return result[0]
                
                # Try partial hash match
                result = conn.execute("""
                    SELECT dataset_id FROM datasets 
                    WHERE dataset_id LIKE ?
                    LIMIT 1
                """, [f"{identifier}%"]).fetchone()
                
                if result:
                    return result[0]
                
                return None
                
        except Exception:
            return None