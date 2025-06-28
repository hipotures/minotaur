#!/usr/bin/env python3
"""
Datasets Module - Dataset registry and management

Provides commands for managing datasets, viewing dataset information,
and analyzing dataset usage across sessions.
"""

import argparse
import json
import os
import sys
import re
import hashlib
import shutil
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from . import ModuleInterface

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from dataset_manager import DatasetManager
from dataset_importer import DatasetImporter

import duckdb

class DatasetsModule(ModuleInterface):
    """Module for managing and analyzing datasets."""
    
    def __init__(self):
        """Initialize datasets module with logging."""
        self.logger = logging.getLogger(__name__)
        
        # Configure logging to match main system
        if not self.logger.handlers:
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            
            # File handler for database operations log
            try:
                log_dir = Path('logs')
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(log_dir / 'db.log')
                # Import SafeFormatter from db config
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
                    from db.config.logging_config import SafeFormatter
                    file_formatter = SafeFormatter('%(asctime)s.%(msecs)03d - [%(dataset_name)s] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                except ImportError:
                    file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d - [dataset] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                file_handler.setFormatter(file_formatter)
                
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO)
            except Exception as e:
                # Fallback to console only
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO)
                print(f"⚠️ Could not setup file logging: {e}")
    
    @property
    def name(self) -> str:
        return "datasets"
    
    @property
    def description(self) -> str:
        return "Manage dataset registry and analyze dataset usage"
    
    @property
    def commands(self) -> Dict[str, str]:
        return {
            "--list": "List all registered datasets with usage statistics",
            "--details": "Show detailed information about a specific dataset",
            "--register": "Register a new dataset manually",
            "--cleanup": "Safely remove dataset and all related data",
            "--stats": "Show dataset usage statistics and comparisons",
            "--sessions": "Show sessions using a specific dataset",
            "--update": "Update dataset metadata",
            "--search": "Search datasets by name or path",
            "--help": "Show detailed help for datasets module"
        }
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add dataset-specific arguments."""
        datasets_group = parser.add_argument_group('Datasets Module')
        datasets_group.add_argument('--list', action='store_true',
                                  help='List all registered datasets')
        datasets_group.add_argument('--active-only', action='store_true',
                                  help='Show only active datasets (with --list)')
        datasets_group.add_argument('--details', type=str, metavar='IDENTIFIER',
                                  help='Show detailed dataset information (name or ID)')
        datasets_group.add_argument('--register', action='store_true',
                                  help='Register a new dataset')
        datasets_group.add_argument('--cleanup', type=str, metavar='IDENTIFIER',
                                  help='Safely remove dataset and all related data')
        
        # Registration-specific arguments
        reg_group = parser.add_argument_group('Dataset Registration')
        reg_group.add_argument('--dataset-name', type=str, required=False,
                              help='Dataset name (required for registration)')
        reg_group.add_argument('--dataset-path', type=str,
                              help='Path to dataset directory for auto-detection')
        reg_group.add_argument('--auto', action='store_true',
                              help='Automatically detect dataset files')
        reg_group.add_argument('--train', type=str, help='Path to train file')
        reg_group.add_argument('--test', type=str, help='Path to test file')
        reg_group.add_argument('--submission', type=str, help='Path to submission file')
        reg_group.add_argument('--validation', type=str, help='Path to validation file')
        reg_group.add_argument('--target-column', type=str, 
                              help='Target column name (required)')
        reg_group.add_argument('--id-column', type=str,
                              help='ID column name (optional)')
        reg_group.add_argument('--competition-name', type=str,
                              help='Kaggle competition name')
        reg_group.add_argument('--description', type=str,
                              help='Dataset description')
        reg_group.add_argument('--force', action='store_true',
                              help='Force overwrite existing dataset')
        datasets_group.add_argument('--stats', action='store_true',
                                  help='Show dataset usage statistics')
        datasets_group.add_argument('--sessions', type=str, metavar='DATASET_NAME',
                                  help='Show sessions using specific dataset')
        datasets_group.add_argument('--update', type=str, metavar='DATASET_NAME',
                                  help='Update dataset metadata')
        datasets_group.add_argument('--update-features', type=str, metavar='DATASET_NAME',
                                  help='Regenerate features for dataset if new features are available')
        datasets_group.add_argument('--search', type=str, metavar='QUERY',
                                  help='Search datasets by name or path')
        datasets_group.add_argument('--format', choices=['table', 'json'],
                                  default='table', help='Output format')
        datasets_group.add_argument('--dry-run', action='store_true',
                                  help='Show what would be changed without making changes')
        datasets_group.add_argument('--force-update', action='store_true',
                                  help='Force feature regeneration even if no changes detected')
    
    def execute(self, args: argparse.Namespace, manager) -> None:
        """Execute datasets module commands."""
        
        if getattr(args, 'help', False):
            self._show_help()
        elif getattr(args, 'list', False):
            self._list_datasets(args, manager)
        elif args.details:
            self._show_dataset_details(args.details, args, manager)
        elif args.register:
            self._register_dataset(args, manager)
        elif args.cleanup:
            self._cleanup_dataset(args.cleanup, args, manager)
        elif args.stats:
            self._show_dataset_stats(args, manager)
        elif args.sessions:
            self._show_dataset_sessions(args.sessions, args, manager)
        elif args.update:
            self._update_dataset(args.update, args, manager)
        elif args.update_features:
            self._update_features(args.update_features, args, manager)
        elif args.search:
            self._search_datasets(args.search, args, manager)
        else:
            print("❌ No datasets command specified. Use --help for options.")
    
    def _show_help(self) -> None:
        """Show detailed help for datasets module."""
        print("📊 DATASETS MODULE HELP")
        print("=" * 50)
        print()
        print("Manage dataset registry and analyze dataset usage across MCTS sessions.")
        print()
        print("📋 AVAILABLE COMMANDS:")
        for command, description in self.commands.items():
            print(f"  {command:<15} {description}")
        print()
        print("💡 EXAMPLES:")
        print("  python scripts/duckdb_manager.py datasets --list")
        print("  python scripts/duckdb_manager.py datasets --details titanic")
        print("  python scripts/duckdb_manager.py datasets --stats")
        print("  python scripts/duckdb_manager.py datasets --sessions titanic")
    
    def _resolve_dataset_identifier(self, identifier: str) -> Tuple[str, str]:
        """Resolve dataset identifier as ID or name.
        
        Args:
            identifier: User input (ID hash or dataset name)
            
        Returns:
            Tuple of (type, identifier) where type is 'id' or 'name'
        """
        # ID/hash: 8+ hexadecimal characters (MD5 is 32 chars)
        if re.match(r'^[a-f0-9]{8,32}$', identifier.lower()):
            return 'id', identifier.lower()
        else:
            return 'name', identifier
    
    def _find_dataset_by_identifier(self, identifier: str, manager) -> Optional[Tuple]:
        """Find dataset by ID or name with smart matching.
        
        Args:
            identifier: Dataset ID (full/partial) or name
            manager: Database manager
            
        Returns:
            Dataset record tuple or None if not found
        """
        id_type, clean_identifier = self._resolve_dataset_identifier(identifier)
        
        try:
            with manager._connect() as conn:
                if id_type == 'id':
                    # Search by ID (full match first, then partial)
                    result = conn.execute("""
                        SELECT 
                            dataset_id, dataset_name, train_path, test_path, target_column, id_column,
                            competition_name, description, 
                            train_records, train_columns, train_format,
                            test_records, test_columns, test_format,
                            submission_records, submission_columns, submission_format,
                            validation_records, validation_columns, validation_format,
                            column_count, train_row_count, test_row_count,
                            created_at, last_used, is_active, data_size_mb, feature_types
                        FROM datasets 
                        WHERE dataset_id = ? OR dataset_id LIKE ?
                        ORDER BY 
                            CASE WHEN dataset_id = ? THEN 1 ELSE 2 END,
                            dataset_name
                        LIMIT 1
                    """, [clean_identifier, f"{clean_identifier}%", clean_identifier]).fetchone()
                else:
                    # Search by name (exact match first, then partial)
                    result = conn.execute("""
                        SELECT 
                            dataset_id, dataset_name, train_path, test_path, target_column, id_column,
                            competition_name, description, 
                            train_records, train_columns, train_format,
                            test_records, test_columns, test_format,
                            submission_records, submission_columns, submission_format,
                            validation_records, validation_columns, validation_format,
                            column_count, train_row_count, test_row_count,
                            created_at, last_used, is_active, data_size_mb, feature_types
                        FROM datasets 
                        WHERE dataset_name = ? OR dataset_name ILIKE ?
                        ORDER BY 
                            CASE WHEN dataset_name = ? THEN 1 ELSE 2 END,
                            dataset_name
                        LIMIT 1
                    """, [clean_identifier, f"%{clean_identifier}%", clean_identifier]).fetchone()
                
                return result
                
        except Exception as e:
            print(f"❌ Error searching for dataset: {e}")
            return None
    
    def _list_datasets(self, args: argparse.Namespace, manager) -> None:
        """List all registered datasets with usage statistics."""
        print("📊 REGISTERED DATASETS")
        # Calculate dynamic separator width
        header_width = 8 + 1 + 30 + 1 + 15 + 1 + 14 + 1 + 8 + 1 + 8  # columns + spaces
        print("=" * header_width)
        
        try:
            with manager._connect() as conn:
                # Build query with optional filters
                where_clauses = []
                params = []
                
                if args.active_only:
                    where_clauses.append("d.is_active = TRUE")
                
                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                # Get datasets with session counts (using new schema columns)
                query = f"""
                    SELECT 
                        d.dataset_id,
                        d.dataset_name,
                        d.target_column,
                        d.train_records,
                        d.train_columns,
                        d.train_format,
                        d.test_records,
                        d.test_columns,
                        d.test_format,
                        d.train_row_count,  -- fallback
                        d.column_count,     -- fallback
                        d.last_used,
                        d.is_active,
                        COUNT(s.session_id) as session_count,
                        MAX(s.best_score) as best_score
                    FROM datasets d
                    LEFT JOIN sessions s ON d.dataset_id = s.dataset_hash
                    {where_clause}
                    GROUP BY d.dataset_id, d.dataset_name, d.target_column,
                             d.train_records, d.train_columns, d.train_format,
                             d.test_records, d.test_columns, d.test_format,
                             d.train_row_count, d.column_count, d.last_used, d.is_active
                    ORDER BY d.last_used DESC NULLS LAST, d.dataset_name
                """
                
                datasets = conn.execute(query, params).fetchall()
                
                if not datasets:
                    print("⚠️  No datasets registered.")
                    print("💡 Register a dataset: python scripts/duckdb_manager.py datasets --register --help")
                    return
                
                if args.format == 'json':
                    # JSON output
                    datasets_json = []
                    for row in datasets:
                        dataset_dict = {
                            'dataset_id': row[0],
                            'dataset_name': row[1],
                            'train_path': row[2],
                            'test_path': row[3],
                            'target_column': row[4],
                            'competition_name': row[5],
                            'train_row_count': row[6],
                            'test_row_count': row[7],
                            'data_size_mb': row[8],
                            'last_used': row[9],
                            'is_active': row[10],
                            'session_count': row[11],
                            'best_score': row[12]
                        }
                        datasets_json.append(dataset_dict)
                    
                    print(json.dumps(datasets_json, indent=2, default=str))
                else:
                    # New improved table output with ID and data info
                    header_line = f"{'ID':<8} {'Name':<30} {'Train Data':<15} {'Test Data':<14} {'Sessions':<8} {'Status':<8}"
                    print(header_line)
                    print("-" * len(header_line))
                    
                    for row in datasets:
                        (dataset_id, name, target_col, train_records, train_columns, train_format,
                         test_records, test_columns, test_format, train_rows_legacy, columns_legacy,
                         last_used, is_active, session_count, best_score) = row
                        
                        # Format components
                        id_short = dataset_id[:8] if dataset_id else "????????"
                        name_short = (name or "Unknown")[:29]
                        
                        # Train data info (use new columns, fallback to legacy)
                        if train_records is not None and train_columns is not None:
                            # Better formatting for record counts
                            if train_records >= 1000000:
                                count_str = f"{train_records//1000000}M"
                            elif train_records >= 1000:
                                count_str = f"{train_records//1000}K"
                            else:
                                count_str = str(train_records)
                            train_info = f"{count_str}×{train_columns}"
                            if train_format:
                                train_info += f" ({train_format[:3]})"
                        elif train_rows_legacy:
                            if train_rows_legacy >= 1000000:
                                count_str = f"{train_rows_legacy//1000000}M"
                            elif train_rows_legacy >= 1000:
                                count_str = f"{train_rows_legacy//1000}K"
                            else:
                                count_str = str(train_rows_legacy)
                            train_info = f"{count_str}×{columns_legacy or '?'}"
                        else:
                            train_info = "No data"
                        train_info = train_info[:14]
                        
                        # Test data info
                        if test_records is not None and test_columns is not None:
                            if test_records >= 1000000:
                                count_str = f"{test_records//1000000}M"
                            elif test_records >= 1000:
                                count_str = f"{test_records//1000}K"
                            else:
                                count_str = str(test_records)
                            test_info = f"{count_str}×{test_columns}"
                            if test_format:
                                test_info += f" ({test_format[:3]})"
                        else:
                            test_info = "N/A"
                        test_info = test_info[:13]
                        
                        # Other fields
                        score_display = f"{best_score:.5f}" if best_score else "N/A"
                        status = "Active" if is_active else "Inactive"
                        
                        print(f"{id_short:<8} {name_short:<30} {train_info:<15} {test_info:<14} {session_count:<8} {status:<8}")
                    
                    print(f"\nTotal datasets: {len(datasets)}")
                    
                    # Summary statistics (adjusted for new column positions)
                    active_count = sum(1 for row in datasets if row[12])  # is_active is now at index 12
                    total_sessions = sum(row[13] for row in datasets)      # session_count is at index 13
                    
                    print(f"Active datasets: {active_count}/{len(datasets)}")
                    print(f"Total sessions: {total_sessions}")
                    
                    # Show tip for more details
                    if datasets:
                        print(f"\n💡 Use --details <ID|NAME> for more information")
                
        except Exception as e:
            print(f"❌ Error listing datasets: {e}")
    
    def _show_dataset_details(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Show detailed dataset information including paths, usage, and statistics."""
        id_type, clean_id = self._resolve_dataset_identifier(dataset_identifier)
        print(f"📊 DATASET DETAILS: {dataset_identifier}")
        print("=" * 70)
        
        try:
            dataset_info = self._find_dataset_by_identifier(dataset_identifier, manager)
            
            if not dataset_info:
                print(f"❌ Dataset not found: {dataset_identifier}")
                print("💡 Use dataset name (e.g., 'MyDataset') or ID (e.g., '516d8e2b')")
                return
            
            # Unpack dataset info (23 columns)
            (dataset_id, name, train_path, test_path, target_col, id_col, 
             competition, description, 
             train_records, train_columns, train_format,
             test_records, test_columns, test_format,
             submission_records, submission_columns, submission_format,
             validation_records, validation_columns, validation_format,
             col_count, train_rows, test_rows,
             created_at, last_used, is_active, size_mb, feature_types) = dataset_info
            
            # Basic Information
            print(f"🏷️  IDENTIFICATION:")
            print(f"   Dataset ID: {dataset_id}")
            print(f"   Name: {name}")
            print(f"   Competition: {competition or 'N/A'}")
            print(f"   Description: {description or 'No description provided'}")
            print(f"   Status: {'🟢 Active' if is_active else '🔴 Inactive'}")
            print()
            
            # ML Configuration
            print(f"🎯 ML CONFIGURATION:")
            print(f"   Target Column: {target_col}")
            print(f"   ID Column: {id_col or 'Not specified'}")
            print()
            
            # Data Information
            print(f"📂 DATA INFORMATION:")
            
            # Train file
            if train_records is not None and train_columns is not None:
                print(f"   📊 Train: {train_records:,} records × {train_columns} columns ({train_format or 'unknown'})")
            else:
                print(f"   📊 Train: {train_rows:,} records × {col_count or '?'} columns (legacy)" if train_rows else "   📊 Train: No metadata")
            print(f"          Path: {train_path}")
            
            # Test file
            if test_records is not None and test_columns is not None:
                print(f"   🧪 Test: {test_records:,} records × {test_columns} columns ({test_format or 'unknown'})")
            elif test_rows:
                print(f"   🧪 Test: {test_rows:,} records (legacy)")
            else:
                print(f"   🧪 Test: Not available")
            if test_path:
                print(f"          Path: {test_path}")
            
            # Submission file
            if submission_records is not None:
                print(f"   📝 Submission: {submission_records:,} records × {submission_columns or '?'} columns ({submission_format or 'unknown'})")
            
            # Validation file
            if validation_records is not None:
                print(f"   ✔️ Validation: {validation_records:,} records × {validation_columns or '?'} columns ({validation_format or 'unknown'})")
            
            print()
            
            # Storage Information
            print(f"💾 STORAGE:")
            if size_mb:
                print(f"   Source Size: {size_mb:.1f} MB")
            
            # Check for DuckDB file
            duckdb_path = Path("data") / "datasets" / name / "dataset.duckdb"
            if duckdb_path.exists():
                duckdb_size = duckdb_path.stat().st_size / (1024 * 1024)
                print(f"   DuckDB File: {duckdb_size:.1f} MB ({duckdb_path})")
                print(f"   Status: 🔄 Imported and ready")
            else:
                print(f"   DuckDB File: Not found ({duckdb_path})")
                print(f"   Status: ⚠️  Legacy dataset - consider re-registering")
            print()
            
            # Usage Statistics
            with manager._connect() as conn:
                session_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                    MIN(best_score) as min_score,
                    MAX(best_score) as max_score,
                    AVG(best_score) as avg_score,
                    MIN(start_time) as first_session,
                    MAX(start_time) as latest_session
                FROM sessions 
                WHERE dataset_hash = ?
                """, [dataset_id]).fetchone()
                
                if session_stats and session_stats[0] > 0:
                    total_sessions, completed, min_score, max_score, avg_score, first_time, latest_time = session_stats
                    
                    print(f"📊 USAGE STATISTICS:")
                    print(f"   Total Sessions: {total_sessions}")
                    print(f"   Completed: {completed} ({100*completed/total_sessions:.1f}%)")
                    if min_score and max_score:
                        print(f"   Score Range: {min_score:.5f} - {max_score:.5f}")
                        print(f"   Average Score: {avg_score:.5f}")
                    print(f"   First Used: {first_time}")
                    print(f"   Last Used: {latest_time}")
                    print()
                    
                    # Show recent sessions
                    recent_sessions = conn.execute("""
                        SELECT session_name, best_score, status, start_time, total_iterations
                        FROM sessions 
                        WHERE dataset_hash = ?
                        ORDER BY start_time DESC
                        LIMIT 5
                    """, [dataset_id]).fetchall()
                    
                    if recent_sessions:
                        print(f"🕒 RECENT SESSIONS:")
                        for sess_name, score, status, start_time, iterations in recent_sessions:
                            score_str = f"{score:.5f}" if score else "No score"
                            print(f"   • {sess_name or 'Unnamed'}: {score_str} ({status}, {iterations} iter) - {start_time}")
                        print()
                else:
                    print(f"📊 USAGE STATISTICS:")
                    print(f"   ⚠️  No usage data - dataset has not been used in any sessions")
                    print()
                
                # Show top features for this dataset
                top_features = conn.execute("""
                    SELECT 
                        fc.feature_name,
                        fc.feature_category,
                        fi.impact_delta,
                        fc.computational_cost
                    FROM feature_catalog fc
                    JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                    JOIN sessions s ON fi.session_id = s.session_id
                    WHERE s.dataset_hash = ?
                    ORDER BY fi.impact_delta DESC
                    LIMIT 5
                """, [dataset_id]).fetchall()
                
                if top_features:
                    print(f"🏆 TOP FEATURES:")
                    for name, category, impact, cost in top_features:
                        print(f"   • {name} ({category}): +{impact:.5f} impact, cost {cost:.1f}")
                    print()
                else:
                    print("⚠️  No feature data found for this dataset.")
                    print()
            
            # Timestamps
            print(f"🕰️ TIMELINE:")
            print(f"   Created: {created_at}")
            print(f"   Last Used: {last_used or 'Never used'}")
            
        except Exception as e:
            print(f"❌ Error showing comprehensive dataset details: {e}")
    
    def _show_dataset_stats(self, args: argparse.Namespace, manager) -> None:
        """Show dataset usage statistics and comparisons."""
        print("📈 DATASET USAGE STATISTICS")
        print("=" * 60)
        
        try:
            with manager._connect() as conn:
                # Overall statistics
                overall_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_datasets,
                        COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_datasets,
                        SUM(train_row_count) as total_rows,
                        SUM(data_size_mb) as total_size,
                        COUNT(CASE WHEN last_used IS NOT NULL THEN 1 END) as used_datasets
                    FROM datasets
                """).fetchone()
                
                if overall_stats:
                    total, active, total_rows, total_size, used = overall_stats
                    print(f"📊 OVERVIEW:")
                    print(f"   Total Datasets: {total}")
                    print(f"   Active: {active}")
                    print(f"   Used in Sessions: {used}")
                    print(f"   Total Rows: {total_rows:,}" if total_rows else "   Total Rows: Unknown")
                    print(f"   Total Size: {total_size:.1f} MB" if total_size else "   Total Size: Unknown")
                    print()
                
                # Usage by dataset
                usage_stats = conn.execute("""
                    SELECT 
                        d.dataset_name,
                        COUNT(s.session_id) as session_count,
                        MAX(s.best_score) as best_score,
                        AVG(s.best_score) as avg_score,
                        COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as completed_count,
                        MAX(s.start_time) as last_used
                    FROM datasets d
                    LEFT JOIN sessions s ON d.dataset_id = s.dataset_hash
                    WHERE d.is_active = TRUE
                    GROUP BY d.dataset_id, d.dataset_name
                    ORDER BY session_count DESC, best_score DESC NULLS LAST
                """).fetchall()
                
                if usage_stats:
                    print(f"📋 USAGE BY DATASET:")
                    print(f"{'Dataset':<20} {'Sessions':<8} {'Completed':<9} {'Best Score':<10} {'Avg Score':<10} {'Last Used':<12}")
                    print("-" * 83)
                    
                    for name, sessions, best_score, avg_score, completed, last_used in usage_stats:
                        name_short = name[:19] if name else "Unknown"
                        best_str = f"{best_score:.5f}" if best_score else "N/A"
                        avg_str = f"{avg_score:.5f}" if avg_score else "N/A"
                        last_used_str = str(last_used).split()[0] if last_used else "Never"
                        
                        print(f"{name_short:<20} {sessions:<8} {completed:<9} {best_str:<10} {avg_str:<10} {last_used_str:<12}")
                    print()
                
                # Competition breakdown
                competition_stats = conn.execute("""
                    SELECT 
                        COALESCE(competition_name, 'Unknown') as competition,
                        COUNT(*) as dataset_count,
                        COUNT(s.session_id) as total_sessions
                    FROM datasets d
                    LEFT JOIN sessions s ON d.dataset_id = s.dataset_hash
                    WHERE d.is_active = TRUE
                    GROUP BY competition_name
                    ORDER BY dataset_count DESC
                """).fetchall()
                
                if competition_stats:
                    print(f"🏆 BY COMPETITION:")
                    for competition, dataset_count, session_count in competition_stats:
                        comp_name = competition or "Unknown"
                        print(f"   • {comp_name}: {dataset_count} dataset(s), {session_count} sessions")
                
        except Exception as e:
            print(f"❌ Error showing dataset statistics: {e}")
    
    def _show_specific_dataset_stats(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Show detailed stats for a specific dataset including new columns."""
        print(f"📊 DATASET STATISTICS: {dataset_identifier}")
        print("="*60)
        
        try:
            with manager._connect() as conn:
                # Find dataset
                dataset_info = conn.execute("""
                    SELECT 
                        dataset_id, dataset_name, train_path, test_path, target_column, id_column,
                        train_records, train_columns, train_format,
                        test_records, test_columns, test_format,
                        submission_records, submission_columns, submission_format,
                        validation_records, validation_columns, validation_format,
                        competition_name, description, created_at, last_used, is_active
                    FROM datasets 
                    WHERE dataset_name ILIKE ? OR dataset_id LIKE ?
                """, [f"%{dataset_identifier}%", f"{dataset_identifier}%"]).fetchone()
                
                if not dataset_info:
                    print(f"❌ Dataset not found: {dataset_identifier}")
                    return
                
                # Unpack dataset info
                (dataset_id, name, train_path, test_path, target_col, id_col,
                 train_records, train_columns, train_format,
                 test_records, test_columns, test_format,
                 submission_records, submission_columns, submission_format,
                 validation_records, validation_columns, validation_format,
                 competition, description, created_at, last_used, is_active) = dataset_info
                
                print(f"📋 DATASET: {name}")
                print(f"   ID: {dataset_id}")
                print(f"   Competition: {competition or 'N/A'}")
                print(f"   Target Column: {target_col}")
                print(f"   ID Column: {id_col or 'N/A'}")
                print(f"   Status: {'Active' if is_active else 'Inactive'}")
                print()
                
                print(f"📁 FILE DETAILS:")
                
                # Train file stats
                if train_records is not None:
                    print(f"   Train: {train_records:,} records, {train_columns} columns ({train_format or 'unknown'} format)")
                    print(f"          {train_path}")
                else:
                    print(f"   Train: No metadata - {train_path}")
                
                # Test file stats
                if test_records is not None:
                    print(f"   Test:  {test_records:,} records, {test_columns} columns ({test_format or 'unknown'} format)")
                    print(f"          {test_path or 'N/A'}")
                elif test_path:
                    print(f"   Test:  No metadata - {test_path}")
                else:
                    print(f"   Test:  Not available")
                
                # Submission file stats  
                if submission_records is not None:
                    print(f"   Submission: {submission_records:,} records, {submission_columns} columns ({submission_format or 'unknown'} format)")
                elif submission_format:
                    print(f"   Submission: Available ({submission_format} format)")
                else:
                    print(f"   Submission: Not available")
                
                # Validation file stats
                if validation_records is not None:
                    print(f"   Validation: {validation_records:,} records, {validation_columns} columns ({validation_format or 'unknown'} format)")
                elif validation_format:
                    print(f"   Validation: Available ({validation_format} format)")
                else:
                    print(f"   Validation: Not available")
                
                print()
                
                # Calculate dataset size estimate
                total_records = (train_records or 0) + (test_records or 0)
                total_columns = max(train_columns or 0, test_columns or 0)
                
                if total_records > 0 and total_columns > 0:
                    # Rough estimate: 8 bytes per numeric value
                    estimated_size_mb = (total_records * total_columns * 8) / (1024 * 1024)
                    print(f"📊 ESTIMATED SIZE: {estimated_size_mb:.1f} MB")
                    print(f"   Total Records: {total_records:,}")
                    print(f"   Max Columns: {total_columns}")
                    print()
                
                # Show DuckDB location
                duckdb_path = Path("data") / "datasets" / name / "dataset.duckdb"
                if duckdb_path.exists():
                    file_size = duckdb_path.stat().st_size / (1024 * 1024)
                    print(f"🖾 DUCKDB STORAGE:")
                    print(f"   Location: {duckdb_path}")
                    print(f"   Size: {file_size:.1f} MB")
                    print()
                
                # Usage statistics
                session_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                        MIN(best_score) as min_score,
                        MAX(best_score) as max_score,
                        AVG(best_score) as avg_score
                    FROM sessions 
                    WHERE dataset_hash = ?
                """, [dataset_id]).fetchone()
                
                if session_stats and session_stats[0] > 0:
                    total_sessions, completed, min_score, max_score, avg_score = session_stats
                    print(f"📊 USAGE STATISTICS:")
                    print(f"   Sessions: {total_sessions} ({completed} completed)")
                    if min_score and max_score:
                        print(f"   Score Range: {min_score:.5f} - {max_score:.5f}")
                        print(f"   Average Score: {avg_score:.5f}")
                    print(f"   Created: {created_at}")
                    print(f"   Last Used: {last_used or 'Never'}")
                else:
                    print(f"⚠️  No usage statistics available (dataset not used in any sessions)")
                
        except Exception as e:
            print(f"❌ Error showing dataset statistics: {e}")
    
    def _show_dataset_sessions(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Show sessions using a specific dataset."""
        print(f"📅 SESSIONS FOR DATASET: {dataset_identifier}")
        print("=" * 70)
        
        try:
            with manager._connect() as conn:
                # Find dataset
                dataset_info = conn.execute("""
                    SELECT dataset_id, dataset_name FROM datasets 
                    WHERE dataset_name ILIKE ? OR dataset_id LIKE ?
                """, [f"%{dataset_identifier}%", f"{dataset_identifier}%"]).fetchone()
                
                if not dataset_info:
                    print(f"❌ Dataset not found: {dataset_identifier}")
                    return
                
                dataset_id, dataset_name = dataset_info
                print(f"Dataset: {dataset_name} ({dataset_id[:8]}...)")
                print()
                
                # Get sessions for this dataset
                sessions = conn.execute("""
                    SELECT 
                        session_name,
                        session_id,
                        start_time,
                        end_time,
                        status,
                        total_iterations,
                        best_score,
                        strategy,
                        is_test_mode
                    FROM sessions 
                    WHERE dataset_hash = ?
                    ORDER BY start_time DESC
                """, [dataset_id]).fetchall()
                
                if not sessions:
                    print("⚠️  No sessions found for this dataset.")
                    return
                
                print(f"{'Session Name':<25} {'Status':<10} {'Score':<10} {'Iter':<5} {'Mode':<6} {'Started':<12}")
                print("-" * 75)
                
                for sess_name, sess_id, start_time, end_time, status, iterations, score, strategy, is_test in sessions:
                    name_short = (sess_name or "Unnamed")[:24]
                    score_str = f"{score:.5f}" if score else "N/A"
                    mode_str = "Test" if is_test else "Prod"
                    start_str = start_time.split()[0] if start_time else "Unknown"
                    
                    print(f"{name_short:<25} {status:<10} {score_str:<10} {iterations:<5} {mode_str:<6} {start_str:<12}")
                
                print(f"\\nTotal sessions: {len(sessions)}")
                
                # Session summary
                completed = sum(1 for s in sessions if s[4] == 'completed')
                test_sessions = sum(1 for s in sessions if s[8])
                best_session = max((s for s in sessions if s[6]), key=lambda x: x[6], default=None)
                
                print(f"Completed: {completed}/{len(sessions)}")
                print(f"Test sessions: {test_sessions}")
                if best_session:
                    print(f"Best session: {best_session[0]} (score: {best_session[6]:.5f})")
                
        except Exception as e:
            print(f"❌ Error showing dataset sessions: {e}")
    
    def _search_datasets(self, query: str, args: argparse.Namespace, manager) -> None:
        """Search datasets by name or path."""
        print(f"🔍 SEARCHING DATASETS: '{query}'")
        print("=" * 50)
        
        try:
            with manager._connect() as conn:
                # Search in name, competition, description, and paths
                results = conn.execute("""
                    SELECT 
                        dataset_name,
                        competition_name,
                        train_path,
                        dataset_id,
                        description,
                        is_active
                    FROM datasets
                    WHERE dataset_name ILIKE ?
                       OR competition_name ILIKE ?
                       OR description ILIKE ?
                       OR train_path ILIKE ?
                       OR test_path ILIKE ?
                    ORDER BY 
                        CASE WHEN dataset_name ILIKE ? THEN 1
                             WHEN competition_name ILIKE ? THEN 2
                             ELSE 3 END,
                        dataset_name
                """, [f"%{query}%"] * 7).fetchall()
                
                if not results:
                    print(f"No datasets found matching '{query}'")
                    return
                
                print(f"Found {len(results)} datasets:")
                print()
                
                for i, (name, competition, train_path, dataset_id, description, is_active) in enumerate(results, 1):
                    status = "Active" if is_active else "Inactive"
                    print(f"{i:2}. {name} [{status}]")
                    if competition:
                        print(f"    Competition: {competition}")
                    if description:
                        desc_preview = description[:80] + "..." if len(description) > 80 else description
                        print(f"    Description: {desc_preview}")
                    print(f"    Train Path: {train_path}")
                    print(f"    ID: {dataset_id[:8]}...")
                    print()
                
        except Exception as e:
            print(f"❌ Error searching datasets: {e}")
    
    def _register_dataset(self, args: argparse.Namespace, manager) -> None:
        """Register a new dataset."""
        print("📝 REGISTER NEW DATASET")
        print("="*50)
        
        # Validate required arguments
        if not args.dataset_name:
            print("❌ --dataset-name is required for registration")
            print("💡 Example: --dataset-name TITANIC")
            return
        
        try:
            # Create dataset-specific logger adapter
            dataset_logger = logging.LoggerAdapter(self.logger, {'dataset_name': args.dataset_name})
            
            # Log the start of registration
            dataset_logger.info(f"Starting dataset registration: {args.dataset_name}")
            
            # Check if dataset already exists
            if not args.force:
                if not self._validate_dataset_name_uniqueness(args.dataset_name, manager):
                    return
            
            # Determine registration mode
            if args.auto or args.dataset_path:
                dataset_logger.info("Using auto-detection mode for registration")
                success = self._register_auto(args, manager, dataset_logger)
            else:
                dataset_logger.info("Using manual mode for registration")
                success = self._register_manual(args, manager, dataset_logger)
            
            if success:
                dataset_logger.info(f"✅ Dataset '{args.dataset_name}' registered successfully")
                print("")
                print("✅ Dataset registration completed successfully!")
                print(f"📊 Use 'datasets --details {args.dataset_name}' to view details")
            else:
                self.logger.error(f"❌ Dataset registration failed for '{args.dataset_name}'")
            
        except Exception as e:
            self.logger.error(f"Registration failed for '{args.dataset_name}': {e}", exc_info=True)
            print(f"❌ Registration failed: {e}")
    
    def _check_dataset_exists(self, dataset_name: str, manager) -> Optional[str]:
        """Check if dataset already exists, return dataset_id if found."""
        try:
            with manager._connect() as conn:
                result = conn.execute(
                    "SELECT dataset_id FROM datasets WHERE dataset_name = ?",
                    [dataset_name]
                ).fetchone()
                return result[0] if result else None
        except Exception as e:
            print(f"⚠️  Warning: Could not check for existing dataset: {e}")
            return None
    
    def _validate_dataset_name_uniqueness(self, dataset_name: str, manager) -> bool:
        """Validate that dataset name is unique."""
        existing_id = self._check_dataset_exists(dataset_name, manager)
        if existing_id:
            print(f"❌ Dataset name '{dataset_name}' already exists (ID: {existing_id[:8]}...)")
            print("💡 Choose a different name or use --force to overwrite")
            print(f"   Examples: {dataset_name}_v2, {dataset_name}_New, {dataset_name}_Updated")
            return False
        return True
    
    def _register_auto(self, args: argparse.Namespace, manager, dataset_logger) -> bool:
        """Auto-registration with file detection."""
        print("🔍 AUTO-DETECTION MODE")
        
        dataset_path = args.dataset_path or '.'
        
        print(f"📂 Scanning directory: {dataset_path}")
        
        try:
            # Initialize importer
            dataset_logger.info(f"Initializing DatasetImporter for: {args.dataset_name}")
            importer = DatasetImporter(args.dataset_name)
            
            # Auto-detect files
            dataset_logger.info(f"Auto-detecting files in: {dataset_path}")
            file_mappings = importer.auto_detect_files(dataset_path)
            
            dataset_logger.info(f"Detected {len(file_mappings)} files: {list(file_mappings.keys())}")
            print("📋 Detected files:")
            for table_name, file_path in file_mappings.items():
                print(f"   • {table_name}: {file_path}")
                dataset_logger.debug(f"File mapping: {table_name} -> {file_path}")
            
            # Extract target column
            target_column = args.target_column
            if not target_column:
                target_column = importer.detect_target_column(file_mappings['train'])
                if target_column:
                    print(f"🎯 Auto-detected target column: {target_column}")
                else:
                    print("❌ Could not auto-detect target column")
                    print("💡 Please specify --target-column manually")
                    return False
            
            # Auto-detect ID column if not specified
            id_column = args.id_column
            if not id_column and 'train' in file_mappings:
                id_column = importer.detect_id_column(file_mappings['train'])
                if id_column:
                    print(f"🆔 Auto-detected ID column: {id_column}")
            
            # Analyze files for metadata
            print("")
            print("🔍 Analyzing dataset files...")
            metadata = {}
            for table_name, file_path in file_mappings.items():
                try:
                    file_metadata = importer.analyze_file(file_path)
                    metadata[table_name] = file_metadata
                    print(f"   ✅ {table_name}: {file_metadata['records']:,} rows, {file_metadata['columns']} columns")
                except Exception as e:
                    print(f"   ⚠️ {table_name}: Analysis failed - {e}")
                    metadata[table_name] = {'error': str(e)}
            
            # Import to DuckDB
            print("")
            print("📥 Importing data to DuckDB...")
            duckdb_path = importer.create_duckdb_dataset(file_mappings, target_column, id_column)
            
            # Generate dataset ID
            dataset_id = importer.generate_dataset_id(file_mappings)
            
            # Register in database
            registration_data = {
                'dataset_id': dataset_id,
                'dataset_name': args.dataset_name,
                'train_path': file_mappings.get('train', ''),
                'test_path': file_mappings.get('test', ''),
                'submission_path': file_mappings.get('submission', ''),
                'validation_path': file_mappings.get('validation', ''),
                'target_column': target_column,
                'id_column': id_column,
                'competition_name': args.competition_name or '',
                'description': args.description or '',
                'duckdb_path': duckdb_path,
                'metadata': metadata
            }
            
            self._insert_dataset_record(registration_data, manager, dataset_logger)
            
            return True
            
        except Exception as e:
            print(f"❌ Auto-detection failed: {e}")
            print("💡 Try manual registration with specific file paths")
            return False
    
    def _register_manual(self, args: argparse.Namespace, manager, dataset_logger) -> bool:
        """Manual registration with specified file paths."""
        print("✋ MANUAL REGISTRATION MODE")
        
        # Validate required files
        if not args.train:
            print("❌ --train is required for manual registration")
            return False
        
        if not args.target_column:
            print("❌ --target-column is required for manual registration")
            return False
        
        # Check file existence
        file_mappings = {}
        for table_name, file_arg in [('train', args.train), ('test', args.test), 
                                    ('submission', args.submission), ('validation', args.validation)]:
            if file_arg:
                if not Path(file_arg).exists():
                    print(f"❌ File not found: {file_arg}")
                    return False
                file_mappings[table_name] = file_arg
        
        print("📋 Files to import:")
        for table_name, file_path in file_mappings.items():
            print(f"   • {table_name}: {file_path}")
        
        try:
            # Initialize importer
            importer = DatasetImporter(args.dataset_name)
            
            # Analyze files for metadata
            print("")
            print("🔍 Analyzing dataset files...")
            metadata = {}
            for table_name, file_path in file_mappings.items():
                try:
                    file_metadata = importer.analyze_file(file_path)
                    metadata[table_name] = file_metadata
                    print(f"   ✅ {table_name}: {file_metadata['records']:,} rows, {file_metadata['columns']} columns")
                except Exception as e:
                    print(f"   ⚠️ {table_name}: Analysis failed - {e}")
                    metadata[table_name] = {'error': str(e)}
            
            # Import to DuckDB
            print("")
            print("📥 Importing data to DuckDB...")
            duckdb_path = importer.create_duckdb_dataset(file_mappings, args.target_column, args.id_column)
            
            # Generate dataset ID
            dataset_id = importer.generate_dataset_id(file_mappings)
            
            # Register in database
            registration_data = {
                'dataset_id': dataset_id,
                'dataset_name': args.dataset_name,
                'train_path': file_mappings.get('train', ''),
                'test_path': file_mappings.get('test', ''),
                'submission_path': file_mappings.get('submission', ''),
                'validation_path': file_mappings.get('validation', ''),
                'target_column': args.target_column,
                'id_column': args.id_column,
                'competition_name': args.competition_name or '',
                'description': args.description or '',
                'duckdb_path': duckdb_path,
                'metadata': metadata
            }
            
            self._insert_dataset_record(registration_data, manager, dataset_logger)
            
            return True
            
        except Exception as e:
            print(f"❌ Manual registration failed: {e}")
            return False
    
    def _calculate_dataset_hash(self, train_path: str, test_path: str = '') -> str:
        """Calculate dataset hash from paths (same as discovery_db.py)."""
        path_string = f"{train_path}|{test_path or ''}"
        return hashlib.md5(path_string.encode()).hexdigest()
    
    def _insert_dataset_record(self, dataset_info: Dict[str, Any], manager, dataset_logger) -> None:
        """Insert dataset record into database using proper repository pattern."""
        metadata = dataset_info['metadata']
        
        # Extract metadata for each table type
        train_meta = metadata.get('train', {})
        test_meta = metadata.get('test', {})
        submission_meta = metadata.get('submission', {})
        validation_meta = metadata.get('validation', {})
        
        try:
            # Use the proper repository pattern instead of manual SQL
            from datetime import datetime
            
            # Import the repository and models
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            from db import DuckDBConnectionManager, DatasetRepository
            from db.models.dataset import Dataset
            
            # Initialize connection manager and repository
            config = {'database': {'db_path': manager.duckdb_path}}
            conn_manager = DuckDBConnectionManager(config)
            dataset_repo = DatasetRepository(conn_manager)
            
            # Create Dataset object
            dataset = Dataset(
                dataset_id=dataset_info['dataset_id'],
                dataset_name=dataset_info['dataset_name'],
                train_path=dataset_info['train_path'],
                test_path=dataset_info.get('test_path'),
                submission_path=dataset_info.get('submission_path'),
                validation_path=dataset_info.get('validation_path'),
                target_column=dataset_info['target_column'],
                id_column=dataset_info.get('id_column'),
                competition_name=dataset_info.get('competition_name', ''),
                description=dataset_info.get('description', ''),
                train_records=train_meta.get('records'),
                train_columns=train_meta.get('columns'),
                train_format=train_meta.get('file_format'),
                test_records=test_meta.get('records'),
                test_columns=test_meta.get('columns'),
                test_format=test_meta.get('file_format'),
                submission_records=submission_meta.get('records'),
                submission_columns=submission_meta.get('columns'),
                submission_format=submission_meta.get('file_format'),
                validation_records=validation_meta.get('records'),
                validation_columns=validation_meta.get('columns'),
                validation_format=validation_meta.get('file_format'),
                column_count=train_meta.get('columns'),  # Legacy column_count
                train_row_count=train_meta.get('records'),  # Legacy train_row_count
                test_row_count=test_meta.get('records'),   # Legacy test_row_count
                data_size_mb=None,
                feature_types=None,
                created_at=datetime.now(),
                last_used=None,
                is_active=True
            )
            
            # Save using repository (this will use proper conflict resolution)
            saved_dataset = dataset_repo.save(dataset, update_on_conflict=True)
            
            print(f"✅ Dataset registered with ID: {dataset_info['dataset_id'][:8]}...")
            
        except Exception as e:
            # No fallback - repository approach should work
            dataset_logger.error(f"Failed to save dataset using repository: {e}")
            raise Exception(f"Dataset registration failed: {e}")
    
    def _update_dataset(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Update dataset metadata."""
        print(f"✏️  UPDATE DATASET: {dataset_identifier}")
        print("=" * 40)
        print("Dataset metadata update not implemented yet.")
        print("Metadata is automatically updated by the migration script.")
    
    def _cleanup_dataset(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Safely remove dataset and all related data with backup."""
        print(f"🗑️  DATASET CLEANUP: {dataset_identifier}")
        print("=" * 50)
        
        try:
            # Find dataset
            dataset_info = self._find_dataset_by_identifier(dataset_identifier, manager)
            if not dataset_info:
                print(f"❌ Dataset not found: {dataset_identifier}")
                return
            
            # Unpack dataset info (23 columns)
            (dataset_id, name, train_path, test_path, target_col, id_col, 
             competition, description, 
             train_records, train_columns, train_format,
             test_records, test_columns, test_format,
             submission_records, submission_columns, submission_format,
             validation_records, validation_columns, validation_format,
             col_count, train_rows, test_rows,
             created_at, last_used, is_active, size_mb, feature_types) = dataset_info
            
            print(f"📋 Dataset to remove:")
            print(f"   Name: {name}")
            print(f"   ID: {dataset_id[:8]}...")
            print(f"   Competition: {competition or 'N/A'}")
            print(f"   Created: {created_at}")
            print()
            
            # Count related sessions
            with manager._connect() as conn:
                session_count = conn.execute("""
                    SELECT COUNT(*) FROM sessions WHERE dataset_hash = ?
                """, [dataset_id]).fetchone()[0]
                
                # Count related features  
                feature_count = conn.execute("""
                    SELECT COUNT(DISTINCT fc.feature_name)
                    FROM feature_catalog fc
                    JOIN feature_impact fi ON fc.feature_name = fi.feature_name
                    JOIN sessions s ON fi.session_id = s.session_id
                    WHERE s.dataset_hash = ?
                """, [dataset_id]).fetchone()[0]
                
                # Count exploration history
                exploration_count = conn.execute("""
                    SELECT COUNT(*)
                    FROM exploration_history eh
                    JOIN sessions s ON eh.session_id = s.session_id
                    WHERE s.dataset_hash = ?
                """, [dataset_id]).fetchone()[0]
            
            print(f"📊 Related data to remove:")
            print(f"   Sessions: {session_count}")
            print(f"   Features discovered: {feature_count}")
            print(f"   Exploration records: {exploration_count}")
            
            # Check dataset files directory
            dataset_dir = Path(f"cache/{name}")
            duckdb_file = dataset_dir / "dataset.duckdb"
            files_to_remove = []
            
            if dataset_dir.exists():
                files_to_remove = list(dataset_dir.glob("*"))
                print(f"   Dataset files: {len(files_to_remove)} files in {dataset_dir}")
                for file in files_to_remove[:3]:  # Show first 3 files
                    file_size = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
                    print(f"     • {file.name} ({file_size:.1f} MB)")
                if len(files_to_remove) > 3:
                    print(f"     • ... and {len(files_to_remove) - 3} more")
            else:
                print(f"   Dataset files: No directory found ({dataset_dir})")
            
            print()
            
            # If no sessions, allow cleanup without confirmation
            if session_count == 0:
                print("✅ No sessions found - proceeding with cleanup without confirmation")
                proceed = True
            else:
                # Ask for confirmation
                print("⚠️  WARNING: This will permanently delete:")
                print(f"   • Dataset '{name}' from database")
                print(f"   • {session_count} session(s) and their results")
                print(f"   • {feature_count} discovered feature(s)")
                print(f"   • {exploration_count} exploration record(s)")
                print(f"   • All files in {dataset_dir}")
                print()
                
                response = input("❓ Continue with cleanup? (yes/no): ")
                proceed = response.lower() in ['yes', 'y']
                
                if not proceed:
                    print("❌ Cleanup cancelled")
                    return
            
            # Create backup before cleanup
            print(f"\n💾 Creating backup before cleanup...")
            backup_success = self._create_cleanup_backup(manager)
            
            if not backup_success:
                print("❌ Backup failed - cleanup cancelled for safety")
                return
            
            # Perform cleanup
            print(f"\n🗑️  Performing cleanup...")
            
            with manager._connect() as conn:
                cursor = conn.cursor()
                
                # Delete in correct order (foreign key dependencies)
                
                # 1. Delete exploration history for sessions
                cursor.execute("""
                    DELETE FROM exploration_history 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions WHERE dataset_hash = ?
                    )
                """, [dataset_id])
                exploration_deleted = cursor.rowcount
                print(f"   ✅ Deleted {exploration_deleted} exploration records")
                
                # 2. Delete feature impact for sessions
                cursor.execute("""
                    DELETE FROM feature_impact 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions WHERE dataset_hash = ?
                    )
                """, [dataset_id])
                impact_deleted = cursor.rowcount
                print(f"   ✅ Deleted {impact_deleted} feature impact records")
                
                # 3. Delete sessions
                cursor.execute("DELETE FROM sessions WHERE dataset_hash = ?", [dataset_id])
                sessions_deleted = cursor.rowcount
                print(f"   ✅ Deleted {sessions_deleted} sessions")
                
                # 4. Delete dataset
                cursor.execute("DELETE FROM datasets WHERE dataset_id = ?", [dataset_id])
                dataset_deleted = cursor.rowcount
                print(f"   ✅ Deleted dataset record")
                
            
            # Remove dataset cache directory (cache/NAZWA)
            # Note: This removes the CACHE in cache/, NOT the source files
            if dataset_dir.exists():
                try:
                    shutil.rmtree(dataset_dir)
                    print(f"   ✅ Deleted dataset cache directory: {dataset_dir}")
                except Exception as e:
                    print(f"   ⚠️ Failed to delete cache directory {dataset_dir}: {e}")
            else:
                print(f"   ℹ️ No cache directory found: {dataset_dir}")
            
            print(f"\n✅ Dataset cleanup completed successfully!")
            print(f"   Removed dataset '{name}' and all {session_count} related sessions")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def _create_cleanup_backup(self, manager) -> bool:
        """Create backup before cleanup with verification."""
        try:
            from .backup import BackupModule
            from datetime import datetime
            
            # Get current backup count for verification
            backup_config = manager.get_backup_config()
            backup_dir = manager.project_root / backup_config['backup_path']
            backup_prefix = backup_config['backup_prefix']
            
            # Count existing backups before
            existing_backups = list(backup_dir.glob(f"{backup_prefix}*.duckdb*"))
            backup_count_before = len(existing_backups)
            
            # Create backup module instance
            backup_module = BackupModule()
            
            # Create mock args for backup
            class MockArgs:
                compress = False
            
            backup_args = MockArgs()
            
            # Create backup
            backup_module._create_backup(backup_args, manager)
            
            # Verify backup was created - check if we have one more backup
            new_backups = list(backup_dir.glob(f"{backup_prefix}*.duckdb*"))
            backup_count_after = len(new_backups)
            
            if backup_count_after > backup_count_before:
                # Find the newest backup
                latest_backup = max(new_backups, key=lambda f: f.stat().st_mtime)
                print(f"   ✅ Backup created: {latest_backup}")
                
                # Quick verification - check if backup is readable
                try:
                    import duckdb
                    test_conn = duckdb.connect(str(latest_backup))
                    tables = test_conn.execute("SHOW TABLES").fetchall()
                    test_conn.close()
                    print(f"   ✅ Backup verified: {len(tables)} tables found")
                    return True
                except Exception as e:
                    print(f"   ❌ Backup verification failed: {e}")
                    return False
            else:
                print(f"   ❌ No new backup detected (before: {backup_count_before}, after: {backup_count_after})")
                return False
                
        except Exception as e:
            print(f"   ❌ Backup creation failed: {e}")
            return False
    
    def _update_features(self, dataset_identifier: str, args: argparse.Namespace, manager) -> None:
        """Update features for a dataset if new features are available."""
        print(f"🔍 CHECKING FEATURES FOR DATASET: {dataset_identifier}")
        print("="*50)
        
        # Find dataset
        dataset_info = self._find_dataset_by_identifier(dataset_identifier, manager)
        if not dataset_info:
            print(f"❌ Dataset '{dataset_identifier}' not found")
            return
        
        dataset_id = dataset_info[0]
        dataset_name = dataset_info[1]
        
        # Get cached dataset path
        cache_dir = Path('cache') / dataset_name
        dataset_db = cache_dir / 'dataset.duckdb'
        
        if not dataset_db.exists():
            print(f"❌ Dataset cache not found: {dataset_db}")
            return
        
        try:
            # Initialize DuckDB connection
            import duckdb
            conn = duckdb.connect(str(dataset_db))
            
            # Check what tables exist
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            
            print(f"📊 Current tables in database: {', '.join(table_names)}")
            
            # Check if train_features exists
            has_train_features = 'train_features' in table_names
            has_test_features = 'test_features' in table_names
            
            if not has_train_features and not has_test_features:
                print("✨ No feature tables found - features will be generated on first registration")
                if not args.force_update:
                    print("💡 Use --force-update to generate features now")
                    conn.close()
                    return
            
            # Get current feature columns
            current_features = set()
            if has_train_features:
                train_cols = conn.execute("PRAGMA table_info(train_features)").fetchall()
                current_features.update([col[1] for col in train_cols])
            
            print(f"📊 Current features in DB: {len(current_features)}")
            
            # Initialize FeatureSpace to check available features
            from src.feature_space import FeatureSpace
            from src.dataset_importer import DatasetImporter
            
            # Create minimal config
            config = {
                'feature_space': {
                    'custom_domain_module': None,
                    'max_features_per_node': 1000,
                    'enable_caching': False,
                    'enabled_categories': ['feature_transformations', 'feature_selection'],
                    'category_weights': {
                        'feature_transformations': 0.4,
                        'feature_selection': 0.1
                    },
                    'lazy_loading': False,
                    'cache_features': False,
                    'max_cache_size_mb': 100,
                    'min_improvement_threshold': 0.001,
                    'feature_timeout': 60,
                    'generic_operations': {
                        'statistical_aggregations': True,
                        'polynomial_features': True,
                        'binning_features': True,
                        'ranking_features': True
                    }
                },
                'autogluon': {
                    'dataset_name': dataset_name
                }
            }
            
            # Determine custom domain module
            importer = DatasetImporter(dataset_name)
            domain_mapping = {
                's5e6': 'domains.fertilizer',
                'fertilizer': 'domains.fertilizer',
                'titanic': 'domains.titanic',
            }
            
            for key, domain in domain_mapping.items():
                if key in dataset_name.lower():
                    config['feature_space']['custom_domain_module'] = domain
                    break
            
            feature_space = FeatureSpace(config)
            
            # Get sample data to determine possible features
            if 'train' in table_names:
                sample_df = conn.execute("SELECT * FROM train LIMIT 100").df()
                
                # Get all possible feature names
                # This is approximate - real features depend on data
                possible_features = set()
                import numpy as np
                numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Add generic feature patterns
                for col in numeric_cols:
                    possible_features.update([
                        f"{col}_squared", f"{col}_log", f"{col}_sqrt",
                        f"{col}_rank", f"{col}_rank_pct", f"{col}_bin_5"
                    ])
                
                for cat_col in categorical_cols[:5]:
                    for num_col in numeric_cols[:10]:
                        possible_features.update([
                            f"{num_col}_mean_by_{cat_col}",
                            f"{num_col}_std_by_{cat_col}",
                            f"{num_col}_dev_from_{cat_col}_mean"
                        ])
                
                # Check for new features
                new_features = possible_features - current_features
                missing_features = len(new_features)
                
                print(f"🧩 Possible features in code: ~{len(possible_features)}")
                print(f"✨ New features detected: ~{missing_features}")
                
                if missing_features > 0 and not args.dry_run:
                    print("\nSome example new features:")
                    for feat in list(new_features)[:5]:
                        print(f"   - {feat}")
                    if len(new_features) > 5:
                        print(f"   ... and {len(new_features) - 5} more")
            
            # Decide whether to update
            if args.force_update or (missing_features > 0 and not args.dry_run):
                print("\n🔄 Regenerating all features...")
                
                # Load data
                train_df = conn.execute("SELECT * FROM train").df() if 'train' in table_names else None
                test_df = conn.execute("SELECT * FROM test").df() if 'test' in table_names else None
                
                # Get target column from dataset info
                target_column = dataset_info[4]  # target_column from database
                
                # Generate all features
                if train_df is not None:
                    print("   Generating features for train data...")
                    train_features_df = feature_space.generate_all_features(train_df, dataset_name)
                    
                    # Save to database
                    conn.execute("DROP TABLE IF EXISTS train_features")
                    conn.register('train_features_df', train_features_df)
                    conn.execute("CREATE TABLE train_features AS SELECT * FROM train_features_df")
                    conn.unregister('train_features_df')
                    
                    feature_count = len(train_features_df.columns)
                    print(f"   ✅ Created train_features with {feature_count} columns")
                
                if test_df is not None:
                    print("   Generating features for test data...")
                    # Remove target column if it exists in test
                    if target_column in test_df.columns:
                        test_df = test_df.drop(columns=[target_column])
                    
                    test_features_df = feature_space.generate_all_features(test_df, dataset_name)
                    
                    # Save to database
                    conn.execute("DROP TABLE IF EXISTS test_features")
                    conn.register('test_features_df', test_features_df)
                    conn.execute("CREATE TABLE test_features AS SELECT * FROM test_features_df")
                    conn.unregister('test_features_df')
                    
                    feature_count = len(test_features_df.columns)
                    print(f"   ✅ Created test_features with {feature_count} columns")
                
                print("\n✅ Feature update completed successfully!")
                
                # Update last_used timestamp
                with manager._connect() as db_conn:
                    db_conn.execute(
                        "UPDATE datasets SET last_used = CURRENT_TIMESTAMP WHERE dataset_id = ?",
                        [dataset_id]
                    )
            
            elif args.dry_run:
                print("\n🔍 DRY RUN - No changes made")
                print(f"Would regenerate features adding ~{missing_features} new features")
            
            else:
                print("\n✅ Features are up to date!")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Feature update failed: {e}")
            import traceback
            traceback.print_exc()