#!/usr/bin/env python3
"""
Datasets Migration Script

Creates and populates the datasets table from existing session configurations.
Automatically detects dataset types and assigns human-readable names.

Usage:
    python scripts/datasets_migration.py [--dry-run] [--db-path PATH]
"""

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    import duckdb
    DUCKDB_AVAILABLE = False
    print("Warning: DuckDB not available, falling back to SQLite")

def calculate_dataset_hash(train_path: str, test_path: str) -> str:
    """Calculate dataset hash from train/test paths."""
    train_path = train_path or ''
    test_path = test_path or ''
    
    # Remove quotes if present
    train_path = train_path.strip('"')
    test_path = test_path.strip('"')
    
    # Create hash from train and test paths
    path_string = f"{train_path}|{test_path}"
    path_hash = hashlib.md5(path_string.encode()).hexdigest()
    
    return path_hash

def detect_dataset_info(train_path: str, test_path: str, target_metric: str) -> Dict[str, any]:
    """Automatically detect dataset information based on paths and metrics."""
    
    # Clean paths
    train_clean = (train_path or '').strip('"').lower()
    test_clean = (test_path or '').strip('"').lower()
    metric_clean = (target_metric or '').strip('"').lower()
    
    # Default values
    dataset_info = {
        'dataset_name': 'Unknown Dataset',
        'competition_name': None,
        'description': 'Automatically detected dataset',
        'target_column': 'target',
        'target_metrics': [target_metric] if target_metric else ['unknown']
    }
    
    # Titanic detection
    if 'titanic' in train_clean:
        dataset_info.update({
            'dataset_name': 'Titanic',
            'competition_name': 'Titanic - Machine Learning from Disaster',
            'description': 'Predict survival on the Titanic based on passenger information',
            'target_column': 'Survived',
            'target_metrics': ['accuracy', 'roc_auc'] if metric_clean in ['accuracy', 'roc_auc'] else [target_metric]
        })
    
    # Fertilizer S5E6 detection
    elif 'playground-series-s5e6' in train_clean or 'fertilizer' in train_clean:
        dataset_info.update({
            'dataset_name': 'Fertilizer S5E6',
            'competition_name': 'Playground Series S5E6 - Predicting Optimal Fertilizers',
            'description': 'Predict optimal fertilizer recommendations for agricultural yields',
            'target_column': 'yield',
            'target_metrics': ['MAP@3'] if metric_clean == 'map@3' else [target_metric]
        })
    
    # Boston Housing detection
    elif 'boston' in train_clean or 'housing' in train_clean:
        dataset_info.update({
            'dataset_name': 'Boston Housing',
            'competition_name': 'Boston Housing Price Prediction',
            'description': 'Predict housing prices in Boston area',
            'target_column': 'price',
            'target_metrics': ['rmse', 'mae'] if metric_clean in ['rmse', 'mae'] else [target_metric]
        })
    
    # Generic competition detection
    elif 'competition' in train_clean or 'kaggle' in train_clean:
        # Extract competition name from path if possible
        path_parts = train_clean.split('/')
        comp_part = None
        for part in path_parts:
            if 'competition' in part or 'kaggle' in part:
                comp_part = part.replace('-', ' ').replace('_', ' ').title()
                break
        
        dataset_info.update({
            'dataset_name': comp_part or 'Kaggle Competition',
            'competition_name': comp_part,
            'description': f'Kaggle competition dataset: {comp_part or "Unknown"}',
            'target_metrics': [target_metric] if target_metric else ['unknown']
        })
    
    return dataset_info

def get_file_stats(file_path: str) -> Tuple[Optional[int], Optional[float]]:
    """Get basic file statistics (row count and size)."""
    try:
        if not file_path or not os.path.exists(file_path):
            return None, None
        
        # Get file size in MB
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Try to count rows using basic file operations
        row_count = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            # If we can't read the file, estimate based on size
            # Rough estimate: 100 bytes per row average
            row_count = int(size_mb * 1024 * 1024 / 100) if size_mb > 0 else None
        
        return row_count, size_mb
        
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Migrate sessions to datasets table')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--db-path', default='data/minotaur.duckdb',
                       help='Path to database file')
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    print(f"🗄️  Processing database: {db_path}")
    
    try:
        # Connect to database
        if DUCKDB_AVAILABLE:
            conn = duckdb.connect(str(db_path))
            db_type = 'DuckDB'
        else:
            conn = duckdb.connect(str(db_path))
            db_type = 'SQLite'
        
        print(f"✅ Connected to {db_type}")
        
        # Check if datasets table exists
        if DUCKDB_AVAILABLE:
            tables_info = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables_info]
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [table[0] for table in cursor.fetchall()]
        
        has_datasets_table = 'datasets' in table_names
        print(f"📋 Datasets table exists: {has_datasets_table}")
        
        if not has_datasets_table:
            print("❌ Datasets table not found. Please run MCTS to create the schema first.")
            return 1
        
        # Get all sessions with config snapshots
        print("🔍 Analyzing sessions for dataset information...")
        
        sessions_query = """
            SELECT session_id, session_name,
                   json_extract(config_snapshot, '$.autogluon.train_path') as train_path,
                   json_extract(config_snapshot, '$.autogluon.test_path') as test_path,
                   json_extract(config_snapshot, '$.autogluon.target_metric') as metric,
                   start_time,
                   dataset_hash
            FROM sessions 
            WHERE config_snapshot IS NOT NULL
            ORDER BY start_time DESC
        """
        
        sessions = conn.execute(sessions_query).fetchall()
        print(f"📊 Found {len(sessions)} sessions with config snapshots")
        
        # Group sessions by dataset hash
        datasets_info = {}
        session_updates = []
        
        for session_id, session_name, train_path, test_path, metric, start_time, existing_hash in sessions:
            # Clean paths and calculate hash
            train_clean = (train_path or '').strip('"') if train_path else ''
            test_clean = (test_path or '').strip('"') if test_path else ''
            metric_clean = (metric or '').strip('"') if metric else 'unknown'
            
            dataset_hash = calculate_dataset_hash(train_clean, test_clean)
            
            # Update session if hash is missing or different
            if existing_hash != dataset_hash:
                session_updates.append((dataset_hash, session_id))
            
            # Collect dataset information
            if dataset_hash not in datasets_info:
                # Detect dataset information
                detected_info = detect_dataset_info(train_clean, test_clean, metric_clean)
                
                # Get file statistics
                train_rows, train_size = get_file_stats(train_clean)
                test_rows, test_size = get_file_stats(test_clean)
                
                datasets_info[dataset_hash] = {
                    'dataset_id': dataset_hash,
                    'dataset_name': detected_info['dataset_name'],
                    'train_path': train_clean,
                    'test_path': test_clean,
                    'target_column': detected_info['target_column'],
                    'target_metrics': json.dumps(detected_info['target_metrics']),
                    'competition_name': detected_info['competition_name'],
                    'description': detected_info['description'],
                    'train_row_count': train_rows,
                    'test_row_count': test_rows,
                    'data_size_mb': (train_size or 0) + (test_size or 0),
                    'sessions': [],
                    'latest_session': start_time
                }
            
            # Add session to dataset
            datasets_info[dataset_hash]['sessions'].append((session_id, session_name, start_time))
            
            # Update latest session time
            if start_time and (not datasets_info[dataset_hash]['latest_session'] or 
                              start_time > datasets_info[dataset_hash]['latest_session']):
                datasets_info[dataset_hash]['latest_session'] = start_time
        
        # Show dataset summary
        print("\\n📈 DETECTED DATASETS:")
        print("=" * 80)
        
        for dataset_hash, data in datasets_info.items():
            print(f"\\n🎯 Dataset: {data['dataset_name']} ({dataset_hash[:8]}...)")
            print(f"   Train: {data['train_path'] or 'N/A'}")
            print(f"   Test:  {data['test_path'] or 'N/A'}")
            print(f"   Target: {data['target_column']} ({json.loads(data['target_metrics'])})")
            print(f"   Competition: {data['competition_name'] or 'N/A'}")
            print(f"   Rows: {data['train_row_count'] or 'Unknown'} train, {data['test_row_count'] or 'Unknown'} test")
            print(f"   Size: {data['data_size_mb']:.1f} MB" if data['data_size_mb'] else "   Size: Unknown")
            print(f"   Sessions: {len(data['sessions'])}")
            
            # Show recent sessions
            recent_sessions = sorted(data['sessions'], key=lambda x: x[2] or '', reverse=True)[:3]
            for session_id, session_name, session_time in recent_sessions:
                print(f"     • {session_name or 'Unnamed'} ({session_id[:8]}...)")
            
            if len(data['sessions']) > 3:
                print(f"     ... and {len(data['sessions']) - 3} more")
        
        # Perform migration
        if args.dry_run:
            print(f"\\n🧪 DRY RUN: Would create {len(datasets_info)} datasets and update {len(session_updates)} sessions")
        else:
            print(f"\\n💾 Creating {len(datasets_info)} datasets...")
            
            # Insert datasets
            for dataset_hash, data in datasets_info.items():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO datasets (
                            dataset_id, dataset_name, train_path, test_path,
                            target_column, target_metrics, competition_name, description,
                            train_row_count, test_row_count, data_size_mb, last_used, is_active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        data['dataset_id'],
                        data['dataset_name'],
                        data['train_path'],
                        data['test_path'],
                        data['target_column'],
                        data['target_metrics'],
                        data['competition_name'],
                        data['description'],
                        data['train_row_count'],
                        data['test_row_count'],
                        data['data_size_mb'],
                        data['latest_session'],
                        True
                    ])
                    print(f"   ✅ Created dataset: {data['dataset_name']}")
                except Exception as e:
                    print(f"   ❌ Failed to create {data['dataset_name']}: {e}")
            
            # Update sessions with correct dataset hash
            if session_updates:
                print(f"\\n💾 Updating {len(session_updates)} sessions with dataset hash...")
                for dataset_hash, session_id in session_updates:
                    conn.execute(
                        "UPDATE sessions SET dataset_hash = ? WHERE session_id = ?",
                        [dataset_hash, session_id]
                    )
                print("   ✅ Sessions updated successfully")
            
            print("\\n✅ Migration completed successfully!")
        
        conn.close()
        
        # Show usage examples
        if not args.dry_run:
            print("\\n💡 Usage examples:")
            print("# List all datasets:")
            print("python scripts/duckdb_manager.py datasets --list")
            print("\\n# Show specific dataset:")
            if datasets_info:
                first_dataset = list(datasets_info.keys())[0]
                first_name = datasets_info[first_dataset]['dataset_name']
                print(f"python scripts/duckdb_manager.py datasets --show {first_name}")
            print("\\n# Features for specific dataset:")
            if datasets_info:
                first_dataset = list(datasets_info.keys())[0]
                print(f"python scripts/duckdb_manager.py features --dataset {first_dataset[:8]}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())