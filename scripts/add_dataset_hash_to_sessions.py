#!/usr/bin/env python3
"""
Add Dataset Hash to Sessions Table

Calculates dataset hash from train/test paths for all existing sessions
and adds dataset_hash column to sessions table.

Usage:
    python scripts/add_dataset_hash_to_sessions.py [--dry-run]
"""

import argparse
import hashlib
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    import sqlite3
    DUCKDB_AVAILABLE = False
    print("Warning: DuckDB not available, falling back to SQLite")

def calculate_dataset_hash(train_path: str, test_path: str) -> str:
    """Calculate dataset hash from train/test paths (same as DuckDBDataManager)."""
    # Handle null values
    train_path = train_path or ''
    test_path = test_path or ''
    
    # Remove quotes if present
    train_path = train_path.strip('"')
    test_path = test_path.strip('"')
    
    # Create hash from train and test paths
    path_string = f"{train_path}|{test_path}"
    path_hash = hashlib.md5(path_string.encode()).hexdigest()
    
    return path_hash

def main():
    parser = argparse.ArgumentParser(description='Add dataset hash to sessions table')
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
            conn = sqlite3.connect(str(db_path))
            db_type = 'SQLite'
        
        print(f"✅ Connected to {db_type}")
        
        # Check if dataset_hash column exists
        if DUCKDB_AVAILABLE:
            columns_info = conn.execute("DESCRIBE sessions").fetchall()
            columns = [col[0] for col in columns_info]
        else:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col[1] for col in cursor.fetchall()]
        
        has_dataset_hash = 'dataset_hash' in columns
        print(f"📋 Dataset hash column exists: {has_dataset_hash}")
        
        if not has_dataset_hash and not args.dry_run:
            print("➕ Adding dataset_hash column to sessions table...")
            conn.execute("ALTER TABLE sessions ADD COLUMN dataset_hash VARCHAR")
            print("✅ Column added successfully")
        
        # Get all sessions with config snapshots
        print("🔍 Analyzing sessions...")
        
        sessions_query = """
            SELECT session_id, session_name,
                   json_extract(config_snapshot, '$.autogluon.train_path') as train_path,
                   json_extract(config_snapshot, '$.autogluon.test_path') as test_path,
                   json_extract(config_snapshot, '$.autogluon.target_metric') as metric
            FROM sessions 
            WHERE config_snapshot IS NOT NULL
            ORDER BY start_time DESC
        """
        
        sessions = conn.execute(sessions_query).fetchall()
        print(f"📊 Found {len(sessions)} sessions with config snapshots")
        
        # Group by dataset hash
        datasets = {}
        updates = []
        
        for session_id, session_name, train_path, test_path, metric in sessions:
            # Clean paths
            train_clean = (train_path or '').strip('"') if train_path else ''
            test_clean = (test_path or '').strip('"') if test_path else ''
            metric_clean = (metric or '').strip('"') if metric else 'unknown'
            
            # Calculate hash
            dataset_hash = calculate_dataset_hash(train_clean, test_clean)
            
            # Group by hash for summary
            if dataset_hash not in datasets:
                datasets[dataset_hash] = {
                    'train_path': train_clean,
                    'test_path': test_clean,
                    'metrics': set(),
                    'sessions': []
                }
            
            datasets[dataset_hash]['metrics'].add(metric_clean)
            datasets[dataset_hash]['sessions'].append((session_id, session_name))
            updates.append((dataset_hash, session_id))
        
        # Show dataset summary
        print("\n📈 DATASET SUMMARY:")
        print("=" * 80)
        
        for hash_key, data in datasets.items():
            train = data['train_path'] or 'N/A'
            test = data['test_path'] or 'N/A'
            metrics = ', '.join(sorted(data['metrics']))
            session_count = len(data['sessions'])
            
            print(f"\n🎯 Dataset Hash: {hash_key}")
            print(f"   Train: {train}")
            print(f"   Test:  {test}")
            print(f"   Metrics: {metrics}")
            print(f"   Sessions: {session_count}")
            
            # Show recent sessions for this dataset
            recent_sessions = data['sessions'][:3]
            for session_id, session_name in recent_sessions:
                print(f"     • {session_name} ({session_id[:8]}...)")
            
            if len(data['sessions']) > 3:
                print(f"     ... and {len(data['sessions']) - 3} more")
        
        # Perform updates
        if args.dry_run:
            print(f"\n🧪 DRY RUN: Would update {len(updates)} sessions with dataset hashes")
        else:
            print(f"\n💾 Updating {len(updates)} sessions with dataset hashes...")
            
            for dataset_hash, session_id in updates:
                conn.execute(
                    "UPDATE sessions SET dataset_hash = ? WHERE session_id = ?",
                    [dataset_hash, session_id]
                )
            
            print("✅ All sessions updated successfully")
        
        conn.close()
        print(f"\n🎉 Operation completed successfully!")
        
        # Show usage examples
        if not args.dry_run:
            print("\n💡 Usage examples:")
            print("# Show features for Titanic dataset only:")
            print(f"python scripts/duckdb_manager.py features --dataset {list(datasets.keys())[0][:8]}")
            print("\n# Show sessions for specific dataset:")
            print(f"python scripts/duckdb_manager.py sessions --dataset {list(datasets.keys())[0][:8]}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())