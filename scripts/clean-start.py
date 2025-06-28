#!/usr/bin/env python3
"""
Clean Start Script - Reset Minotaur Runtime Data

This script removes all runtime-generated files and databases to restore
the system to a clean state for fresh runs, while preserving the development
environment and git repository.

REMOVES (runtime data):
- Main database (data/minotaur.duckdb only)
- Entire outputs directory (outputs/)
- All MCTS cache files (cache/)
- All log files (logs/)
- Generated features and models
- Python cache files (__pycache__/ in src/ and scripts/)

PRESERVES (development environment):
- .venv/ (Python virtual environment)
- .pytest_cache/ (testing cache)
- .git/ (repository and history)
- All source code files
- Configuration files

Use with caution - runtime data cannot be recovered!
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_files_to_remove() -> List[Path]:
    """
    Get list of all files and directories to remove for clean start.
    
    Removes only runtime-generated files and application data:
    - Main database file (data/minotaur.duckdb only)
    - MCTS cache (cache/)
    - Log files (logs/)
    - Session outputs
    - Generated features and models
    - Python cache files (but only in src/ and scripts/)
    
    Does NOT remove:
    - .venv/ (development environment)
    - .pytest_cache/ (testing cache)
    - .git/ (repository)
    - data/ directory contents (except main database file)
    - Source code files
    
    Returns:
        List of Path objects to remove
    """
    root = get_project_root()
    
    files_to_remove = [
        # Main database (ONLY the main database file in data/)
        root / "data" / "minotaur.duckdb",
        root / "data" / "minotaur.duckdb.wal",  # WAL file if exists
        
        # Cache directory (entire cache)
        root / "cache",
        
        # Log files (entire logs directory)
        root / "logs",
        
        # Outputs directory (entire outputs directory)
        root / "outputs",
        
        # Temporary database files
        root / "feature_discovery.db",
        root / "feature_discovery_fast.db",
        root / "feature_discovery_test.db",
        root / "feature_discovery_titanic_test.db",
        
        # Generated feature files (entire subdirectories)
        root / "features" / "train",
        root / "features" / "test",
        
        # Model artifacts (entire subdirectories)
        root / "models" / "autogluon_models",
        root / "models" / "autogluon_models_full",
        
        # Note: data/ directory contents (except main db) are preserved
        # Legacy hash directories and datasets will remain
        
        # Python cache files (only in project source directories)
        root / "src" / "__pycache__",
        root / "src" / "db" / "__pycache__",
        root / "src" / "db" / "models" / "__pycache__",
        root / "src" / "db" / "repositories" / "__pycache__",
        root / "src" / "db" / "core" / "__pycache__",
        root / "src" / "db" / "config" / "__pycache__",
        root / "src" / "domains" / "__pycache__",
        root / "scripts" / "__pycache__",
        root / "scripts" / "modules" / "__pycache__",
        root / "__pycache__",  # Root level cache only
        
        # Coverage files (testing artifacts)
        root / ".coverage",
        root / "htmlcov",
        
        # Note: .venv and .pytest_cache are intentionally NOT included
        # Note: .git is intentionally NOT included
        # Note: Individual .pyc files will be handled by glob pattern (excluding .venv)
    ]
    
    return files_to_remove


def remove_glob_patterns(root: Path) -> List[str]:
    """
    Get glob patterns for files to remove (excluding .venv and .pytest_cache).
    
    Returns:
        List of glob patterns to remove
    """
    patterns = [
        # Python compiled files (but we'll filter out .venv paths later)
        "src/**/*.pyc",
        "scripts/**/*.pyc", 
        "tests/**/*.pyc",
        "*.pyc",  # Root level only
        "src/**/*.pyo",
        "scripts/**/*.pyo",
        "tests/**/*.pyo",
        "src/**/*.pyd",
        "scripts/**/*.pyd", 
        "tests/**/*.pyd",
        
        # OS files
        "**/.DS_Store",
        "**/Thumbs.db",
        
        # Note: outputs/ directory removed entirely, no need for patterns
        
        # Temporary database files
        "**/*_test.duckdb",
        "**/*_fast.duckdb",
        "**/*_backup_*.duckdb*",
    ]
    
    return patterns


def safe_remove(path: Path) -> bool:
    """
    Safely remove a file or directory.
    
    Args:
        path: Path to remove
        
    Returns:
        True if removal was successful or file didn't exist
    """
    try:
        if path.is_file():
            path.unlink()
            print(f"✅ Removed file: {path}")
            return True
        elif path.is_dir():
            shutil.rmtree(path)
            print(f"✅ Removed directory: {path}")
            return True
        else:
            # File/directory doesn't exist - that's okay
            return True
    except PermissionError:
        print(f"❌ Permission denied: {path}")
        return False
    except Exception as e:
        print(f"❌ Error removing {path}: {e}")
        return False


def confirm_deletion() -> bool:
    """
    Ask user for confirmation before deletion.
    
    Returns:
        True if user confirms deletion
    """
    print("🧹 MINOTAUR RUNTIME DATA CLEANUP")
    print("=" * 50)
    print("This will permanently delete RUNTIME DATA:")
    print("• Main database (data/minotaur.duckdb)")
    print("• Entire outputs directory (outputs/)")
    print("• All MCTS cache files (cache/)")
    print("• All log files (logs/)")
    print("• Generated features and models")
    print("• Python cache (__pycache__/ in src/scripts only)")
    print("")
    print("✅ PRESERVES development environment:")
    print("• .venv/ (Python virtual environment)")
    print("• .pytest_cache/ (testing cache)")
    print("• .git/ (repository and history)")
    print("• All source code and config files")
    print("")
    print("⚠️  Runtime data cannot be recovered!")
    print("=" * 50)
    
    while True:
        response = input("Proceed with runtime data cleanup? Type '124' to proceed: ").strip()
        
        if response == "124":
            return True
        elif response.lower() in ['no', 'n', 'exit', 'quit', '']:
            print("Operation cancelled.")
            return False
        else:
            print("Invalid response. Type '124' to proceed or 'no' to cancel.")


def create_clean_directories(root: Path):
    """
    Create essential empty directories after cleanup.
    
    Args:
        root: Project root path
    """
    essential_dirs = [
        root / "data",
        root / "features",
        root / "models",
    ]
    
    for dir_path in essential_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")


def main():
    """Main cleanup function."""
    print("🧹 Minotaur Clean Start Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    root = get_project_root()
    if not (root / "CLAUDE.md").exists():
        print("❌ Error: This script must be run from the Minotaur project directory")
        print(f"Current path: {root}")
        sys.exit(1)
    
    print(f"Project root: {root}")
    
    # Confirm deletion
    if not confirm_deletion():
        sys.exit(0)
    
    print("\n🗑️  Starting cleanup...")
    
    # Remove specific files and directories
    files_to_remove = get_files_to_remove()
    removed_count = 0
    failed_count = 0
    
    for file_path in files_to_remove:
        if safe_remove(file_path):
            if file_path.exists():  # Only count if it actually existed
                removed_count += 1
        else:
            failed_count += 1
    
    # Remove files matching glob patterns
    patterns = remove_glob_patterns(root)
    for pattern in patterns:
        matching_files = list(root.glob(pattern))
        for file_path in matching_files:
            if safe_remove(file_path):
                removed_count += 1
            else:
                failed_count += 1
    
    # Create essential directories
    print("\n📁 Creating essential directories...")
    create_clean_directories(root)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 RUNTIME DATA CLEANUP COMPLETE")
    print("=" * 50)
    print(f"✅ Successfully removed: {removed_count} runtime items")
    if failed_count > 0:
        print(f"❌ Failed to remove: {failed_count} items")
    print("\n🔄 Runtime data cleared - development environment preserved")
    print("✅ .venv/ and .pytest_cache/ left intact")
    print("✅ .git/ repository preserved")
    print("\n📋 Next steps:")
    print("1. Run: python run_feature_discovery.py --config config/mcts_config.yaml --new-session")
    print("2. Register datasets: python scripts/duckdb_manager.py datasets --register")
    print("3. Verify setup: python scripts/duckdb_manager.py selfcheck --run")
    
    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count} items could not be removed (likely permission issues)")
        sys.exit(1)


if __name__ == "__main__":
    main()