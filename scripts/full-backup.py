#!/usr/bin/env python3
"""
Full Minotaur Project Backup Script

Creates a complete compressed backup of the entire Minotaur project to ~/DEV/BAK/
with timestamp, excluding only .venv and .pytest_cache directories.

Backup includes:
- All source code and configuration files
- Git repository and history (.git/)
- Databases and data files
- Logs and outputs
- Cache and temporary files
- Documentation and requirements
- Everything except .venv and .pytest_cache

Output: ~/DEV/BAK/minotaur-YYYYMMDD_HHMM.tar.gz
"""

import os
import tarfile
import sys
import gzip
from datetime import datetime
from pathlib import Path
from typing import Set, List


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_backup_dir() -> Path:
    """Get the backup directory (~/DEV/BAK/)."""
    home = Path.home()
    backup_dir = home / "DEV" / "BAK"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMM format."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def get_excluded_paths() -> Set[str]:
    """
    Get set of paths to exclude from backup.
    
    Returns:
        Set of directory/file names to exclude
    """
    excluded = {
        '.venv',
        '.pytest_cache',
        # Note: We deliberately include everything else:
        # '.git',           # INCLUDE - git repository and history
        # '__pycache__',    # INCLUDE - Python cache files
        # 'cache',          # INCLUDE - MCTS cache
        # 'data',           # INCLUDE - databases and data
        # 'logs',           # INCLUDE - log files
        # 'outputs',        # INCLUDE - generated outputs
        # '.coverage',      # INCLUDE - coverage files
        # 'htmlcov',        # INCLUDE - coverage reports
    }
    return excluded


def should_exclude_path(path: Path, excluded_paths: Set[str]) -> bool:
    """
    Check if a path should be excluded from backup.
    
    Args:
        path: Path to check
        excluded_paths: Set of excluded directory names
        
    Returns:
        True if path should be excluded
    """
    # Check if any part of the path matches excluded directories
    for part in path.parts:
        if part in excluded_paths:
            return True
    return False


def get_project_size(project_root: Path, excluded_paths: Set[str]) -> tuple:
    """
    Calculate total size of project files to backup.
    
    Args:
        project_root: Root directory of project
        excluded_paths: Set of excluded paths
        
    Returns:
        Tuple of (total_files, total_size_mb)
    """
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(project_root):
        current_path = Path(root)
        
        # Skip excluded directories
        if should_exclude_path(current_path, excluded_paths):
            continue
        
        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if not should_exclude_path(current_path / d, excluded_paths)]
        
        for file in files:
            file_path = current_path / file
            try:
                file_size = file_path.stat().st_size
                total_size += file_size
                total_files += 1
            except (OSError, PermissionError):
                # Skip files we can't access
                pass
    
    total_size_mb = total_size / (1024 * 1024)
    return total_files, total_size_mb


def create_backup(project_root: Path, backup_path: Path, excluded_paths: Set[str]) -> bool:
    """
    Create compressed backup of the project.
    
    Args:
        project_root: Root directory of project
        backup_path: Output backup file path
        excluded_paths: Set of excluded paths
        
    Returns:
        True if backup was successful
    """
    try:
        files_added = 0
        
        print(f"📦 Creating backup: {backup_path}")
        print(f"📂 Source: {project_root}")
        
        with tarfile.open(backup_path, 'w:gz', compresslevel=6) as tar:
            for root, dirs, files in os.walk(project_root):
                current_path = Path(root)
                
                # Skip excluded directories
                if should_exclude_path(current_path, excluded_paths):
                    continue
                
                # Remove excluded directories from dirs to prevent walking into them
                dirs[:] = [d for d in dirs if not should_exclude_path(current_path / d, excluded_paths)]
                
                # Add directories
                relative_dir = current_path.relative_to(project_root.parent)
                try:
                    tar.add(current_path, arcname=relative_dir, recursive=False)
                except (OSError, PermissionError) as e:
                    print(f"⚠️  Warning: Could not add directory {current_path}: {e}")
                
                # Add files
                for file in files:
                    file_path = current_path / file
                    relative_file = file_path.relative_to(project_root.parent)
                    
                    try:
                        tar.add(file_path, arcname=relative_file)
                        files_added += 1
                        
                        # Progress indicator every 100 files
                        if files_added % 100 == 0:
                            print(f"📄 Added {files_added} files...")
                            
                    except (OSError, PermissionError) as e:
                        print(f"⚠️  Warning: Could not add file {file_path}: {e}")
        
        print(f"✅ Successfully added {files_added} files to backup")
        return True
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False


def verify_backup(backup_path: Path) -> bool:
    """
    Verify the backup file integrity.
    
    Args:
        backup_path: Path to backup file
        
    Returns:
        True if backup is valid
    """
    try:
        print(f"🔍 Verifying backup integrity...")
        
        with tarfile.open(backup_path, 'r:gz') as tar:
            # Try to list contents - this will fail if archive is corrupted
            members = tar.getmembers()
            file_count = len([m for m in members if m.isfile()])
            dir_count = len([m for m in members if m.isdir()])
            
        backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Backup verification successful")
        print(f"📁 Directories: {dir_count}")
        print(f"📄 Files: {file_count}")
        print(f"💾 Compressed size: {backup_size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Backup verification failed: {e}")
        return False


def list_excluded_content(project_root: Path, excluded_paths: Set[str]):
    """
    Show what will be excluded from backup.
    
    Args:
        project_root: Root directory of project
        excluded_paths: Set of excluded paths
    """
    print("🚫 Excluded from backup:")
    
    excluded_found = []
    for item in project_root.iterdir():
        if item.name in excluded_paths and item.exists():
            if item.is_dir():
                try:
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    excluded_found.append(f"   📁 {item.name}/ ({size_mb:.1f} MB)")
                except:
                    excluded_found.append(f"   📁 {item.name}/ (size unknown)")
            else:
                try:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    excluded_found.append(f"   📄 {item.name} ({size_mb:.1f} MB)")
                except:
                    excluded_found.append(f"   📄 {item.name} (size unknown)")
    
    if excluded_found:
        for item in excluded_found:
            print(item)
    else:
        print("   (none found)")


def main():
    """Main backup function."""
    print("📦 Minotaur Full Project Backup")
    print("=" * 40)
    
    # Check if we're in the right directory
    project_root = get_project_root()
    if not (project_root / "CLAUDE.md").exists():
        print("❌ Error: This script must be run from the Minotaur project directory")
        print(f"Current path: {project_root}")
        sys.exit(1)
    
    # Setup paths
    backup_dir = get_backup_dir()
    timestamp = get_timestamp()
    backup_filename = f"minotaur-{timestamp}.tar.gz"
    backup_path = backup_dir / backup_filename
    
    # Get excluded paths
    excluded_paths = get_excluded_paths()
    
    print(f"📂 Project root: {project_root}")
    print(f"💾 Backup location: {backup_path}")
    print(f"⏰ Timestamp: {timestamp}")
    
    # Show what will be excluded
    list_excluded_content(project_root, excluded_paths)
    
    # Calculate project size
    print("\n📊 Analyzing project size...")
    total_files, total_size_mb = get_project_size(project_root, excluded_paths)
    print(f"📄 Files to backup: {total_files:,}")
    print(f"💾 Estimated size: {total_size_mb:.1f} MB")
    
    # Confirm backup
    print("\n" + "=" * 40)
    print("🎯 BACKUP SUMMARY")
    print("=" * 40)
    print(f"Source: {project_root}")
    print(f"Destination: {backup_path}")
    print(f"Files: {total_files:,} ({total_size_mb:.1f} MB)")
    print(f"Excluded: {', '.join(sorted(excluded_paths))}")
    print("=" * 40)
    
    response = input("Proceed with backup? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Backup cancelled.")
        sys.exit(0)
    
    # Create backup
    print(f"\n🚀 Starting backup at {datetime.now().strftime('%H:%M:%S')}...")
    
    if backup_path.exists():
        print(f"⚠️  Backup file already exists: {backup_path}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("Backup cancelled.")
            sys.exit(0)
        backup_path.unlink()
    
    # Create the backup
    success = create_backup(project_root, backup_path, excluded_paths)
    
    if not success:
        print("❌ Backup failed!")
        sys.exit(1)
    
    # Verify backup
    if not verify_backup(backup_path):
        print("❌ Backup verification failed!")
        sys.exit(1)
    
    # Final summary
    final_size_mb = backup_path.stat().st_size / (1024 * 1024)
    compression_ratio = (total_size_mb / final_size_mb) if final_size_mb > 0 else 0
    
    print("\n" + "=" * 50)
    print("🎉 BACKUP COMPLETE")
    print("=" * 50)
    print(f"✅ Backup created: {backup_path}")
    print(f"📄 Files backed up: {total_files:,}")
    print(f"💾 Original size: {total_size_mb:.1f} MB")
    print(f"🗜️  Compressed size: {final_size_mb:.1f} MB")
    print(f"📊 Compression ratio: {compression_ratio:.1f}x")
    print(f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    print("\n📋 Backup contents:")
    print(f"• All source code and configuration")
    print(f"• Git repository and history (.git/)")
    print(f"• Databases and data files")
    print(f"• Logs and session outputs")
    print(f"• Cache and temporary files")
    print(f"• Documentation and requirements")
    print(f"• Everything except: {', '.join(sorted(excluded_paths))}")
    
    print(f"\n🔄 To restore:")
    print(f"cd ~/DEV && tar -xzf {backup_path}")


if __name__ == "__main__":
    main()