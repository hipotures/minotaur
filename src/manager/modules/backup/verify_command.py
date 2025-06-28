"""
Verify Command - Verify backup file integrity.

Provides backup verification including:
- File format validation
- Compression integrity checking
- Database connectivity testing
- Content verification and metadata extraction
"""

from pathlib import Path
from .base import BaseBackupCommand


class VerifyCommand(BaseBackupCommand):
    """Handle --verify command for backup."""
    
    def execute(self, args) -> None:
        """Execute the backup verification command."""
        try:
            backup_file = args.verify
            
            print("🔍 VERIFYING BACKUP INTEGRITY")
            print("=" * 40)
            
            # Resolve backup file path
            backup_path = self.resolve_backup_path(backup_file)
            
            if not backup_path.exists():
                self.print_error(f"Backup file not found: {backup_path}")
                self._suggest_available_backups()
                return
            
            # Show backup information
            self._show_backup_details(backup_path)
            
            # Perform comprehensive verification
            verification_result = self._perform_comprehensive_verification(backup_path)
            
            # Show verification results
            self._show_verification_results(verification_result)
            
        except Exception as e:
            self.print_error(f"Backup verification failed: {e}")
    
    def _show_backup_details(self, backup_path: Path) -> None:
        """Show backup file details before verification."""
        info = self.format_backup_info(backup_path)
        
        print(f"Backup file: {backup_path}")
        print(f"File size: {info['size']}")
        print(f"Created: {info['created']}")
        print(f"Type: {info['type']}")
        print()
    
    def _perform_comprehensive_verification(self, backup_path: Path) -> dict:
        """Perform comprehensive backup verification."""
        results = {
            'file_exists': False,
            'file_size_valid': False,
            'format_valid': False,
            'compression_valid': False,
            'database_valid': False,
            'content_verified': False,
            'database_info': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Test 1: File existence and basic properties
            print("🔍 Step 1: Checking file existence and properties...")
            if self._verify_file_properties(backup_path, results):
                print("✅ File properties verification passed")
            else:
                print("❌ File properties verification failed")
                return results
            
            # Test 2: Format verification
            print("🔍 Step 2: Verifying file format...")
            if self._verify_file_format(backup_path, results):
                print("✅ File format verification passed")
            else:
                print("❌ File format verification failed")
                return results
            
            # Test 3: Compression verification (if applicable)
            if backup_path.suffix == '.gz':
                print("🔍 Step 3: Verifying compression integrity...")
                if self._verify_compression(backup_path, results):
                    print("✅ Compression verification passed")
                else:
                    print("❌ Compression verification failed")
                    return results
            else:
                print("🔍 Step 3: Skipping compression verification (uncompressed file)")
                results['compression_valid'] = True
            
            # Test 4: Database connectivity and content
            print("🔍 Step 4: Verifying database content...")
            if self._verify_database_content(backup_path, results):
                print("✅ Database content verification passed")
            else:
                print("❌ Database content verification failed")
                return results
            
            results['content_verified'] = True
            
        except Exception as e:
            results['errors'].append(f"Verification process failed: {e}")
        
        return results
    
    def _verify_file_properties(self, backup_path: Path, results: dict) -> bool:
        """Verify basic file properties."""
        try:
            if backup_path.exists():
                results['file_exists'] = True
                
                file_size = backup_path.stat().st_size
                if file_size > 0:
                    results['file_size_valid'] = True
                    return True
                else:
                    results['errors'].append("Backup file is empty")
            else:
                results['errors'].append("Backup file does not exist")
            
        except Exception as e:
            results['errors'].append(f"File property check failed: {e}")
        
        return False
    
    def _verify_file_format(self, backup_path: Path, results: dict) -> bool:
        """Verify file format is valid."""
        try:
            # Check if it's a DuckDB file or compressed DuckDB file
            if backup_path.suffix == '.duckdb' or backup_path.name.endswith('.duckdb.gz'):
                results['format_valid'] = True
                return True
            else:
                results['errors'].append(f"Invalid backup file format: {backup_path.suffix}")
                
        except Exception as e:
            results['errors'].append(f"Format verification failed: {e}")
        
        return False
    
    def _verify_compression(self, backup_path: Path, results: dict) -> bool:
        """Verify compression integrity for gzipped files."""
        try:
            import gzip
            
            with gzip.open(backup_path, 'rb') as f:
                # Try to read the entire file to verify compression
                chunk_size = 1024 * 1024  # 1MB chunks
                total_read = 0
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    total_read += len(chunk)
                
                if total_read > 0:
                    results['compression_valid'] = True
                    return True
                else:
                    results['errors'].append("Compressed file appears to be empty")
            
        except Exception as e:
            results['errors'].append(f"Compression verification failed: {e}")
        
        return False
    
    def _verify_database_content(self, backup_path: Path, results: dict) -> bool:
        """Verify database content and structure."""
        try:
            # For compressed files, we need to extract temporarily
            if backup_path.suffix == '.gz':
                temp_db_path = backup_path.with_suffix('.tmp')
                try:
                    self.extract_from_backup(backup_path, temp_db_path)
                    db_info = self.verify_database_connectivity(temp_db_path)
                    temp_db_path.unlink()  # Clean up temp file
                except Exception as e:
                    if temp_db_path.exists():
                        temp_db_path.unlink()
                    raise e
            else:
                db_info = self.verify_database_connectivity(backup_path)
            
            if db_info:
                results['database_valid'] = True
                results['database_info'] = db_info
                return True
            else:
                results['errors'].append("Database connectivity test failed")
            
        except Exception as e:
            results['errors'].append(f"Database content verification failed: {e}")
        
        return False
    
    def _show_verification_results(self, results: dict) -> None:
        """Show comprehensive verification results."""
        print("\n📊 VERIFICATION RESULTS")
        print("=" * 40)
        
        # Overall status
        all_checks_passed = (
            results['file_exists'] and 
            results['file_size_valid'] and 
            results['format_valid'] and 
            results['compression_valid'] and 
            results['database_valid'] and
            results['content_verified']
        )
        
        if all_checks_passed:
            self.print_success("✅ ALL VERIFICATION CHECKS PASSED")
        else:
            self.print_error("❌ VERIFICATION FAILED")
        
        print()
        
        # Detailed results
        self._print_check_result("File Existence", results['file_exists'])
        self._print_check_result("File Size Valid", results['file_size_valid'])
        self._print_check_result("Format Valid", results['format_valid'])
        self._print_check_result("Compression Valid", results['compression_valid'])
        self._print_check_result("Database Valid", results['database_valid'])
        self._print_check_result("Content Verified", results['content_verified'])
        
        # Database information
        if results['database_info']:
            print(f"\n📋 Database Information:")
            db_info = results['database_info']
            print(f"   Tables found: {db_info['table_count']}")
            if 'session_count' in db_info:
                print(f"   Sessions: {db_info['session_count']}")
            if 'tables' in db_info and db_info['tables']:
                print(f"   Table names: {', '.join(db_info['tables'])}")
        
        # Errors and warnings
        if results['errors']:
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"   • {error}")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"   • {warning}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if all_checks_passed:
            print("   • Backup file is valid and ready for restoration")
            print("   • No issues detected")
        else:
            print("   • Do not use this backup for restoration")
            print("   • Create a new backup to replace this corrupted file")
            print("   • Check disk space and file system integrity")
    
    def _print_check_result(self, check_name: str, passed: bool) -> None:
        """Print individual check result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:<20}: {status}")
    
    def _suggest_available_backups(self) -> None:
        """Suggest available backup files when requested file not found."""
        backup_files = self.find_backup_files()
        
        if backup_files:
            print("\n💡 Available backup files:")
            for backup_file in backup_files[:5]:  # Show first 5
                info = self.format_backup_info(backup_file)
                print(f"   • {backup_file.name} ({info['size']}, {info['created']})")
            
            if len(backup_files) > 5:
                print(f"   ... and {len(backup_files) - 5} more")
            
            print("\n💡 Usage: python manager.py backup --verify <filename>")
        else:
            print("\n💡 No backup files found. Create one with: python manager.py backup --create")