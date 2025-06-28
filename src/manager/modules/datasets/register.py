"""
Register Command - Register new datasets in the system.

Handles both automatic and manual dataset registration:
- Auto-detection of dataset files in directory
- Manual specification of individual files
- Metadata extraction and validation
- Database registration and integrity checking
"""

from typing import Dict, Any
from .base import BaseDatasetsCommand


class RegisterCommand(BaseDatasetsCommand):
    """Handle --register command for datasets."""
    
    def execute(self, args) -> None:
        """Execute the register dataset command."""
        try:
            # Validate required arguments
            if not args.dataset_name:
                self.print_error("Dataset name is required for registration.")
                self.print_info("Usage: python manager.py datasets --register --dataset-name NAME [options]")
                return
            
            if args.auto:
                self._register_auto(args)
            else:
                self._register_manual(args)
                
        except Exception as e:
            self.print_error(f"Failed to register dataset: {e}")
    
    def _register_auto(self, args) -> None:
        """Register dataset with auto-detection."""
        if not args.dataset_path:
            self.print_error("Dataset path is required for auto-registration.")
            return
        
        self.print_info(f"Auto-registering dataset '{args.dataset_name}' from {args.dataset_path}")
        
        # Use dataset service to register
        try:
            result = self.dataset_service.register_dataset_auto(
                name=args.dataset_name,
                path=args.dataset_path,
                target_column=getattr(args, 'target_column', None),
                id_column=getattr(args, 'id_column', None),
                competition_name=getattr(args, 'competition_name', None),
                description=getattr(args, 'description', None)
            )
            
            if result:
                self.print_success(f"Dataset '{args.dataset_name}' registered successfully!")
                dataset_id = result.get('dataset_id', '')[:8]
                self.print_info(f"Dataset ID: {dataset_id}")
                self.print_info(f"Show details: python manager.py datasets --details {args.dataset_name}")
            else:
                self.print_error("Registration failed - no result returned")
                
        except Exception as e:
            self.print_error(f"Auto-registration failed: {e}")
    
    def _register_manual(self, args) -> None:
        """Register dataset with manual file specification."""
        # Validate required files
        if not args.train:
            self.print_error("Train file path is required for manual registration.")
            return
        
        self.print_info(f"Manually registering dataset '{args.dataset_name}'")
        
        try:
            result = self.dataset_service.register_dataset_manual(
                name=args.dataset_name,
                train_path=args.train,
                test_path=getattr(args, 'test', None),
                submission_path=getattr(args, 'submission', None),
                validation_path=getattr(args, 'validation', None),
                target_column=getattr(args, 'target_column', None),
                id_column=getattr(args, 'id_column', None),
                competition_name=getattr(args, 'competition_name', None),
                description=getattr(args, 'description', None)
            )
            
            if result:
                self.print_success(f"Dataset '{args.dataset_name}' registered successfully!")
                dataset_id = result.get('dataset_id', '')[:8]
                self.print_info(f"Dataset ID: {dataset_id}")
                self.print_info(f"Show details: python manager.py datasets --details {args.dataset_name}")
            else:
                self.print_error("Registration failed - no result returned")
                
        except Exception as e:
            self.print_error(f"Manual registration failed: {e}")