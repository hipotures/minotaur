"""
Register Command - Register new datasets in the system.

Handles both automatic and manual dataset registration:
- Auto-detection of dataset files in directory
- Manual specification of individual files
- Metadata extraction and validation
- Database registration and integrity checking
"""

import threading
import time
import sys
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
            
            if args.auto or args.dataset_path:
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
        
        # Animation control
        animation_running = threading.Event()
        animation_running.set()
        
        def animate():
            """Show animated progress indicator"""
            chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while animation_running.is_set():
                print(f"\r🔄 Processing dataset files and generating features {chars[i % len(chars)]}", end="", flush=True)
                time.sleep(0.1)
                i += 1
        
        # Start animation in background
        animation_thread = threading.Thread(target=animate, daemon=True)
        animation_thread.start()
        
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
            
            # Stop animation
            animation_running.clear()
            
            if result.get('success'):
                print(f"\r✅ Dataset processing completed successfully!" + " " * 20, flush=True)  # Clear animation chars
                self.print_success(f"Dataset '{args.dataset_name}' registered successfully!")
                dataset_id = result.get('dataset_id', '')[:8]
                self.print_info(f"Dataset ID: {dataset_id}")
                self.print_info(f"Show details: python manager.py datasets --details {args.dataset_name}")
            else:
                print(f"\r❌ Dataset processing failed" + " " * 30, flush=True)  # Clear animation chars
                self.print_error(f"Registration failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            # Stop animation on error
            animation_running.clear()
            print(f"\r❌ Dataset processing failed" + " " * 30, flush=True)  # Clear animation chars
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
            
            if result.get('success'):
                self.print_success(f"Dataset '{args.dataset_name}' registered successfully!")
                dataset_id = result.get('dataset_id', '')[:8]
                self.print_info(f"Dataset ID: {dataset_id}")
                self.print_info(f"Show details: python manager.py datasets --details {args.dataset_name}")
            else:
                self.print_error(f"Registration failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.print_error(f"Manual registration failed: {e}")