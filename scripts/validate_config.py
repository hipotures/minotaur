#!/usr/bin/env python3
"""
Configuration Validation Tool for Minotaur MCTS System

Standalone tool for validating MCTS configuration files.
Useful for debugging configuration issues and understanding validation rules.

Usage:
    python scripts/validate_config.py config/mcts_config.yaml
    python scripts/validate_config.py --all  # Validate all configs in config/
    python scripts/validate_config.py --show-schema  # Show validation schema info
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_validator import (
    ConfigValidator, validate_configuration, calculate_configuration_hash,
    ValidationResult, CompatibilityResult, CompatibilityLevel
)
from src.utils.display_formatter import get_formatter


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")


def validate_single_config(config_path: Path, 
                         validator: ConfigValidator,
                         formatter) -> bool:
    """
    Validate a single configuration file.
    
    Returns:
        True if validation passed, False otherwise
    """
    try:
        config = load_config_file(config_path)
    except ValueError as e:
        formatter.print(formatter.error(f"Failed to load {config_path}: {e}"))
        return False
    
    # Validate configuration
    result = validate_configuration(config)
    
    # Calculate config hash
    config_hash = calculate_configuration_hash(config)
    
    # Print results
    formatter.print(formatter.section_header(f"Validation Results: {config_path.name}"))
    
    if result.is_valid:
        formatter.print(formatter.success("Configuration is valid"))
    else:
        formatter.print(formatter.error("Configuration validation failed"))
    
    # Show configuration hash
    hash_info = {
        "Config Hash": config_hash[:16] + "...",
        "File Path": str(config_path)
    }
    formatter.print(formatter.key_value_pairs(hash_info, "Configuration Info"))
    
    # Show validation details
    if result.errors:
        formatter.print(formatter.info_block(
            "\n".join(f"• {error}" for error in result.errors),
            "❌ Validation Errors",
            "red"
        ))
    
    if result.warnings:
        formatter.print(formatter.info_block(
            "\n".join(f"• {warning}" for warning in result.warnings),
            "⚠️  Warnings",
            "yellow"
        ))
    
    return result.is_valid


def compare_configurations(config1_path: Path, 
                         config2_path: Path,
                         validator: ConfigValidator,
                         formatter) -> None:
    """Compare two configurations for compatibility."""
    try:
        config1 = load_config_file(config1_path)
        config2 = load_config_file(config2_path)
    except ValueError as e:
        formatter.print(formatter.error(f"Failed to load configuration: {e}"))
        return
    
    # Check compatibility
    compatibility = validator.check_compatibility(config1, config2)
    
    # Print comparison results
    formatter.print(formatter.header(f"Configuration Compatibility Analysis"))
    
    formatter.print(f"\n📁 Config 1: {config1_path.name}")
    formatter.print(f"📁 Config 2: {config2_path.name}")
    
    # Show compatibility level
    level_info = {
        "Compatibility Level": compatibility.level.value.title(),
        "Total Changes": str(len(compatibility.changes)),
        "Critical Changes": str(len(compatibility.critical_changes)),
        "Warning Changes": str(len(compatibility.warning_changes))
    }
    
    if compatibility.level == CompatibilityLevel.COMPATIBLE:
        level_color = "green"
    elif compatibility.level == CompatibilityLevel.WARNING:
        level_color = "yellow"
    else:
        level_color = "red"
    
    formatter.print(formatter.key_value_pairs(level_info, "Compatibility Summary", level_color))
    
    # Show detailed changes
    if compatibility.changes:
        changes_text = "\n".join(f"• {change}" for change in compatibility.changes)
        formatter.print(formatter.info_block(changes_text, "All Changes", "blue"))
    
    # Show messages
    if compatibility.messages:
        messages_text = "\n".join(compatibility.messages)
        formatter.print(formatter.info_block(messages_text, "Analysis Messages", level_color))


def show_validation_schema(validator: ConfigValidator, formatter) -> None:
    """Show information about the validation schema."""
    formatter.print(formatter.header("Configuration Validation Schema Information"))
    
    # Get parameter categorization
    param_info = validator.get_parameter_info()
    
    # Show critical parameters
    if param_info['critical']:
        critical_text = "\n".join(f"• {param}" for param in param_info['critical'])
        formatter.print(formatter.info_block(
            critical_text,
            "🚫 Critical Parameters (Block Resumption)",
            "red"
        ))
    
    # Show warning parameters
    if param_info['warning']:
        warning_text = "\n".join(f"• {param}" for param in param_info['warning'])
        formatter.print(formatter.info_block(
            warning_text,
            "⚠️  Warning Parameters (Require --force-resume)",
            "yellow"
        ))
    
    # Show safe parameters
    if param_info['safe']:
        safe_text = "\n".join(f"• {param}" for param in param_info['safe'][:10])  # Show first 10
        if len(param_info['safe']) > 10:
            safe_text += f"\n... and {len(param_info['safe']) - 10} more"
        
        formatter.print(formatter.info_block(
            safe_text,
            "✅ Safe Parameters (Can Change Freely)",
            "green"
        ))
    
    # Show usage examples
    usage_examples = """
Examples:
  • Changing 'mcts.max_tree_depth' blocks session resumption
  • Changing 'autogluon.time_limit' requires --force-resume
  • Changing 'logging.level' is always safe
  • Configuration hash only includes critical parameters
    """
    
    formatter.print(formatter.info_block(usage_examples.strip(), "Usage Guidelines", "blue"))


def find_all_configs(config_dir: Path) -> List[Path]:
    """Find all YAML configuration files in a directory."""
    configs = []
    for pattern in ['*.yaml', '*.yml']:
        configs.extend(config_dir.glob(pattern))
    return sorted(configs)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate MCTS configuration files")
    
    parser.add_argument(
        'config_file',
        nargs='?',
        help='Configuration file to validate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all configuration files in config/ directory'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('CONFIG1', 'CONFIG2'),
        help='Compare two configuration files for compatibility'
    )
    parser.add_argument(
        '--show-schema',
        action='store_true',
        help='Show validation schema information'
    )
    parser.add_argument(
        '--plain',
        action='store_true',
        help='Plain text output (no Rich formatting)'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default='config',
        help='Directory to search for config files (default: config/)'
    )
    
    args = parser.parse_args()
    
    # Setup formatter
    formatter = get_formatter(force_plain=args.plain)
    validator = ConfigValidator()
    
    # Handle different modes
    if args.show_schema:
        show_validation_schema(validator, formatter)
        return 0
    
    if args.compare:
        config1_path = Path(args.compare[0])
        config2_path = Path(args.compare[1])
        compare_configurations(config1_path, config2_path, validator, formatter)
        return 0
    
    if args.all:
        # Validate all configs in directory
        config_dir = args.config_dir
        if not config_dir.exists():
            formatter.print(formatter.error(f"Config directory not found: {config_dir}"))
            return 1
        
        configs = find_all_configs(config_dir)
        if not configs:
            formatter.print(formatter.warning(f"No configuration files found in {config_dir}"))
            return 1
        
        formatter.print(formatter.header(f"Validating All Configurations in {config_dir}"))
        
        success_count = 0
        for config_path in configs:
            formatter.print(f"\n{'='*50}")
            if validate_single_config(config_path, validator, formatter):
                success_count += 1
        
        # Summary
        total = len(configs)
        formatter.print(f"\n{'='*50}")
        if success_count == total:
            formatter.print(formatter.success(f"All {total} configurations are valid"))
        else:
            failed = total - success_count
            formatter.print(formatter.error(f"{failed}/{total} configurations failed validation"))
        
        return 0 if success_count == total else 1
    
    # Validate single config file
    if not args.config_file:
        parser.print_help()
        return 1
    
    config_path = Path(args.config_file)
    if not config_path.exists():
        formatter.print(formatter.error(f"Configuration file not found: {config_path}"))
        return 1
    
    success = validate_single_config(config_path, validator, formatter)
    return 0 if success else 1


if __name__ == "__main__":
    # Suppress logging output for cleaner tool output
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('DB').setLevel(logging.WARNING)
    logging.getLogger('db').setLevel(logging.WARNING)  # Database connection manager logger
    logging.getLogger('src').setLevel(logging.WARNING)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)