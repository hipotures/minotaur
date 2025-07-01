"""
Custom Feature Operations

Domain-specific feature operations for different datasets.
"""

# Import available custom feature modules dynamically
import importlib
import logging

logger = logging.getLogger(__name__)

# Available custom domains
CUSTOM_DOMAINS = {
    'titanic': '.titanic',
    'kaggle_s5e6': '.kaggle_s5e6',
    'fertilizer': '.kaggle_s5e6',  # Alias for compatibility
    's5e6': '.kaggle_s5e6',  # Alias for fertilizer dataset
}

def get_custom_features(domain_name: str):
    """
    Dynamically load custom feature operations for a specific domain.
    
    Args:
        domain_name: Name of the domain (e.g., 'titanic', 'fertilizer')
        
    Returns:
        CustomFeatureOperations class for the domain, or None if not found
    """
    # Clean the domain name
    clean_name = domain_name.lower().strip()
    
    # Check for aliases
    module_name = CUSTOM_DOMAINS.get(clean_name)
    
    if not module_name:
        logger.warning(f"No custom features available for domain: {domain_name}")
        return None
    
    try:
        # Import the module
        module = importlib.import_module(module_name, package='src.features.custom')
        
        # Get the CustomFeatureOperations class
        if hasattr(module, 'CustomFeatureOperations'):
            return module.CustomFeatureOperations
        else:
            logger.error(f"Module {module_name} does not have CustomFeatureOperations class")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import custom features for {domain_name}: {e}")
        return None