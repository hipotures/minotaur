"""
Color scheme constants for Rich terminal UI.

Provides consistent color palette across all modules for professional appearance.
Based on Rich library best practices and terminal UI design patterns.
"""

# =============================================================================
# RICH COLOR SCHEME - Professional Terminal UI
# =============================================================================

# Table Headers and Structure
HEADERS = "bold magenta"           # All table headers
TABLE_TITLE = "bold blue"         # Table titles
BORDERS = "bright_black"          # Table borders (if needed)

# Primary Information Hierarchy  
PRIMARY = "bold yellow"            # Main identifiers (names, IDs)
SECONDARY = "cyan"                 # Secondary information
TERTIARY = "white"                 # Standard text

# Data Types
NUMBERS = "green"                  # All metrics, counts, sizes
DATES = "blue"                     # Timestamps, dates
PATHS = "dim blue"                 # File paths, URLs
TEXT_DATA = "white"                # General text content

# Status and State Colors (Semantic)
STATUS_SUCCESS = "bold green"      # ✅ Active, completed, success
STATUS_WARNING = "bold yellow"     # ⚠️ Warnings, pending
STATUS_ERROR = "bold red"          # ❌ Errors, failed, inactive
STATUS_INFO = "cyan"               # ℹ️ Information, neutral
STATUS_INACTIVE = "dim"            # Disabled, unused

# Interactive Elements
ACCENT = "magenta"                 # Commands, links, actions
HIGHLIGHT = "reverse"              # Selected items, emphasis
MUTED = "dim white"                # Helper text, descriptions

# Special Purpose
PROGRESS = "bright_green"          # Progress indicators
DEBUG = "dim cyan"                 # Debug information
SEPARATOR = "bright_black"         # Dividers, separators

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def status_color(status: str) -> str:
    """Get appropriate color for status text."""
    status_lower = status.lower()
    
    if status_lower in ['active', 'completed', 'success', 'passed', 'ok', 'running']:
        return STATUS_SUCCESS
    elif status_lower in ['warning', 'pending', 'partial', 'limited']:
        return STATUS_WARNING  
    elif status_lower in ['error', 'failed', 'inactive', 'disabled', 'offline']:
        return STATUS_ERROR
    elif status_lower in ['info', 'unknown', 'neutral']:
        return STATUS_INFO
    else:
        return STATUS_INACTIVE

def format_status(status: str) -> str:
    """Format status with appropriate color markup."""
    color = status_color(status)
    return f"[{color}]{status}[/{color}]"

def format_number(value, suffix: str = "") -> str:
    """Format number with consistent styling."""
    return f"[{NUMBERS}]{value}{suffix}[/{NUMBERS}]"

def format_primary(text: str) -> str:
    """Format primary identifier with consistent styling."""
    return f"[{PRIMARY}]{text}[/{PRIMARY}]"

def format_secondary(text: str) -> str:
    """Format secondary information with consistent styling."""
    return f"[{SECONDARY}]{text}[/{SECONDARY}]"