"""
Defines the absolute path to the project root directory.
"""

from pathlib import Path

# The project root is defined as the parent directory of the 'src' directory.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
