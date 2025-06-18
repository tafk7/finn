"""
FINN Unified Code Generation - Consolidated Architecture

Simple, explicit, fast.
"""

from .template_engine import TemplateEngine
from .backend_registry import BackendRegistry
from .backend_registration import get_backend_registry, register_all_backends
from .config import CodegenConfig, get_global_config
from .codegen import (
    Codegen,
    UnsupportedTemplateError,
    TemplateValidationError,
    CodeGenerationError
)

# Replace complex components with simplified versions
from .simple_library_resolver import SimpleLibraryResolver as LibraryResolver
from .simple_file_manager import SimpleFileManager as FileManager

# Backend classes are imported separately to avoid circular imports
# Import them directly from their modules when needed

__all__ = [
    # Core components
    'TemplateEngine',
    'BackendRegistry',
    'CodegenConfig',
    
    # Utilities (simplified versions)
    'FileManager',
    'LibraryResolver',
    
    # Registration
    'get_backend_registry',
    'register_all_backends',
    'get_global_config',
    
    # Base interface
    'Codegen',
    
    # Exceptions
    'UnsupportedTemplateError',
    'TemplateValidationError',
    'CodeGenerationError'
]