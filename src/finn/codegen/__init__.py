"""
FINN Unified Code Generation Framework

This module provides a unified architecture for code generation across HLS and RTL backends,
adopting RTL's successful operation-driven pattern while providing modern infrastructure.
"""

from .base import BaseCodeGenerator
from .template_engine import TemplateEngine
from .file_manager import FileManager
from .library_resolver import LibraryResolver
from .hls_generator import ModernHLSGenerator
from .rtl_generator import ModernRTLGenerator

__all__ = [
    'BaseCodeGenerator',
    'TemplateEngine', 
    'FileManager',
    'LibraryResolver',
    'ModernHLSGenerator',
    'ModernRTLGenerator'
]