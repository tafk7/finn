"""
Simple Configuration for FINN Code Generation

This module provides a simplified configuration system without complex layering.
Replaces the complex configuration system with a straightforward dataclass approach.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodegenConfig:
    """
    Simple configuration without layered complexity.
    
    Eliminates complex environment variable processing and file-based
    configuration in favor of simple, explicit settings.
    """
    
    # Template configuration
    template_dirs: Optional[List[str]] = None
    
    # Debug and logging
    debug_mode: bool = False
    log_level: str = 'INFO'
    
    # Performance settings
    cache_templates: bool = True
    max_template_cache_size: int = 50
    
    # Output settings
    debug_output_dir: Optional[str] = None
    save_intermediate_files: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default template directories if not provided
        if self.template_dirs is None:
            self.template_dirs = self._get_default_template_dirs()
        
        # Simple environment variable override (no complex layering)
        if os.getenv('FINN_DEBUG', '').lower() in ('true', '1', 'yes'):
            self.debug_mode = True
        
        if os.getenv('FINN_LOG_LEVEL'):
            self.log_level = os.getenv('FINN_LOG_LEVEL')
        
        if os.getenv('FINN_CODEGEN_CACHE_SIZE'):
            try:
                self.max_template_cache_size = int(os.getenv('FINN_CODEGEN_CACHE_SIZE'))
            except ValueError:
                pass  # Keep default value
        
        if os.getenv('FINN_DEBUG_OUTPUT_DIR'):
            self.debug_output_dir = os.getenv('FINN_DEBUG_OUTPUT_DIR')
            self.save_intermediate_files = True
    
    def _get_default_template_dirs(self) -> List[str]:
        """
        Get default template directories for FINN.
        
        Returns:
            List of template directory paths
        """
        finn_root = os.environ.get('FINN_ROOT', '.')
        
        template_dirs = [
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'hls'),
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'rtl'),
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'common'),
            # Backward compatibility with existing template locations
            os.path.join(finn_root, 'custom_hls'),
            os.path.join(finn_root, 'finn-rtllib'),
        ]
        
        # Filter to only existing directories
        existing_dirs = [d for d in template_dirs if os.path.exists(d)]
        
        return existing_dirs
    
    @classmethod
    def from_env(cls) -> 'CodegenConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            CodegenConfig with environment-based settings
        """
        return cls()  # __post_init__ handles environment variables
    
    @classmethod
    def for_testing(cls, debug_mode: bool = True) -> 'CodegenConfig':
        """
        Create configuration for testing.
        
        Args:
            debug_mode: Enable debug mode for testing
            
        Returns:
            CodegenConfig configured for testing
        """
        return cls(
            debug_mode=debug_mode,
            log_level='DEBUG',
            cache_templates=False,  # Disable caching for deterministic tests
            save_intermediate_files=False
        )
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        if self.debug_mode:
            level = logging.DEBUG
            
        # Configure root logger for FINN codegen
        logger = logging.getLogger('finn.codegen')
        logger.setLevel(level)
        
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def get_debug_output_dir(self) -> str:
        """
        Get debug output directory, creating if necessary.
        
        Returns:
            Path to debug output directory
        """
        if self.debug_output_dir:
            debug_dir = self.debug_output_dir
        else:
            debug_dir = os.path.join(os.getcwd(), 'finn_debug')
        
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir


# Global configuration instance
_global_config = None


def get_global_config() -> CodegenConfig:
    """
    Get the global configuration instance.
    
    Returns:
        Global CodegenConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = CodegenConfig.from_env()
        _global_config.setup_logging()
    return _global_config


def set_global_config(config: CodegenConfig):
    """
    Set the global configuration instance.
    
    Args:
        config: CodegenConfig instance to use globally
    """
    global _global_config
    _global_config = config
    config.setup_logging()


def reset_global_config():
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None