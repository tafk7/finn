"""
Template Engine for Unified FINN Code Generation

This module provides a modern Jinja2-based template engine that replaces
the basic string replacement approach, while maintaining simplicity and
backward compatibility.
"""

import jinja2
from typing import Dict, Any, List, Optional
from pathlib import Path
import os


class TemplateEngine:
    """
    Modern template engine using Jinja2 for FINN code generation.
    
    Replaces the basic string replacement approach with a more powerful
    template system that supports conditionals, loops, inheritance, and
    other modern template features while maintaining simplicity.
    """
    
    def __init__(self, template_dirs: Optional[List[str]] = None):
        """
        Initialize the template engine.
        
        Args:
            template_dirs: List of directories to search for templates.
                          If None, uses default FINN template locations.
        """
        if template_dirs is None:
            template_dirs = self._get_default_template_dirs()
            
        # Create Jinja2 environment with appropriate settings for C/C++ code
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dirs, followlinks=True),
            # Keep whitespace control similar to RTL templates
            trim_blocks=True,
            lstrip_blocks=True,
            # Note: Do NOT set line_statement_prefix='#' as it conflicts with C preprocessor directives
            # Add useful extensions
            extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols']
        )
        
        # Add custom filters for FINN-specific operations
        self._register_custom_filters()
        
    def _get_default_template_dirs(self) -> List[str]:
        """
        Get default template directories for FINN.
        
        Returns:
            List of template directory paths
        """
        finn_root = os.environ.get('FINN_ROOT', '.')
        
        return [
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'hls'),
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'rtl'),
            os.path.join(finn_root, 'src', 'finn', 'codegen', 'templates', 'common'),
            # Backward compatibility with existing template locations
            os.path.join(finn_root, 'custom_hls'),
            os.path.join(finn_root, 'finn-rtllib'),
        ]
    
    def _register_custom_filters(self):
        """Register FINN-specific Jinja2 filters."""
        
        def format_define(value, define_name):
            """Format a #define statement for C/C++/SystemVerilog."""
            if isinstance(value, bool):
                return f"#define {define_name} {1 if value else 0}"
            elif isinstance(value, str):
                return f'#define {define_name} "{value}"'
            else:
                return f"#define {define_name} {value}"
        
        def format_parameter(value, param_name, param_type="parameter"):
            """Format a parameter statement for SystemVerilog."""
            return f"{param_type} {param_name} = {value}"
        
        def format_port(direction, width, name):
            """Format a port declaration for SystemVerilog."""
            if width == 1:
                return f"{direction} logic {name}"
            else:
                return f"{direction} logic [{width-1}:0] {name}"
        
        def format_array_size(size):
            """Format array size for C++ templates."""
            return f"[{size}]" if size > 1 else ""
        
        def cpp_type_name(finn_datatype):
            """Convert FINN datatype to C++ type name."""
            # This would integrate with existing FINN datatype system
            type_map = {
                'BIPOLAR': 'ap_int<1>',
                'BINARY': 'ap_uint<1>',
                'INT2': 'ap_int<2>',
                'INT4': 'ap_int<4>',
                'INT8': 'ap_int<8>',
                'INT16': 'ap_int<16>',
                'INT32': 'ap_int<32>',
                'UINT2': 'ap_uint<2>',
                'UINT4': 'ap_uint<4>',
                'UINT8': 'ap_uint<8>',
                'UINT16': 'ap_uint<16>',
                'UINT32': 'ap_uint<32>',
            }
            if hasattr(finn_datatype, 'bitwidth'):
                return f'ap_int<{finn_datatype.bitwidth()}>'
            return type_map.get(str(finn_datatype), 'ap_int<8>')
        
        def regex_replace(value, pattern, replacement=''):
            """Replace regex pattern in string value."""
            import re
            return re.sub(pattern, replacement, str(value))
        
        # Register filters with Jinja2 environment
        self.jinja_env.filters['format_define'] = format_define
        self.jinja_env.filters['format_parameter'] = format_parameter
        self.jinja_env.filters['format_port'] = format_port
        self.jinja_env.filters['format_array_size'] = format_array_size
        self.jinja_env.filters['cpp_type_name'] = cpp_type_name
        self.jinja_env.filters['regex_replace'] = regex_replace
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name/path of template file
            context: Dictionary containing template variables
            
        Returns:
            Rendered template content as string
            
        Raises:
            jinja2.TemplateNotFound: If template cannot be found
            jinja2.TemplateSyntaxError: If template has syntax errors
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except jinja2.TemplateNotFound as e:
            # Provide helpful error message with search paths
            search_paths = "\n  ".join(self.jinja_env.loader.searchpath)
            raise jinja2.TemplateNotFound(
                f"Template '{template_name}' not found.\n"
                f"Searched in:\n  {search_paths}"
            )
    
    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render a template from a string rather than a file.
        
        This method provides backward compatibility with existing code
        that uses inline template strings.
        
        Args:
            template_string: Template content as string
            context: Dictionary containing template variables
            
        Returns:
            Rendered template content as string
        """
        template = self.jinja_env.from_string(template_string)
        return template.render(**context)
    
    def render_legacy(self, template_content: str, replacements: Dict[str, Any]) -> str:
        """
        Render using legacy string replacement method.
        
        This method provides backward compatibility with existing RTL
        operations that use simple string replacement (e.g., $KEY$ -> value).
        
        Args:
            template_content: Template content with $KEY$ placeholders
            replacements: Dictionary mapping keys to replacement values
            
        Returns:
            Template with replacements applied
        """
        result = template_content
        for key, value in replacements.items():
            # Handle both $KEY$ and KEY formats
            placeholder = f"${key}$" if not key.startswith('$') else key
            if not placeholder.endswith('$'):
                placeholder += '$'
            result = result.replace(placeholder, str(value))
        return result
    
    def list_templates(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            pattern: Optional glob pattern to filter templates
            
        Returns:
            List of available template names
        """
        templates = self.jinja_env.list_templates()
        if pattern:
            import fnmatch
            templates = [t for t in templates if fnmatch.fnmatch(t, pattern)]
        return sorted(templates)
    
    def add_template_dir(self, template_dir: str):
        """
        Add an additional template directory to the search path.
        
        Args:
            template_dir: Path to template directory
        """
        if os.path.exists(template_dir):
            current_paths = list(self.jinja_env.loader.searchpath)
            if template_dir not in current_paths:
                current_paths.append(template_dir)
                self.jinja_env.loader.searchpath = current_paths
    
    def template_exists(self, template_name: str) -> bool:
        """
        Check if a template exists.
        
        Args:
            template_name: Name/path of template file
            
        Returns:
            True if template exists, False otherwise
        """
        try:
            self.jinja_env.get_template(template_name)
            return True
        except jinja2.TemplateNotFound:
            return False