# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import time


class UnsupportedTemplateError(Exception):
    """Raised when operation doesn't support requested template."""
    pass


class TemplateValidationError(Exception):
    """Raised when template values fail validation."""
    pass


class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass


class Codegen(ABC):
    """Abstract base class for all code generation backends.
    
    Provides simplified infrastructure for template-based code generation
    with explicit template selection and minimal complexity.
    """
    
    def __init__(self):
        """Initialize simplified code generation backend."""
        # Initialize simple configuration
        try:
            from .config import get_global_config
            self.config = get_global_config()
        except ImportError:
            self.config = None
            
        # Initialize logging
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        
        # Initialize simple template engine
        self.template_engine = self._initialize_simple_template_engine()
        
        self.logger.debug(f"Initialized {self.__class__.__name__}")
    
    # ===== Abstract Methods for Subclasses =====
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Get explicit template name for this backend.
        
        Returns:
            Name of template to use for code generation
        """
        pass
    
    @abstractmethod
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for specified template.
        
        Args:
            template_name: Name of template to extract values for
            
        Returns:
            Dictionary mapping template placeholders to values
            
        Raises:
            UnsupportedTemplateError: If template not supported
        """
        pass
    
    # ===== Shared Code Generation Infrastructure =====
    
    def generate_code(self) -> str:
        """Simplified code generation flow.
        
        Returns:
            Generated code as string
            
        Raises:
            CodeGenerationError: If code generation fails
        """
        start_time = time.time()
        operation_name = getattr(self, 'onnx_node', {}).get('op_type', 'unknown')
        
        try:
            self.logger.info(f"Starting code generation for {operation_name}")
            
            # 1. Get template name (explicit from backend)
            template_name = self.get_template_name()
            self.logger.info(f"Using template: {template_name}")
            
            # 2. Extract values for template
            template_values = self.get_template_values(template_name)
            self.logger.debug(f"Extracted {len(template_values)} template values")
            
            # 3. Save debug values if enabled
            if self.config and self.config.save_intermediate_files:
                self._save_debug_values(template_name, template_values)
            
            # 4. Render template
            code = self._render_template(template_name, template_values)
            self.logger.debug(f"Template rendered ({len(code)} chars)")
            
            # 5. Simple post-processing
            processed_code = self._post_process_code(code)
            
            total_time = time.time() - start_time
            self.logger.info(f"Code generation completed in {total_time:.3f}s")
            
            return processed_code
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"Code generation failed after {error_time:.3f}s: {e}")
            raise CodeGenerationError(f"Failed to generate code: {e}") from e
    
    def _initialize_simple_template_engine(self):
        """Initialize simplified template engine.
        
        Returns:
            Initialized TemplateEngine instance
        """
        try:
            from .template_engine import TemplateEngine
            
            # Get template paths from config if available
            template_dirs = None
            if self.config and hasattr(self.config, 'template_dirs'):
                template_dirs = self.config.template_dirs
                
            engine = TemplateEngine(template_dirs)
            self.logger.debug("Template engine initialized")
            return engine
        except ImportError:
            self.logger.warning("Template engine not available, using mock")
            return MockTemplateEngine()
    
    def _render_template(self, template_name: str, values: Dict[str, Any]) -> str:
        """Render template - simplified rendering logic.
        
        Args:
            template_name: Name of template to render
            values: Template values to use
            
        Returns:
            Rendered template as string
            
        Raises:
            CodeGenerationError: If rendering fails
        """
        try:
            rendered = self.template_engine.render(template_name, values)
            self.logger.debug(f"Template {template_name} rendered successfully")
            return rendered
        except Exception as e:
            self.logger.error(f"Template rendering failed for {template_name}: {e}")
            raise CodeGenerationError(f"Template rendering failed: {e}") from e
    
    def _post_process_code(self, code: str) -> str:
        """Post-process generated code - simple processing.
        
        Args:
            code: Raw generated code
            
        Returns:
            Post-processed code
        """
        # Simple post-processing: basic cleanup
        processed = code.strip()
        
        # Remove excessive blank lines
        lines = processed.split('\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            is_blank = line.strip() == ''
            if is_blank and prev_blank:
                continue  # Skip consecutive blank lines
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        processed = '\n'.join(cleaned_lines)
        self.logger.debug("Code post-processing completed")
        return processed
    
    # ===== Shared Utility Methods =====
    
    def _safe_extract_value(self, operation, attr_name: str, default_value=None):
        """Safely extract value with logging.
        
        Args:
            operation: Operation instance to extract from
            attr_name: Name of attribute to extract
            default_value: Default value if attribute missing
            
        Returns:
            Extracted value or default
        """
        try:
            value = operation.get_nodeattr(attr_name)
            self.logger.debug(f"Extracted {attr_name}: {value}")
            return value
        except (AttributeError, KeyError) as e:
            if default_value is not None:
                self.logger.debug(f"Using default for {attr_name}: {default_value}")
                return default_value
            else:
                self.logger.error(f"Required attribute {attr_name} missing: {e}")
                raise
    
    def _extract_common_values(self, operation) -> Dict[str, Any]:
        """Extract common template values - shared extraction.
        
        Args:
            operation: Operation instance to extract from
            
        Returns:
            Dictionary of common template values
        """
        try:
            common_values = {
                'op_type': operation.onnx_node.op_type,
                'input_width': operation.get_instream_width(),
                'output_width': operation.get_outstream_width(),
                'exp_cycles': operation.get_exp_cycles(),
            }
            self.logger.debug(f"Extracted common values: {list(common_values.keys())}")
            return common_values
        except Exception as e:
            self.logger.error(f"Failed to extract common values: {e}")
            return {
                'op_type': getattr(operation.onnx_node, 'op_type', 'unknown'),
                'input_width': 32,  # Safe defaults
                'output_width': 32,
                'exp_cycles': 1,
            }
    
    def _has_attr(self, operation, attr_name: str) -> bool:
        """Check if operation has specific attribute.
        
        Args:
            operation: Operation instance to check
            attr_name: Name of attribute to check
            
        Returns:
            True if attribute exists
        """
        try:
            operation.get_nodeattr(attr_name)
            return True
        except (AttributeError, KeyError):
            return False


    def _save_debug_values(self, template_name: str, values: Dict[str, Any]):
        """Save template values for debugging.
        
        Args:
            template_name: Name of template
            values: Template values to save
        """
        try:
            if self.config and self.config.save_intermediate_files:
                import os
                import json
                
                debug_dir = self.config.get_debug_output_dir()
                debug_file = os.path.join(debug_dir, f"{self.__class__.__name__}_{template_name}_values.json")
                
                # Convert values to JSON-serializable format
                serializable_values = {}
                for key, value in values.items():
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_values[key] = value
                    except (TypeError, ValueError):
                        serializable_values[key] = str(value)
                
                with open(debug_file, 'w') as f:
                    json.dump(serializable_values, f, indent=2)
                    
                self.logger.debug(f"Debug values saved to: {debug_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save debug values: {e}")


class MockTemplateEngine:
    """Mock template engine for testing when real engine unavailable."""
    
    def render(self, template_name: str, values: Dict[str, Any]) -> str:
        """Return mock rendered template."""
        return f"// Mock template: {template_name}\n// Values: {list(values.keys())}\n"
    
    def template_exists(self, template_name: str) -> bool:
        """Mock template exists check."""
        return True
    
    def clear_cache(self):
        """Mock cache clear."""
        pass