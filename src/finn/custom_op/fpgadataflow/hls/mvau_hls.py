"""
MVAU HLS Backend Implementation

This module implements the HLS backend for MatrixVectorActivation (MVAU) operations,
providing template value extraction and code generation specific to MVAU characteristics
including PE+SIMD parallelization and weight memory management.
"""

import logging
from typing import Dict, Any, Set, List
from ..hlsbackend import HLSBackend

try:
    # Import the actual MVAU operation class
    from finn.custom_op.fpgadataflow.matrixvectoractivation import MatrixVectorActivation
    MVAU_AVAILABLE = True
except ImportError:
    # Define a mock class for testing when MVAU not available
    class MatrixVectorActivation:
        pass
    MVAU_AVAILABLE = False


class MVAU_HLS(MatrixVectorActivation, HLSBackend):
    """
    HLS Backend implementation for MatrixVectorActivation operations.
    
    Combines MVAU domain logic with HLS code generation capabilities.
    Handles MVAU-specific characteristics:
    - PE (Processing Element) parallelization
    - SIMD (Single Instruction Multiple Data) parallelization  
    - Weight memory management (BRAM/UltraRAM/distributed)
    - Activation function integration
    - Precision configurations
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize MVAU HLS backend.
        
        Args:
            onnx_node: ONNX node for this operation
            **kwargs: Additional initialization arguments
        """
        # Initialize both parent classes
        MatrixVectorActivation.__init__(self, onnx_node, **kwargs)
        HLSBackend.__init__(self)
        
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        self.logger.debug(f"Initialized MVAU_HLS for node: {onnx_node.name}")
    
    def get_supported_templates(self) -> Set[str]:
        """Return templates this MVAU HLS backend supports.
        
        Returns:
            Set of template names this backend can handle
        """
        templates = {
            # Basic templates
            'hls_mvau_basic.cpp.j2',
            'hls_mvau_simple.cpp.j2',
            
            # Streaming templates (most common for MVAU)
            'hls_mvau_streaming.cpp.j2',
            'hls_mvau_streaming_optimized.cpp.j2',
            
            # Parallel templates
            'hls_mvau_parallel.cpp.j2',
            'hls_mvau_parallel_simd.cpp.j2',
            
            # Memory-optimized templates
            'hls_mvau_bram.cpp.j2',
            'hls_mvau_uram.cpp.j2',
            'hls_mvau_distributed.cpp.j2',
            
            # Activation-integrated templates
            'hls_mvau_with_activation.cpp.j2',
            'hls_mvau_relu.cpp.j2',
            'hls_mvau_bipolar.cpp.j2',
            
            # High-performance templates
            'hls_mvau_optimized.cpp.j2',
            'hls_mvau_high_throughput.cpp.j2',
        }
        
        self.logger.debug(f"MVAU HLS supports {len(templates)} templates")
        return templates
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for specified template.
        
        Args:
            template_name: Name of template to extract values for
            
        Returns:
            Dictionary mapping template placeholders to values
            
        Raises:
            UnsupportedTemplateError: If template not supported
        """
        if template_name not in self.get_supported_templates():
            from ..codegen import UnsupportedTemplateError
            raise UnsupportedTemplateError(f"MVAU HLS backend does not support template: {template_name}")
        
        self.logger.debug(f"Extracting template values for: {template_name}")
        
        # Start with common values
        template_values = self._extract_common_values(self)
        
        # Add MVAU-specific parallelization values
        template_values.update(self._extract_mvau_parallelization_values())
        
        # Add MVAU weight and memory configuration
        template_values.update(self._extract_mvau_memory_config())
        
        # Add MVAU precision configuration  
        template_values.update(self._extract_mvau_precision_config())
        
        # Add MVAU activation configuration
        template_values.update(self._extract_mvau_activation_config())
        
        # Add template-specific optimizations
        template_values.update(self._extract_template_specific_values(template_name))
        
        self.logger.debug(f"Extracted {len(template_values)} template values for MVAU")
        return template_values
    
    def _get_template_priority_order(self) -> List[str]:
        """Return template priority order for MVAU HLS.
        
        MVAU prioritizes streaming templates for high throughput,
        then parallel templates, then basic templates.
        
        Returns:
            List of template names in priority order (highest first)
        """
        return [
            # High-performance streaming (highest priority)
            'hls_mvau_streaming_optimized.cpp.j2',
            'hls_mvau_high_throughput.cpp.j2',
            'hls_mvau_optimized.cpp.j2',
            
            # Standard streaming
            'hls_mvau_streaming.cpp.j2',
            'hls_mvau_with_activation.cpp.j2',
            
            # Parallel templates
            'hls_mvau_parallel_simd.cpp.j2',
            'hls_mvau_parallel.cpp.j2',
            
            # Memory-specific templates
            'hls_mvau_bram.cpp.j2',
            'hls_mvau_uram.cpp.j2',
            'hls_mvau_distributed.cpp.j2',
            
            # Basic templates (fallback)
            'hls_mvau_basic.cpp.j2',
            'hls_mvau_simple.cpp.j2',
        ]
    
    def _extract_mvau_parallelization_values(self) -> Dict[str, Any]:
        """Extract MVAU-specific parallelization values.
        
        MVAU uses both PE (Processing Element) and SIMD parallelization.
        
        Returns:
            Dictionary containing parallelization configuration
        """
        values = {}
        
        # PE factor - number of parallel processing elements
        values['pe_factor'] = self._safe_extract_value(self, 'PE', default_value=1)
        
        # SIMD factor - SIMD parallelization within each PE
        values['simd_factor'] = self._safe_extract_value(self, 'SIMD', default_value=1)
        
        # Total parallelization
        values['total_parallelization'] = values['pe_factor'] * values['simd_factor']
        
        # Parallelization strategy
        if values['pe_factor'] > 1 and values['simd_factor'] > 1:
            values['parallelization_strategy'] = 'pe_and_simd'
        elif values['pe_factor'] > 1:
            values['parallelization_strategy'] = 'pe_only'
        elif values['simd_factor'] > 1:
            values['parallelization_strategy'] = 'simd_only'
        else:
            values['parallelization_strategy'] = 'sequential'
        
        # Matrix dimensions for parallelization planning
        mw = self._safe_extract_value(self, 'MW', default_value=32)
        mh = self._safe_extract_value(self, 'MH', default_value=32)
        values['matrix_width'] = mw
        values['matrix_height'] = mh
        
        # Compute workload distribution
        values['pe_workload'] = max(1, mh // values['pe_factor'])
        values['simd_workload'] = max(1, mw // values['simd_factor'])
        
        self.logger.debug(f"MVAU parallelization: PE={values['pe_factor']}, SIMD={values['simd_factor']}")
        return values
    
    def _extract_mvau_memory_config(self) -> Dict[str, Any]:
        """Extract MVAU memory configuration.
        
        Returns:
            Dictionary containing memory configuration
        """
        values = {}
        
        # Memory mode for weights
        values['mem_mode'] = self._safe_extract_value(self, 'mem_mode', default_value='external')
        
        # RAM style for weight storage
        values['ram_style'] = self._safe_extract_value(self, 'ram_style', default_value='block')
        
        # Weight memory characteristics
        weight_memory_mode = self._safe_extract_value(self, 'resType', default_value='lut')
        values['weight_memory_mode'] = weight_memory_mode
        
        # Memory bandwidth requirements
        pe_factor = values.get('pe_factor', 1)
        simd_factor = values.get('simd_factor', 1)
        values['memory_bandwidth_factor'] = pe_factor * simd_factor
        
        # Memory addressing
        if values['mem_mode'] == 'const':
            values['memory_addressing'] = 'constant'
            values['memory_access_pattern'] = 'sequential'
        elif values['mem_mode'] == 'decoupled':
            values['memory_addressing'] = 'decoupled'
            values['memory_access_pattern'] = 'streaming'
        else:
            values['memory_addressing'] = 'external'
            values['memory_access_pattern'] = 'random'
        
        # Memory optimization hints
        if weight_memory_mode == 'lut':
            values['memory_optimization'] = 'distribute_weights'
        elif weight_memory_mode == 'bram':
            values['memory_optimization'] = 'block_memory'
        elif weight_memory_mode == 'uram':
            values['memory_optimization'] = 'ultra_memory'
        else:
            values['memory_optimization'] = 'generic'
        
        self.logger.debug(f"MVAU memory config: mode={values['mem_mode']}, ram_style={values['ram_style']}")
        return values
    
    def _extract_mvau_precision_config(self) -> Dict[str, Any]:
        """Extract MVAU precision configuration.
        
        Returns:
            Dictionary containing precision settings
        """
        values = {}
        
        # Input precision
        input_datatype = self.get_input_datatype()
        if hasattr(input_datatype, 'bitwidth'):
            values['input_width'] = input_datatype.bitwidth()
            values['input_signed'] = input_datatype.signed if hasattr(input_datatype, 'signed') else True
        else:
            values['input_width'] = 8
            values['input_signed'] = True
        
        # Weight precision  
        weight_datatype = self.get_weight_datatype()
        if hasattr(weight_datatype, 'bitwidth'):
            values['weight_width'] = weight_datatype.bitwidth()
            values['weight_signed'] = weight_datatype.signed if hasattr(weight_datatype, 'signed') else True
        else:
            values['weight_width'] = 8
            values['weight_signed'] = True
        
        # Accumulator precision (typically wider)
        values['accumulator_width'] = values['input_width'] + values['weight_width'] + 8  # Safe margin
        values['accumulator_signed'] = True
        
        # Output precision
        output_datatype = self.get_output_datatype()
        if hasattr(output_datatype, 'bitwidth'):
            values['output_width'] = output_datatype.bitwidth()
            values['output_signed'] = output_datatype.signed if hasattr(output_datatype, 'signed') else True
        else:
            values['output_width'] = values['input_width']
            values['output_signed'] = values['input_signed']
        
        # Precision optimization flags
        values['narrow_weights'] = values['weight_width'] <= 4
        values['narrow_inputs'] = values['input_width'] <= 4
        values['mixed_precision'] = values['input_width'] != values['weight_width']
        
        self.logger.debug(f"MVAU precision: input={values['input_width']}, weight={values['weight_width']}, output={values['output_width']}")
        return values
    
    def _extract_mvau_activation_config(self) -> Dict[str, Any]:
        """Extract MVAU activation function configuration.
        
        Returns:
            Dictionary containing activation settings
        """
        values = {}
        
        # Check for activation function
        activation_type = self._safe_extract_value(self, 'activation', default_value='linear')
        values['activation_type'] = activation_type
        values['has_activation'] = activation_type != 'linear'
        
        # Activation-specific parameters
        if activation_type == 'relu':
            values['activation_threshold'] = 0
            values['activation_implementation'] = 'clamp_negative'
        elif activation_type == 'bipolar':
            values['activation_threshold'] = 0
            values['activation_implementation'] = 'sign_function'
        elif activation_type == 'multithreshold':
            # Multi-threshold activation
            values['activation_implementation'] = 'lookup_table'
            threshold_values = self._safe_extract_value(self, 'threshold', default_value=[])
            values['threshold_values'] = threshold_values
            values['num_thresholds'] = len(threshold_values) if threshold_values else 0
        else:
            values['activation_implementation'] = 'passthrough'
        
        # Integration flags
        values['fused_activation'] = values['has_activation']
        values['separate_activation'] = not values['has_activation']
        
        self.logger.debug(f"MVAU activation: type={activation_type}, fused={values['fused_activation']}")
        return values
    
    def _extract_template_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Extract template-specific optimization values.
        
        Args:
            template_name: Name of template to optimize for
            
        Returns:
            Dictionary containing template-specific values
        """
        values = {}
        
        # Template category detection
        if 'streaming' in template_name:
            values['template_category'] = 'streaming'
            values['optimization_target'] = 'throughput'
            values['pipeline_depth'] = 'deep'
            values['buffer_strategy'] = 'ping_pong'
        elif 'parallel' in template_name:
            values['template_category'] = 'parallel'
            values['optimization_target'] = 'latency'
            values['pipeline_depth'] = 'shallow'
            values['buffer_strategy'] = 'parallel_access'
        elif 'bram' in template_name:
            values['template_category'] = 'memory_optimized'
            values['optimization_target'] = 'resource_efficiency'
            values['preferred_memory'] = 'block_ram'
            values['memory_banking'] = True
        elif 'uram' in template_name:
            values['template_category'] = 'high_capacity'
            values['optimization_target'] = 'capacity'
            values['preferred_memory'] = 'ultra_ram'
            values['memory_banking'] = True
        elif 'distributed' in template_name:
            values['template_category'] = 'distributed'
            values['optimization_target'] = 'speed'
            values['preferred_memory'] = 'lutram'
            values['memory_banking'] = False
        else:
            values['template_category'] = 'generic'
            values['optimization_target'] = 'balanced'
            values['pipeline_depth'] = 'medium'
        
        # Performance characteristics based on template
        if 'optimized' in template_name or 'high_throughput' in template_name:
            values['performance_level'] = 'high'
            values['resource_usage'] = 'high'
            values['enable_optimizations'] = True
        else:
            values['performance_level'] = 'standard'
            values['resource_usage'] = 'moderate'
            values['enable_optimizations'] = False
        
        # Implementation directives
        values['pragma_pipeline'] = values['template_category'] in ['streaming', 'parallel']
        values['pragma_unroll'] = values['template_category'] in ['parallel', 'high_performance']
        values['pragma_dataflow'] = values['template_category'] == 'streaming'
        
        self.logger.debug(f"Template-specific config for {template_name}: category={values.get('template_category')}")
        return values
    
    def generate_hls_code(self) -> str:
        """Generate HLS code for MVAU operation.
        
        Returns:
            Generated HLS C++ code as string
            
        Raises:
            CodeGenerationError: If code generation fails
        """
        self.logger.info("Generating HLS code for MVAU operation")
        
        try:
            # Use the shared generate_code method from Codegen base class
            return self.generate_code()
        except Exception as e:
            self.logger.error(f"MVAU HLS code generation failed: {e}")
            raise


# Convenience function for creating MVAU HLS backend
def create_mvau_hls_backend(onnx_node, **kwargs) -> MVAU_HLS:
    """
    Factory function to create MVAU HLS backend.
    
    Args:
        onnx_node: ONNX node for MVAU operation
        **kwargs: Additional arguments
        
    Returns:
        Configured MVAU_HLS backend instance
    """
    return MVAU_HLS(onnx_node, **kwargs)


# Export the main class and factory function
__all__ = ['MVAU_HLS', 'create_mvau_hls_backend']