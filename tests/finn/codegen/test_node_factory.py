"""
FINN Codegen Test Node Factory
Creates realistic ONNX nodes for backend testing and validation.
"""

import logging
from typing import Dict, Any, Optional


class MockONNXAttribute:
    """Mock ONNX attribute for testing purposes."""
    def __init__(self, name: str, value: Any):
        self.name = name
        self.i = value if isinstance(value, int) else 0
        self.f = value if isinstance(value, float) else 0.0
        self.s = value.encode() if isinstance(value, str) else b""
        self.type = 1  # INT type by default


class MockONNXNode:
    """Mock ONNX node for testing purposes."""
    
    def __init__(self):
        self.op_type = ""
        self.name = ""
        self.attribute = []  # List of MockONNXAttribute objects
        self.input = []
        self.output = []
        self._attr_dict = {}  # Internal dict for easy access
        
    def get_nodeattr(self, attr_name: str, default=None):
        """Get node attribute value (compatibility with FINN backends)."""
        return self._attr_dict.get(attr_name, default)
        
    def set_nodeattr(self, attr_name: str, value: Any):
        """Set node attribute value."""
        self._attr_dict[attr_name] = value
        
        # Update the attribute list for ONNX compatibility
        # Remove existing attribute with same name
        self.attribute = [attr for attr in self.attribute if attr.name != attr_name]
        # Add new attribute
        self.attribute.append(MockONNXAttribute(attr_name, value))
    
    def add_attributes(self, attr_dict: Dict[str, Any]):
        """Add multiple attributes at once."""
        for name, value in attr_dict.items():
            self.set_nodeattr(name, value)


class TestNodeFactory:
    """Factory for creating realistic ONNX nodes for backend testing."""
    
    def __init__(self):
        self.logger = logging.getLogger("finn.codegen.test_node_factory")
    
    def create_test_node(self, operation_type: str, **kwargs) -> MockONNXNode:
        """Create test node based on operation type."""
        operation_type_lower = operation_type.lower()
        
        if 'thresholding' in operation_type_lower:
            return self.create_thresholding_node(**kwargs)
        elif operation_type_lower in ['mvau', 'matrixvectoractivation']:
            return self.create_mvau_node(**kwargs)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def create_thresholding_node(self, **kwargs) -> MockONNXNode:
        """Create Thresholding ONNX node with realistic attributes."""
        node = MockONNXNode()
        node.op_type = "Thresholding"
        node.name = kwargs.get('node_name', 'Thresholding_0')
        
        # Set realistic Thresholding attributes
        node.attribute = {
            'NumChannels': kwargs.get('NumChannels', 32),
            'PE': kwargs.get('PE', 4), 
            'NumSteps': kwargs.get('NumSteps', 8),
            'ram_style': kwargs.get('ram_style', 'block'),
            'runtime_writeable_weights': kwargs.get('runtime_writeable_weights', 0),
            'mem_mode': kwargs.get('mem_mode', 'const'),
            'code_gen_dir_cppsim': kwargs.get('code_gen_dir_cppsim', '/tmp/test_cppsim'),
            'code_gen_dir_ipgen': kwargs.get('code_gen_dir_ipgen', '/tmp/test_ipgen'),
            'executable_path': kwargs.get('executable_path', '/tmp/test_exec')
        }
        
        # Add input/output shape info
        node.input = ['input_tensor']
        node.output = ['output_tensor']
        
        self.logger.debug(f"Created Thresholding node with attributes: {node.attribute}")
        return node
    
    def create_mvau_node(self, **kwargs) -> MockONNXNode:
        """Create MVAU ONNX node with realistic attributes.""" 
        node = MockONNXNode()
        node.op_type = "MatrixVectorActivation"
        node.name = kwargs.get('node_name', 'MVAU_0')
        
        # Set realistic MVAU attributes
        node.attribute = {
            'MW': kwargs.get('MW', 64),
            'MH': kwargs.get('MH', 32),
            'PE': kwargs.get('PE', 4),
            'SIMD': kwargs.get('SIMD', 8),
            'mem_mode': kwargs.get('mem_mode', 'internal_embedded'),
            'resType': kwargs.get('resType', 'lut'),
            'ram_style': kwargs.get('ram_style', 'block'),
            'code_gen_dir_cppsim': kwargs.get('code_gen_dir_cppsim', '/tmp/test_cppsim'),
            'code_gen_dir_ipgen': kwargs.get('code_gen_dir_ipgen', '/tmp/test_ipgen'),
            'executable_path': kwargs.get('executable_path', '/tmp/test_exec'),
            'runtime_writeable_weights': kwargs.get('runtime_writeable_weights', 0)
        }
        
        # Add input/output tensors
        node.input = ['input_tensor', 'weight_tensor']
        node.output = ['output_tensor']
        
        self.logger.debug(f"Created MVAU node with attributes: {node.attribute}")
        return node
    
    def create_rtl_node(self, operation_type: str, **kwargs) -> MockONNXNode:
        """Create RTL-specific test node."""
        if 'thresholding' in operation_type.lower():
            node = self.create_thresholding_node(**kwargs)
        elif operation_type.lower() in ['mvau', 'matrixvectoractivation']:
            node = self.create_mvau_node(**kwargs)
        else:
            raise ValueError(f"Unsupported RTL operation type: {operation_type}")
        
        # Add RTL-specific attributes
        node.attribute.update({
            'backend_type': 'rtl',
            'clk_freq': kwargs.get('clk_freq', 250.0),
            'target_clk_ns': kwargs.get('target_clk_ns', 4.0)
        })
        
        return node


def test_node_factory():
    """Test function to verify node factory functionality."""
    factory = TestNodeFactory()
    
    # Test Thresholding node creation
    thres_node = factory.create_thresholding_node(
        NumChannels=64,
        PE=8,
        NumSteps=16
    )
    print(f"✅ Thresholding node created: {thres_node.name}")
    print(f"   Attributes: {thres_node.attribute}")
    
    # Test MVAU node creation
    mvau_node = factory.create_mvau_node(
        MW=128,
        MH=64,
        PE=8,
        SIMD=16,
        mem_mode='external'
    )
    print(f"✅ MVAU node created: {mvau_node.name}")
    print(f"   Attributes: {mvau_node.attribute}")
    
    # Test generic creation
    test_node = factory.create_test_node('Thresholding', NumChannels=32)
    print(f"✅ Generic node created: {test_node.name}")
    
    return True


if __name__ == "__main__":
    test_node_factory()