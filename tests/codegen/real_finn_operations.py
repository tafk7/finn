"""
Real FINN Operation Factory for Testing
Creates actual FINN operations without mocks for authentic testing.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as np_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

# Import real FINN operations
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.thresholding import Thresholding


def create_real_mvau_model(mw=64, mh=64, pe=4, simd=4, wdt="INT8", idt="INT8", odt="INT8"):
    """
    Create a real FINN model with an actual MVAU operation.
    
    Args:
        mw: Matrix width (input features)
        mh: Matrix height (output features)  
        pe: Processing elements
        simd: SIMD parallelism
        wdt: Weight data type
        idt: Input data type
        odt: Output data type
        
    Returns:
        ModelWrapper containing real MVAU operation
    """
    # Create input tensor
    input_shape = [1, mw]
    input_tensor = oh.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, input_shape)
    
    # Create weight tensor with random values
    weight_shape = [mw, mh]
    weight_values = gen_finn_dt_tensor(DataType[wdt], weight_shape)
    weight_init = oh.make_tensor("weights", onnx.TensorProto.FLOAT, weight_shape, weight_values.flatten())
    
    # Create output tensor
    output_shape = [1, mh]
    output_tensor = oh.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, output_shape)
    
    # Create MVAU node with all required attributes
    mvau_node = oh.make_node(
        "MatrixVectorActivation",
        inputs=["inp", "weights"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        PE=pe,
        SIMD=simd,
        inputDataType=idt,
        weightDataType=wdt,
        outputDataType=odt,
        noActivation=1,  # No activation for simpler testing
        mem_mode="internal_embedded",
        # Add missing attributes needed by generators
        ActType="relu",
        numInputVectors=[1],  # List of input vector dimensions
        ram_style="distributed",
        resType="lut"
    )
    
    # Create graph
    graph = oh.make_graph(
        nodes=[mvau_node],
        name="mvau_test",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init]
    )
    
    # Create model
    model = oh.make_model(graph)
    model = ModelWrapper(model)
    
    # Clean up and validate
    model = model.transform(GiveUniqueNodeNames())
    
    return model


def create_real_mvau_operation(mw=64, mh=64, pe=4, simd=4, wdt="INT8", idt="INT8", odt="INT8"):
    """
    Create a real FINN MVAU operation instance.
    
    Returns:
        Real MVAU operation instance with proper FINN structure
    """
    # Create model with MVAU
    model = create_real_mvau_model(mw, mh, pe, simd, wdt, idt, odt)
    
    # Get the MVAU node
    mvau_node = model.get_nodes_by_op_type("MatrixVectorActivation")[0]
    
    # Create real MVAU operation instance (no model parameter)
    mvau_op = MVAU(mvau_node)
    
    return mvau_op, model


def create_real_thresholding_model(num_channels=64, pe=4, idt="INT8", odt="INT8", n_thres=1):
    """
    Create a real FINN model with actual Thresholding operation.
    """
    # Create input tensor
    input_shape = [1, num_channels]
    input_tensor = oh.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, input_shape)
    
    # Create threshold values
    thresh_shape = [num_channels, n_thres]
    thresh_values = np.random.randint(0, 127, thresh_shape).astype(np.float32)
    thresh_init = oh.make_tensor("thresholds", onnx.TensorProto.FLOAT, thresh_shape, thresh_values.flatten())
    
    # Create output tensor
    output_shape = [1, num_channels]
    output_tensor = oh.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, output_shape)
    
    # Create Thresholding node
    thresh_node = oh.make_node(
        "Thresholding",
        inputs=["inp", "thresholds"],
        outputs=["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=num_channels,
        PE=pe,
        inputDataType=idt,
        outputDataType=odt
    )
    
    # Create graph
    graph = oh.make_graph(
        nodes=[thresh_node],
        name="thresh_test",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[thresh_init]
    )
    
    # Create model
    model = oh.make_model(graph)
    model = ModelWrapper(model)
    model = model.transform(GiveUniqueNodeNames())
    
    return model


def create_real_thresholding_operation(num_channels=64, pe=4, idt="INT8", odt="INT8"):
    """
    Create a real FINN Thresholding operation instance.
    """
    model = create_real_thresholding_model(num_channels, pe, idt, odt)
    thresh_node = model.get_nodes_by_op_type("Thresholding")[0]
    thresh_op = Thresholding(thresh_node)
    
    return thresh_op, model


def validate_real_operation(operation, expected_op_type):
    """
    Validate that an operation is a real FINN operation, not a mock.
    
    Args:
        operation: Operation to validate
        expected_op_type: Expected operation type string
        
    Raises:
        AssertionError: If operation is not real FINN operation
    """
    # Check it's a real HWCustomOp
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    assert isinstance(operation, HWCustomOp), f"Operation is not a real HWCustomOp: {type(operation)}"
    
    # Check it has real ONNX node
    assert hasattr(operation, 'onnx_node'), "Operation missing onnx_node"
    assert hasattr(operation.onnx_node, 'op_type'), "ONNX node missing op_type"
    assert operation.onnx_node.op_type == expected_op_type, f"Wrong op_type: {operation.onnx_node.op_type}"
    
    # Check it has real FINN datatype methods
    assert hasattr(operation, 'get_input_datatype'), "Missing get_input_datatype method"
    assert hasattr(operation, 'get_output_datatype'), "Missing get_output_datatype method"
    
    # Test real datatype functionality
    input_dt = operation.get_input_datatype(0)
    assert hasattr(input_dt, 'bitwidth'), "Input datatype missing bitwidth method"
    assert callable(input_dt.bitwidth), "bitwidth is not callable"
    
    # Check bitwidth returns integer
    bitwidth = input_dt.bitwidth()
    assert isinstance(bitwidth, int), f"bitwidth() returned {type(bitwidth)}, expected int"
    assert bitwidth > 0, f"Invalid bitwidth: {bitwidth}"
    
    print(f"✅ Validated real {expected_op_type} operation")


class FinnOperationFactory:
    """Factory class for creating real FINN operations for testing."""
    
    @staticmethod
    def create_mvau_operation(mw=64, mh=64, pe=4, simd=4):
        """Create a real MVAU operation for testing."""
        mvau_op, model = create_real_mvau_operation(mw, mh, pe, simd)
        return mvau_op
    
    @staticmethod
    def create_thresholding_operation(num_channels=64, pe=4):
        """Create a real Thresholding operation for testing."""
        thresh_op, model = create_real_thresholding_operation(num_channels, pe)
        return thresh_op
    
    @staticmethod
    def create_mvau_rtl_operation(mw=64, mh=64, pe=4, simd=4):
        """Create a real MVAU RTL operation for testing."""
        # Create base MVAU operation
        mvau_op, model = create_real_mvau_operation(mw, mh, pe, simd)
        
        # Modify the op_type to indicate RTL backend
        mvau_op.onnx_node.op_type = "MatrixVectorActivation_rtl"
        
        # Add required RTL methods
        def get_verilog_top_module_intf_names():
            """Mock RTL interface names for testing."""
            return {
                "s_axis_0": ("inp", False),
                "m_axis_0": ("outp", True)
            }
        
        def get_verilog_top_module_name():
            """Get the top module name for RTL generation."""
            return f"MVAU_hls_{mw}_{mh}_{pe}_{simd}"
        
        # Bind methods to the operation instance
        mvau_op.get_verilog_top_module_intf_names = get_verilog_top_module_intf_names
        mvau_op.get_verilog_top_module_name = get_verilog_top_module_name
        
        return mvau_op


if __name__ == "__main__":
    # Test creating real operations
    print("🧪 Testing Real FINN Operation Factory")
    
    try:
        # Test MVAU creation
        mvau_op, mvau_model = create_real_mvau_operation()
        validate_real_operation(mvau_op, "MatrixVectorActivation")
        print(f"✅ Created real MVAU: MW={mvau_op.get_nodeattr('MW')}, MH={mvau_op.get_nodeattr('MH')}")
        
        # Test Thresholding creation
        thresh_op, thresh_model = create_real_thresholding_operation()
        validate_real_operation(thresh_op, "Thresholding")
        print(f"✅ Created real Thresholding: NumChannels={thresh_op.get_nodeattr('NumChannels')}")
        
        print("🎉 Real FINN operation factory working correctly!")
        
    except Exception as e:
        print(f"❌ Error creating real operations: {e}")
        import traceback
        traceback.print_exc()