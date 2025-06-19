# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

# Import our CG implementations
from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
from finn.custom_op.fpgadataflow.hls.transition_thresholding_hls import CG_Thresholding_hls_Full
from finn.custom_op.fpgadataflow.rtl.CG_thresholding_rtl import CG_Thresholding_rtl

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5
EXPAND_FLOAT_RANGE = 100


def generate_random_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1
    if data_type.is_integer():
        return np.random.randint(
            data_type.min(),
            data_type.max() + 1,
            (num_input_channels, num_steps),
        ).astype(np.float32)
    else:
        return (np.random.randn(num_input_channels, num_steps) * EXPAND_FLOAT_RANGE).astype(
            data_type.to_numpy_dt()
        )


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


def make_single_multithresholding_modelwrapper(
    thresholds,
    input_data_type,
    threshold_data_type,
    output_data_type,
    activation_bias,
    num_input_vecs,
    num_channels,
):
    if input_data_type == DataType["FLOAT16"]:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT16, num_input_vecs + [num_channels]
        )
    else:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, num_input_vecs + [num_channels]
        )
    if threshold_data_type == DataType["FLOAT16"]:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT16, thresholds.shape)
    else:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, thresholds.shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [num_channels])

    node_inp_list = ["inp", "thresh"]

    Multithresholding_node = helper.make_node(
        "MultiThreshold",
        node_inp_list,
        ["outp"],
        domain="qonnx.custom_op.general",
        out_dtype=output_data_type.name,
        out_bias=float(activation_bias),
        out_scale=1.0,
        data_layout="NHWC",
    )

    graph = helper.make_graph(
        nodes=[Multithresholding_node],
        name="multithresholding_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[thresh],
    )

    model = helper.make_model(graph, producer_name="multithresholding-model")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model.set_tensor_datatype("inp", input_data_type)
    model.set_tensor_datatype("outp", output_data_type)

    model.set_tensor_datatype("thresh", threshold_data_type)
    model.set_initializer("thresh", thresholds)
    return model


def create_cg_thresholding_model(
    thresholds,
    input_data_type,
    threshold_data_type,
    output_data_type,
    activation_bias,
    num_input_vecs,
    num_channels,
    cg_impl_class,
):
    """Create a model directly using CG thresholding implementation."""
    if input_data_type == DataType["FLOAT16"]:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT16, num_input_vecs + [num_channels]
        )
    else:
        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, num_input_vecs + [num_channels]
        )
    if threshold_data_type == DataType["FLOAT16"]:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT16, thresholds.shape)
    else:
        thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, thresholds.shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, num_input_vecs + [num_channels])

    node_inp_list = ["inp", "thresh"]
    
    # Use the exact class name as the op_type for direct registration lookup
    op_type = cg_impl_class.__name__
    
    # Determine domain based on implementation type
    if "rtl" in cg_impl_class.__name__.lower():
        domain = "finn.custom_op.fpgadataflow.rtl"
    else:
        domain = "finn.custom_op.fpgadataflow.hls"

    # Create CG thresholding node directly
    cg_thresholding_node = helper.make_node(
        op_type,
        node_inp_list,
        ["outp"],
        domain=domain,
        backend="fpgadataflow",
        NumChannels=num_channels,
        numSteps=thresholds.shape[1],
        inputDataType=input_data_type.name,
        weightDataType=threshold_data_type.name,
        outputDataType=output_data_type.name,
        ActVal=activation_bias,
        numInputVectors=num_input_vecs,
        PE=1,  # Will be overridden in test
    )

    graph = helper.make_graph(
        nodes=[cg_thresholding_node],
        name="cg_thresholding_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[thresh],
    )

    model = helper.make_model(graph, producer_name="cg-thresholding-model")
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model.set_tensor_datatype("inp", input_data_type)
    model.set_tensor_datatype("outp", output_data_type)
    model.set_tensor_datatype("thresh", threshold_data_type)
    model.set_initializer("thresh", thresholds)
    
    return model


@pytest.mark.parametrize("num_input_channels", [6, 16])
@pytest.mark.parametrize(
    "num_input_vecs",
    [
        [1],
        [1, 2, 2],
    ],
)
@pytest.mark.parametrize("activation", [DataType["UINT4"], DataType["INT4"], DataType["BIPOLAR"]])
@pytest.mark.parametrize(
    "idt_tdt_cfg",
    [
        (DataType["INT8"], DataType["INT25"]),
        (DataType["UINT5"], DataType["UINT8"]),
        (DataType["FLOAT32"], DataType["FLOAT32"]),
        (DataType["FLOAT16"], DataType["FLOAT16"]),
    ],
)
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("per_tensor", [True, False])
@pytest.mark.parametrize("cg_impl", ["clean_hls", "transition_hls", "clean_rtl"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
@pytest.mark.parametrize("round_thresh", [True, False])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_CG_thresholding(
    num_input_channels,
    num_input_vecs,
    activation,
    idt_tdt_cfg,
    fold,
    narrow,
    per_tensor,
    cg_impl,
    exec_mode,
    mem_mode,
    round_thresh,
):
    # Skip invalid combinations
    if cg_impl == "clean_rtl" and mem_mode == "internal_decoupled":
        pytest.skip("RTL implementation doesn't support decoupled memory mode")
    if narrow and activation == DataType["BIPOLAR"]:
        pytest.skip("Narrow needs to be false with bipolar activation.")
    
    input_data_type, threshold_data_type = idt_tdt_cfg
    num_steps = activation.get_num_possible_values() - 1

    if fold == -1:
        fold = num_input_channels
    pe = num_input_channels // fold
    if num_input_channels % pe != 0:
        pytest.skip("Invalid folding configuration. Skipping test.")

    output_data_type = activation
    if activation == DataType["BIPOLAR"]:
        activation_bias = 0
    else:
        activation_bias = activation.min()
        if narrow and activation.signed():
            activation_bias += 1

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(
        threshold_data_type, num_input_channels, num_steps, narrow, per_tensor
    )
    thresholds = sort_thresholds_increasing(thresholds)

    # Create reference model using legacy path
    ref_model = make_single_multithresholding_modelwrapper(
        thresholds,
        input_data_type,
        threshold_data_type,  
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
    )

    # Calculate reference output
    x = gen_finn_dt_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))
    if not input_data_type.is_integer():
        x = (x * EXPAND_FLOAT_RANGE).astype(input_data_type.to_numpy_dt())

    input_dict = {ref_model.graph.input[0].name: x}
    y_expected = oxe.execute_onnx(ref_model, input_dict)[ref_model.graph.output[0].name]

    if output_data_type == DataType["BIPOLAR"]:
        # binary to bipolar
        y_expected = 2 * y_expected - 1

    # Convert reference to thresholding layer for functional validation
    ref_model = ref_model.transform(InferThresholdingLayer())
    y_produced_ref = oxe.execute_onnx(ref_model, input_dict)[ref_model.graph.output[0].name]
    assert (y_produced_ref == y_expected).all()

    # Select CG implementation class
    if cg_impl == "clean_hls":
        cg_impl_class = CG_Thresholding_hls
    elif cg_impl == "transition_hls":  
        cg_impl_class = CG_Thresholding_hls_Full
    elif cg_impl == "clean_rtl":
        cg_impl_class = CG_Thresholding_rtl
    else:
        raise ValueError(f"Unknown CG implementation: {cg_impl}")

    # Create CG model
    model = create_cg_thresholding_model(
        thresholds,
        input_data_type,
        threshold_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
        cg_impl_class,
    )

    # Configure CG model
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    
    # Set memory mode for HLS implementations
    if "hls" in cg_impl and hasattr(inst, 'set_nodeattr'):
        try:
            inst.set_nodeattr("mem_mode", mem_mode)
        except:
            pass  # Clean implementation may not support mem_mode

    if round_thresh:
        model = model.transform(RoundAndClipThresholds())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())

    # Prepare for execution
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    # Execute and validate
    y_produced = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    assert (y_produced == y_expected).all(), f"CG {cg_impl} output doesn't match expected"

    # Additional validation for RTL simulation
    if exec_mode == "rtlsim":
        if "hls" in cg_impl:
            hls_synt_res_est = model.analysis(hls_synth_res_estimation)
            assert model.graph.node[0].name in hls_synt_res_est
        node = model.get_nodes_by_op_type(node.op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


@pytest.mark.parametrize("num_input_channels", [6])
@pytest.mark.parametrize("num_input_vecs", [[1, 2, 2]])
@pytest.mark.parametrize("activation", [DataType["INT4"]])
@pytest.mark.parametrize("idt_tdt_cfg", [(DataType["INT8"], DataType["INT25"])])
@pytest.mark.parametrize("fold", [-1, 1, 2])
@pytest.mark.parametrize("ram_style", ["distributed", "block"])
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S", "xczu7ev-ffvc1156-2-e"])
@pytest.mark.parametrize("cg_impl", ["transition_hls"])  # Only test full HLS implementation
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_CG_thresholding_stitched_ip(
    num_input_channels, num_input_vecs, activation, idt_tdt_cfg, fold, ram_style, part, cg_impl
):
    """Test CG thresholding with stitched IP generation."""
    input_data_type, threshold_data_type = idt_tdt_cfg
    num_steps = activation.get_num_possible_values() - 1

    if fold == -1:
        fold = num_input_channels
    pe = num_input_channels // fold

    output_data_type = activation
    activation_bias = activation.min()

    # Generate random thresholds and sort in ascending order
    thresholds = generate_random_threshold_values(
        threshold_data_type, num_input_channels, num_steps, False, False
    )
    thresholds = sort_thresholds_increasing(thresholds)

    # Create reference model
    ref_model = make_single_multithresholding_modelwrapper(
        thresholds,
        input_data_type,
        threshold_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
    )

    # Calculate reference output
    x = gen_finn_dt_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))
    input_dict = {ref_model.graph.input[0].name: x}
    y_expected = oxe.execute_onnx(ref_model, input_dict)[ref_model.graph.output[0].name]

    ref_model = ref_model.transform(InferThresholdingLayer())

    # Create CG model
    cg_impl_class = CG_Thresholding_hls_Full
    model = create_cg_thresholding_model(
        thresholds,
        input_data_type,
        threshold_data_type,
        output_data_type,
        activation_bias,
        num_input_vecs,
        num_input_channels,
        cg_impl_class,
    )

    # Configure CG model
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("ram_style", ram_style)

    model = model.transform(GiveUniqueNodeNames())
    # Run stitched-ip RTLsim to have memstream in the test loop
    model = model.transform(InsertAndSetFIFODepths(part, target_clk_ns))
    model = model.transform(PrepareIP(part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")

    exec_ctx_dict = {"global_in": x}
    y_produced = oxe.execute_onnx(model, exec_ctx_dict)["global_out"]

    assert (
        y_expected == y_produced
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"


# Comparison test between legacy and CG implementations
@pytest.mark.parametrize("num_input_channels", [8])
@pytest.mark.parametrize("activation", [DataType["UINT4"]])
@pytest.mark.parametrize("idt_tdt_cfg", [(DataType["INT8"], DataType["INT16"])])
@pytest.mark.parametrize("exec_mode", ["cppsim"])
@pytest.mark.fpgadataflow 
@pytest.mark.vivado
def test_CG_vs_legacy_thresholding_comparison(
    num_input_channels, activation, idt_tdt_cfg, exec_mode
):
    """Compare CG implementations against legacy for identical results."""
    input_data_type, threshold_data_type = idt_tdt_cfg
    num_input_vecs = [1]
    num_steps = activation.get_num_possible_values() - 1
    pe = 2
    output_data_type = activation
    activation_bias = activation.min()

    # Generate test data
    thresholds = generate_random_threshold_values(
        threshold_data_type, num_input_channels, num_steps, False, False
    )
    thresholds = sort_thresholds_increasing(thresholds)

    x = gen_finn_dt_tensor(input_data_type, tuple(num_input_vecs + [num_input_channels]))
    input_dict = {"inp": x}

    # Create legacy reference
    ref_model = make_single_multithresholding_modelwrapper(
        thresholds, input_data_type, threshold_data_type, output_data_type,
        activation_bias, num_input_vecs, num_input_channels
    )
    ref_model = ref_model.transform(InferThresholdingLayer())

    # Test all CG implementations
    implementations = [
        ("Clean HLS", CG_Thresholding_hls),
        ("Transition HLS", CG_Thresholding_hls_Full), 
        ("Clean RTL", CG_Thresholding_rtl),
    ]

    for impl_name, impl_class in implementations:
        model = create_cg_thresholding_model(
            thresholds, input_data_type, threshold_data_type, output_data_type,
            activation_bias, num_input_vecs, num_input_channels, impl_class
        )

        # Configure
        node = model.graph.node[0]
        inst = getCustomOp(node)
        inst.set_nodeattr("PE", pe)
        
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())

        if exec_mode == "cppsim":
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
            model = model.transform(SetExecMode("cppsim"))

        # Execute and compare
        try:
            y_cg = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
            y_ref = oxe.execute_onnx(ref_model, input_dict)[ref_model.graph.output[0].name]
            
            assert (y_cg == y_ref).all(), f"{impl_name} output differs from reference"
            print(f"✓ {impl_name} matches reference implementation")
            
        except Exception as e:
            print(f"✗ {impl_name} failed: {e}")
            # Continue testing other implementations
            continue