# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.shells.pynq.iodma``: the pynq shell's IODMA, ``IODMA_hls``, a
finn-hlslib DMA between an AXI-MM port and an AXI stream (``direction`` "in": memory to
stream; "out": stream to memory), its stream width converted where its widths differ.

This is a frozen extraction of the HWCustomOp flow's ``IODMA_hls``
(finn.custom_op.fpgadataflow.hls.iodma_hls) with only what the shell's runner reads of
it from ``HWCustomOp`` and ``HLSBackend``: its widths, its HLS code and Vitis HLS script
(``code_generation_ipgen``), its synthesis (``ipgen_singlenode_code``), its block-design
cell and its interface names. It generates the code that op generates, byte for byte
(``tests/kernel_ops/test_builder_phase.py``'s Z0 captures), and changes only when the
IODMA becomes an HLS kernel. Each end's IP is generated from a scratch model of one node
of this op (``runner.iodma_model``, ``ipgen``).

qonnx resolves this domain by importing it and reads its op classes from ``__all__``.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import numpy as np
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import roundup_to_integer_multiple

from finn import resources
from finn.shells.pynq import templates
from finn.util.hls import CallHLS
from finn.util.resources import tcl_quote
from finn.util.toolchain import Toolchain

opset_version = 1


class IODMA_hls(CustomOp):  # noqa: N801 (the op type, as the HWCustomOp flow's)
    """One IODMA: ``NumChannels`` elements of ``dataType`` a vector, ``numInputVectors``
    vectors a frame, moved between an AXI-MM port of ``intfWidth`` bits (``burstMode``
    "increment" or "wrap") and a stream of ``streamWidth`` bits."""

    def __init__(self, onnx_node: Any, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict: dict[str, list[str]] = {}

    def get_nodeattr_types(self) -> dict[str, Any]:
        return {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # Width of input or output stream
            "streamWidth": ("i", False, 32),
            # DMA-specific parameters
            # width of axi-mm interface
            "intfWidth": ("i", False, 32),
            # burst mode for axi-mm interface (wrap used for DRAM weights)
            "burstMode": ("s", False, "increment", {"wrap", "increment"}),
            # IODMA direction: in = read from DRAM, out = write to DRAM
            "direction": ("s", False, "in", {"in", "out"}),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
            # name of axi-mm interface
            "intfName": ("s", False, ""),
            # what IP generation made: the code, the HLS project, its IP and VLNV
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
        }

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        vecs = list(self.get_nodeattr("numInputVectors"))
        num_ch = self.get_nodeattr("NumChannels")
        ishape = tuple(vecs + [num_ch])
        return ishape

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        return self.get_normal_input_shape()

    def make_shape_compatible_op(self, model: ModelWrapper) -> Any:
        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        pass

    def verify_node(self) -> list[str]:
        return []

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_instream_width(self, ind: int = 0) -> int:
        if self.get_nodeattr("direction") == "in":
            return int(self.get_nodeattr("intfWidth"))
        elif self.get_nodeattr("direction") == "out":
            return int(self.get_nodeattr("streamWidth"))
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_outstream_width(self, ind: int = 0) -> int:
        if self.get_nodeattr("direction") == "out":
            return int(self.get_nodeattr("intfWidth"))
        elif self.get_nodeattr("direction") == "in":
            return int(self.get_nodeattr("streamWidth"))
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_instream_width_padded(self, ind: int = 0) -> int:
        """Returns input stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        in_width = self.get_instream_width(ind=ind)
        if in_width != 0:
            return int(roundup_to_integer_multiple(in_width, 8))
        else:
            return 0

    def get_outstream_width_padded(self, ind: int = 0) -> int:
        """Returns output stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        out_width = self.get_outstream_width(ind=ind)
        return int(roundup_to_integer_multiple(out_width, 8))

    def get_ap_int_max_w(self) -> int:
        "Return the maximum width of any ap_int used in this module."
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        width_lcm = (instream * outstream) // math.gcd(instream, outstream)
        return width_lcm

    def global_includes(self) -> None:
        self.code_gen_dict["$GLOBALS$"] = ['#include "dma.h"']
        self.code_gen_dict["$GLOBALS$"].append('#include "streamtools.h"')

    def defines(self, var: str) -> None:
        itype_bits = self.get_input_datatype().bitwidth()
        total_bits = itype_bits * np.prod(self.get_normal_input_shape())
        assert total_bits % 8 == 0, "DMA input not a multiple of 1 Byte"
        total_bytes = total_bits // 8
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumBytes1 {}\n#define DataWidth1 {}\n""".format(
                total_bytes, self.get_nodeattr("intfWidth")
            )
        ]

    def docompute(self) -> None:
        direction = self.get_nodeattr("direction")
        mode = self.get_nodeattr("burstMode")
        dwc_func = "StreamingDataWidthConverter_Batch"
        if direction == "in":
            if mode == "wrap":
                func = "Mem2Stream_Batch_external_wmem"
            else:
                func = "Mem2Stream_Batch"
        elif direction == "out":
            func = "Stream2Mem_Batch"
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")
        # define templates for instantiation
        dma_inst_template = func + "<DataWidth1, NumBytes1>(%s, %s, numReps);"
        dwc_inst_template = dwc_func + "<%d, %d, %d>(%s, %s, numReps);"
        # do stream infrastructure and instantiations
        intfw = self.get_nodeattr("intfWidth")
        strmw = self.get_nodeattr("streamWidth")
        width_lcm = (strmw * intfw) // math.gcd(strmw, intfw)
        # we always need two streams: one of width_lcm, and one of intfw width
        # because we use WidthAdjustedInputStream,
        dtype_bits = self.get_input_datatype().bitwidth()
        total_bits = dtype_bits * np.prod(self.get_normal_input_shape())

        if direction == "in":
            # AXI MM -> IODMA -> (DWCs) -> out
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: AXI MM width = out width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by out width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2dwc;" % intfw,
                    dma_inst_template % ("in0_V", "dma2dwc"),
                    dwc_inst_template
                    % (
                        intfw,
                        strmw,
                        total_bits // intfw,
                        "dma2dwc",
                        "out0_V",
                    ),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2lcm;" % intfw,
                    "hls::stream<ap_uint<%d> > lcm2out;" % width_lcm,
                    dma_inst_template % ("in0_V", "dma2lcm"),
                    dwc_inst_template
                    % (intfw, width_lcm, total_bits // intfw, "dma2lcm", "lcm2out"),
                    dwc_inst_template
                    % (
                        width_lcm,
                        strmw,
                        total_bits // width_lcm,
                        "lcm2out",
                        "out0_V",
                    ),
                ]
        elif direction == "out":
            # in0 -> (DWCs) -> IODMA -> AXI MM
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: in width = AXI MM width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by in width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dwc2dma;" % intfw,
                    dwc_inst_template
                    % (
                        strmw,
                        intfw,
                        total_bits // strmw,
                        "in0_V",
                        "dwc2dma",
                    ),
                    dma_inst_template % ("dwc2dma", "out0_V"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > in2lcm;" % width_lcm,
                    "hls::stream<ap_uint<%d> > lcm2dma;" % intfw,
                    dwc_inst_template
                    % (
                        strmw,
                        width_lcm,
                        total_bits // strmw,
                        "in0_V",
                        "in2lcm",
                    ),
                    dwc_inst_template
                    % (width_lcm, intfw, total_bits // width_lcm, "in2lcm", "lcm2dma"),
                    dma_inst_template % ("lcm2dma", "out0_V"),
                ]
        else:
            raise Exception("Unknown IODMA direction: %s" % direction)

    def blackboxfunction(self) -> None:
        packed_ibits = self.get_instream_width()
        packed_hls_type_in = "ap_uint<%d>" % packed_ibits
        packed_obits = self.get_outstream_width()
        packed_hls_type_out = "ap_uint<%d>" % packed_obits
        direction = self.get_nodeattr("direction")
        if direction == "in":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(%s *in0_V, hls::stream<%s > &out0_V, unsigned int numReps)"
                % (
                    self.onnx_node.name,
                    packed_hls_type_in,
                    packed_hls_type_out,
                )
            ]
        elif direction == "out":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0_V, %s *out0_V, unsigned int numReps)"
                % (
                    self.onnx_node.name,
                    packed_hls_type_in,
                    packed_hls_type_out,
                )
            ]
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def pragmas(self) -> None:
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE s_axilite port=numReps bundle=control"
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=return bundle=control"
        )
        direction = self.get_nodeattr("direction")
        intfname = self.get_nodeattr("intfName")
        if direction == "in":
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=in0_V"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=in0_V bundle=control"
            )
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        elif direction == "out":
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in0_V")
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=out0_V"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=out0_V bundle=control"
            )
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW")

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generates c++ code and tcl script for ip generation."""
        node = self.onnx_node

        # generate top cpp file for ip generation
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.global_includes()
        self.defines("ipgen")
        self.blackboxfunction()
        self.pragmas()
        self.docompute()

        template = templates.ipgen_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, "top_{}.cpp".format(node.name)), "w") as f:
            f.write(template)
        self.code_gen_dict.clear()

        # generate tcl script for ip generation
        self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
        self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
        self.code_gen_dict["$FPGAPART$"] = [fpgapart]
        self.code_gen_dict["$TOPFXN$"] = [node.name]
        self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]
        self.code_gen_dict["$DEFAULT_DIRECTIVES$"] = self.ipgen_default_directives()
        self.code_gen_dict["$EXTRA_DIRECTIVES$"] = []

        template = templates.ipgentcl_template.replace(
            "$HLSLIB$", tcl_quote(resources.path("hlslib"))[1:-1]
        )

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        with open(os.path.join(code_gen_dir, "hls_syn_{}.tcl".format(node.name)), "w") as f:
            f.write(template)
        self.code_gen_dict.clear()

    def ipgen_default_directives(self) -> list[str]:
        """Return list of default HLS synthesis directives"""

        default_directives = [
            "set_param hls.enable_hidden_option_error false",
            "config_compile -disable_unroll_code_size_check -pipeline_style flp",
            "config_interface -m_axi_addr64",
            "config_rtl -module_auto_prefix",
            "config_rtl -deadlock_detection none",
        ]
        return default_directives

    def ipgen_singlenode_code(self, toolchain: Toolchain | None = None) -> None:
        """Builds the bash script for IP generation using the CallHLS utility, and runs
        it in ``toolchain`` (by default the machine's)."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        builder = CallHLS(toolchain=toolchain)
        builder.append_tcl(code_gen_dir + "/hls_syn_{}.tcl".format(node.name))
        builder.set_ipgen_path(code_gen_dir + "/project_{}".format(node.name))
        builder.build(code_gen_dir)
        ipgen_path = builder.ipgen_path
        assert os.path.isdir(ipgen_path), "IPGen failed: %s not found" % (ipgen_path)
        self.set_nodeattr("ipgen_path", ipgen_path)
        ip_path = ipgen_path + "/sol1/impl/ip"
        assert os.path.isdir(ip_path), "IPGen failed: %s not found. Check log under %s" % (
            ip_path,
            code_gen_dir,
        )
        self.set_nodeattr("ip_path", ip_path)
        vlnv = "xilinx.com:hls:%s:1.0" % node.name
        self.set_nodeattr("ip_vlnv", vlnv)

    def code_generation_ipi(self) -> list[str]:
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def get_verilog_top_module_intf_names(self) -> dict[str, list[Any]]:
        """The names of the HLS top's interfaces by protocol ('clk', 'rst', 'm_axis',
        's_axis', 'aximm', 'axilite', 'ap_none'): each stream and memory port as
        ``(name, width in bits)``, each AXI-Lite port by name."""
        node = self.onnx_node
        intf_names: dict[str, list[Any]] = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        for i in range(len(node.input)):
            # not every node input will result in an interface of the produced HW
            # filter out inputs that have no stream width associated with them
            width = self.get_instream_width_padded(i)
            if width != 0:
                intf_names["s_axis"].append(("in%d_V" % (i), self.get_instream_width_padded(i)))
        intf_names["m_axis"] = []
        for i in range(len(node.output)):
            intf_names["m_axis"].append(("out%d_V" % (i), self.get_outstream_width_padded(i)))
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        if self.get_nodeattr("direction") == "out":
            intf_names["m_axis"] = []
        else:
            intf_names["s_axis"] = []
        intf_names["axilite"] = ["s_axi_control"]
        intf_names["aximm"] = [("m_axi_gmem", self.get_nodeattr("intfWidth"))]
        return intf_names


__all__ = ["IODMA_hls"]
