# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""TFC_W2A2 through the HWCustomOp flow's builder, for the TFC probes: the streamlined
capture (``tfc_streamlined``) through the builder's ``phase_convert_to_hardware`` and
``phase_optimize_hardware``, as a build for Ultra96 at 5 ns and 1,000,000 frames a
second with standalone thresholds runs them (SetFolding at 200 cycles a frame), its
estimate reports the only output (so no FIFO sizing)."""

from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)
from finn.builder.build_dataflow_phases import (
    phase_convert_to_hardware,
    phase_optimize_hardware,
)

BOARD = "Ultra96"
PERIOD_NS = 5.0
FPS = 1_000_000


def folded(captures, build):
    """The streamlined capture's dataflow partition, folded: the partition's model and
    the build's configuration."""
    cfg = DataflowBuildConfig(
        output_dir=str(build / "out"),
        synth_clk_period_ns=PERIOD_NS,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        board=BOARD,
        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
        target_fps=FPS,
        standalone_thresholds=True,
        auto_fifo_depths=False,
        save_intermediate_models=False,
    )
    model = ModelWrapper(str(captures / "tfc_w2a2_streamlined.onnx"))
    model = phase_convert_to_hardware(model, cfg)
    return phase_optimize_hardware(model, cfg), cfg
