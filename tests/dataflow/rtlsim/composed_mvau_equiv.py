############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 5: composed replay + dot product is bit-equivalent to the fused MVAU.

Runs INSIDE the FINN Docker container; needs Vivado's ``xelab`` and ``finn_xsi``.

The decomposition claims that ``mvu_vvu_axi`` is an activation replay followed
by a dot-product core, and that naming the two halves separately changes
nothing observable.  This drives both with the same random stream and requires
bit-identical output.

**The composed DUT is not written here.**  It comes out of the production path:
a real ``MvauDataflowOp`` over a real graph selects the decomposed Kernel, the
provider elaborates it, and ``write_decomposed_artifact`` emits the Verilog and
the source manifest.  Anything this fixture proves is therefore a property of
what the compiler actually builds, not of a lookalike assembled beside it.  The
*fused* top is written here, because that one is the oracle.

Both DUTs are compared to each other, not to a numeric golden, so the stimulus
is raw random integers packed to the stream widths.  Any value reaches both
identically; a mismatch means the two disagree -- and this fixture cannot say
which of them is right.  Fixture 8 can, and once had to.

**The stimulus must keep the promises the design point makes.**  Adding DSP48E1
in Phase 6e made this fail, and the cause was here rather than in any core: the
declared ``NARROW_WEIGHTS`` is derived from the model's weight *initializer*,
while the streamed weights were an independent random draw that could carry the
datatype minimum.  ``mvu.sv`` packs only ``WEIGHT_WIDTH-1`` magnitude bits under
that promise and warns when it is broken; FinnLib's ``dotp`` tolerates it.  Two
cores differing on a value neither is obliged to handle is not a result.
Weights are now drawn lane by lane in the declared range.

Baseline FINN cannot reach that state at all -- ``specialize_layers`` refuses
the RTL MVAU for non-narrow weights on DSP48E1 -- and the production wrapper
computes correctly on this design point, which is what settled it.

**Each simulation runs in its own process**, which is ``rtl_transport``'s job
and no longer this file's.  That module holds the marshalling, the
backpressure collector and the watchdog handling, so fixture 8 -- which drives
the same composed RTL against arithmetic instead of against the fused core --
reuses the transport without reusing this comparison.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_equiv.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign
from finn.dataflow.ops.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.binding import finnlib_root
from finn.dataflow.ops.mvau.artifacts._implementation import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    write_decomposed_artifact,
)
from finn.dataflow.ops.mvau.elaboration import elaborate_mvau
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.input_supply import EXTERNAL_SUPPLY
from finn.dataflow.ops.mvau.op import MVAUDataflowBuildContext, MvauDataflowOp

from dataflow.rtlsim.rtl_transport import drive, random_word

#: The fused golden's sources, relative to ``finn-rtllib/mvu``.
FUSED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
)

ACCU_WIDTH = 16
CLOCK_PERIOD_NS = 4.0


def finn_root() -> str:
    """The FINN checkout, from the environment or from where this file sits.

    The container sets ``FINN_ROOT``; a plain pytest run does not, and the
    parts of this harness that need no simulator should still work there.
    """

    return os.environ.get("FINN_ROOT") or str(Path(__file__).resolve().parents[3])


@dataclass(frozen=True)
class Config:
    """One design point to compare, named the way the design space names it."""

    label: str
    target: DspBlock
    repetitions: int
    matrix_width: int
    matrix_height: int
    pe: int
    simd: int
    pumping: bool = False
    activation_bits: int = 8
    weight_bits: int = 8
    #: A specific device, when the default one for this DSP generation is not
    #: the point.  Two parts of the same generation take the same parameters
    #: and therefore the same generated source, which is what makes cross-part
    #: reuse demonstrable at all.
    part_override: str | None = None

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe

    @property
    def fpga_part(self) -> str:
        return self.part_override or _PART_FOR_TARGET[self.target]


#: A part per DSP family, so the design point's target is the real target and
#: not an assertion the fixture makes on the side.
_PART_FOR_TARGET = {
    DspBlock.DSP48E1: "xc7z020clg400-1",
    DspBlock.DSP48E2: "xczu3eg-sbva484-1-e",
    DspBlock.DSP58: "xcvc1902-vsva2197-2MP-e-S",
}

#: DSP48E2 exercises the soft-vector core, DSP58 the packed one.  The last
#: three cover several repetitions (so the replay buffer has to reset between
#: frames) and pumped compute.
CONFIGS = [
    Config("softvec", DspBlock.DSP48E2, 1, 4, 4, 2, 2),
    Config("packed", DspBlock.DSP58, 1, 4, 4, 2, 2),
    Config("one_neuron_fold", DspBlock.DSP58, 1, 4, 4, 4, 2),
    Config("one_synapse_fold", DspBlock.DSP58, 1, 4, 4, 2, 4),
    Config("three_repetitions", DspBlock.DSP58, 3, 4, 4, 2, 2),
    Config("repetitions_softvec", DspBlock.DSP48E2, 2, 8, 6, 3, 2),
    Config("pumped", DspBlock.DSP58, 2, 8, 4, 2, 4, pumping=True),
    # DSP48E1 was in the part table from the start and in no configuration, so
    # the oldest generation this Kernel covers had never been simulated at all.
    # Phase 4 corrected the narrow-weight rule in both directions by reading
    # ``sliceLanes()`` and recorded that it rested on arithmetic rather than on
    # a measurement; this is where that is paid.
    #
    # Running it found a defect in this fixture rather than in either core: the
    # weight stimulus did not honour the NARROW_WEIGHTS the point declares.
    # See ``_random_weight_beat``.
    Config("dsp48e1", DspBlock.DSP48E1, 2, 8, 4, 2, 2, activation_bits=4),
]

CONFIGS_BY_LABEL = {config.label: config for config in CONFIGS}


# -- the production path -----------------------------------------------------


class _BuildConfig:
    """The narrow slice of a build configuration the MVAU operation reads."""

    def __init__(self, part: str) -> None:
        self.synth_clk_period_ns = CLOCK_PERIOD_NS
        self.fpga_part = part

    def _resolve_fpga_part(self) -> str:
        return self.fpga_part


def _model(config: Config) -> ModelWrapper:
    """A one-node graph whose shapes and types are the configuration."""

    node_id = f"mvau_{config.label}"
    node = helper.make_node(
        "MvauDataflowOp",
        ["activation", "weights"],
        ["output"],
        name=node_id,
        domain="finn.custom_op.dataflow",
        dataflow_scope_id=f"{node_id}_scope",
        accDataType=f"INT{ACCU_WIDTH}",
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [node],
                f"composed-{config.label}",
                [
                    helper.make_tensor_value_info(
                        "activation",
                        TensorProto.FLOAT,
                        [config.repetitions, config.matrix_width],
                    ),
                    helper.make_tensor_value_info(
                        "weights",
                        TensorProto.FLOAT,
                        [config.matrix_width, config.matrix_height],
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "output",
                        TensorProto.FLOAT,
                        [config.repetitions, config.matrix_height],
                    )
                ],
            ),
            producer_name="fixture5",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType[f"INT{config.activation_bits}"])
    model.set_tensor_datatype("weights", DataType[f"INT{config.weight_bits}"])
    model.set_tensor_datatype("output", DataType[f"INT{ACCU_WIDTH}"])
    generator = np.random.RandomState(1)
    model.set_initializer(
        "weights",
        generator.randint(
            -(2 ** (config.weight_bits - 1)),
            2 ** (config.weight_bits - 1),
            size=(config.matrix_width, config.matrix_height),
        ).astype(np.float32),
    )
    return model


def decomposed_requirements(config: Config) -> MVAUDecomposedArtifactRequirements:
    """Everything the compiler says this configuration should be built from.

    Straight through the production path: select the decomposed Kernel on a
    real operation, elaborate through the declared provider, and build the
    requirements.  Nothing about the RTL is decided here.
    """

    model = _model(config)
    context = MVAUDataflowBuildContext(_BuildConfig(config.fpga_part))
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, MvauDataflowOp)
    operation.initialize_dataflow_scope_id()
    assert MVAU_DESIGN_INVENTORY.inventory.design_path is not None
    operation.commit_dataflow_assignments(
        context,
        {
            MVAU_DESIGN_INVENTORY.inventory.design_path: DotProductDesign.id,
            MVAU_DESIGN_INVENTORY.dot_product.pe.path: config.pe,
            MVAU_DESIGN_INVENTORY.dot_product.simd.path: config.simd,
            MVAU_DESIGN_INVENTORY.compute_pumping.path: config.pumping,
            MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: EXTERNAL_SUPPLY,
        },
    )
    resolved = operation.resolve_dataflow(context)
    root = finn_root()
    return build_decomposed_artifact_requirements(
        resolved,
        elaborate_mvau(resolved),
        root,
        finnlib_root(root),
    )


def declared_parameters(config: Config) -> dict[str, object]:
    """The RTL parameter values the design point declares, by name."""

    return dict(decomposed_requirements(config).parameters)


# -- the fused oracle --------------------------------------------------------


def _verilog(value: object) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def _fused_top(name: str, config: Config, values: Mapping[str, object]) -> str:
    """The golden: one fused wrapper, driven by the same declared values.

    ``IS_MVU``, ``MW`` and ``MH`` are the fused wrapper's own parameters -- it
    needs the geometry to size the replay it contains.  They are not in the
    declared dot-product parameter set, which is the decomposition showing up
    in the parameter list.
    """

    weight_bits = config.pe * config.simd * config.weight_bits
    input_bits = config.simd * config.activation_bits
    output_bits = config.pe * ACCU_WIDTH
    shared = ",\n        ".join(
        f".{key}({_verilog(values[key])})"
        for key in (
            "VERSION",
            "PE",
            "SIMD",
            "SEGMENTLEN",
            "ACTIVATION_WIDTH",
            "WEIGHT_WIDTH",
            "ACCU_WIDTH",
            "NARROW_WEIGHTS",
            "SIGNED_ACTIVATIONS",
            "PUMPED_COMPUTE",
            "FORCE_BEHAVIORAL",
        )
    )
    return f"""
module {name} #(
    parameter WSTREAM = {(weight_bits + 7) // 8 * 8},
    parameter ISTREAM = {(input_bits + 7) // 8 * 8},
    parameter OSTREAM = {(output_bits + 7) // 8 * 8}
)(
    input  logic ap_clk,
    input  logic ap_clk2x,
    input  logic ap_rst_n,
    input  logic [WSTREAM-1:0] in1_V_tdata,
    input  logic in1_V_tvalid,
    output logic in1_V_tready,
    input  logic [ISTREAM-1:0] in0_V_tdata,
    input  logic in0_V_tvalid,
    output logic in0_V_tready,
    output logic [OSTREAM-1:0] out0_V_tdata,
    output logic out0_V_tvalid,
    input  logic out0_V_tready
);
    mvu_vvu_axi #(
        .IS_MVU(1),
        .MW({config.matrix_width}), .MH({config.matrix_height}),
        {shared}
    ) core (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(in0_V_tdata),
        .s_axis_input_tvalid(in0_V_tvalid),
        .s_axis_input_tready(in0_V_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


# -- staging -----------------------------------------------------------------


def _write(directory: str, name: str, text: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _sources(root: str, subdirectory: str, names: tuple[str, ...]) -> list[str]:
    resolved = []
    for name in names:
        path = os.path.join(root, subdirectory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing RTL source: {path}")
        resolved.append(path)
    return resolved


def _random_weight_beat(generator: np.random.RandomState, config: Config, *, narrow: bool) -> int:
    """One ``PE * SIMD`` weight beat, drawn per lane in the declared range.

    ``narrow`` excludes the datatype minimum, which is exactly what
    ``NARROW_WEIGHTS`` promises the core.  A beat assembled from opaque random
    bits cannot make that promise, and fixture 5 spent its whole life declaring
    it and not keeping it.
    """

    width = config.weight_bits
    low = -(2 ** (width - 1)) + (1 if narrow else 0)
    high = 2 ** (width - 1) - 1
    beat = 0
    for lane in range(config.pe * config.simd):
        value = int(generator.randint(low, high + 1))
        beat |= (value & ((1 << width) - 1)) << (lane * width)
    return beat


def run_one(config: Config, finn_root: str) -> bool:
    print(f"\n========== fixture 5: {config.label} ==========")
    requirements = decomposed_requirements(config)
    values = dict(requirements.parameters)
    synapse_folds, neuron_folds = config.synapse_folds, config.neuron_folds
    passes = config.repetitions
    expected = passes * neuron_folds

    generator = np.random.RandomState(0)
    activation = [
        random_word(generator, config.simd * config.activation_bits)
        for _ in range(passes * synapse_folds)
    ]
    # Weights are drawn *lane by lane*, not as an opaque word, so the stimulus
    # can honour the ``NARROW_WEIGHTS`` this design point declares.
    #
    # It did not, and that is what the DSP48E1 configuration exposed.  The flag
    # is derived from the model's weight *initializer*, while the streamed bits
    # were an independent random draw -- so a beat could carry the datatype
    # minimum after the point had promised it never would.  ``mvu.sv`` packs
    # only ``WEIGHT_WIDTH-1`` magnitude bits under that promise and warns
    # "violates NARROW_WEIGHTS commitment"; the FinnLib core happens to
    # tolerate it.  The two then differ on a value neither is obliged to
    # handle, which is a difference in undefined behaviour and not a result.
    #
    # Baseline FINN cannot even reach that state: ``specialize_layers`` refuses
    # the RTL MVAU outright for non-narrow weights on DSP48E1.
    narrow = bool(values["NARROW_WEIGHTS"])
    weight = [
        _random_weight_beat(generator, config, narrow=narrow)
        for _ in range(passes * synapse_folds * neuron_folds)
    ]
    stimulus = {"in0": activation, "in1": weight}
    print(
        f"  geometry: R={passes} SF={synapse_folds} NF={neuron_folds} "
        f"in0={len(activation)} in1={len(weight)} expect {expected} outputs"
    )
    print(f"  top:      {requirements.top_module_name} on {requirements.target_fpga_part}")
    print(
        "  declared: "
        + " ".join(f"{name}={_verilog(value)}" for name, value in sorted(values.items()))
    )

    fused_sources = _sources(finn_root, "finn-rtllib/mvu", FUSED_SOURCES)

    ok = True
    for stalls in (False, True):
        mode = "stalled" if stalls else "free-running"
        with tempfile.TemporaryDirectory() as scratch:
            # The composed DUT is whatever the compiler emits, verbatim.
            composed_sources = list(write_decomposed_artifact(requirements, scratch))
            fused_path = _write(scratch, "mvau_fused.sv", _fused_top("mvau_fused", config, values))
            fused = drive(
                "mvau_fused", [*fused_sources, fused_path], stimulus, expected, stalls=stalls
            )
            composed = drive(
                requirements.top_module_name,
                composed_sources,
                stimulus,
                expected,
                stalls=stalls,
            )
        matched = fused == composed and len(fused) == expected
        print(f"  {mode:12} fused={fused}")
        print(f"  {mode:12} composed={composed}")
        if not matched:
            print(f"  {config.label.upper()} ({mode}): FAIL")
            ok = False
    if ok:
        print(f"  {config.label.upper()}: PASS (bit-identical, {expected} outputs, both modes)")
    return ok


def record_identity(finn_root: str, library_root: str) -> None:
    """Print what was actually compiled.

    A passing run means nothing unless the result says which revisions it
    passed against.  Public because fixture 6 owes its log the same header.
    """

    for label, root in (("finn", finn_root), ("finnlib", library_root)):
        try:
            revision = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            revision, dirty = "unknown", ""
        print(f"{label:8} {revision}{' (dirty)' if dirty else ''}  {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS_BY_LABEL),
        help="run only this configuration",
    )
    arguments = parser.parse_args(argv)

    root = finn_root()
    library_root = str(finnlib_root(root))
    if not os.path.isdir(os.path.join(library_root, "rtl")):
        print(f"FinnLib RTL not found under {library_root}; set FINNLIB_ROOT or fetch-repos.sh")
        return 2
    record_identity(root, library_root)

    configs = (
        [CONFIGS_BY_LABEL[arguments.config]] if arguments.config is not None else list(CONFIGS)
    )
    ok = all([run_one(config, root) for config in configs])
    print("\nRESULT:", "FIXTURE 5 PASS" if ok else "FIXTURE 5 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
