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
which of them is right.  Fixture 8 can, and one configuration now needs it.

**DSP48E1 diverges, and the fused core is the one that is wrong.**  Adding the
oldest DSP generation to the matrix in Phase 6e made this fail: the two agree
on the first frame and disagree from the second, on a shape every other
generation matches.  Fixture 8's ``dsp48e1_fixture5_point`` runs that exact
geometry against ``MvauDataflowOp.execute_node`` and the composed core matches
on every value in both modes, so the defect is in ``mvu_vvu_axi``/``mvu.sv``
rather than in the decomposition.

The configuration stays in the matrix and stays simulated.  It carries a
``known_divergence`` that inverts the comparison and **fails if it ever agrees
again**, so the note cannot outlive the defect.  Recording it as unsupported
instead would have been the one outcome the Phase 6 plan rules out.

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

from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import ActivationReplayKernel, DotProductKernel
from finn.dataflow.mvau.hardware.binding import finnlib_root
from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    write_decomposed_artifact,
)
from finn.dataflow.mvau.providers import elaborate_mvau
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.ops.mvau import MVAU_WEIGHT_SUPPLY_SELECTION
from finn.dataflow.ops.mvau_op import MVAUDataflowBuildContext, MvauDataflowOp

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
    target: MVAUDspBlock
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
    #: Why the two DUTs are *known* not to agree here, and what settled which
    #: of them is right.
    #:
    #: Not an exemption.  A configuration carrying this is still simulated,
    #: still prints both readings, and **fails if it stops diverging** -- which
    #: is what keeps the note from outliving the defect it describes.  What it
    #: changes is only the sign of the comparison, and only where an oracle
    #: outside this fixture has already said which side is wrong.  A divergence
    #: recorded without that adjudication would be exactly the "unsupported to
    #: keep the gate green" the Phase 6 plan rules out.
    known_divergence: str = ""

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
    MVAUDspBlock.DSP48E1: "xc7z020clg400-1",
    MVAUDspBlock.DSP48E2: "xczu3eg-sbva484-1-e",
    MVAUDspBlock.DSP58: "xcvc1902-vsva2197-2MP-e-S",
}

#: DSP48E2 exercises the soft-vector core, DSP58 the packed one.  The last
#: three cover several repetitions (so the replay buffer has to reset between
#: frames) and pumped compute.
CONFIGS = [
    Config("softvec", MVAUDspBlock.DSP48E2, 1, 4, 4, 2, 2),
    Config("packed", MVAUDspBlock.DSP58, 1, 4, 4, 2, 2),
    Config("one_neuron_fold", MVAUDspBlock.DSP58, 1, 4, 4, 4, 2),
    Config("one_synapse_fold", MVAUDspBlock.DSP58, 1, 4, 4, 2, 4),
    Config("three_repetitions", MVAUDspBlock.DSP58, 3, 4, 4, 2, 2),
    Config("repetitions_softvec", MVAUDspBlock.DSP48E2, 2, 8, 6, 3, 2),
    Config("pumped", MVAUDspBlock.DSP58, 2, 8, 4, 2, 4, pumping=True),
    # DSP48E1 was in the part table from the start and in no configuration, so
    # the oldest generation this Kernel covers had never been simulated at all.
    # Phase 4 corrected the narrow-weight rule in both directions by reading
    # ``sliceLanes()`` and recorded that it rested on arithmetic rather than on
    # a measurement; this is where that is paid.
    #
    # Running it found something else: the two cores do not agree here, and the
    # composed one is right.  See ``known_divergence``.
    Config(
        "dsp48e1",
        MVAUDspBlock.DSP48E1,
        2,
        8,
        4,
        2,
        2,
        activation_bits=4,
        known_divergence=(
            "the fused mvu_vvu_axi core computes the wrong values on VERSION=1 "
            "from the second frame onward; fixture 8's dsp48e1_fixture5_point "
            "runs this exact geometry against execute_node and the composed "
            "core matches on every value, in both modes"
        ),
    ),
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
    operation.commit_dataflow_assignments(
        context,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            DECOMPOSED_MVAU_KERNELS.pe.path: config.pe,
            DECOMPOSED_MVAU_KERNELS.simd.path: config.simd,
            DECOMPOSED_MVAU_KERNELS.compute_pumping.path: config.pumping,
            # Weights arrive at the boundary; re-attaching the supplier is
            # increment G.
            MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NO_KERNEL,
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
    weight = [
        random_word(generator, config.pe * config.simd * config.weight_bits)
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
        complete = len(fused) == len(composed) == expected
        matched = fused == composed and complete
        print(f"  {mode:12} fused={fused}")
        print(f"  {mode:12} composed={composed}")
        if config.known_divergence:
            # The sign of the comparison is inverted, and only here.  A recorded
            # divergence that stopped diverging would mean the defect was fixed
            # and this note is now describing something that is not true.
            if not complete:
                print(f"  {config.label.upper()} ({mode}): FAIL (wrong number of outputs)")
                ok = False
            elif matched:
                print(
                    f"  {config.label.upper()} ({mode}): FAIL -- the recorded divergence is "
                    "gone, so the note above is stale and must be removed"
                )
                ok = False
            continue
        if not matched:
            print(f"  {config.label.upper()} ({mode}): FAIL")
            ok = False
    if not ok:
        return False
    if config.known_divergence:
        # Printed in full every run.  A divergence nobody reads is a divergence
        # nobody fixes, and the log is the only place this is visible.
        print(f"  {config.label.upper()}: DIVERGES AS RECORDED, {expected} outputs, both modes")
        print(f"    {config.known_divergence}")
        return True
    print(f"  {config.label.upper()}: PASS (bit-identical, {expected} outputs, both modes)")
    return True


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
    diverging = [config.label for config in configs if config.known_divergence]
    if diverging:
        # In the summary, not only beside the configuration.  A green RESULT
        # line with a defect buried two hundred lines above is how a known
        # divergence becomes an unknown one.
        print(f"\n{len(diverging)} configuration(s) diverge as recorded: {', '.join(diverging)}")
    print("\nRESULT:", "FIXTURE 5 PASS" if ok else "FIXTURE 5 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
