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
identically; a mismatch can only mean the composition altered behaviour.

**Each simulation runs in its own process.**  XSI keeps state that outlives
``close_rtlsim``: the third ``load_sim_obj`` in a process hangs -- not the
third load of the same object, the third load at all -- with no diagnostic and
without the watchdog firing.  Isolating per configuration is not enough,
because one configuration is already four simulations.  Compiling and running
each one in a fresh interpreter removes the variable entirely and costs only
process startup, which is nothing beside ``xelab``.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_equiv.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
from finn.xsi import close_rtlsim, compile_sim_obj, load_sim_obj, reset_rtlsim

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
LIVENESS = 4000


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


# -- simulation --------------------------------------------------------------


def _random_word(generator: np.random.RandomState, bits: int) -> int:
    """A random integer of exactly ``bits`` bits.

    Built from bytes rather than ``randint`` because a packed weight beat is
    ``PE * SIMD * WEIGHT_WIDTH`` wide and reaches 64 bits at modest folding,
    which ``randint`` cannot represent.
    """

    raw = int.from_bytes(bytes(generator.randint(0, 256, size=(bits + 7) // 8)), "little")
    return raw & ((1 << bits) - 1)


#: Output beats accepted between stalls, and how long each stall lasts.
BACKPRESSURE_PERIOD = 1
BACKPRESSURE_TICKS = 5


def _collect_with_backpressure(sim: object, stream: str, size: int, watchdog: object) -> object:
    """Collect outputs while de-asserting ready every few accepted beats.

    ``rtlsim_multi_io`` holds ready high forever, which never exercises the
    stall path: the obligation is that the composition does not drop, duplicate
    or reorder anything when the consumer is not listening, and a consumer that
    always listens cannot show that.

    The watchdog is reset on every accepted beat, exactly as the stock
    collector does; a deliberate stall must not read as a hang.
    """

    class ThrottledCollector:
        def __init__(self) -> None:
            # The bus-port accessor lives on the engine, which is what
            # SimEngine.collect_output passes its own collector as ``top``.
            self.vld = sim.get_bus_port(stream, "tvalid")  # type: ignore[attr-defined]
            self.rdy = sim.get_bus_port(stream, "tready")  # type: ignore[attr-defined]
            self.dat = sim.get_bus_port(stream, "tdata")  # type: ignore[attr-defined]
            self.buf: list[str] = []
            self.stall = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.buf)

        def __call__(self, _sim: object) -> object:
            if self.stall > 0:
                self.stall -= 1
                return {self.rdy: "0"} if self.rdy.as_bool() else {}
            if self.rdy.as_bool():
                if self.vld.read().as_bool():
                    watchdog.reset()  # type: ignore[attr-defined]
                    self.buf.append(self.dat.read().as_hexstr())
                    if len(self.buf) == size:
                        return {self.rdy: "0"}
                    if len(self.buf) % BACKPRESSURE_PERIOD == 0:
                        self.stall = BACKPRESSURE_TICKS
                        return {self.rdy: "0"}
                return {}
            if len(self.buf) < size:
                return {self.rdy: "1"}
            return None

    collector = ThrottledCollector()
    sim.enlist(collector)  # type: ignore[attr-defined]
    return collector


def _drive(
    top_module: str,
    sources: list[str],
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    stalls: bool,
) -> list[int]:
    """Compile and run one DUT, optionally stalling both sides of it.

    **One simulation per process.**  XSI keeps state that ``close_rtlsim`` does
    not release: the third ``load_sim_obj`` in a process hangs -- not the third
    *of the same object*, the third at all -- with no diagnostic and without the
    watchdog firing.  Isolating per configuration was not enough, because one
    configuration is already four simulations.  So each one is compiled and run
    by :func:`simulate_once` in a fresh interpreter, and this function only
    marshals arguments across that boundary.
    """

    with tempfile.TemporaryDirectory() as scratch:
        request = Path(scratch) / "request.json"
        response = Path(scratch) / "response.json"
        request.write_text(
            json.dumps(
                {
                    "top_module": top_module,
                    "sources": sources,
                    "stimulus": stimulus,
                    "expected": expected,
                    "stalls": stalls,
                }
            )
        )
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--simulate",
                str(request),
                "--out",
                str(response),
            ],
            check=False,
        )
        if completed.returncode != 0 or not response.is_file():
            raise AssertionError(
                f"{top_module}: simulation subprocess failed (exit {completed.returncode})"
            )
        payload = json.loads(response.read_text())
        if payload.get("error"):
            raise AssertionError(f"{top_module}: {payload['error']}")
        return cast("list[int]", payload["output"])


def simulate_once(request_path: str, response_path: str) -> int:
    """Run exactly one simulation described by a JSON request, then exit.

    The whole body of the process: compile, load, run, write the outputs.
    Nothing else may load a simulation object here, which is the point.
    """

    request = json.loads(Path(request_path).read_text())
    payload: dict[str, object]
    try:
        with tempfile.TemporaryDirectory() as scratch:
            sim_dir, so_rel = compile_sim_obj(
                request["top_module"], request["sources"], scratch, behav=True
            )
            payload = {
                "output": _simulate(
                    sim_dir,
                    so_rel,
                    request["stimulus"],
                    request["expected"],
                    stalls=request["stalls"],
                    label=request["top_module"],
                )
            }
    except Exception as failure:  # noqa: BLE001 - reported across the boundary
        payload = {"error": f"{type(failure).__name__}: {failure}"}
    Path(response_path).write_text(json.dumps(payload))
    return 1 if payload.get("error") else 0


def _simulate(
    sim_dir: str,
    so_rel: str,
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    stalls: bool,
    label: str,
) -> list[int]:
    sim = load_sim_obj(sim_dir, so_rel)
    reset_rtlsim(sim)
    # Different throttles per stream, so the two inputs also arrive out of step
    # with each other rather than in lockstep.
    throttles = {"in0": (2, 3), "in1": (3, 2)} if stalls else {}
    for name, values in stimulus.items():
        sim.stream_input(
            f"{name}_V",
            map(lambda value: f"{value:0x}", list(values)),
            throttle=throttles.get(name, (float("inf"), 0)),
        )
    watchdog = sim.create_watchdog("out0_V timeout", LIVENESS)
    if stalls:
        collected = _collect_with_backpressure(sim, "out0_V", expected, watchdog)
    else:
        collected = sim.collect_output("out0_V", expected, watchdog=watchdog)
    timeouts = sim.run()
    if timeouts:
        raise AssertionError(f"{label}: deadlock, watchdogs fired: {timeouts}")
    # Both collectors are iterables of hex strings; neither is typed as one.
    result = [int(value, base=16) for value in cast("Iterable[str]", collected)]
    if watchdog in sim.watchdogs:
        sim.remove_watchdog(watchdog)
    close_rtlsim(sim)
    return result


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
        _random_word(generator, config.simd * config.activation_bits)
        for _ in range(passes * synapse_folds)
    ]
    weight = [
        _random_word(generator, config.pe * config.simd * config.weight_bits)
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
            fused = _drive(
                "mvau_fused", [*fused_sources, fused_path], stimulus, expected, stalls=stalls
            )
            composed = _drive(
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
    parser.add_argument(
        "--simulate",
        metavar="REQUEST",
        help="internal: run the one simulation this JSON request describes",
    )
    parser.add_argument("--out", metavar="RESPONSE", help="internal: where to write the result")
    arguments = parser.parse_args(argv)

    # The simulation worker does nothing else -- no resolving, no printing --
    # because loading a second simulation object in the same process is what
    # this split exists to prevent.
    if arguments.simulate is not None:
        if arguments.out is None:
            parser.error("--simulate requires --out")
        return simulate_once(arguments.simulate, arguments.out)

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
