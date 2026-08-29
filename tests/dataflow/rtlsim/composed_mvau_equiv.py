############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Fixture 5: composed replay + dot product is bit-equivalent to the fused MVAU.

Runs INSIDE the FINN Docker container; needs Vivado's ``xelab`` and ``finn_xsi``.

The decomposition claims that ``mvu_vvu_axi`` is an activation replay followed
by a dot-product core, and that naming the two halves separately changes
nothing observable.  The RTL says the same thing: ``dotp_axi`` is
``mvu_vvu_axi`` with the ``replay_buffer`` lifted out and ``alast`` taken from
``s_axis_input_tlast`` instead of the replay's ``olast``.  This test drives both
with the same random stream and requires bit-identical output.

Both DUTs are compared to *each other*, not to a numeric golden, so the
stimulus is raw random integers packed to the stream widths.  Any value reaches
both identically; a mismatch can only mean the composition altered behaviour.

Usage, from ``finn/``::

    bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_equiv.sh
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np  # type: ignore[import-not-found]

from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, Engine
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import (
    ActivationReplayKernel,
    DotProductKernel,
    ParameterOwnership,
)
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM_SPEC,
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblemPaths,
)
from finn.dataflow.design import DesignSpaceSpec, ProblemSchema
from finn.dataflow.region import NumericElementType
from finn.xsi import close_rtlsim, compile_sim_obj, load_sim_obj, reset_rtlsim


def mvau_operation_context() -> DesignSpaceSpec:
    """The operation-owned declarations the two pools read.

    The same context the pool tests assemble, restated here because this script
    runs inside the container without the test tree on the path.
    """

    fields = tuple(
        item
        for item in MVAU_PROBLEM_SPEC.problem_schema.fields
        if item.path != MVAUProblemPaths.SOURCE_DESCRIPTION
    )
    return DesignSpaceSpec(ProblemSchema(fields), properties=MVAU_PROBLEM_SPEC.properties)


#: These designs settle in tens of ticks, so a generous budget only converts a
#: deadlock into an hour of simulation before anyone finds out.
LIVENESS = 20000

#: What the fused golden compiles against: FINN's own ship-everything list.
FUSED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
)

#: What the composed DUT compiles against: the same cores, reached through
#: FinnLib's dot-product wrapper, plus FINN's replay buffer.
COMPOSED_FINN_SOURCES = ("mvu_pkg.sv", "replay_buffer.sv")
COMPOSED_FINNLIB_SOURCES = (
    "add_multi_pkg.sv",
    "add_multi.sv",
    "dotp_axi.sv",
    "dotp.sv",
    "dotp_8sx9_dsp58.sv",
)

ACCU_WIDTH = 16
CLOCK_PERIOD_NS = 4.0


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

    @property
    def synapse_folds(self) -> int:
        return self.matrix_width // self.simd

    @property
    def neuron_folds(self) -> int:
        return self.matrix_height // self.pe


#: DSP48E2 exercises the soft-vector core, DSP58 the packed one.  The last four
#: cover what the earlier version of this harness did not: several repetitions
#: (so the replay buffer has to reset between frames), and pumped compute.
CONFIGS = [
    Config("softvec", MVAUDspBlock.DSP48E2, 1, 4, 4, 2, 2),
    Config("packed", MVAUDspBlock.DSP58, 1, 4, 4, 2, 2),
    Config("one_neuron_fold", MVAUDspBlock.DSP58, 1, 4, 4, 4, 2),
    Config("one_synapse_fold", MVAUDspBlock.DSP58, 1, 4, 4, 2, 4),
    Config("three_repetitions", MVAUDspBlock.DSP58, 3, 4, 4, 2, 2),
    Config("repetitions_softvec", MVAUDspBlock.DSP48E2, 2, 8, 6, 3, 2),
    Config("pumped", MVAUDspBlock.DSP58, 2, 8, 4, 2, 4, pumping=True),
]


def declared_parameters(config: Config) -> dict[str, object]:
    """Resolve a real design point and read the values it declares.

    This is the difference between "an arrangement that behaves the same" and
    "the arrangement this design space asks for".  Every value below comes from
    the point or the provider table; none is restated here.
    """

    pools = DECOMPOSED_MVAU_KERNELS
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                mvau_operation_context(),
                # The operation's real compute pool, not a private one: the
                # values driven into the RTL must be the values the shipped
                # design space produces for this point.
                MVAU_COMPUTE_SELECTION.build_spec(),
                MVAU_REPLAY_SELECTION.build_spec(),
            )
        )
    )
    activation = NumericElementType("int", config.activation_bits)
    weight = NumericElementType("int", config.weight_bits)
    accumulator = NumericElementType("int", ACCU_WIDTH)
    point = engine.start(
        space,
        {
            MVAUProblemPaths.REPETITIONS: config.repetitions,
            MVAUProblemPaths.MATRIX_WIDTH: config.matrix_width,
            MVAUProblemPaths.MATRIX_HEIGHT: config.matrix_height,
            MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: activation,
            MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: weight,
            MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: accumulator,
            MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: accumulator,
            MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
            MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
            MVAUProblemPaths.RUNTIME_WRITABLE: False,
            MVAUProblemPaths.TARGET_DSP_BLOCK: config.target,
            MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS: CLOCK_PERIOD_NS,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            pools.pe.path: config.pe,
            pools.simd.path: config.simd,
            pools.compute_pumping.path: config.pumping,
        },
    ).point

    feasible = engine.evaluate_constraint_set(
        point, MVAU_COMPUTE_SELECTION.feasibility_constraint_set
    )
    if feasible.verdict is not True:
        raise AssertionError(f"{config.label}: the design point is not feasible: {feasible}")

    values: dict[str, object] = {}
    for parameter in pools.provider_parameters():
        if parameter.ownership is ParameterOwnership.CONSTANT:
            values[parameter.name] = parameter.value
            continue
        assert parameter.source is not None
        if parameter.ownership is ParameterOwnership.DECISION:
            values[parameter.name] = point.assignments[parameter.source]
        elif parameter.ownership is ParameterOwnership.PROBLEM:
            raw = point.problem[parameter.source]
            values[parameter.name] = raw.bit_width if isinstance(raw, NumericElementType) else raw
        else:
            answer = engine.query_property(point, parameter.source)
            if not isinstance(answer, Decided):
                raise AssertionError(f"{config.label}: {parameter.name} did not resolve: {answer}")
            values[parameter.name] = answer.value
    return values


def _verilog(value: object) -> str:
    return str(int(value)) if isinstance(value, bool) else str(value)


def _ports(config: Config, values: dict[str, object]) -> str:
    weight_bits = config.pe * config.simd * config.weight_bits
    input_bits = config.simd * config.activation_bits
    output_bits = config.pe * ACCU_WIDTH
    del values
    return f"""#(
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
)"""


def _shared_parameters(values: dict[str, object]) -> str:
    """The parameters both wrappers take, from the declared values verbatim."""

    return f""".VERSION({_verilog(values["VERSION"])}),
        .PE({_verilog(values["PE"])}), .SIMD({_verilog(values["SIMD"])}),
        .SEGMENTLEN({_verilog(values["SEGMENTLEN"])}),
        .ACTIVATION_WIDTH({_verilog(values["ACTIVATION_WIDTH"])}),
        .WEIGHT_WIDTH({_verilog(values["WEIGHT_WIDTH"])}),
        .ACCU_WIDTH({_verilog(values["ACCU_WIDTH"])}),
        .NARROW_WEIGHTS({_verilog(values["NARROW_WEIGHTS"])}),
        .SIGNED_ACTIVATIONS({_verilog(values["SIGNED_ACTIVATIONS"])}),
        .PUMPED_COMPUTE({_verilog(values["PUMPED_COMPUTE"])}),
        .FORCE_BEHAVIORAL({_verilog(values["FORCE_BEHAVIORAL"])})"""


def _fused_top(name: str, config: Config, values: dict[str, object]) -> str:
    """The golden: one fused wrapper.

    ``IS_MVU``, ``MW`` and ``MH`` are the fused wrapper's own parameters -- it
    needs the geometry to size the replay it contains.  They are not part of
    the declared dot-product parameter set, which is the decomposition showing
    up in the parameter list.
    """

    return f"""
module {name} {_ports(config, values)};
    mvu_vvu_axi #(
        .IS_MVU(1),
        .MW({config.matrix_width}), .MH({config.matrix_height}),
        {_shared_parameters(values)}
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


def _composed_top(name: str, config: Config, values: dict[str, object]) -> str:
    """Replay feeding a dot product, wired exactly as the fused core wires them.

    Every parameter is the value the design point declares.  ``replay_buffer``
    takes ``LEN``/``REP``/``W`` from the replay Kernel's derived properties, and
    its ``olast`` becomes the dot product's ``tlast``.  ``ofin`` is left
    unconnected because the fused core never reads it either.
    """

    return f"""
module {name} {_ports(config, values)};
    localparam int unsigned REPLAY_W = {_verilog(values["W"])};

    uwire rst = !ap_rst_n;
    uwire [REPLAY_W-1:0] replayed_tdata;
    uwire replayed_tvalid;
    uwire replayed_tlast;
    uwire replayed_tready;

    replay_buffer #(
        .LEN({_verilog(values["LEN"])}), .REP({_verilog(values["REP"])}), .W(REPLAY_W)
    ) activation_replay (
        .clk(ap_clk), .rst(rst),
        .idat(in0_V_tdata[REPLAY_W-1:0]),
        .ivld(in0_V_tvalid),
        .irdy(in0_V_tready),
        .odat(replayed_tdata),
        .olast(replayed_tlast),
        .ofin(),
        .ovld(replayed_tvalid),
        .ordy(replayed_tready)
    );

    dotp_axi #(
        .ACTIVATION_BROADCASTING({_verilog(values["ACTIVATION_BROADCASTING"])}),
        {_shared_parameters(values)}
    ) dot_product (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(replayed_tdata),
        .s_axis_input_tvalid(replayed_tvalid),
        .s_axis_input_tlast(replayed_tlast),
        .s_axis_input_tready(replayed_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
"""


def _random_word(generator: np.random.RandomState, bits: int) -> int:
    """A random integer of exactly ``bits`` bits.

    Built from bytes rather than ``randint`` because a packed weight beat is
    ``PE * SIMD * WEIGHT_WIDTH`` wide and reaches 64 bits at modest folding,
    which ``randint`` cannot represent.
    """

    raw = int.from_bytes(bytes(generator.randint(0, 256, size=(bits + 7) // 8)), "little")
    return raw & ((1 << bits) - 1)


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
    top_path: str,
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    stalls: bool,
) -> list[int]:
    """Compile and run one DUT, optionally stalling both sides of it.

    Each call gets its own scratch directory and its own simulation object.
    Sharing either -- two tops compiled into one directory, or one compiled
    object loaded twice in a process -- segfaults inside XSI, so the compile
    cost is paid per run deliberately rather than optimised away.
    """

    with tempfile.TemporaryDirectory() as scratch:
        sim_dir, so_rel = compile_sim_obj(top_module, [*sources, top_path], scratch, behav=True)
        return _simulate(sim_dir, so_rel, stimulus, expected, stalls=stalls, label=top_module)


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


def _run(config: Config, finn_root: str, finnlib_root: str) -> bool:
    print(f"\n========== fixture 5: {config.label} ==========")
    values = declared_parameters(config)
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
    print(
        "  declared: "
        + " ".join(f"{name}={_verilog(value)}" for name, value in sorted(values.items()))
    )

    fused_sources = _sources(finn_root, "finn-rtllib/mvu", FUSED_SOURCES)
    composed_sources = [
        *_sources(finn_root, "finn-rtllib/mvu", COMPOSED_FINN_SOURCES),
        *_sources(finnlib_root, "rtl", COMPOSED_FINNLIB_SOURCES),
    ]

    ok = True
    for stalls in (False, True):
        mode = "stalled" if stalls else "free-running"
        with tempfile.TemporaryDirectory() as sources_dir:
            fused_path = _write(
                sources_dir, "mvau_fused.sv", _fused_top("mvau_fused", config, values)
            )
            composed_path = _write(
                sources_dir, "mvau_composed.sv", _composed_top("mvau_composed", config, values)
            )
            fused = _drive(
                "mvau_fused", fused_sources, fused_path, stimulus, expected, stalls=stalls
            )
            composed = _drive(
                "mvau_composed", composed_sources, composed_path, stimulus, expected, stalls=stalls
            )
        matched = fused == composed and len(fused) == expected
        print(f"  {mode:12} fused={fused}")
        print(f"  {mode:12} composed={composed}")
        if not matched:
            print(f"  {config.label.upper()} ({mode}): FAIL")
            ok = False
    if ok:
        print(f"  {config.label.upper()}: PASS (bit-identical, {expected} outputs, both modes)")
        return True
    return False


def _record_identity(finn_root: str, finnlib_root: str) -> None:
    """Print what was actually compiled.

    FinnLib comes from an ambient checkout, so a passing run means nothing
    unless the result says which revision it passed against.
    """

    for label, root in (("finn", finn_root), ("finnlib", finnlib_root)):
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


def main() -> int:
    finn_root = os.environ["FINN_ROOT"]
    finnlib_root = os.environ.get("FINNLIB_ROOT", os.path.join(finn_root, "..", "finnlib"))
    if not os.path.isdir(os.path.join(finnlib_root, "rtl")):
        print(f"FinnLib RTL not found under {finnlib_root}; set FINNLIB_ROOT")
        return 2
    _record_identity(finn_root, finnlib_root)
    ok = True
    for config in CONFIGS:
        ok &= _run(config, finn_root, finnlib_root)
    print("\nRESULT:", "FIXTURE 5 PASS" if ok else "FIXTURE 5 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
