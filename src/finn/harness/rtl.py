# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The RTL harness: build a module's sources, run a testbench in XSim, stream words
through a module, and measure its cycles.

``materialize`` builds a ``module`` into a directory and places each
memory's INIT_FILE where ``$readmemh`` reads it. ``simulate`` elaborates a
testbench module ``check`` against sources, with the tools of FINN's toolchain
(the machine's unless one is given), and requires it to display ``PASS``;
otherwise it raises ``SimulationFailed``. ``stream_through`` drives a module's
AXIS inputs with words (under stalls on both sides unless ``stalled`` is False)
and checks that each AXIS output presents its words, compared on their payload
bits (an AXIS word is padded to bytes); every other top input is held at zero. A
``repeating`` design (fed by a cyclic source) never stops producing: each
output then takes exactly its words and holds its ready low after them. A port
name the module does not present on that side is refused, naming the ports it
has: a stale name would otherwise leave the real port idle until the watchdog.

``measure`` streams identical frames back to back, never stalled, and records
the cycle each handshake completes in, on the root's streams and on each link
between the module's instances, as ``Measured``. Every frame presents the same
beats, so a stream's handshakes split evenly into frames. From them, with every
count inclusive of both ends (one beat in and out in the same cycle spans one
cycle):

- a frame's **latency**: from its first input beat accepted to its last output
  beat, over the root's streams, or between any two observed streams (``span``);
- the **interval** between frames: their last beats apart on a stream, the root's
  over its outputs; the last interval is the steady state's (II);
- the **total**: from the first input beat to the last output beat of all frames;
- a stream's **busy** span in a frame: its first to its last beat.

``Measured`` is pure arithmetic on the recorded cycles: the fast gate tests it
without Vivado.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Direction,
    Endpoint,
    Reset,
    StandardProtocol,
    abi_pins,
)
from finn.kernels.artifacts.build import emit_module, instance_net
from finn.kernels.artifacts.module import Composed, Link, Module
from finn.kernels.artifacts.sources import include_directories, is_header
from finn.util.toolchain import Toolchain, machine_toolchain

Words = tuple[Sequence[int], int]
"""A port's words, and the payload bits of each."""


def pack(values: Sequence[int], bits: int) -> int:
    """Lanes low first, each ``bits`` wide."""
    mask = (1 << bits) - 1
    return sum((value & mask) << (index * bits) for index, value in enumerate(values))


def materialize(module: Module, directory: Path) -> tuple[str, list[str], dict[str, str]]:
    """The top module, its HDL sources, and each INIT_FILE's name and contents.

    FinnLib is the ``finnlib`` resource (``FINN_RESOURCES_FINNLIB`` overrides it).
    """
    emitted = emit_module(module, directory / "module", roots={"finnlib": finnlib_root()})
    sources = [str(emitted.directory / path) for path in emitted.sources]
    data = {path: (emitted.directory / path).read_text() for path in emitted.data}
    return emitted.entry_point, sources, data


class SimulationFailed(AssertionError):
    """A simulator tool exited with an error, or the testbench did not display PASS;
    the message is what the simulator printed. An ``AssertionError``: a check that the
    hardware does what its testbench expects failed."""


def simulate(
    sources: Sequence[str | Path],
    testbench: str,
    directory: Path,
    *,
    toolchain: Toolchain | None = None,
) -> str:
    """Elaborate the testbench module ``check`` over ``sources``; it must display PASS.

    Each tool runs through ``toolchain`` (the machine's by default), whose Vivado
    (XILINX_VIVADO) provides ``glbl.v``. The simulator's output, which the testbench
    may have displayed more in."""
    toolchain = toolchain or machine_toolchain()
    vivado = toolchain.environment.get("XILINX_VIVADO")
    if not vivado:
        raise LookupError("the toolchain names no Vivado (XILINX_VIVADO) for glbl.v")
    bench = directory / "check.sv"
    bench.write_text("`timescale 1ns/1ps\n" + testbench)
    commands = (
        # Relaxed, as FINN's own flow elaborates (finn_xsi: ``xelab -relax``).
        (
            "xvlog",
            [
                "--sv",
                "--relax",
                *(f"--include={directory}" for directory in include_directories(sources)),
                *(str(source) for source in sources if not is_header(source)),
                str(Path(vivado) / "data/verilog/src/glbl.v"),
                str(bench),
            ],
        ),
        (
            "xelab",
            [
                "work.check",
                "work.glbl",
                "--mt",
                "2",
                "-L",
                "unisims_ver",
                "-L",
                "unimacro_ver",
                "--snapshot",
                "check",
                "--timescale",
                "1ns/1ps",
            ],
        ),
        ("xsim", ["check", "--runall"]),
    )
    output = printed = ""
    for tool, args in commands:
        result = toolchain.run(tool, args, cwd=directory, timeout=300, check=False)
        output = result.stdout.decode(errors="replace")
        printed = output + result.stderr.decode(errors="replace")
        if result.returncode != 0:
            raise SimulationFailed(printed)
    if "PASS" not in output:
        raise SimulationFailed(printed)
    return output


def _table(name: str, bits: int, words: Sequence[int]) -> str:
    values = ", ".join(f"{bits}'h{word:x}" for word in words)
    return f"logic [{bits - 1}:0] {name} [{len(words)}] = '{{{values}}};"


def stream_through(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    stalled: bool = True,
    repeating: bool = False,
) -> None:
    _check_streams(module, inputs, outputs)
    _stream(module, directory, inputs, outputs, stalled=stalled, repeating=repeating)


def measure(
    module: Module,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    frames: int = 4,
) -> Measured:
    """The cycles of ``frames`` frames streamed back to back, never stalled; printed too.

    Each input presents its words ``frames`` times over and each output must
    present its words as many times, checked as ``stream_through`` checks them.
    Every handshake is recorded by the cycle it completes in: on the root's
    streams and on each link between the module's instances, observed at the
    sink's nets (``instance_net``), so a kernel's own ports are measured where the
    module composes several.
    """
    if frames < 1:
        raise ValueError(f"at least one frame, not {frames}")
    _check_streams(module, inputs, outputs)
    observed = {name: (f"{name}_tvalid", f"{name}_tready") for name in (*inputs, *outputs)}
    for link in module.fragment.links if isinstance(module, Composed) else ():
        name = link_stream(link)
        if name is not None:
            sink = link.sink.instance
            assert sink is not None
            observed[name] = (
                f"dut.{instance_net(sink, link.sink.valid)}",
                f"dut.{instance_net(sink, link.sink.ready)}",
            )
    log = _stream(
        module,
        directory,
        {port: (list(words) * frames, bits) for port, (words, bits) in inputs.items()},
        {port: (list(words) * frames, bits) for port, (words, bits) in outputs.items()},
        stalled=False,
        repeating=False,
        observed=observed,
    )
    beats: dict[str, list[int]] = {name: [] for name in observed}
    for line in log.splitlines():
        if line.startswith("BEAT "):
            _, name, cycle = line.split()
            beats[name].append(int(cycle))
    measured = Measured(
        frames=frames,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        beats={name: tuple(cycles) for name, cycles in beats.items()},
    )
    print(measured.table(), flush=True)
    return measured


def link_stream(link: Link) -> str | None:
    """The name ``measure`` records a link's handshakes under: its sink's data pin,
    ``instance.pin``; None for a link into the root, recorded under the root's output
    port. A link from the root is recorded under both: its sink's pin and the root's input
    port, one handshake."""
    sink = link.sink
    return None if sink.instance is None else f"{sink.instance}.{sink.data}"


def _check_streams(
    module: Module, inputs: Mapping[str, Words], outputs: Mapping[str, Words]
) -> None:
    buses = [
        port
        for port in module.abi.pins
        if isinstance(port, Bus) and port.protocol is StandardProtocol.AXIS
    ]
    for side, names, endpoint in (
        ("input", inputs, Endpoint.TARGET),
        ("output", outputs, Endpoint.INITIATOR),
    ):
        has = [bus.name for bus in buses if bus.endpoint is endpoint]
        unknown = sorted(set(names) - set(has))
        if unknown:
            raise ValueError(
                f"the module has no {side} stream {', '.join(unknown)}; its {side}s: "
                + (", ".join(has) or "none")
            )


def _stream(
    module: Module,
    directory: Path,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    *,
    stalled: bool,
    repeating: bool,
    observed: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """Stream ``inputs`` through ``module`` checking ``outputs``; the simulator's output.

    Each ``observed`` stream, by name, its valid and ready signals: a line ``BEAT
    <name> <cycle>`` is displayed for each handshake on it.
    """
    top, sources, data = materialize(module, directory)
    for name, text in data.items():  # $readmemh reads an INIT_FILE from the simulator's directory
        (directory / name).write_text(text)
    valid, ready = ("cycle % 3 != 0", "cycle % 4 != 1") if stalled else ("1", "1")
    streams = {**inputs, **outputs}
    lines: list[str] = []
    for name, info in abi_pins(module.abi.pins).items():
        if isinstance(info.role, (Clock, Reset)) or info.bus in streams:
            continue
        width = "" if info.width == 1 else f"[{info.width - 1}:0] "
        held = info.direction is Direction.IN
        lines.append(f"logic {width}{name} = 0;" if held else f"wire {width}{name};")
    drive: list[str] = []
    count: list[str] = []
    done: list[str] = []
    for port, (words, bits) in inputs.items():
        carrier, total = (bits + 7) // 8 * 8, len(words)
        lines += [
            f"logic [{carrier - 1}:0] {port}_tdata; logic {port}_tvalid = 0; wire {port}_tready;",
            _table(f"{port}_words", carrier, words),
            f"int {port}_sent = 0;",
        ]
        count.append(f"if ({port}_tvalid && {port}_tready) {port}_sent <= {port}_sent + 1;")
        drive += [
            f"{port}_tvalid = ap_rst_n && {port}_sent < {total} && ({valid});",
            f"{port}_tdata = {port}_words[{port}_sent < {total} ? {port}_sent : 0];",
        ]
    for port, (words, bits) in outputs.items():
        carrier = (bits + 7) // 8 * 8
        lines += [
            f"wire [{carrier - 1}:0] {port}_tdata; wire {port}_tvalid; logic {port}_tready = 0;",
            _table(f"{port}_words", bits, words),
            f"int {port}_received = 0;",
        ]
        count.append(
            f"""if ({port}_tvalid && {port}_tready) begin
                if ({port}_tdata[{bits - 1}:0] !== {port}_words[{port}_received])
                    $fatal(1, "{port} word %0d: %h != %h", {port}_received,
                        {port}_tdata, {port}_words[{port}_received]);
                {port}_received <= {port}_received + 1;
            end"""
        )
        taken = f"{port}_received < {len(words)} && " if repeating else ""
        drive.append(f"{port}_tready = {taken}({ready});")
        done.append(f"{port}_received == {len(words)}")
    for name, (valid, ready) in (observed or {}).items():
        count.append(f'if ({valid} && {ready}) $display("BEAT {name} %0d", cycle);')
    newline = "\n    "
    return simulate(
        sources,
        f"""module check;
    logic ap_clk = 0, ap_rst_n = 0;
    {newline.join(lines)}
    always #5 ap_clk = !ap_clk;
    {top} dut (.*);
    int cycle = 0;
    always @(posedge ap_clk) begin
        cycle <= cycle + 1;
        if (ap_rst_n) begin
            {(newline + "        ").join(count)}
        end
    end
    always @(negedge ap_clk) begin
        {(newline + "    ").join(drive)}
    end
    initial begin
        repeat (16) @(posedge ap_clk);  // past the DSP models' startup recovery (GSR)
        ap_rst_n = 1;
        wait ({" && ".join(done)});
        repeat (4) @(posedge ap_clk);
        $display("STREAM_PASS");
        $finish;
    end
    initial begin #400000; $fatal(1, "watchdog"); end
endmodule
""",
        directory,
    )


# -- measurement ------------------------------------------------------------------------

Frame = tuple[int, ...]
"""The cycles of one frame's beats on one stream, in order."""


@dataclass(frozen=True)
class Measured:
    """``frames`` frames' handshakes, by stream name: the root's ``inputs`` and
    ``outputs`` by port, each link by its sink (``instance.pin``)."""

    frames: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    beats: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        for name in (*self.inputs, *self.outputs):
            if name not in self.beats:
                raise ValueError(f"no beats recorded on the root's stream {name}")
        for name in self.beats:
            self.of(name)  # refuses uneven frames now

    def of(self, name: str) -> tuple[Frame, ...]:
        """``name``'s beats, frame by frame."""
        cycles = self.beats[name]
        if not cycles or len(cycles) % self.frames:
            raise ValueError(f"{name}: {len(cycles)} beats do not split into {self.frames} frames")
        count = len(cycles) // self.frames
        return tuple(tuple(cycles[f * count : (f + 1) * count]) for f in range(self.frames))

    def per_frame(self, name: str) -> int:
        """The beats a frame presents on ``name``."""
        return len(self.beats[name]) // self.frames

    def busy(self, name: str) -> tuple[int, ...]:
        """Each frame's first to last beat on ``name``."""
        return tuple(frame[-1] - frame[0] + 1 for frame in self.of(name))

    def _first(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(min(self.of(name)[f][0] for name in names) for f in range(self.frames))

    def _last(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(max(self.of(name)[f][-1] for name in names) for f in range(self.frames))

    def span(self, source: str, sink: str) -> tuple[int, ...]:
        """Each frame's first beat on ``source`` to its last beat on ``sink``."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first([source]), self._last([sink]), strict=True)
        )

    @property
    def latencies(self) -> tuple[int, ...]:
        """Each frame's first input beat to its last output beat."""
        return tuple(
            last - first + 1
            for first, last in zip(self._first(self.inputs), self._last(self.outputs), strict=True)
        )

    def intervals(self, name: str | None = None) -> tuple[int, ...]:
        """Successive frames' last beats apart on ``name``; on the root's outputs by default."""
        ends = self._last(self.outputs if name is None else [name])
        return tuple(later - earlier for earlier, later in zip(ends, ends[1:]))

    def interval(self, name: str | None = None) -> int:
        """The steady state's interval (the last), on ``name`` or the root's outputs."""
        if self.frames < 2:
            raise ValueError("an interval needs at least two frames")
        return self.intervals(name)[-1]

    @property
    def total(self) -> int:
        """The first input beat to the last output beat of the last frame."""
        return self._last(self.outputs)[-1] - self._first(self.inputs)[0] + 1

    def table(self) -> str:
        """The measurement as text: the root's latency, interval and total, then per
        stream its beats per frame, busy span, interval and first beat."""
        start = self._first(self.inputs)[0]
        lines = [
            f"frames {self.frames}; latency per frame {list(self.latencies)}; "
            f"intervals {list(self.intervals()) if self.frames > 1 else []}; "
            f"total {self.total}",
            f"{'stream':<56} {'beats':>6} {'busy':>6} {'II':>6} {'first':>6}",
        ]
        for name in self.beats:
            interval = self.interval(name) if self.frames > 1 else "-"
            lines.append(
                f"{name:<56} {self.per_frame(name):>6} {self.busy(name)[-1]:>6} "
                f"{interval:>6} {self.of(name)[0][0] - start:>6}"
            )
        return "\n".join(lines)
