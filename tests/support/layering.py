# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's layers, what each may import, and the one import walker that checks them.

```text
finn.core.reporting  <-  finn.core.space, finn.toolchain, and every layer above them that
                         states a finding or records a run
finn.core.space <- finn.kernels <- finn.platform <- finn.core.graph <- finn.custom_op.kernels
finn.dataflow   <-                                                  <- finn.transformation.kernels
                                                                       <- finn.shells
                                                                    <- the XSim testbench
                                                                    <- the executors
                                                                    <- the partition node
                                                                    <- finn.harness
                                                                    <- the flow (all other finn)
finn.core.graph.partition,           <-  finn.core.graph, finn.custom_op.kernels,
finn.core.graph.annotations.outputs,     finn.transformation.kernels, the executors, the
finn.core.graph.annotations.choices      partition node, finn.harness, finn.shells, the flow
finn.toolchain  <-  finn.util (with finn.xsi, finn.resources)  <-  finn.transformation.kernels,
                                                                the XSim testbench, the
                                                                executors, finn.harness,
                                                                finn.shells, the flow
finn.core.space, finn.util  <-  preparation (finn.transformation.qonnx, .streamline)
                                <-  the flow
finn.dataflow  <-  the draws (finn.util.draws)  <-  finn.analysis, finn.harness, the flow
finn.core.space, the draws, the executors  <-  finn.analysis  <-  the flow
finn.kernels, finn.platform, finn.toolchain, finn.util  <-  finn.platform.generate (the
                                                            catalog generator; nothing
                                                            imports it)
finn.platform.capabilities  <-  finn.kernels, and every row above them
```

Each row imports every row to its left. What a run says (finn.core.reporting: its
findings and the refusals that carry them, its log and its manifest) sits below the
engine, the standard library only, and every row that states a finding or records a run
imports it: the toolchain logs its tool runs there. The platform's capabilities
(finn.platform.capabilities: what a target gives kernels, plain data) sit beside the
engine and the values, below the kernels; the rest of finn.platform, the registry with
the build's target, sits above them. FINN's view of the ONNX model (finn.core.graph:
its metadata, the facts a graph states of a tensor, the walks over its dataflow) reads
the kernels' values and types, the XSim testbench
(finn.core.rtlsim_bench, with finn.core.rtlsim_pacing) builds and simulates a kernel
module, the executors (finn.core.executor, the XSim executor finn.core.rtlsim_exec, and
finn.core.onnx_exec, which runs a model with them) run the KernelOps and a
partition's hardware with it, the partition node (finn.custom_op.partition) runs its
body with the executors of the run that reached it,
the harness (the hardware against the KernelOps' oracle, two runs of the
executors) imports the testbench and the executors, and the flow imports all.

``LAYERS`` is the one statement of that order. A module belongs to the layer
with the longest module prefix that names it, so the flow (prefix ``finn``)
holds every FINN module no other layer claims, and a new module falls into its
package's layer without an edit here. A module may import its own layer, the
layers its row names, the standard library, and the third-party packages its
row names (``ANY``: all of them). Every import counts: module level,
function-local and under ``TYPE_CHECKING`` alike, and ``import_module`` or
``__import__`` of a literal name.

The rows are declared lowest first and name only earlier rows, so the order
has no cycle. Each row names the test tree whose ``test_layering.py`` checks
it, so each gate checks the layers it owns.

A test helper, imported as ``support.layering`` (the gates put ``tests`` on
``PYTHONPATH``).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
TESTS = ROOT / "tests"

#: ``Layer.packages`` of a layer that may import any third-party package.
ANY = None


@dataclass(frozen=True)
class Layer:
    """One row of the table.

    ``modules`` are dotted prefixes: ``finn.*`` under ``src``, anything else a
    test package under ``tests``. ``also`` names single modules of a layer this
    one may not otherwise import, each with its reason at the row.
    """

    name: str
    modules: tuple[str, ...]
    imports: tuple[str, ...]
    packages: tuple[str, ...] | None
    tree: str
    also: tuple[str, ...] = ()


# The kernel stack: the engine, the values, the module build values, the platform's
# capabilities, the kernels. The engine and the values are independent: the kernels are
# the first layer to use both.
_KERNEL_STACK = (
    "reporting",
    "space",
    "dataflow",
    "kernels.artifacts",
    "platform.capabilities",
    "kernels",
)

LAYERS: tuple[Layer, ...] = (
    # What a run says: its findings and the refusals that carry them, its log and its
    # manifest. The standard library only, below the engine.
    Layer("reporting", ("finn.core.reporting",), (), (), "tests/core/reporting"),
    # The generic Space engine: the standard library only.
    Layer("space", ("finn.core.space",), ("reporting",), (), "tests/core/space"),
    # Canonical logical values: QONNX's datatypes and their ranges (a tensor's range is
    # part of its datatype fact, qonnx.core.ranges) and numpy, no engine; the findings
    # their refusals carry.
    Layer(
        "dataflow",
        ("finn.dataflow",),
        ("reporting",),
        ("qonnx.core.datatype", "qonnx.core.ranges", "numpy"),
        "tests/dataflow",
    ),
    # Module build values and their emission, below every Space; pyslang checks
    # declared pins against the RTL.
    # Its checks state their findings (finn.core.reporting).
    Layer(
        "kernels.artifacts",
        ("finn.kernels.artifacts",),
        ("reporting",),
        ("pyslang",),
        "tests/kernels",
    ),
    # What a target gives kernels: its clock, DSP block, fabric and resource units. Plain
    # data, the standard library only.
    Layer("platform.capabilities", ("finn.platform.capabilities",), (), (), "tests/kernels"),
    # Kernels bound to RTL/HLS sources. They never read an ONNX graph. One module of
    # util: the resource store (standard library only), where an input_gen's buffer
    # is read from FinnLib's RTL as FINN resolves FinnLib (decision FS6).
    Layer(
        "kernels",
        ("finn.kernels",),
        ("reporting", "space", "dataflow", "kernels.artifacts", "platform.capabilities"),
        ("qonnx.core.datatype", "qonnx.core.ranges", "numpy", "pyslang"),
        "tests/kernels",
        also=("finn.resources",),
    ),
    # The platform registry: parts, boards and shell rows, resolved to the build's target
    # (finn.platform.target), the capabilities kernels read with the part, shell and
    # board. A partition's network in its shell reads its row, whose budgets a module's pins
    # (finn.kernels.artifacts) are held to. A refusal carries its finding
    # (finn.core.reporting).
    Layer(
        "platform",
        ("finn.platform",),
        ("reporting", "space", "kernels.artifacts", "platform.capabilities", "kernels"),
        (),
        "tests/kernel_ops",
    ),
    # What is read of a partition, what a build made of it (finn.outputs) and the keys its
    # body holds (finn.choices, which no design digest covers): below both its writers
    # (the cut, PackagePartition) and its readers (the channel's annotation, whose choices
    # a partition's body states; the builder, the integration export, the shell's build,
    # the executors). qonnx (and onnx, which qonnx depends on: its nodes are NodeProto)
    # and the findings only, so the builder reads it without loading the kernel stack.
    Layer(
        "graph.partition",
        (
            "finn.core.graph.partition",
            "finn.core.graph.annotations.outputs",
            "finn.core.graph.annotations.choices",
        ),
        ("reporting",),
        ("onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # FINN's view of the ONNX model: its graph metadata (the target, a channel's choices,
    # a body's pins), the facts a graph states of a tensor, the walks over its dataflow.
    # Above the kernels: a channel's keys are generated from Channel's Decisions, and the
    # facts are read as the kernels' values.
    Layer(
        "graph",
        ("finn.core.graph",),
        (*_KERNEL_STACK, "platform", "graph.partition"),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The KernelOps, qonnx custom ops that each bind one kernel point on an ONNX node, and
    # the networks they compose: a selection of their nodes bound as one network, and a
    # partition's in its shell. A channel's choices are stated in a partition's body only
    # (finn.core.graph.partition says what one is).
    Layer(
        "custom_op.kernels",
        ("finn.custom_op.kernels",),
        (*_KERNEL_STACK, "platform", "graph.partition", "graph"),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # What FINN knows of the external tools it runs: the launcher, the host file, one
    # module per tool. The standard library only, the findings and the records its tool
    # runs are logged to (finn.core.reporting), and one module of util: the resource store
    # (standard library only), on whose primitives the HLS cache builds.
    Layer(
        "toolchain",
        ("finn.toolchain",),
        ("reporting",),
        (),
        "tests/kernel_ops",
        also=("finn.resources",),
    ),
    # What the flow and the kernel tests build on: helpers, the resource store and
    # the XSI binding (finn.xsi runs the simulator through finn.toolchain). Below the
    # flow: no module here imports a flow module.
    Layer(
        "util",
        ("finn.util", "finn.xsi", "finn.resources"),
        ("toolchain",),
        ANY,
        "tests/kernel_ops",
    ),
    # A graph's inputs drawn from their annotations, as int64: what the harness and the
    # analyses run a graph on, and the testbench reads its inputs by.
    Layer(
        "draws",
        ("finn.util.draws",),
        ("dataflow",),
        ("numpy", "qonnx"),
        "tests/kernel_ops",
    ),
    # Graph preparation's transforms: the front end that takes an export to the graph the
    # build converts (Quant lowering, streamlining); the builder's preparation steps
    # run them. It reads no target: neither the platform registry nor the KernelOps.
    Layer(
        "preparation",
        (
            "finn.transformation.qonnx",
            "finn.transformation.streamline",
        ),
        ("reporting", "util"),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The part catalog's generator: a tool above the registry, which runs Vivado
    # through finn.toolchain. Nothing imports it but its tests.
    Layer(
        "platform.generate",
        ("finn.platform.generate",),
        ("platform.capabilities", "kernels", "platform", "toolchain", "util"),
        (),
        "tests/kernel_ops",
    ),
    # The KernelOps' graph transformations. PackagePartition runs the toolchain
    # through finn.toolchain.
    Layer(
        "transformation.kernels",
        ("finn.transformation.kernels",),
        (
            *_KERNEL_STACK,
            "platform",
            "graph.partition",
            "graph",
            "custom_op.kernels",
            "toolchain",
            "util",
        ),
        ("onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The XSim testbench: a module's sources built (an HLS leaf's product from
    # transformation.kernels' HLS cache), the stream testbench written and simulated
    # through finn.toolchain, cycles measured; and how a simulation paces its streams.
    # The XSim executor's testbench and pacing, apart from the executor: it reads no
    # graph, so the kernel tests simulate with it.
    Layer(
        "xsim",
        ("finn.core.rtlsim_bench", "finn.core.rtlsim_pacing"),
        (*_KERNEL_STACK, "toolchain", "util", "transformation.kernels"),
        (),
        "tests/kernel_ops",
    ),
    # The executors and the execution that runs a model with them (finn.core.onnx_exec):
    # what runs a model's nodes, chosen by the caller. They run the KernelOps (by their
    # domain) and a partition's hardware (XSim: configured_network, the testbench).
    # Nothing below the flow imports them but the partition node, which executes its
    # body with them.
    Layer(
        "executors",
        (
            "finn.core.executor",
            "finn.core.rtlsim_exec",
            "finn.core.onnx_exec",
        ),
        (
            *_KERNEL_STACK,
            "platform",
            "graph.partition",
            "graph",
            "custom_op.kernels",
            "toolchain",
            "util",
            "transformation.kernels",
            "xsim",
        ),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The build's partition node, its ONNX domain (finn.custom_op.partition): it
    # runs its body with the executors of the run that reached it
    # (finn.core.onnx_exec.running), imported when it runs.
    Layer(
        "custom_op.partition",
        ("finn.custom_op.partition",),
        ("graph.partition", "executors"),
        ("qonnx",),
        "tests/kernel_ops",
    ),
    # Analysis passes: generic functions of a graph returning their findings, which no
    # build step runs. Neither the harness nor the builder: what a finding stops is the
    # caller's. The graph comparison runs both graphs with FINN's execution (the
    # executors' finn.core.onnx_exec), which computes integers exactly.
    Layer(
        "analysis",
        ("finn.analysis",),
        ("reporting", "space", "draws", "executors"),
        ("numpy", "qonnx"),
        "tests/kernel_ops",
    ),
    # The kernel harness: what checks a kernel's hardware (the hardware against its
    # KernelOps' oracle, execute_node, on a model of them: two runs of the executors) and
    # a KernelOp's reference against qonnx's execution of the ONNX it covers, on the
    # draws. It reads no test tree and no pytest; the tests that use it stay in tests/.
    # Nothing below the flow imports it.
    Layer(
        "harness",
        ("finn.harness",),
        (
            *_KERNEL_STACK,
            "platform",
            "graph.partition",
            "graph",
            "custom_op.kernels",
            "toolchain",
            "util",
            "transformation.kernels",
            "draws",
            "xsim",
            "executors",
        ),
        ("numpy", "qonnx"),
        "tests/kernel_ops",
    ),
    # The shells' builds: what builds the partition into a shell (the pynq shell's block
    # design, its movers' IODMAs, its driver), from the integration export. Below the
    # flow: the builder runs them. The board's driver files (finn.shells.pynq.data)
    # import pynq, each other and, to validate, dataset_loading.
    Layer(
        "shells",
        ("finn.shells",),
        (
            *_KERNEL_STACK,
            "platform",
            "graph.partition",
            "graph",
            "custom_op.kernels",
            "toolchain",
            "util",
            "transformation.kernels",
        ),
        ("numpy", "onnx", "qonnx", "pynq", "driver", "driver_base", "dataset_loading"),
        "tests/kernel_ops",
    ),
    # The flow: every FINN module no other layer claims (the builder).
    Layer(
        "flow",
        ("finn",),
        (
            *_KERNEL_STACK,
            "platform",
            "graph.partition",
            "graph",
            "custom_op.kernels",
            "toolchain",
            "util",
            "platform.generate",
            "transformation.kernels",
            "preparation",
            "draws",
            "analysis",
            "harness",
            "xsim",
            "executors",
            "custom_op.partition",
            "shells",
        ),
        ANY,
        "tests/kernel_ops",
    ),
    # This table: the standard library only. Every tree's tests import it.
    Layer("tests.layering", ("support.layering",), (), (), "tests/core/space"),
    # A refusal asserted by its code (finn.core.reporting's findings) and pytest. Every
    # tree's tests import it.
    Layer(
        "tests.refusals", ("support.refusals",), ("reporting",), ("pytest",), "tests/core/reporting"
    ),
    # The default snapshot's contract on value classes: the standard library only.
    # The engine, dataflow and kernel trees check their own classes with it.
    Layer("tests.value_classes", ("support.value_classes",), (), (), "tests/core/space"),
    # The Space engine's reference model: an oracle written independently of the engine,
    # so the standard library only, never finn. The engine's tests compare with it.
    Layer("tests.space_reference", ("core.space.reference",), (), (), "tests/core/space"),
    # The finn-dev oracle's captures (scripts/oracle): the standard library only. The
    # kernel tests compare with the HWCustomOp flow's values through it.
    Layer("tests.oracle", ("oracle",), (), (), "tests/kernels"),
    # Each test one build: the member cache its fixture opens (finn.kernels.members), which
    # the kernel and the kernel-op tests' conftests import.
    Layer(
        "tests.member_cache", ("support.member_cache",), ("kernels",), ("pytest",), "tests/kernels"
    ),
    # The shell's bus counts (finn.platform.shells.bus_counts) checked against the top's
    # pins on every netlist a test reads, the fixture the kernel and the kernel-op tests'
    # conftests import: the one place the kernel tests reach the platform registry. The
    # counts' oracle, until lane 4 (budgets as constraints) retires it.
    Layer(
        "tests.top_pins",
        ("support.top_pins",),
        (*_KERNEL_STACK, "platform"),
        ("pytest",),
        "tests/kernels",
    ),
    # The engine's purity check off for a test that counts the calls its memo saves
    # (_world.check_purity), which every tree's conftests import.
    Layer("tests.purity", ("support.purity",), ("space",), ("pytest",), "tests/core/space"),
    # The tests of the lower layers use only those layers and this table
    # (_typeshed: stubs named under TYPE_CHECKING).
    Layer(
        "tests.core.reporting",
        ("core.reporting",),
        ("reporting", "tests.layering", "tests.refusals"),
        ("pytest",),
        "tests/core/reporting",
    ),
    Layer(
        "tests.core.space",
        (
            "core.space",
            "support.space_cores",
            "support.space_oracle",
            "support.space_pipeline",
            "support.space_walk",
        ),
        (
            "reporting",
            "space",
            "tests.layering",
            "tests.refusals",
            "tests.value_classes",
            "tests.space_reference",
            "tests.purity",
        ),
        ("pytest", "_typeshed"),
        "tests/core/space",
    ),
    Layer(
        "tests.dataflow",
        ("dataflow", "support.tiled_mvu"),
        ("dataflow", "tests.layering", "tests.refusals", "tests.value_classes"),
        ("pytest", "numpy", "qonnx.core.datatype", "qonnx.core.ranges"),
        "tests/dataflow",
    ),
    # The kernel tests need no graph either: the kernel stack, the harness and the
    # integers it decodes (the draws' type), the XSim testbench, the toolchain, util for
    # the XSI runtime and the resource store, the space tests' helpers and the oracle's
    # captures.
    Layer(
        "tests.kernels",
        (
            "kernels",
            "support.adapted",
            "support.chain",
            "support.conformance",
            "support.emitted_netlist",
            "support.helpers",
            "support.ipxact_ports",
            "support.ipxact_script",
            "support.kernel_specs",
            "support.netlist",
            "support.xsim",
        ),
        (
            *_KERNEL_STACK,
            "toolchain",
            "util",
            "draws",
            "harness",
            "xsim",
            "tests.layering",
            "tests.refusals",
            "tests.value_classes",
            "tests.core.space",
            "tests.dataflow",
            "tests.oracle",
            "tests.member_cache",
            "tests.top_pins",
            "tests.purity",
        ),
        ("pytest", "numpy", "pyslang", "qonnx.core.datatype", "qonnx.core.ranges"),
        "tests/kernels",
    ),
    # The class-based oracle's targets and its tool (support.space_oracle): the design
    # spaces of FINN's kernels it compares (the toys, stage 7's chain). The rest of the
    # package is the engine's (tests.core.space). Its TFC target, which reads FINN's builder,
    # is a support module of its own in no row (as support.tfc is): a module this table
    # names is in the import closure the XSim groups are keyed by (scripts/emitted_text.py).
    Layer(
        "tests.space_oracle_targets",
        (
            "support.space_oracle.targets",
            "support.space_oracle.tool",
            "support.space_oracle.__main__",
        ),
        ("space", "kernels", "tests.core.space"),
        ("support.one_space",),
        "tests/kernel_ops",
    ),
)

BY_NAME = {layer.name: layer for layer in LAYERS}


def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def layer_of(name: str) -> Layer | None:
    """The layer a dotted name belongs to, or ``None`` for a third-party name."""

    owners = [
        (len(prefix), layer) for layer in LAYERS for prefix in layer.modules if within(name, prefix)
    ]
    return max(owners, key=lambda owner: owner[0])[1] if owners else None


def permits(layer: Layer, name: str) -> bool:
    """Whether a module of ``layer`` may import ``name`` (a module or an attribute)."""

    if any(within(name, module) for module in layer.also):
        return True
    owner = layer_of(name)
    if owner is not None:
        return owner is layer or owner.name in layer.imports
    return (
        name.partition(".")[0] in sys.stdlib_module_names
        or layer.packages is ANY
        or any(within(name, package) for package in layer.packages)
    )


def module_name(path: Path) -> str:
    """The dotted name a file under ``src`` or ``tests`` imports as."""

    base = SOURCE if path.is_relative_to(SOURCE) else TESTS
    parts = path.relative_to(base).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(source: str, module: str, is_package: bool = False) -> list[tuple[int, str]]:
    """Every name the code of ``module`` imports, as ``(line, absolute name)``.

    Relative imports resolve against the module's package. ``from a.b import c``
    names ``a.b.c``, which lies within ``a.b`` whether ``c`` is a module or an
    attribute. Literal ``import_module`` and ``__import__`` calls count, with
    ``import_module``'s ``package`` argument.
    """

    package = module if is_package else module.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package)
            found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            name = _literal_import(node, package)
            if name is not None:
                found.append((node.lineno, name))
    return found


def _literal_import(node: ast.Call, package: str) -> str | None:
    function = node.func
    call = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else ""
    )
    argument = node.args[0]
    if call not in ("import_module", "__import__") or not (
        isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ):
        return None
    name = argument.value
    if call == "import_module" and name.startswith("."):
        anchor = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "package"),
            node.args[1] if len(node.args) > 1 else None,
        )
        if isinstance(anchor, ast.Constant) and isinstance(anchor.value, str):
            package = anchor.value
        name = resolve_name(name, package)
    return name


def path_imports(path: Path) -> list[tuple[int, str]]:
    return imported_modules(path.read_text(), module_name(path), path.name == "__init__.py")


def sources(layer: Layer) -> tuple[Path, ...]:
    """The files of ``layer``: under its prefixes, less those a longer prefix claims."""

    found: set[Path] = set()
    for module in layer.modules:
        root = (SOURCE if within(module, "finn") else TESTS).joinpath(*module.split("."))
        candidates = root.rglob("*.py") if root.is_dir() else (root.with_suffix(".py"),)
        found.update(
            path
            for path in candidates
            if path.is_file()
            and "__pycache__" not in path.parts
            and layer_of(module_name(path)) is layer
        )
    return tuple(sorted(found))


def violations(layer: Layer) -> list[str]:
    """``file:line: name (its layer)`` for each import the table does not permit."""

    return [
        f"{path.relative_to(ROOT)}:{line}: {name} ({_owner(name)})"
        for path in sources(layer)
        for line, name in path_imports(path)
        if not permits(layer, name)
    ]


def _owner(name: str) -> str:
    owner = layer_of(name)
    return "third-party" if owner is None else owner.name


def checked_by(tree: Path) -> tuple[Layer, ...]:
    """The rows the ``test_layering.py`` in ``tree`` checks."""

    name = tree.resolve().relative_to(ROOT).as_posix()
    return tuple(layer for layer in LAYERS if layer.tree == name)
