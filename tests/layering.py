# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's layers, what each may import, and the one import walker that checks them.

```text
finn.core.space  <-  finn.kernels  <-  finn.platform  <-  finn.custom_op.kernels
finn.dataflow    <-                                   <-  finn.transformation.kernels
                                                          <-  finn.shells
                                                      <-  the XSim testbench
                                                      <-  the executors
                                                      <-  the partition node
                                                      <-  finn.harness
                                                      <-  the flow (all other finn)
finn.custom_op.partition.kernel_partitions  <-  finn.custom_op.kernels,
                                                finn.transformation.kernels, the
                                                executors, the partition node,
                                                finn.harness, finn.shells, the flow
finn.util (with finn.xsi, finn.resources)  <-  finn.transformation.kernels, the XSim
                                               testbench, the executors, finn.harness,
                                               finn.shells, the flow
finn.core.space, finn.util  <-  preparation (finn.transformation.qonnx, .streamline,
                                .prepare)  <-  finn.harness, the flow
finn.core.containers  <-  finn.custom_op.kernels, preparation, finn.harness, the flow
finn.kernels, finn.platform, finn.util  <-  finn.platform.generate (the catalog
                                            generator; nothing imports it)
```

Each row imports every row to its left: the XSim testbench
(finn.core.executors.xsim, less its executor) builds and simulates a kernel
module, the executors (finn.core.executors, the XSim executor among them, and
finn.core.onnx_exec, which runs a model with them) run the KernelOps and a
partition's hardware with it, the partition node (finn.custom_op.partition, less
kernel_partitions) runs its body with the executors of the run that reached it,
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

A test helper, imported as ``layering`` (the gates put ``tests`` on
``PYTHONPATH``).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


# The kernel stack: the engine, the values, the module build values, the kernels.
# The engine and the values are independent: the kernels are the first layer to
# use both.
_KERNEL_STACK = ("space", "dataflow", "kernels.artifacts", "kernels")

LAYERS: tuple[Layer, ...] = (
    # The generic Space engine: the standard library and its native dependency.
    Layer("space", ("finn.core.space",), (), ("greenlet",), "tests/core/space"),
    # Canonical logical values: QONNX's datatypes and numpy, no engine.
    Layer("dataflow", ("finn.dataflow",), (), ("qonnx.core.datatype", "numpy"), "tests/dataflow"),
    # Module build values and their emission, below every Space; pyslang checks
    # declared pins against the RTL.
    Layer("kernels.artifacts", ("finn.kernels.artifacts",), (), ("pyslang",), "tests/kernels"),
    # Kernels bound to RTL/HLS sources. They never read an ONNX graph. One module of
    # util: the resource store (standard library only), where an input_gen's buffer
    # is read from FinnLib's RTL as FINN resolves FinnLib (decision FS6).
    Layer(
        "kernels",
        ("finn.kernels",),
        ("space", "dataflow", "kernels.artifacts"),
        ("qonnx.core.datatype", "numpy", "pyslang"),
        "tests/kernels",
        also=("finn.resources",),
    ),
    # Containers: the ONNX element types execution holds a tensor's values in, and the
    # integers each holds exactly. The KernelOps' domain step, graph preparation's
    # container pass and checkpoint, and the harness's draws read the one table.
    Layer(
        "containers", ("finn.core.containers",), (), ("numpy", "onnx", "qonnx"), "tests/kernel_ops"
    ),
    # The platform registry: parts, boards and shell rows, resolved to the capabilities
    # kernels read. The shell root reads its shell's row.
    Layer("platform", ("finn.platform",), ("kernels",), (), "tests/kernel_ops"),
    # What is read and built of a partition (kernel_partitions): below both its writers
    # (the cut, PackagePartition) and its readers (the KernelOps, whose channel choices a
    # partition's body states; the builder, the integration export, the shell's build,
    # the executors). qonnx (and onnx, which qonnx depends on: its nodes are NodeProto)
    # only, so the builder reads it without loading the kernel stack.
    Layer(
        "partition",
        ("finn.custom_op.partition.kernel_partitions",),
        (),
        ("onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The KernelOps: qonnx custom ops that each bind one kernel point, on ONNX nodes.
    # A channel's choices are stated in a partition's body only (kernel_partitions says
    # what one is).
    Layer(
        "custom_op.kernels",
        ("finn.custom_op.kernels",),
        (*_KERNEL_STACK, "platform", "containers", "partition"),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # What the flow and the kernel tests build on: helpers, the resource store and
    # the XSI binding (finn.util and finn.xsi import each other). Below the flow:
    # no module here imports a flow module.
    Layer(
        "util",
        ("finn.util", "finn.xsi", "finn.resources"),
        (),
        ANY,
        "tests/kernel_ops",
    ),
    # Graph preparation: the front end that takes an export to the graph the kernel
    # path converts (Quant lowering, streamlining) and the phase over it, with its
    # checkpoint's findings. It reads no target: neither the platform registry nor
    # the KernelOps, whose anchors the checkpoint is given.
    Layer(
        "preparation",
        (
            "finn.transformation.qonnx",
            "finn.transformation.streamline",
            "finn.transformation.prepare",
        ),
        ("space", "util", "containers"),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The part catalog's generator: a tool above the registry, which runs Vivado
    # through util's toolchain. Nothing imports it but its tests.
    Layer(
        "platform.generate",
        ("finn.platform.generate",),
        ("kernels", "platform", "util"),
        (),
        "tests/kernel_ops",
    ),
    # The KernelOps' graph transformations. PackagePartition runs the toolchain
    # through util.
    Layer(
        "transformation.kernels",
        ("finn.transformation.kernels",),
        (*_KERNEL_STACK, "platform", "custom_op.kernels", "partition", "util"),
        ("onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The XSim testbench: a module's sources built (an HLS leaf's product from
    # transformation.kernels' HLS cache), the stream testbench written and simulated
    # through util's toolchain, cycles measured; and how a simulation paces its streams.
    # The XSim executor's package, less the executor: it reads no graph, so the kernel
    # tests simulate with it.
    Layer(
        "xsim",
        ("finn.core.executors.xsim",),
        (*_KERNEL_STACK, "util", "transformation.kernels"),
        (),
        "tests/kernel_ops",
    ),
    # The executors and the execution that runs a model with them (finn.core.onnx_exec):
    # what runs a model's nodes, chosen by the caller. They run the KernelOps (by their
    # domain) and a partition's hardware (XSim: configured_root, the testbench).
    # finn.core.rtlsim_exec is the legacy executor the model's exec_mode metadata
    # selects, here until the legacy executors are deleted. Nothing below the flow
    # imports them but the partition node, which executes its body with them.
    Layer(
        "executors",
        (
            "finn.core.executors",
            "finn.core.executors.xsim.executor",
            "finn.core.onnx_exec",
            "finn.core.rtlsim_exec",
        ),
        (
            *_KERNEL_STACK,
            "platform",
            "custom_op.kernels",
            "partition",
            "util",
            "transformation.kernels",
            "xsim",
        ),
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The kernel path's partition node, its ONNX domain (finn.custom_op.partition, less
    # kernel_partitions): it runs its body with the executors of the run that reached it
    # (finn.core.onnx_exec.executing), imported when it runs, since the executors read
    # kernel_partitions, under this package.
    Layer(
        "custom_op.partition",
        ("finn.custom_op.partition",),
        ("partition", "executors"),
        ("qonnx",),
        "tests/kernel_ops",
    ),
    # The kernel harness: what checks a kernel's hardware (the hardware against its
    # KernelOps' oracle, execute_node, on a model of them: two runs of the executors), a
    # KernelOp's reference against qonnx's execution of the ONNX it covers, and the
    # prepared graph against the export. It reads no test tree and no pytest;
    # the tests that use it stay in tests/. Nothing below the flow imports it. ONNX
    # Runtime: which ops it cannot run in float64 (the equivalence check's float64 run).
    Layer(
        "harness",
        ("finn.harness",),
        (
            *_KERNEL_STACK,
            "platform",
            "custom_op.kernels",
            "partition",
            "util",
            "transformation.kernels",
            "preparation",
            "containers",
            "xsim",
            "executors",
        ),
        ("numpy", "onnxruntime", "qonnx"),
        "tests/kernel_ops",
    ),
    # The shells' builds: what builds the partition into a shell (the pynq shell's block
    # design, its ends' IODMAs, its driver), from the integration export. Below the
    # flow: the builder runs them, and nothing here imports the HWCustomOp flow. The
    # board's driver files (finn.shells.pynq.data) import pynq, each other and, to
    # validate, dataset_loading.
    Layer(
        "shells",
        ("finn.shells",),
        (
            *_KERNEL_STACK,
            "platform",
            "custom_op.kernels",
            "partition",
            "util",
            "transformation.kernels",
        ),
        ("numpy", "onnx", "qonnx", "pynq", "driver", "driver_base", "dataset_loading"),
        "tests/kernel_ops",
    ),
    # The flow: every FINN module no other layer claims. finn.util.torch_hw_modules
    # is here by its imports: the PyTorch twin of the PWPolyF custom op, it reads
    # that op's constants (finn.custom_op.general). Upstream FINN documents it at
    # its util path, so it stays there.
    Layer(
        "flow",
        ("finn", "finn.util.torch_hw_modules"),
        (
            *_KERNEL_STACK,
            "platform",
            "custom_op.kernels",
            "partition",
            "util",
            "platform.generate",
            "transformation.kernels",
            "preparation",
            "harness",
            "xsim",
            "executors",
            "custom_op.partition",
            "containers",
            "shells",
        ),
        ANY,
        "tests/kernel_ops",
    ),
    # This table: the standard library only. Every tree's tests import it.
    Layer("tests.layering", ("layering",), (), (), "tests/core/space"),
    # The default snapshot's contract on value classes: the standard library only.
    # The engine, dataflow and kernel trees check their own classes with it.
    Layer("tests.value_classes", ("value_classes",), (), (), "tests/core/space"),
    # The finn-dev oracle's captures (scripts/oracle): the standard library only. The
    # kernel tests compare with the HWCustomOp flow's values through it.
    Layer("tests.oracle", ("oracle",), (), (), "tests/kernels"),
    # The tests of the lower layers use only those layers and this table
    # (_typeshed: stubs named under TYPE_CHECKING).
    Layer(
        "tests.core.space",
        ("core.space",),
        ("space", "tests.layering", "tests.value_classes"),
        ("pytest", "_typeshed"),
        "tests/core/space",
    ),
    Layer(
        "tests.dataflow",
        ("dataflow",),
        ("dataflow", "tests.layering", "tests.value_classes"),
        ("pytest", "qonnx.core.datatype"),
        "tests/dataflow",
    ),
    # The kernel tests need no graph either: the kernel stack, the harness, the XSim
    # testbench, util for the XSI runtime and the resource store, the space tests'
    # helpers and the oracle's captures. The one flow module is a live oracle: the
    # stream contracts are compared with the shuffle decomposition that baseline FINN
    # hard-codes.
    Layer(
        "tests.kernels",
        ("kernels",),
        (
            *_KERNEL_STACK,
            "util",
            "harness",
            "xsim",
            "tests.layering",
            "tests.value_classes",
            "tests.core.space",
            "tests.oracle",
        ),
        ("pytest", "numpy", "pyslang", "qonnx.core.datatype"),
        "tests/kernels",
        also=("finn.transformation.fpgadataflow.transpose_decomposition",),
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
