# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's layers, what each may import, and the one import walker that checks them.

```text
finn.core.space  <-  finn.dataflow  <-  finn.kernels  <-  finn.custom_op.kernels
                                                     <-  finn.transformation.kernels
                                                     <-  the flow (all other finn)
finn.util (with finn.xsi, finn.resources)  <-  finn.transformation.kernels, the flow
finn.parked: imported by nothing
```

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
it, so each gate checks the layers it owns. ``finn.parked`` is reference code:
no tree checks it, and no row may import it.

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
    tree: str | None
    also: tuple[str, ...] = ()


# The kernel stack: the engine, the values, the module build values, the kernels.
_KERNEL_STACK = ("space", "dataflow", "kernels.artifacts", "kernels")

LAYERS: tuple[Layer, ...] = (
    # The generic Space engine: the standard library and its native dependency.
    Layer("space", ("finn.core.space",), (), ("greenlet",), "tests/core/space"),
    # Canonical logical values: the engine and QONNX's datatypes.
    Layer("dataflow", ("finn.dataflow",), ("space",), ("qonnx.core.datatype",), "tests/dataflow"),
    # Module build values and their emission, below every Space; pyslang checks
    # declared pins against the RTL.
    Layer("kernels.artifacts", ("finn.kernels.artifacts",), (), ("pyslang",), "tests/kernels"),
    # Kernels bound to RTL/HLS sources. They never read an ONNX graph.
    Layer(
        "kernels",
        ("finn.kernels",),
        ("space", "dataflow", "kernels.artifacts"),
        ("qonnx.core.datatype", "pyslang"),
        "tests/kernels",
    ),
    # The KernelOps: qonnx custom ops that each bind one kernel point, on ONNX nodes.
    Layer(
        "custom_op.kernels",
        ("finn.custom_op.kernels",),
        _KERNEL_STACK,
        ("numpy", "onnx", "qonnx"),
        "tests/kernel_ops",
    ),
    # The kernel-partition facts. The module sits in the flow's package but below
    # both its writer (PackagePartition) and its readers (InsertIODMA, the
    # driver): it imports qonnx only, so the flow imports it without loading the
    # kernel stack.
    Layer(
        "kernel_partitions",
        ("finn.transformation.fpgadataflow.kernel_partitions",),
        (),
        ("qonnx",),
        "tests/kernel_ops",
    ),
    # What the flow and the kernel tests build on: helpers, the resource store and
    # the XSI binding (finn.util and finn.xsi import each other). Below the flow:
    # no module here imports a flow module.
    Layer("util", ("finn.util", "finn.xsi", "finn.resources"), (), ANY, "tests/kernel_ops"),
    # The KernelOps' graph transformations. PackagePartition runs the toolchain
    # through util.
    Layer(
        "transformation.kernels",
        ("finn.transformation.kernels",),
        (*_KERNEL_STACK, "custom_op.kernels", "kernel_partitions", "util"),
        ("onnx", "qonnx"),
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
            "custom_op.kernels",
            "kernel_partitions",
            "util",
            "transformation.kernels",
        ),
        ANY,
        "tests/kernel_ops",
    ),
    # Retired code, reference only.
    Layer("parked", ("finn.parked",), (), ANY, None),
    # This table: the standard library only. Every tree's tests import it.
    Layer("tests.layering", ("layering",), (), (), "tests/core/space"),
    # The tests of the lower layers use only those layers and this table
    # (_typeshed: stubs named under TYPE_CHECKING).
    Layer(
        "tests.core.space",
        ("core.space",),
        ("space", "tests.layering"),
        ("pytest", "_typeshed"),
        "tests/core/space",
    ),
    Layer(
        "tests.dataflow",
        ("dataflow",),
        ("space", "dataflow", "tests.layering"),
        ("pytest", "qonnx.core.datatype"),
        "tests/dataflow",
    ),
    # The kernel tests need no graph either: the kernel stack, util for the XSim
    # harness and the resource store, and the space tests' helpers. The one flow
    # module is an oracle: the stream contracts are compared with the shuffle
    # decomposition that baseline FINN hard-codes.
    Layer(
        "tests.kernels",
        ("kernels",),
        (*_KERNEL_STACK, "util", "tests.layering", "tests.core.space"),
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


def importers(target: Layer, root: Path) -> list[str]:
    """``file:line: name`` for each import of a ``target`` module from outside it under ``root``."""

    return [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts and layer_of(module_name(path)) is not target
        for line, name in path_imports(path)
        if layer_of(name) is target
    ]


def checked_by(tree: Path) -> tuple[Layer, ...]:
    """The rows the ``test_layering.py`` in ``tree`` checks."""

    name = tree.resolve().relative_to(ROOT).as_posix()
    return tuple(layer for layer in LAYERS if layer.tree == name)
