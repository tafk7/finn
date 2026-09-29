"""P0.1 count: the explicit `semantics=` in finn.kernels and finn.dataflow that D2 makes redundant.

Every Space family in the two packages is scanned for members that carry an explicit
`semantics=` (derived, view, Param, Decision). Each is classified by its annotation and
by whether the explicit semantics differs from `default_semantics(T)`:

  redundant-today  plain `T` annotation, default-equivalent semantics: removable already
  redundant-D2     `T | Rejected` (engine markers), default-equivalent: removable with D2
  name-only        as above but a custom ValueSemantics differing from the default only in
                   name/snapshot (BEAT_SEQUENCE, TRAVERSAL): removable per plan D2
  stays            Protocol, genuine union, QueryResult[T], or genuinely custom semantics

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests <venv python> \
        docs/kernel-authoring-2026-09-29/p0/count_semantics.py
"""

from __future__ import annotations

import collections
import importlib
import inspect
import pkgutil
import types
from copy import deepcopy
from pathlib import Path
from typing import Union, get_args, get_origin

from finn.core.space import Space
from finn.core.space._signatures import _answer_value_type, resolve_annotations
from finn.core.space.collection import _annotation_namespace
from finn.core.space.declarations import Decision, Derived, Param, View, declared_annotation
from finn.core.space.results import Inapplicable, Rejected, Unresolved
from finn.core.space.semantics import ValueSemantics

PACKAGES = ("finn.kernels", "finn.dataflow")
MARKERS = (Inapplicable, Rejected, Unresolved)
ROOT = Path(__file__).resolve().parents[3]

for package in PACKAGES:
    module = importlib.import_module(package)
    for info in pkgutil.walk_packages(module.__path__, package + "."):
        importlib.import_module(info.name)

constant_names: dict[int, str] = {}
for name, module in list(__import__("sys").modules.items()):
    if name.startswith(PACKAGES):
        for attr, value in vars(module).items():
            if isinstance(value, ValueSemantics) and attr.isupper():
                constant_names.setdefault(id(value), attr)


def families() -> list[type[Space]]:
    seen, stack = [], [Space]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return [f for f in seen if f.__module__.startswith(PACKAGES)]


def is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def nominal(annotation: object) -> object:
    origin = get_origin(annotation)
    return origin if origin is not None else annotation


def classify(annotation: object, explicit: ValueSemantics[object]) -> tuple[str, str]:
    """(bucket, annotation shape)."""
    shape = "plain"
    value = annotation
    if _answer_value_type(annotation) is not None:
        return "stays", "QueryResult[T]"
    if is_union(annotation):
        values = [a for a in get_args(annotation) if a not in MARKERS]
        if len(values) != 1:
            return "stays", "union"
        value, shape = values[0], "T | marker"
    kind = nominal(value)
    if getattr(kind, "_is_protocol", False):
        return "stays", "Protocol" if shape == "plain" else "Protocol | marker"
    if not isinstance(kind, type) or is_union(value):
        return "stays", "special form"
    token = nominal(explicit.type_token)
    if token is not kind:
        return "stays", f"{shape}, token {getattr(token, '__name__', token)} is not {kind.__name__}"
    default_like = explicit.snapshot is deepcopy and explicit.name == kind.__qualname__
    if not default_like:
        # recognizes/equality may differ (INTEGER_TENSOR, ...) or only name/snapshot
        name = constant_names.get(id(explicit), explicit.name)
        if name in ("BEAT_SEQUENCE", "TRAVERSAL"):
            return "name-only", shape
        return "stays", f"{shape}, custom {name}"
    return ("redundant-today" if shape == "plain" else "redundant-D2"), shape


def where(function: object) -> str:
    code = getattr(function, "__code__")
    return f"{Path(code.co_filename).resolve().relative_to(ROOT)}:{code.co_firstlineno}"


def param_where(family: type[Space], name: str) -> str:
    lines, start = inspect.getsourcelines(family)
    for offset, line in enumerate(lines):
        if line.strip().startswith(f"{name}:"):
            path = Path(inspect.getsourcefile(family) or "").resolve().relative_to(ROOT)
            return f"{path}:{start + offset}"
    return f"{family.__module__}.{family.__qualname__}.{name}"


rows: list[tuple[str, str, str, str, str]] = []
for family in families():
    namespace = _annotation_namespace(family)
    for name, declaration in vars(family).items():
        if isinstance(declaration, (Derived, View)) and declaration.function is not None:
            explicit = declaration.semantics
            if explicit is None:
                continue
            owner = f"{family.__qualname__}.{name}"
            hints = resolve_annotations(declaration.function, namespace, owner)
            annotation = hints["return"]
            site = where(declaration.function)
            kind = "view" if isinstance(declaration, View) else "derived"
        elif isinstance(declaration, (Param, Decision)):
            explicit = getattr(declaration, "explicit", None)
            if explicit is None:
                continue
            annotation = declared_annotation(declaration, type(declaration).__name__)
            site = param_where(family, name)
            kind = type(declaration).__name__
        else:
            continue
        bucket, shape = classify(annotation, explicit)
        label = constant_names.get(id(explicit), f"<{explicit.name}>")
        rows.append((bucket, site, kind, label, shape))

rows.sort(key=lambda r: (r[0], r[1]))
for bucket, site, kind, label, shape in rows:
    print(f"{bucket:<16} {site:<48} {kind:<8} {label:<32} {shape}")

print()
sites = {r[1] for r in rows}
print(f"scanned sites: {len(rows)} ({len(sites)} distinct lines)")
by_bucket = collections.Counter(r[0] for r in rows)
for bucket in ("redundant-today", "redundant-D2", "name-only", "stays"):
    constants = collections.Counter(r[3] for r in rows if r[0] == bucket)
    print(f"  {bucket:<16} {by_bucket[bucket]:>3}  {dict(constants.most_common())}")
