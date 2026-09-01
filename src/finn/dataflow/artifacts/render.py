# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering: templates as files, contexts as flat scalars, output as bytes.

Renderers are the one producer class held to **determinism** as well as
closure.  A vendor tool may vary with threading, seeds and timestamps; a
renderer may not, and there is a cross-process test that says so.

Four settings are pinned and none is taste:

``StrictUndefined``
    a missing context value is an error, not an empty string.  The prior
    compiler configured the byte-determinism trio and left this out, so a
    missing value rendered as nothing -- into HDL, where an empty width or an
    empty port name is a silent wrong answer rather than a crash.

``SandboxedEnvironment``
    defence in depth over first-party templates.  Cheap, and it means a
    template cannot reach an attribute nobody meant to expose.

``trim_blocks`` / ``lstrip_blocks`` / ``keep_trailing_newline``
    fixed, because whitespace handling is part of the output bytes and the
    output bytes are in the key.

**The context is flat scalars only.**  Not a limitation to work around: a
template that can walk an object graph grows logic, and the limit worth
holding is that templates stay thin wrappers, shims and HLS tops.  A list of
generated parameters is computed in Python and passed as one string.  When
structural generation appears, that is the signal to emit programmatically,
not to grow the template.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Jinja2 is a real dependency and is present in the container and the working
# venv, but not in the interpreter the mypy gate runs under -- the same
# situation qonnx is in throughout this tree, and handled the same way.
from jinja2 import (  # type: ignore[import-not-found]
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    UndefinedError,
)
from jinja2.sandbox import SandboxedEnvironment  # type: ignore[import-not-found]

#: What a context value may be.  Everything else is refused at the boundary
#: rather than stringified inside a template.
Scalar = bool | int | float | str


class RenderError(Exception):
    """A template could not be rendered, so no text exists."""


def environment(roots: Sequence[Path]) -> Any:
    """The one Jinja2 environment.  Every setting here is load-bearing."""

    return SandboxedEnvironment(
        loader=FileSystemLoader([str(root) for root in roots]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,  # HDL, not HTML; escaping would corrupt the output
    )


def _flat(context: Mapping[str, object]) -> dict[str, Scalar]:
    for name, value in context.items():
        if not isinstance(value, (bool, int, float, str)):
            raise RenderError(
                f"context value {name!r} is a {type(value).__name__}; the render context is "
                "flat scalars only, so anything structural is computed in Python first"
            )
    return dict(context)  # type: ignore[arg-type]


def render_template(roots: Sequence[Path], name: str, context: Mapping[str, object]) -> str:
    """Render one template file with a flat scalar context.

    Templates are *files* because §9.2 needs their content digests in the plan
    key: two source revisions of one configuration must not claim one name.
    An inline string has no digest anybody can take.
    """

    try:
        template = environment(roots).get_template(name)
    except TemplateError as error:
        raise RenderError(f"{name} could not be loaded: {error}") from error
    try:
        rendered = template.render(**_flat(context))
        return str(rendered)
    except UndefinedError as error:
        raise RenderError(
            f"{name} reads {error.message}, which the context does not define; "
            "an undefined value must not render as an empty string into HDL"
        ) from error
    except TemplateError as error:
        raise RenderError(f"{name} failed to render: {error}") from error


#: ``$KEY$`` rather than ``$KEY``: FINN's existing marker convention.
#:
#: Written out rather than subclassing ``string.Template``, whose ``pattern``
#: has to be overridden with a raw string and then means something different
#: from what its own type says.  A named group and one ``sub`` is less code
#: than fighting that.
_MARKER = re.compile(r"\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)\$")


def substitute_markers(text: str, values: Mapping[str, object]) -> str:
    """Flat ``$KEY$`` substitution, for text with no loop in it.

    The stdlib where the stdlib is enough.  What it adds over ``str.replace``
    is that an unsubstituted marker and an unused value are both refused: the
    first ships a literal ``$KEY$`` into HDL, and the second is a value with no
    authority over anything.
    """

    flat = _flat(values)
    markers = {match.group("name") for match in _MARKER.finditer(text)}
    missing = sorted(markers - set(flat))
    if missing:
        raise RenderError(f"the text reads {missing}, which the values do not define")
    unused = sorted(set(flat) - markers)
    if unused:
        raise RenderError(f"{unused} is supplied and never read; it has no authority")
    return _MARKER.sub(lambda match: _text(flat[match.group("name")]), text)


def _text(value: Scalar) -> str:
    """A scalar as HDL spells it.

    ``bool`` before anything else: SystemVerilog has no ``True``, and letting
    Python's ``str()`` decide would put the word ``True`` into a parameter.
    """

    return str(int(value)) if isinstance(value, bool) else str(value)


def parameter_list(values: Mapping[str, Scalar], *, indent: str = "        ") -> str:
    """A Verilog named-parameter list, computed in Python rather than a template.

    Sorted, because a parameter table is a mapping and its order is not a fact
    about the build.  This is the shape of thing that keeps templates thin: a
    loop in a template would be one more place output bytes are decided.
    """

    separator = ",\n" + indent
    return separator.join(f".{name}({_text(values[name])})" for name in sorted(values))


__all__ = [
    "RenderError",
    "Scalar",
    "environment",
    "parameter_list",
    "render_template",
    "substitute_markers",
]
