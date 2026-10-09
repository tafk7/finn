# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a kernel contributes to its module's sources: files it copies, data it generates,
and HLS build requests.

``provides`` and ``requires`` name module symbols (``module:dotp``), from which a
module's sources are ordered (``sources.ordered``).

An ``HlsSource`` is a request, not a file: a top function FINN writes from typed
values, the FinnLib headers it includes, its clock period and its configuration
directives. Its product is RTL, synthesized for a part by FINN's toolchain
(``finn.util.hls``), which ``emit_module`` takes by the request's ``function``
(``finn.kernels.artifacts.build``) and stages as it stages copied sources. The
part is not in the request: kernels see capabilities, never parts, so the part
enters at synthesis and in the synthesis key.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Union

from finn.kernels.artifacts.projection import content_digest, digest


class ContributionError(Exception):
    """A contribution names no file it could stand for."""


def _relative(path: str, label: str) -> None:
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise ContributionError(f"{path!r} is not a name relative to {label}")


@dataclass(frozen=True)
class CopiedSource:
    """A file taken verbatim from a named root (``finnlib``), so it does not move
    with the checkout."""

    root: str
    path: str
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.root:
            raise ContributionError("a copied source names its root")
        _relative(self.path, "its source root")
        object.__setattr__(self, "provides", tuple(sorted(set(self.provides))))
        object.__setattr__(self, "requires", tuple(sorted(set(self.requires))))


@dataclass(frozen=True)
class GeneratedData:
    """A data file whose contents the kernel generates, such as a memory image.

    The contents are part of the module, so they enter its fingerprint.
    ``path`` is the name the RTL reads it under (an ``INIT_FILE``), beside the
    module's sources.
    """

    path: str
    data: bytes

    def __post_init__(self) -> None:
        _relative(self.path, "the module's directory")
        if type(self.data) is not bytes:
            raise ContributionError("generated data is bytes")


def hex_image(prefix: str, words: Iterable[int], bits: int) -> GeneratedData:
    """A ``$readmemh`` memory image, one word of ``bits`` per line, named
    ``<prefix>_<digest>.dat`` by its contents."""

    digits = (bits + 3) // 4
    data = "".join(f"{word:0{digits}x}\n" for word in words).encode()
    return GeneratedData(f"{prefix}_{content_digest(data)[:16]}.dat", data)


# An identifier HLS keeps as a top's RTL name: no leading, trailing or double underscore.
_STEM = re.compile(r"^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)*$")
_KEY = re.compile(r"^[0-9a-f]{16}$")
#: What stands for the top's name where a request is digested: the name is the digest.
NAME_PLACEHOLDER = "__finn_hls_top__"
#: The headers a request may declare: C and C++ headers, which the top includes.
HEADER_SUFFIXES = (".h", ".hpp")


@dataclass(frozen=True)
class HlsSource:
    """An HLS build request: the top's text, the headers it includes, its clock and its
    configuration; one leaf's only source.

    ``top`` is a C++ translation unit defining the top function ``function``,
    ``<stem>_<key16>``, where ``key16`` is the request's own digest with the name left
    out (``request_key``): two requests share a name only when they are one request, so
    two configurations of a kernel in one netlist never share a module, nor, under
    ``config_rtl -module_auto_prefix``, a submodule. The name has no double underscore:
    HLS renames such a top (``SYN 201-103``), and the module would not be the
    request's. Build one with ``request``.

    ``headers`` are the files ``top`` and they include by quoted name, copied from their
    roots, each found on the include path its directory joins (``-I``). ``period_ns`` is
    the clock period HLS schedules for; ``directives`` the configuration commands
    (``config_rtl ...``) it runs before synthesis, in order.
    """

    function: str
    top: str
    headers: tuple[CopiedSource, ...]
    period_ns: float
    directives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        headers, directives = tuple(self.headers), tuple(self.directives)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "directives", directives)
        if any(not isinstance(item, CopiedSource) for item in headers):
            raise ContributionError("an HLS request's headers are copied sources")
        paths = [item.path for item in headers]
        if len(paths) != len(set(paths)):
            raise ContributionError("an HLS request declares one header path twice")
        for item in headers:
            if not item.path.endswith(HEADER_SUFFIXES):
                raise ContributionError(f"{item.path} is not a C or C++ header")
        if isinstance(self.period_ns, bool) or not isinstance(self.period_ns, (int, float)):
            raise ContributionError("an HLS request's clock period is a number of ns")
        object.__setattr__(self, "period_ns", float(self.period_ns))
        if not self.period_ns > 0:
            raise ContributionError("an HLS request's clock period is positive")
        if any(not isinstance(line, str) or "\n" in line for line in directives):
            raise ContributionError("a directive is one line of Tcl")
        stem, key = request_stem(self.function), self.function[-16:]
        if not (stem and _KEY.fullmatch(key)):
            raise ContributionError(f"{self.function!r} is not a top's name, <stem>_<key16>")
        if self.function not in self.top:
            raise ContributionError(f"the top's text does not define {self.function}")
        unnamed = self.top.replace(self.function, NAME_PLACEHOLDER)
        if key != request_key(unnamed, headers, self.period_ns, directives)[:16]:
            raise ContributionError(f"{self.function} is not the request's own digest")

    @classmethod
    def request(
        cls,
        stem: str,
        write: Callable[[str], str],
        headers: tuple[CopiedSource, ...],
        period_ns: float,
        directives: tuple[str, ...] = (),
    ) -> HlsSource:
        """The request whose top ``write(name)`` writes, named ``<stem>_<key16>`` by its
        digest: ``write`` is given the placeholder, then the name."""
        if not _STEM.fullmatch(stem):
            raise ContributionError(f"{stem!r} is no HLS top's stem: an identifier, no '__'")
        key = request_key(write(NAME_PLACEHOLDER), tuple(headers), period_ns, tuple(directives))
        function = f"{stem}_{key[:16]}"
        return cls(function, write(function), headers, period_ns, directives)


def request_stem(function: str) -> str | None:
    """The stem of an HLS top's name ``<stem>_<key16>``, or None when it is not one."""
    stem, separator, key = function[:-17], function[-17:-16], function[-16:]
    if separator != "_" or not _STEM.fullmatch(stem) or not _KEY.fullmatch(key):
        return None
    return stem


def request_key(
    unnamed: str,
    headers: tuple[CopiedSource, ...],
    period_ns: float,
    directives: tuple[str, ...],
) -> str:
    """The digest of a request whose top's name is ``NAME_PLACEHOLDER`` in ``unnamed``."""
    return digest(
        (
            "hls-request-v1",
            unnamed,
            tuple((item.root, item.path) for item in headers),
            float(period_ns),
            tuple(directives),
        )
    )


Contribution = Union[CopiedSource, GeneratedData, HlsSource]
"""What a kernel contributes: a file it copies, data it generates, or an HLS build request."""


__all__ = [
    "Contribution",
    "ContributionError",
    "CopiedSource",
    "GeneratedData",
    "HEADER_SUFFIXES",
    "HlsSource",
    "hex_image",
    "request_stem",
]
