# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``deterministic-tar``: transport, and byte-reproducible.

An archive is an optional transport encoding and **never the primary API** --
tools take paths, so an archive would need unpacking at every consumer.  It
exists so a component can be moved.

Every field a tar header carries that is not a fact about the content is
pinned, because each one is a way two identical components produce two
different archives:

======================  ====================================================
mtime                   ``0``; the clock is not an input
uid / gid               ``0``; who ran the build is not a property of it
uname / gname           empty, for the same reason
mode                    ``0o644``; the umask is ambient state
type / format           regular files, USTAR, so no PAX time headers appear
order                   sorted by name, since a directory listing is not one
======================  ====================================================

No compression.  gzip writes a timestamp into its own header, which would undo
all of the above, and compressing a source tree is not this layer's job.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.derivation import Derivation, OutputLayout, ProducerIdentity
from finn.dataflow.artifacts.formats import _descriptor
from finn.dataflow.artifacts.packaging import (
    PackageOptions,
    PackagePlan,
    PackagingError,
    PortableComponent,
    Realization,
    Support,
    Supported,
    Target,
)
from finn.dataflow.artifacts.projection import content_digest

ARCHIVE_NAME = "component.tar"

#: Pinned so two equal components produce two equal archives.
_MTIME = 0
_MODE = 0o644


@dataclass(frozen=True)
class TarOptions(PackageOptions):
    """The prefix every member sits under, if any."""

    prefix: str = ""

    def as_options(self) -> tuple[tuple[str, str], ...]:
        return (("prefix", self.prefix),)


class DeterministicTar:
    """Transport for anything.  It publishes no interfaces, so it refuses none.

    Worth stating rather than leaving implicit: this format's ``supports()``
    accepting everything is not laxness.  It carries bytes and a descriptor and
    makes no claim about what a consumer can connect, so there is nothing it
    could silently fail to express.  A format that *does* publish interfaces --
    the RTL module directory, IP-XACT -- has something to refuse.
    """

    format_id = "deterministic-tar"
    contract_version = "1"
    required_realization = Realization.SOURCE
    options_schema: type[PackageOptions] = TarOptions

    def supports(self, abi: ComponentABI) -> Support:
        return Supported()

    def plan(
        self, component: PortableComponent, target: Target, options: PackageOptions
    ) -> PackagePlan:
        if not isinstance(options, TarOptions):
            raise PackagingError(f"{self.format_id} takes TarOptions")
        descriptor = _descriptor.encode(component.abi)
        members = {_descriptor.DESCRIPTOR_NAME: descriptor}
        archive = write_archive(members, prefix=options.prefix)
        derivation = Derivation(
            kind="tar-package",
            schema_version=f"{self.format_id}-v{self.contract_version}",
            producer=ProducerIdentity(self.format_id, self.contract_version),
            inputs=(("component", component.artifact),),
            options=(("part", target.part), ("archive", content_digest(archive)))
            + options.as_options(),
            outputs=OutputLayout((ARCHIVE_NAME,)),
        )
        return PackagePlan(derivation, ((ARCHIVE_NAME, archive),))

    def parse(self, contents: Mapping[str, bytes]) -> ComponentABI:
        members = read_archive(contents[ARCHIVE_NAME])
        for name, data in members.items():
            if name.endswith(_descriptor.DESCRIPTOR_NAME):
                return _descriptor.decode(data)
        raise PackagingError(f"{ARCHIVE_NAME} carries no component descriptor")


def write_archive(members: Mapping[str, bytes], *, prefix: str = "") -> bytes:
    """A tar whose bytes are a function of its contents and nothing else."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(f"{prefix}/{name}" if prefix else name)
            info.size = len(data)
            info.mtime = _MTIME
            info.mode = _MODE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def read_archive(data: bytes) -> Mapping[str, bytes]:
    """Read one back, refusing a member that would escape the extraction root."""

    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for info in archive.getmembers():
            if info.name.startswith("/") or ".." in info.name.split("/"):
                raise PackagingError(f"{info.name!r} would extract outside its own directory")
            if not info.isfile():
                continue
            handle = archive.extractfile(info)
            if handle is None:
                continue
            found[info.name] = handle.read()
    return found


__all__ = ["ARCHIVE_NAME", "DeterministicTar", "TarOptions", "read_archive", "write_archive"]
