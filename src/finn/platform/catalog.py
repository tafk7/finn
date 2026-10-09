# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's part catalog: every part of the installed Vivado's supported series, its
device and the device's resources per SLR, generated from Vivado's part database
(``finn.platform.generate``) and committed in ``data/``.

The records:

- a resource record (``Record``): the totals (``lut``, ``ff``, ``bram18``,
  ``uram``, ``dsp``) of one or more devices, their number of SLRs and each SLR's
  counts, which sum to the totals, stored once, keyed by the digest of its facts. A
  device sold on a larger die (an XCZU2EG on the XCZU3EG's) states fewer resources
  than its die's sites, and Vivado does not state how they split over its SLRs: a
  reduced device of several SLRs states each SLR's site capacity, an upper bound for
  that SLR alone, with its totals as the cap on their sum (``capped``,
  ``SLRS_CAPPED``), and which caps a fill proved
  (``finn.platform.architectures.CAP_EVIDENCE``);
- a ``Device``: its name, Vivado's ARCHITECTURE and FAMILY, what FINN builds there
  (its pair's rule, ``finn.platform.architectures``), its record's totals
  (``resources``) and SLRs, and the devices that share them (``shared_with``: equal
  facts, never a name match; a CG, EG and EV device of one die stay three devices);
- a ``Part``: its name as Vivado spells it, its device, package, speed and
  temperature grade (``None`` where Vivado states none), and where its facts come
  from (``source``).

The queries: ``part(name)`` and ``device(name)``, the name compared without case
and answered in Vivado's spelling, never a pattern or a nearby part (an unknown name
is refused, ``unknown-part``, ``unknown-device``, listing close names without
choosing one); ``parts(pattern, family=)``; ``part_report(name)``, a part's facts as
the resources report states them. A device whose pair FINN does not build
for is catalogued with its identity and totals, and states why (``unsupported``);
the resolution refuses its parts (``unsupported-architecture``).

A user adds what the catalog does not ship (a new part, an engineering sample, a
board's custom part) without editing FINN: an overlay file, named by the machine
setting ``FINN_PLATFORM_CATALOG`` (read once, when the catalog is first queried),
in the committed data's schema (``load``). It is validated on load; it may add
resource records, devices and parts, and refuses by name an entry that redefines a
shipped one unless the entry says it ``overrides``. Each entry carries its
``source`` (or the file's), which the part's ``source`` states. A device's fabric
and DSP block are its pair's rule, or stated on the device (``fabric``, ``dsp``,
``uram_init``).
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from finn.kernels.target import DspBlock
from finn.kernels.utilization import RESOURCE_NAMES, Fabric, Resources
from finn.platform.architectures import CAP_EVIDENCE, CapEvidence, rule
from finn.platform.refusal import TargetRefused

DATA = Path(__file__).resolve().parent / "data"
FILES = ("resources.jsonl", "devices.jsonl", "parts.jsonl", "manifest.json")
"""The data files: the resource, device and part records, one a line, then the manifest."""
OVERLAY_SETTING = "FINN_PLATFORM_CATALOG"
"""The machine setting that names a catalog overlay file (unset: none)."""


SLRS_CAPPED = (
    "a reduced die: each SLR's counts are its site capacity, an upper bound for that SLR "
    "alone; the device's totals cap their sum, so not every SLR can be full at once"
)
"""What the SLRs of a ``capped`` record state."""


@dataclass(frozen=True, kw_only=True)
class Record:
    """A resource record: the ``totals``, the number of SLRs (``slr_count``) and each
    SLR's counts (``slrs``, which sum to the totals; ``None``: not stated, an overlay's
    device of several SLRs whose split it does not know). A ``capped`` record is a
    reduced die of several SLRs (``SLRS_CAPPED``): each SLR's counts are its site
    capacity, and the totals cap their sum, which covers them in every resource and
    exceeds them in one at least."""

    totals: Resources
    slr_count: int
    slrs: tuple[Resources, ...] | None
    capped: bool = False

    def __post_init__(self) -> None:
        if type(self.slr_count) is not int or self.slr_count < 1:
            raise ValueError(f"slr_count is a number of SLRs, not {self.slr_count!r}")
        if type(self.capped) is not bool:
            raise ValueError(f"capped is true or false, not {self.capped!r}")
        if self.slrs is None:
            if self.slr_count == 1:
                raise ValueError("a device of one SLR states it: its totals")
            if self.capped:
                raise ValueError("a capped record states each SLR's site capacity")
            return
        if len(self.slrs) != self.slr_count:
            raise ValueError(f"{len(self.slrs)} SLRs stated, not slr_count {self.slr_count}")
        summed = sum(self.slrs, Resources())
        if not self.capped:
            if summed != self.totals:
                raise ValueError(f"the SLRs sum to {summed}, not the totals")
            return
        if self.slr_count == 1:
            raise ValueError("a capped record has several SLRs: one SLR is the device")
        if not self.capped_resources() or any(
            getattr(summed, name) < getattr(self.totals, name) for name in RESOURCE_NAMES
        ):
            raise ValueError(
                f"the SLRs of a capped record sum to {summed}, which does not cover the "
                f"totals {self.totals} and exceed them in one resource"
            )

    def capped_resources(self) -> tuple[str, ...]:
        """The resources whose totals cap the SLRs' sum: those it exceeds (none if the
        record is not ``capped``)."""
        if not self.capped or self.slrs is None:
            return ()
        summed = sum(self.slrs, Resources())
        return tuple(
            name for name in RESOURCE_NAMES if getattr(summed, name) > getattr(self.totals, name)
        )

    @property
    def digest(self) -> str:
        """The digest the record is keyed by: of its normalized facts."""
        encoded = json.dumps(self.stored_facts(), separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]

    def stored_facts(self) -> dict[str, object]:
        """The record as the data stores it, without its digest (``capped`` only where
        it is)."""
        return {
            "slr_count": self.slr_count,
            "totals": _counts(self.totals),
            "slrs": None if self.slrs is None else [_counts(slr) for slr in self.slrs],
            **({"capped": True} if self.capped else {}),
        }

    def stored(self) -> dict[str, object]:
        """The record as the data stores it: its digest, then its facts."""
        return {"digest": self.digest, **self.stored_facts()}


def _counts(resources: Resources) -> dict[str, int]:
    return {name: getattr(resources, name) for name in RESOURCE_NAMES}


@dataclass(frozen=True, kw_only=True)
class Device:
    """A device: its ``name``, Vivado's ``architecture`` and ``family``, what FINN
    builds there (``fabric``, ``dsp``, ``uram_init``; ``unsupported`` says why it
    builds nothing, and they are then ``None`` and ``False``), its totals
    (``resources``), SLRs (``slr_count``) and each SLR's resources (``slrs``;
    ``None``: not stated) under their record's ``digest``, the resources whose
    totals cap the SLRs' sum on a reduced die (``capped``, ``SLRS_CAPPED``; empty:
    the SLRs sum to the totals) and, for a shipped device, which of those caps a
    fill proved (``cap_evidence``, ``finn.platform.architectures.CAP_EVIDENCE``),
    the other devices with the same record (``shared_with``), and where its facts
    come from (``source``)."""

    name: str
    architecture: str
    family: str
    fabric: Fabric | None
    dsp: DspBlock | None
    uram_init: bool
    resources: Resources
    slr_count: int
    slrs: tuple[Resources, ...] | None
    capped: tuple[str, ...]
    cap_evidence: CapEvidence | None
    digest: str
    shared_with: tuple[str, ...]
    source: str
    unsupported: str | None


@dataclass(frozen=True, kw_only=True)
class Part:
    """A part: its ``name`` as Vivado spells it, its ``device``, ``package``,
    ``speed`` and ``temperature`` grade (``None``: not stated), and where its facts
    come from (``source``)."""

    name: str
    device: Device
    package: str | None
    speed: str | None
    temperature: str | None
    source: str


@dataclass(frozen=True)
class Catalog:
    """A loaded catalog: its devices and parts by lower-cased name, the
    ``manifest`` of its committed data, and the ``overlay`` laid over it (if any)."""

    devices: Mapping[str, Device]
    by_name: Mapping[str, Part]
    manifest: Mapping[str, Any]
    overlay: Path | None = None
    _sorted: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_sorted", tuple(sorted(self.by_name)))

    def describe(self) -> str:
        """What the catalog covers, as a refusal names it."""
        tool = self.manifest["tool"]
        series = ", ".join(self.manifest["series"])
        named = f"Vivado {tool['version']} ({tool['build']}): {series}"
        return named if self.overlay is None else f"{named}; overlay {self.overlay}"

    def part(self, name: str) -> Part:
        """The part ``name`` (compared without case); refused if the catalog has no
        such part (``unknown-part``), with close names, none chosen."""
        found = self.by_name.get(name.lower())
        if found is None:
            raise TargetRefused(
                "unknown-part",
                f"{name!r} is not in the part catalog ({self.describe()})"
                + _close(name, (self.by_name[each].name for each in self._sorted)),
            )
        return found

    def device(self, name: str) -> Device:
        """The device ``name`` (compared without case); refused if the catalog has no
        such device (``unknown-device``), with close names."""
        found = self.devices.get(name.lower())
        if found is None:
            raise TargetRefused(
                "unknown-device",
                f"{name!r} is not a device of the part catalog ({self.describe()})"
                + _close(name, (each.name for each in self.devices.values())),
            )
        return found

    def parts(self, pattern: str | None = None, *, family: str | None = None) -> list[Part]:
        """The parts whose lower-cased name matches ``pattern`` (``fnmatch``) and whose
        device is of ``family`` (Vivado's FAMILY), sorted by name."""
        return [
            self.by_name[each]
            for each in self._sorted
            if (pattern is None or fnmatchcase(each, pattern.lower()))
            and (family is None or self.by_name[each].device.family == family)
        ]


def _close(name: str, names: Iterable[str]) -> str:
    spelled = {each.lower(): each for each in names}
    close = difflib.get_close_matches(name.lower(), spelled, n=5, cutoff=0.8)
    return f"; close names: {', '.join(spelled[each] for each in close)}" if close else ""


# -- the records, as stored ---------------------------------------------------------------


class _Invalid(ValueError):
    pass


def _expect(record: object, where: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise _Invalid(f"{where}: a record is a JSON object, not {record!r}")
    missing = required - set(record)
    unknown = set(record) - required - optional
    if missing or unknown:
        raise _Invalid(
            f"{where}: "
            + "; ".join(
                text
                for text in (
                    missing and f"missing {sorted(missing)}",
                    unknown and f"unknown {sorted(unknown)}",
                )
                if text
            )
        )
    return record


def _text(value: object, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise _Invalid(
            f"{where}: a non-empty string{' or null' if optional else ''}, not {value!r}"
        )
    return value


def _resources(stated: object, where: str) -> Resources:
    counts = _expect(stated, where, set(RESOURCE_NAMES), set())
    try:
        return Resources(**counts)
    except ValueError as error:
        raise _Invalid(f"{where}: {error}") from error


_RECORD_FIELDS = {"digest", "slr_count", "totals", "slrs"}


def _record(stated: object, where: str, *, overlay: bool) -> Record:
    """A resource record as stored, every field stated (``capped`` only where it is)
    and its digest checked; an overlay may leave out what follows from the rest: the
    ``digest``, the ``totals`` and ``slr_count`` of stated ``slrs``, and the ``slrs``
    of a device of one SLR (its totals)."""
    if overlay:
        found = _expect(stated, where, set(), _RECORD_FIELDS | {"capped"})
    else:
        found = _expect(stated, where, _RECORD_FIELDS, {"capped"})
        if found.get("capped", True) is not True:
            raise _Invalid(f"{where}: capped is stated only where it is true")
    slrs = found.get("slrs")
    if slrs is not None:
        if not isinstance(slrs, list) or not slrs:
            raise _Invalid(f"{where}: slrs is a non-empty list of SLRs, or null")
        slrs = tuple(_resources(slr, f"{where}: SLR {i}") for i, slr in enumerate(slrs))
    if "totals" in found:
        totals = _resources(found["totals"], f"{where}: totals")
    elif slrs is not None:
        totals = sum(slrs, Resources())
    else:
        raise _Invalid(f"{where}: states its totals or its SLRs")
    count = found.get("slr_count", 1 if slrs is None else len(slrs))
    if slrs is None and count == 1:
        slrs = (totals,)
    try:
        record = Record(
            totals=totals, slr_count=count, slrs=slrs, capped=found.get("capped", False)
        )
    except ValueError as error:
        raise _Invalid(f"{where}: {error}") from error
    if found.get("digest", record.digest) != record.digest:
        raise _Invalid(f"{where}: digest {found['digest']!r} is not its facts' {record.digest!r}")
    return record


@dataclass
class _Entries:
    """The records of the committed data or of an overlay, checked one by one."""

    resources: dict[str, Record] = field(default_factory=dict)
    devices: dict[str, dict[str, Any]] = field(default_factory=dict)
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)


_DEVICE_FIELDS = {"name", "architecture", "family", "resources"}
_PART_FIELDS = {"name", "device", "package", "speed", "temperature"}
_OVERLAY_DEVICE = {"fabric", "dsp", "uram_init", "overrides", "source"}
_OVERLAY_PART = {"overrides", "source"}


def _read_entries(
    resources: Iterable[object],
    devices: Iterable[object],
    parts: Iterable[object],
    *,
    overlay: bool,
    source: str | None = None,
) -> _Entries:
    entries = _Entries()
    for index, record in enumerate(resources):
        where = f"resources[{index}]"
        found = _record(record, where, overlay=overlay)
        entries.resources[found.digest] = found
    for index, record in enumerate(devices):
        where = f"devices[{index}]"
        stated = dict(_expect(record, where, _DEVICE_FIELDS, _OVERLAY_DEVICE if overlay else set()))
        name = _text(stated["name"], f"{where}: name")
        where = f"device {name}"
        for key in ("architecture", "family"):
            _text(stated[key], f"{where}: {key}")
        if isinstance(stated["resources"], dict):  # an overlay may state its record inline
            if not overlay:
                raise _Invalid(f"{where}: resources is a digest")
            found = _record(stated["resources"], f"{where}: resources", overlay=True)
            stated["resources"] = found.digest
            entries.resources[found.digest] = found
        _text(stated["resources"], f"{where}: resources")
        if overlay:
            stated["source"] = _source(stated, source, where)
        _no_duplicate(entries.devices, str(name), where)
        entries.devices[str(name).lower()] = stated
    for index, record in enumerate(parts):
        where = f"parts[{index}]"
        stated = dict(_expect(record, where, _PART_FIELDS, _OVERLAY_PART if overlay else set()))
        name = _text(stated["name"], f"{where}: name")
        where = f"part {name}"
        _text(stated["device"], f"{where}: device")
        for key in ("package", "speed", "temperature"):
            _text(stated[key], f"{where}: {key}", optional=True)
        if overlay:
            stated["source"] = _source(stated, source, where)
        _no_duplicate(entries.parts, str(name), where)
        entries.parts[str(name).lower()] = stated
    return entries


def _source(stated: Mapping[str, object], default: str | None, where: str) -> str:
    found = stated.get("source", default)
    if not isinstance(found, str) or not found:
        raise _Invalid(f"{where}: states no source (nor does the overlay)")
    return found


def _no_duplicate(entries: Mapping[str, object], name: str, where: str) -> None:
    if name.lower() in entries:
        raise _Invalid(f"{where}: stated twice")


def _jsonl(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- loading -----------------------------------------------------------------------------


def load(data: Path = DATA, overlay: Path | None = None) -> Catalog:
    """The catalog in ``data`` (the committed one), with ``overlay`` laid over it.

    The overlay is a JSON object of the committed data's records, each list
    optional: ``resources`` (``{"slrs": [...]}``, a ``digest`` checked if stated),
    ``devices`` (``resources`` the digest of a shipped or overlay record, or the
    record itself; ``fabric``, ``dsp`` and ``uram_init`` stated where the pair has
    no rule or the device is not built as its pair's) and ``parts``; ``source``, the
    file's, or each entry's own; ``overrides: true`` on an entry that replaces a
    shipped one of its name. Anything else is refused (``catalog-overlay-invalid``)."""
    *records, manifest_file = (data / name for name in FILES)
    manifest = json.loads(manifest_file.read_text())
    shipped = _read_entries(*(_jsonl(path) for path in records), overlay=False)
    tool = manifest["tool"]
    origin = (
        f"FINN's part catalog, from Vivado {tool['version']} ({tool['build']}): get_parts and "
        "each SLR's sites in an empty design"
    )
    if overlay is None:
        return _build(shipped, _Entries(), manifest, origin, None)
    try:
        stated = json.loads(Path(overlay).read_text())
        top = _expect(stated, str(overlay), set(), {"source", "resources", "devices", "parts"})
        added = _read_entries(
            top.get("resources", []),
            top.get("devices", []),
            top.get("parts", []),
            overlay=True,
            source=_text(top["source"], "source") if "source" in top else None,
        )
        return _build(shipped, added, manifest, origin, Path(overlay))
    except (OSError, json.JSONDecodeError, _Invalid) as error:
        raise TargetRefused("catalog-overlay-invalid", f"{overlay}: {error}") from error


def _build(
    shipped: _Entries,
    added: _Entries,
    manifest: Mapping[str, object],
    origin: str,
    overlay: Path | None,
) -> Catalog:
    for kind, theirs, ours in (
        ("device", shipped.devices, added.devices),
        ("part", shipped.parts, added.parts),
    ):
        for key, entry in ours.items():
            if key in theirs and not entry.get("overrides", False):
                raise _Invalid(
                    f"{kind} {entry['name']} redefines the catalog's {theirs[key]['name']}; "
                    'an entry that replaces it says "overrides": true'
                )
            if key not in theirs and entry.get("overrides", False):
                raise _Invalid(f"{kind} {entry['name']} overrides nothing the catalog ships")
    records = {**shipped.resources, **added.resources}
    stated_devices = {**shipped.devices, **added.devices}
    by_digest: dict[str, list[str]] = defaultdict(list)
    for entry in stated_devices.values():
        if entry["resources"] not in records:
            raise _Invalid(f"device {entry['name']}: no resource record {entry['resources']}")
        by_digest[entry["resources"]].append(entry["name"])
    devices = {
        key: _device(entry, records, by_digest, origin, overlay)
        for key, entry in sorted(stated_devices.items())
    }
    parts: dict[str, Part] = {}
    for key, entry in sorted({**shipped.parts, **added.parts}.items()):
        device = devices.get(str(entry["device"]).lower())
        if device is None:
            raise _Invalid(f"part {entry['name']}: no device {entry['device']!r}")
        part_source = entry.get("source")
        if part_source is None:
            source = device.source
        else:
            source = f"{part_source} (overlay {overlay})"
            if source != device.source:
                source = f"{source}; its device {device.name}: {device.source}"
        parts[key] = Part(
            name=entry["name"],
            device=device,
            package=entry["package"],
            speed=entry["speed"],
            temperature=entry["temperature"],
            source=source,
        )
    return Catalog(devices, parts, manifest, overlay)


def _device(
    entry: Mapping[str, object],
    records: Mapping[str, Record],
    by_digest: Mapping[str, list[str]],
    origin: str,
    overlay: Path | None,
) -> Device:
    name, architecture, family = (
        str(entry["name"]),
        str(entry["architecture"]),
        str(entry["family"]),
    )
    where = f"device {name}"
    found = rule(architecture, family)
    fabric: Fabric | None
    dsp: DspBlock | None
    if "fabric" in entry or "dsp" in entry:
        try:
            fabric, dsp = Fabric(entry["fabric"]), DspBlock(entry["dsp"])
        except (KeyError, ValueError) as error:
            raise _Invalid(
                f"{where}: states its fabric ({[each.value for each in Fabric]}) and its DSP "
                f"block ({[each.value for each in DspBlock]}) together: {error}"
            ) from error
        uram_init = entry.get("uram_init", False)
        if not isinstance(uram_init, bool):
            raise _Invalid(f"{where}: uram_init is true or false")
        unsupported = None
    elif "uram_init" in entry:
        raise _Invalid(f"{where}: uram_init is stated with the fabric and DSP block it is of")
    elif found is None:
        raise _Invalid(
            f"{where}: FINN has no rule for {architecture}/{family}: state its fabric and dsp"
        )
    else:
        fabric, dsp, uram_init, unsupported = (
            found.fabric,
            found.dsp,
            found.uram_init,
            found.unsupported,
        )
    digest = str(entry["resources"])
    source = origin if "source" not in entry else f"{entry['source']} (overlay {overlay})"
    capped = records[digest].capped_resources()
    return Device(
        name=name,
        architecture=architecture,
        family=family,
        fabric=fabric,
        dsp=dsp,
        uram_init=uram_init,
        resources=records[digest].totals,
        slr_count=records[digest].slr_count,
        slrs=records[digest].slrs,
        capped=capped,
        # The reviewed row is of Vivado's device: an overlay's entry states its own.
        cap_evidence=CAP_EVIDENCE.get(name) if capped and "source" not in entry else None,
        digest=digest,
        shared_with=tuple(sorted(each for each in by_digest[digest] if each != name)),
        source=source,
        unsupported=unsupported,
    )


@functools.cache
def catalog() -> Catalog:
    """The catalog this process resolves by: the committed one, with the overlay the
    machine setting ``FINN_PLATFORM_CATALOG`` names, read once."""
    named = os.environ.get(OVERLAY_SETTING) or None
    return load(DATA, None if named is None else Path(named))


def part(name: str) -> Part:
    """``catalog().part(name)``."""
    return catalog().part(name)


def device(name: str) -> Device:
    """``catalog().device(name)``."""
    return catalog().device(name)


def parts(pattern: str | None = None, *, family: str | None = None) -> list[Part]:
    """``catalog().parts(pattern, family=family)``."""
    return catalog().parts(pattern, family=family)


def part_report(name: str) -> dict[str, object]:
    """The target's part as the part catalog states it: its device, the device's
    resources per SLR (``None`` where an overlay states no split), on a reduced die of
    several SLRs each SLR's site capacity under the device's totals as a cap (``cap``:
    the totals, the resources they cap, which of those caps a fill proved and which are
    inferred, and the evidence), the devices that share them, and where its facts come
    from (``source``); a part the catalog does not have says why, with no source."""
    try:
        found = part(name)
    except TargetRefused as refused:
        return {"name": name, "source": None, "refused": str(refused)}
    device = found.device
    cap: dict[str, object] | None = None
    if device.capped:
        review = device.cap_evidence
        proven = set() if review is None else review.proven
        cap = {
            "slrs": SLRS_CAPPED,
            "totals": asdict(device.resources),
            "proven": [each for each in device.capped if each in proven],
            "inferred": [each for each in device.capped if each not in proven],
            "evidence": found.source if review is None else review.evidence,
        }
    return {
        "name": found.name,
        "device": device.name,
        "architecture": device.architecture,
        "family": device.family,
        "slrs": None if device.slrs is None else [asdict(slr) for slr in device.slrs],
        "cap": cap,
        "shared_with": list(device.shared_with),
        "source": found.source,
    }


__all__ = [
    "DATA",
    "FILES",
    "OVERLAY_SETTING",
    "Catalog",
    "Device",
    "Part",
    "catalog",
    "device",
    "load",
    "part",
    "part_report",
    "parts",
    "SLRS_CAPPED",
    "Record",
]
