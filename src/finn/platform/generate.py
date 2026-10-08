# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The generator of FINN's part catalog (``finn.platform.catalog``), from the
installed Vivado's part database.

::

    python -m finn.platform.generate write [--jobs N] [--work DIR] [--extracted DIR]
    python -m finn.platform.generate check [--jobs N] [--work DIR]

``write`` extracts and writes ``data/`` beside this module; ``check`` extracts into a
temporary directory and compares it with the committed data, reporting the parts
and devices added or removed and the fields that changed; exit 0 is the pass.
``--extracted`` derives the catalog from an earlier extraction's logs (``--work``
keeps them), without Vivado. Resolution never runs Vivado.

The extraction (``catalog.tcl``), in ``--jobs`` Vivado processes at once:

1. every installed part's identity (name, device, package, speed, temperature
   grade, ARCHITECTURE and FAMILY) and its device-level totals (LUT_ELEMENTS,
   FLIPFLOPS, BLOCK_RAMS, ULTRA_RAMS, DSP, SLRS); every part of a device must state
   the same pair and totals;
2. for each device, an empty design linked on its first part (no synthesis): every
   site of each SLR by site type, and one SLICEM's BELs.

From them:

- each SLR's resources: ``lut`` and ``ff`` the slices (SLICEL and SLICEM sites)
  times a slice's LUTs and flip-flops (its LUT6 and FF BELs), ``bram18`` the RAMB18
  sites, ``uram`` the URAM288 sites, ``dsp`` the primary DSP sites. Their number
  must be the device's SLRS and their sum its totals (``2 x BLOCK_RAMS`` RAMB18s; an
  absent ULTRA_RAMS is none). A device sold on a larger die (an XCZU2EG on the
  XCZU3EG's) has more sites than it states: where the sites cover the totals in
  every resource and exceed one, the device is a reduced die, its record states its
  totals, and over one SLR that SLR's; over several, Vivado does not state the split
  and the record states none (``finn.platform.catalog.SLRS_UNSTATED``). Anything
  else is ``slr-sum``. The probe's log names each reduced device;
- the rule probe: each (ARCHITECTURE, FAMILY) pair needs a rule
  (``finn.platform.architectures``; ``unreviewed-architecture`` otherwise). A
  supported rule's sample part must be one of its pair, and every device of the
  pair must show its fabric's SLICEM (LUTs, F7/F8/F9 multiplexers, carry), its DSP
  block's site and an UltraRAM site the fabric has (``architecture-mismatch``
  otherwise). The probe's log (``probe.log``) has a line per pair;
- the records: each distinct per-SLR resource record once, keyed by the digest of
  its normalized facts; each device's identity and its record's digest; each part's
  identity and its device. One record a line, sorted.
- the manifest: the tool's build, the series and families, the counts, the
  digests of the generator and of the rules the probe checked, and the digest of
  the catalog's facts (no timestamp, no path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from finn.kernels.target import DspBlock, Fabric
from finn.kernels.utilization import RESOURCE_NAMES, Resources
from finn.platform.architectures import RULES, SERIES, Rule, rules_digest
from finn.platform.catalog import Record
from finn.util.toolchain import machine_toolchain

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCRIPT = HERE / "catalog.tcl"
FILES = ("resources.jsonl", "devices.jsonl", "parts.jsonl", "manifest.json")

SLICE_SITES = frozenset({"SLICEL", "SLICEM"})
BRAM18_SITES = frozenset({"RAMB18E1", "FIFO18E1", "RAMB181", "RAMBFIFO18", "RAMB18_L", "RAMB18_U"})
"""The sites a RAMB18 is placed on: each 36 Kb block RAM is two of them."""
URAM_SITES = frozenset({"URAM288"})
DSP_SITES: Mapping[str, DspBlock] = {
    "DSP48E1": DspBlock.DSP48E1,
    "DSP48E2": DspBlock.DSP48E2,
    "DSP58_PRIMARY": DspBlock.DSP58,
}
"""The site a DSP is placed on, by its block (a DSP58_CPLX site pairs two DSP58s)."""
COUNTED = SLICE_SITES | BRAM18_SITES | URAM_SITES | frozenset(DSP_SITES)
NOT_COUNTED = frozenset(
    {"RAMBFIFO36E1", "RAMBFIFO36", "RAMB36", "DSP58_CPLX", "URAM_CAS_DLY", "SLICE_FE"}
)
"""Memory and DSP sites that are another view of counted ones (a RAMB36 is two RAMB18
sites; a complex DSP58 pair is two DSP58_PRIMARY sites) or hold no resource."""
_RESOURCE_SITE = re.compile(r"^(SLICE|RAMB|FIFO|URAM|DSP)")


@dataclass(frozen=True)
class Signature:
    """What the probe reads of a fabric: a SLICEM's LUT6 BELs, its read
    multiplexers (F7, F8, F9), its carry BEL, and the UltraRAM site types it may
    have."""

    luts: int
    muxes: frozenset[str]
    carry: str
    uram: frozenset[str]


SIGNATURES: Mapping[Fabric, Signature] = {
    Fabric.SERIES7: Signature(4, frozenset({"F7AMUX", "F7BMUX", "F8MUX"}), "CARRY4", frozenset()),
    Fabric.ULTRASCALE: Signature(
        8,
        frozenset(
            {"F7MUX_AB", "F7MUX_CD", "F7MUX_EF", "F7MUX_GH", "F8MUX_BOT", "F8MUX_TOP", "F9MUX"}
        ),
        "CARRY8",
        frozenset({"URAM288"}),
    ),
    Fabric.VERSAL: Signature(8, frozenset(), "LOOKAHEAD8", frozenset({"URAM288"})),
}
"""What FINN means by each fabric, as the site probe reads it."""

_LUT6 = re.compile(r"^[A-H]6LUT$")
_FF = re.compile(r"^[A-H](5FF|FF|FF2)$")
_MUX = re.compile(r"^F[789]")
_CARRY = frozenset({"CARRY4", "CARRY8", "LOOKAHEAD8"})


class GenerationError(RuntimeError):
    """A catalog that cannot be generated, by name: ``unreviewed-architecture``,
    ``architecture-mismatch``, ``slr-sum``, ``device-inconsistent``, ``extraction``."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name}: {message}")
        self.name = name


# -- extraction --------------------------------------------------------------------------


@dataclass(frozen=True)
class PartRow:
    """A part as Vivado states it (an absent property is ``None``)."""

    name: str
    device: str
    package: str | None
    speed: str | None
    temperature: str | None
    architecture: str
    family: str
    totals: tuple[str | None, ...]  # LUT_ELEMENTS FLIPFLOPS BLOCK_RAMS ULTRA_RAMS DSP SLRS


@dataclass(frozen=True)
class DeviceProbe:
    """A device as the linked design shows it: each SLR's sites by type, in Vivado's
    SLR order, and one SLICEM's BELs."""

    device: str
    part: str
    slrs: tuple[Mapping[str, int], ...]
    bels: frozenset[str]

    def site_types(self) -> set[str]:
        return {each for slr in self.slrs for each in slr}

    def resource_sites(self) -> set[str]:
        return {each for each in self.site_types() if _RESOURCE_SITE.match(each)}

    def observed(self) -> str:
        """What the probe read, as the probe's log states it."""
        luts = sum(1 for bel in self.bels if _LUT6.match(bel))
        muxes = sorted(bel for bel in self.bels if _MUX.match(bel))
        carry = sorted(self.bels & _CARRY)
        dsp = sorted(each for each in self.site_types() if each in DSP_SITES)
        uram = sorted(each for each in self.site_types() if each in URAM_SITES)
        other = sorted(self.resource_sites() - COUNTED - NOT_COUNTED)
        return (
            f"SLICEM {luts} LUTs, muxes {','.join(muxes) or 'none'}, carry "
            f"{','.join(carry) or 'none'}; DSP sites {','.join(dsp) or 'none'}; URAM sites "
            f"{','.join(uram) or 'none'}"
            + (f"; unrecognised sites {','.join(other)}" if other else "")
        )


def _field(value: str) -> str | None:
    return value if value != "" else None


def _lines(text: str, tag: str) -> list[list[str]]:
    prefix = f"CATALOG {tag} "
    return [line[len(prefix) :].split("|") for line in text.splitlines() if line.startswith(prefix)]


def parse_tool(texts: Iterable[str]) -> dict[str, str]:
    """The tool's version and build, which every extraction log states alike."""
    found = {tuple(row) for text in texts for row in _lines(text, "TOOL")}
    if len(found) != 1:
        raise GenerationError("extraction", f"the logs name {len(found)} tools: {sorted(found)}")
    ((version, build),) = found
    return {"version": version, "build": build}


def parse_parts(texts: Iterable[str]) -> list[PartRow]:
    """Every part of the ``parts`` logs, each exactly once, all of them."""
    counts: set[int] = set()
    rows: dict[str, PartRow] = {}
    for text in texts:
        counts.update(int(count) for (count,) in _lines(text, "COUNT"))
        if "CATALOG COMPLETE" not in text:
            raise GenerationError("extraction", "a parts extraction did not complete")
        for fields in _lines(text, "PART"):
            name, device, package, speed, temperature, architecture, family, *totals = fields
            if name in rows:
                raise GenerationError("extraction", f"{name} was extracted twice")
            rows[name] = PartRow(
                name,
                device,
                _field(package),
                _field(speed),
                _field(temperature),
                architecture,
                family,
                tuple(_field(each) for each in totals),
            )
    if len(counts) != 1 or counts.pop() != len(rows):
        raise GenerationError("extraction", f"{len(rows)} parts extracted, not every part")
    return [rows[name] for name in sorted(rows)]


def parse_devices(texts: Iterable[str]) -> dict[str, DeviceProbe]:
    """Every linked device of the ``devices`` logs, by device name."""
    probes: dict[str, DeviceProbe] = {}
    for text in texts:
        failed = _lines(text, "LINK_FAIL")
        if failed:
            raise GenerationError("extraction", f"link_design failed: {failed}")
        devices = {part: device for part, device in _lines(text, "DEVICE")}
        slrs: dict[str, list[dict[str, int]]] = defaultdict(list)
        for part, _slr, _fabric, counts in _lines(text, "SLR"):
            slrs[part].append(
                {k: int(v) for k, v in (each.split("=") for each in counts.split(",") if each)}
            )
        bels = {
            part: frozenset(filter(None, found.split(",")))
            for part, found in _lines(text, "SLICEM")
        }
        for part, device in devices.items():
            if part not in bels:
                raise GenerationError("extraction", f"{part}: the probe did not complete")
            probes[device] = DeviceProbe(device, part, tuple(slrs[part]), bels[part])
    return probes


def _vivado(work: Path, name: str, args: Sequence[str]) -> subprocess.Popen[bytes]:
    toolchain = machine_toolchain()
    command = toolchain.command(
        "vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", SCRIPT, "-tclargs", *args
    )
    log = (work / f"{name}.log").open("wb")
    return subprocess.Popen(
        command, cwd=work, env=dict(toolchain.environment), stdout=log, stderr=subprocess.STDOUT
    )


def _wait(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    failed = [str(process.args) for process in processes if process.wait() != 0]
    if failed:
        raise GenerationError("extraction", f"{len(failed)} Vivado runs failed: {failed[0]}")


def extract(work: Path, jobs: int) -> None:
    """Run the extraction into ``work``: ``parts-<i>.log`` and ``devices-<i>.log``."""
    work.mkdir(parents=True, exist_ok=True)
    _wait([_vivado(work, f"parts-{i}", ["parts", str(i), str(jobs)]) for i in range(jobs)])
    parts = parse_parts(path.read_text() for path in sorted(work.glob("parts-*.log")))
    first: dict[str, str] = {}
    for row in parts:
        first.setdefault(row.device, row.name)
    # The largest devices first, dealt round so each run links a similar share.
    by_size = sorted(first, key=lambda device: -int(_totals(parts, device)[0] or 0))
    batches = [by_size[i::jobs] for i in range(jobs)]
    _wait(
        [
            _vivado(work, f"devices-{i}", ["devices", *(first[device] for device in batch)])
            for i, batch in enumerate(batches)
            if batch
        ]
    )


def _totals(parts: Sequence[PartRow], device: str) -> tuple[str | None, ...]:
    return next(row.totals for row in parts if row.device == device)


# -- derivation --------------------------------------------------------------------------


def slr_resources(probe: DeviceProbe) -> tuple[Resources, ...]:
    """Each SLR's resources from its sites and the device's SLICEM BELs."""
    luts = sum(1 for bel in probe.bels if _LUT6.match(bel))
    ffs = sum(1 for bel in probe.bels if _FF.match(bel))
    found = []
    for sites in probe.slrs:
        slices = sum(sites.get(each, 0) for each in SLICE_SITES)
        found.append(
            Resources(
                lut=slices * luts,
                ff=slices * ffs,
                bram18=sum(sites.get(each, 0) for each in BRAM18_SITES),
                uram=sum(sites.get(each, 0) for each in URAM_SITES),
                dsp=sum(sites.get(each, 0) for each in DSP_SITES),
            )
        )
    return tuple(found)


def device_totals(row: PartRow) -> tuple[Resources, int]:
    """The device-level totals a part states, in the catalog's units, and its SLRs."""
    lut, ff, bram, uram, dsp, slrs = row.totals
    if None in (lut, ff, bram, dsp, slrs):
        raise GenerationError("extraction", f"{row.name} states no totals: {row.totals}")
    totals = Resources(
        lut=int(lut),  # type: ignore[arg-type]
        ff=int(ff),  # type: ignore[arg-type]
        bram18=2 * int(bram),  # type: ignore[arg-type]
        uram=0 if uram is None else int(uram),
        dsp=int(dsp),  # type: ignore[arg-type]
    )
    return totals, int(slrs)  # type: ignore[arg-type]


def confirm(rule: Rule, probe: DeviceProbe) -> str | None:
    """Why ``probe`` contradicts the supported ``rule``, or ``None`` if it confirms it."""
    assert rule.fabric is not None and rule.dsp is not None
    signature = SIGNATURES[rule.fabric]
    luts = sum(1 for bel in probe.bels if _LUT6.match(bel))
    muxes = frozenset(bel for bel in probe.bels if _MUX.match(bel))
    dsp = {each for each in probe.site_types() if each in DSP_SITES}
    uram = {each for each in probe.site_types() if each in URAM_SITES}
    other = probe.resource_sites() - COUNTED - NOT_COUNTED
    wrong = []
    if luts != signature.luts or muxes != signature.muxes or signature.carry not in probe.bels:
        wrong.append(f"its SLICEM is not {rule.fabric.value}'s")
    if dsp != {site for site, block in DSP_SITES.items() if block is rule.dsp}:
        wrong.append(f"its DSP sites are not {rule.dsp.value}")
    if not uram <= signature.uram:
        wrong.append(f"{rule.fabric.value} has no {sorted(uram)} sites")
    if other:
        wrong.append(f"its sites {sorted(other)} are not recognised")
    return "; ".join(wrong) or None


@dataclass(frozen=True)
class Catalog:
    """A generated catalog: its files' records and the probe's log."""

    resources: list[dict[str, object]]
    devices: list[dict[str, object]]
    parts: list[dict[str, object]]
    manifest: dict[str, object]
    probe_log: list[str]

    def files(self) -> dict[str, str]:
        """Each data file's text: one record a line, sorted; the manifest indented."""
        lines = {
            "resources.jsonl": self.resources,
            "devices.jsonl": self.devices,
            "parts.jsonl": self.parts,
        }
        found = {
            name: "".join(json.dumps(record, separators=(", ", ": ")) + "\n" for record in records)
            for name, records in lines.items()
        }
        found["manifest.json"] = json.dumps(self.manifest, indent=2) + "\n"
        return found


def generator_digest() -> str:
    """The digest of the generator: this module and its Tcl."""
    hashed = hashlib.sha256()
    for path in (Path(__file__).resolve(), SCRIPT):
        hashed.update(path.read_bytes())
    return "sha256:" + hashed.hexdigest()


def facts_digest(resources: object, devices: object, parts: object) -> str:
    """The semantic digest of the catalog's normalized facts (no timestamp, no path)."""
    encoded = json.dumps([resources, devices, parts], sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def derive(
    tool: Mapping[str, str],
    parts: Sequence[PartRow],
    probes: Mapping[str, DeviceProbe],
    rules: Mapping[tuple[str, str], Rule] = RULES,
) -> Catalog:
    """The catalog from an extraction, every check applied."""
    by_device: dict[str, list[PartRow]] = defaultdict(list)
    for row in parts:
        by_device[row.device].append(row)
    folded: dict[str, str] = {}
    for name in [row.name for row in parts] + list(by_device):
        if folded.setdefault(name.lower(), name) != name:
            raise GenerationError("extraction", f"{name} and {folded[name.lower()]} collide")
    log: list[str] = []
    failures: list[str] = []
    records: dict[str, Record] = {}
    devices: list[dict[str, object]] = []
    reduced: list[str] = []
    for device in sorted(by_device):
        rows = by_device[device]
        stated = {(row.architecture, row.family, row.totals) for row in rows}
        if len(stated) != 1:
            raise GenerationError("device-inconsistent", f"{device}'s parts differ: {stated}")
        if device not in probes:
            raise GenerationError("extraction", f"{device} was not probed")
        totals, slr_count = device_totals(rows[0])
        sites = slr_resources(probes[device])
        summed = sum(sites, Resources())
        slrs: tuple[Resources, ...] | None = sites
        if len(sites) != slr_count or not all(
            getattr(summed, name) >= getattr(totals, name) for name in RESOURCE_NAMES
        ):
            failures.append(
                f"slr-sum: {device}: {len(sites)} SLRs summing to {summed}, the device "
                f"states {slr_count} and {totals}"
            )
            continue
        if summed != totals:
            # A reduced die: the device is sold with fewer resources than its die's
            # sites. Vivado states the totals, not their split over the SLRs.
            reduced.append(device)
            slrs = (totals,) if slr_count == 1 else None
            log.append(f"{device}|REDUCED|die {summed}|device {totals} over {slr_count} SLRs")
        record = Record(totals=totals, slr_count=slr_count, slrs=slrs)
        if records.setdefault(record.digest, record) != record:
            raise GenerationError("extraction", f"two records share the digest {record.digest}")
        devices.append(
            {
                "name": device,
                "architecture": rows[0].architecture,
                "family": rows[0].family,
                "resources": record.digest,
            }
        )
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for each in devices:
        pairs[(str(each["architecture"]), str(each["family"]))].append(str(each["name"]))
    names = {row.name: row for row in parts}
    for pair in sorted(pairs):
        found = rules.get(pair)
        head = f"{pair[0]}|{pair[1]}"
        if found is None:
            failures.append(f"unreviewed-architecture: {head}: no rule")
            log.append(f"{head}|no rule|UNREVIEWED")
            continue
        sample = names.get(found.sample)
        if sample is None or (sample.architecture, sample.family) != pair:
            failures.append(f"unreviewed-architecture: {head}: sample {found.sample} is not its")
            log.append(f"{head}|{found.sample}|not of the pair|UNREVIEWED")
            continue
        observed = probes[sample.device].observed()
        if found.unsupported is not None:
            log.append(f"{head}|{found.sample}|{observed}|UNSUPPORTED: {found.unsupported}")
            continue
        assert found.fabric is not None and found.dsp is not None
        wrong = {
            device: why
            for device in pairs[pair]
            if (why := confirm(found, probes[device])) is not None
        }
        rule_text = f"{found.fabric.value}/{found.dsp.value}"
        if wrong:
            failures.append(f"architecture-mismatch: {head} ({rule_text}): {wrong}")
            log.append(f"{head}|{found.sample}|{observed}|MISMATCH {rule_text}: {wrong}")
        else:
            log.append(
                f"{head}|{found.sample}|{observed}|CONFIRMED {rule_text} on "
                f"{len(pairs[pair])} devices"
            )
    stale = sorted(set(rules) - set(pairs))
    if stale:
        failures.append(f"unreviewed-architecture: rules for pairs no part has: {stale}")
    if failures:
        raise GenerationError(failures[0].split(":")[0], "\n".join(failures))
    resource_records = [records[digest].stored() for digest in sorted(records)]
    part_records: list[dict[str, object]] = [
        {
            "name": row.name,
            "device": row.device,
            "package": row.package,
            "speed": row.speed,
            "temperature": row.temperature,
        }
        for row in parts
    ]
    series: dict[str, list[str]] = defaultdict(list)
    for architecture, family in sorted(pairs):
        series[rules[(architecture, family)].series].append(family)
    manifest = {
        "tool": dict(tool),
        "series": {name: sorted(series[name]) for name in SERIES if name in series},
        "counts": {
            "parts": len(part_records),
            "devices": len(devices),
            "resources": len(resource_records),
            "pairs": len(pairs),
            "reduced": len(reduced),
            "reduced_without_slrs": sum(
                1 for each in devices if records[str(each["resources"])].slrs is None
            ),
        },
        "checks": [
            "every part of a device states its device's ARCHITECTURE, FAMILY and totals",
            "each device's per-SLR resources, from its SLRs' sites, sum to its "
            "LUT_ELEMENTS, FLIPFLOPS, 2 x BLOCK_RAMS, ULTRA_RAMS (absent: none) and DSP, "
            "over SLRS SLRs; on a reduced die (sites covering the totals in every "
            "resource, more in one) the record states the device's totals, and over one "
            "SLR that SLR's",
            "each supported rule's fabric and DSP block are confirmed on every device of "
            "its pair by the site probe",
        ],
        "generator": generator_digest(),
        "rules": rules_digest(rules),
        "facts": facts_digest(resource_records, devices, part_records),
    }
    return Catalog(resource_records, devices, part_records, manifest, log)


def derive_from(work: Path) -> Catalog:
    """The catalog from an extraction's logs in ``work``."""
    texts = {path.name: path.read_text() for path in sorted(work.glob("*.log"))}
    parts = parse_parts(text for name, text in texts.items() if name.startswith("parts-"))
    probes = parse_devices(text for name, text in texts.items() if name.startswith("devices-"))
    return derive(parse_tool(texts.values()), parts, probes)


def write_files(catalog: Catalog, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in catalog.files().items():
        (directory / name).write_text(text)


# -- check -------------------------------------------------------------------------------


def differences(generated: Mapping[str, str], committed: Path) -> list[str]:
    """What differs between a generated catalog and the committed one: records added
    or removed, and each field changed, by file and record name."""
    found: list[str] = []
    for name in FILES:
        path = committed / name
        theirs = path.read_text() if path.exists() else ""
        ours = generated[name]
        if theirs == ours:
            continue
        if name == "manifest.json":
            old = json.loads(theirs) if theirs else {}
            new = json.loads(ours)
            found += [
                f"{name}: {key}: {old.get(key)!r} -> {new.get(key)!r}"
                for key in sorted(set(old) | set(new))
                if old.get(key) != new.get(key)
            ]
            continue
        key = "digest" if name == "resources.jsonl" else "name"
        old_records = {r[key]: r for r in map(json.loads, theirs.splitlines())}
        new_records = {r[key]: r for r in map(json.loads, ours.splitlines())}
        found += [f"{name}: removed {each}" for each in sorted(set(old_records) - set(new_records))]
        found += [f"{name}: added {each}" for each in sorted(set(new_records) - set(old_records))]
        for each in sorted(set(old_records) & set(new_records)):
            old, new = old_records[each], new_records[each]
            found += [
                f"{name}: {each}: {field}: {old.get(field)!r} -> {new.get(field)!r}"
                for field in sorted(set(old) | set(new))
                if old.get(field) != new.get(field)
            ]
        if not found:
            found.append(f"{name}: the same records, written differently")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m finn.platform.generate")
    parser.add_argument("command", choices=("write", "check"))
    parser.add_argument("--jobs", type=int, default=8, help="Vivado processes at once")
    parser.add_argument("--work", type=Path, help="keep the extraction's logs here")
    parser.add_argument("--extracted", type=Path, help="derive from these logs (write only)")
    parser.add_argument("--data", type=Path, default=DATA, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.extracted is not None and options.command == "check":
        parser.error("check extracts again; --extracted is for write")
    with tempfile.TemporaryDirectory(prefix="finn-catalog-") as scratch:
        work = options.extracted or options.work or Path(scratch)
        if options.extracted is None:
            extract(work, options.jobs)
        try:
            catalog = derive_from(work)
        except GenerationError as error:
            print(f"generation failed: {error}", file=sys.stderr)
            return 1
        (work / "probe.log").write_text("".join(line + "\n" for line in catalog.probe_log))
        print("".join(line + "\n" for line in catalog.probe_log), end="")
        counts = catalog.manifest["counts"]
        if options.command == "write":
            write_files(catalog, options.data)
            print(f"wrote {options.data}: {counts}")
            return 0
        found = differences(catalog.files(), options.data)
        for line in found:
            print(line)
        print(f"check: {counts}; {'differs' if found else 'identical'}")
        return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
