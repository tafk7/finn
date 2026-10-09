# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The part catalog (``finn.platform.catalog``): its loader, queries and refusals on
synthetic data, the overlay a user adds devices with, the committed data's
invariants, and (with Vivado) the generator against the installed part database."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from kernels.xsim import requires_vivado

from finn.kernels.target import DspBlock
from finn.kernels.utilization import RESOURCE_NAMES, Fabric, Resources
from finn.platform import BOARDS, TargetRefused, device, part_report, parts, resolve_target
from finn.platform.architectures import CAP_EVIDENCE, RULES, rules_digest
from finn.platform.catalog import (
    DATA,
    OVERLAY_SETTING,
    SLRS_CAPPED,
    Catalog,
    Record,
    catalog,
    load,
)
from finn.platform.generate import (
    DeviceProbe,
    GenerationError,
    PartRow,
    _vivado,
    _wait,
    confirm,
    derive,
    parse_devices,
    slr_resources,
)

ZU3 = [Resources(lut=70_560, ff=141_120, bram18=432, uram=0, dsp=360)]
ZU7 = [Resources(lut=230_400, ff=460_800, bram18=624, uram=96, dsp=1_728)]
VU5P_SLR = Resources(lut=394_080, ff=788_160, bram18=1_440, uram=320, dsp=2_280)
VU5P = Resources(lut=600_577, ff=1_201_154, bram18=2_048, uram=470, dsp=3_474)


def counts(slrs: list[Resources]) -> list[dict[str, int]]:
    return [asdict(slr) for slr in slrs]


def record(slrs: list[Resources]) -> Record:
    return Record(totals=sum(slrs, Resources()), slr_count=len(slrs), slrs=tuple(slrs))


def write_data(directory: Path) -> Path:
    """A catalog of three devices on two records (ZU3CG and ZU3EG share one) and the
    parts on them, in the committed data's schema."""
    directory.mkdir()
    records = {record(ZU3).digest: record(ZU3), record(ZU7).digest: record(ZU7)}
    devices = [
        ("xczu3cg", "zynquplus", "zynquplus", ZU3),
        ("xczu3eg", "zynquplus", "zynquplus", ZU3),
        ("xczu7ev", "zynquplus", "zynquplus", ZU7),
    ]
    parts = [
        ("xczu3cg-sbva484-1-e", "xczu3cg"),
        ("xczu3eg-sbva484-1-e", "xczu3eg"),
        ("xczu3eg-sbva484-1-i", "xczu3eg"),
        ("xczu7ev-ffvc1156-2-e", "xczu7ev"),
    ]
    lines: dict[str, list[dict[str, Any]]] = {
        "resources.jsonl": [each.stored() for _, each in sorted(records.items())],
        "devices.jsonl": [
            {"name": n, "architecture": a, "family": f, "resources": record(r).digest}
            for n, a, f, r in devices
        ],
        "parts.jsonl": [
            {"name": n, "device": d, "package": "sbva484", "speed": "-1", "temperature": "E"}
            for n, d in parts
        ],
    }
    for name, records_ in lines.items():
        (directory / name).write_text("".join(json.dumps(each) + "\n" for each in records_))
    manifest = {"tool": {"version": "2025.2", "build": "SW Build 1"}, "series": {"UltraScale+": []}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return write_data(tmp_path / "data")


def overlay(tmp_path: Path, stated: object) -> Path:
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(stated))
    return path


# -- the loader and queries --------------------------------------------------------------


def test_a_part_is_found_without_case_and_answered_in_its_spelling(data: Path) -> None:
    found = load(data).part("XCZU3EG-SBVA484-1-I")
    assert found.name == "xczu3eg-sbva484-1-i"
    assert (found.package, found.speed, found.temperature) == ("sbva484", "-1", "E")
    assert found.device.name == "xczu3eg" and found.device.slrs == tuple(ZU3)
    assert found.source.startswith("FINN's part catalog, from Vivado 2025.2 (SW Build 1)")


def test_devices_with_equal_facts_share_one_record_and_stay_distinct(data: Path) -> None:
    loaded = load(data)
    cg, eg, ev = (loaded.device(name) for name in ("xczu3cg", "XCZU3EG", "xczu7ev"))
    assert cg.digest == eg.digest != ev.digest
    assert (cg.shared_with, eg.shared_with, ev.shared_with) == (("xczu3eg",), ("xczu3cg",), ())
    assert cg.name != eg.name and cg.resources == eg.resources == ZU3[0]
    assert (eg.fabric, eg.dsp, eg.uram_init) == (Fabric.ULTRASCALE, DspBlock.DSP48E2, False)


def test_a_name_never_matches_a_pattern_or_a_nearby_part(data: Path) -> None:
    loaded = load(data)
    with pytest.raises(TargetRefused, match="unknown-part: 'xczu3eg' is not in the part") as no:
        loaded.part("xczu3eg")  # a device's name is not a part's
    assert "Vivado 2025.2 (SW Build 1)" in str(no.value)
    with pytest.raises(TargetRefused, match="close names: xczu3eg-sbva484-1-e, xczu3eg-sbva4"):
        loaded.part("xczu3eg-sbva484-2-e")
    with pytest.raises(TargetRefused, match="unknown-device: 'xczu9eg'"):
        loaded.device("xczu9eg")


def test_parts_are_listed_by_pattern_and_family(data: Path) -> None:
    loaded = load(data)
    assert [each.name for each in loaded.parts("XCZU3EG-*")] == [
        "xczu3eg-sbva484-1-e",
        "xczu3eg-sbva484-1-i",
    ]
    assert len(loaded.parts(family="zynquplus")) == 4 and loaded.parts(family="versal") == []


# -- the records -------------------------------------------------------------------------


def test_a_capped_records_slrs_exceed_its_totals_and_are_accepted() -> None:
    """A reduced die of several SLRs states each SLR's site capacity: their sum covers
    the totals and exceeds them, and the totals cap it; an uncapped record's SLRs
    sum to its totals exactly."""
    capped = Record(totals=VU5P, slr_count=2, slrs=(VU5P_SLR,) * 2, capped=True)
    assert sum(capped.slrs or (), Resources()) == VU5P_SLR.times(2) != VU5P
    assert capped.capped_resources() == RESOURCE_NAMES  # VU5P's sites exceed every total
    assert capped.stored()["capped"] is True
    assert "capped" not in record(ZU3).stored() and record(ZU3).capped_resources() == ()
    with pytest.raises(ValueError, match="not the totals"):
        Record(totals=VU5P, slr_count=2, slrs=(VU5P_SLR,) * 2)
    with pytest.raises(ValueError, match="does not cover the totals"):
        Record(totals=VU5P_SLR.times(3), slr_count=2, slrs=(VU5P_SLR,) * 2, capped=True)
    with pytest.raises(ValueError, match="does not cover the totals"):  # nothing to cap
        Record(totals=VU5P_SLR.times(2), slr_count=2, slrs=(VU5P_SLR,) * 2, capped=True)
    with pytest.raises(ValueError, match="one SLR is the device"):
        Record(totals=ZU3[0], slr_count=1, slrs=(ZU7[0],), capped=True)
    with pytest.raises(ValueError, match="states each SLR's site capacity"):
        Record(totals=VU5P, slr_count=2, slrs=None, capped=True)


# -- the overlay -------------------------------------------------------------------------


CUSTOM: dict[str, Any] = {
    "source": "Acme's data sheet for the XCZU3EG-ES1, revision 0.3",
    "devices": [
        {
            "name": "xczu3eg_es1",
            "architecture": "zynquplus",
            "family": "zynquplus",
            "resources": {"slrs": [dict(asdict(ZU3[0]), lut=70_000)]},
        }
    ],
    "parts": [
        {
            "name": "xczu3eg_es1-sbva484-1-e",
            "device": "xczu3eg_es1",
            "package": "sbva484",
            "speed": "-1",
            "temperature": "E",
        },
        {
            "name": "xczu3eg-sbva484-2-x",
            "device": "xczu3eg",
            "package": "sbva484",
            "speed": "-2",
            "temperature": "E",
            "source": "a grade our board vendor ships",
        },
    ],
}


def test_an_overlay_adds_devices_and_parts_with_their_source(data: Path, tmp_path: Path) -> None:
    path = overlay(tmp_path, CUSTOM)
    loaded = load(data, path)
    sample = loaded.part("xczu3eg_es1-sbva484-1-e")
    assert sample.device.resources.lut == 70_000 and sample.device.fabric is Fabric.ULTRASCALE
    assert sample.source == f"{CUSTOM['source']} (overlay {path})"
    assert sample.device.shared_with == ()
    grade = loaded.part("xczu3eg-sbva484-2-x")  # a new part on a shipped device
    assert grade.device is loaded.device("xczu3eg")
    assert grade.source.startswith(f"a grade our board vendor ships (overlay {path}); its device")
    assert "FINN's part catalog" in grade.source
    assert str(path) in loaded.describe()


def test_an_overlay_device_states_its_fabric_where_its_pair_has_no_rule(
    data: Path, tmp_path: Path
) -> None:
    device = {
        "name": "xcnew1",
        "architecture": "newarch",
        "family": "newfamily",
        "resources": {"slrs": counts(ZU3)},
        "source": "an engineering sample",
    }
    with pytest.raises(TargetRefused, match="no rule for newarch/newfamily: state its fabric"):
        load(data, overlay(tmp_path, {"devices": [device]}))
    stated = dict(device, fabric="versal", dsp="DSP58", uram_init=False)
    loaded = load(data, overlay(tmp_path, {"devices": [stated]}))
    found = loaded.device("xcnew1")
    assert (found.fabric, found.dsp) == (Fabric.VERSAL, DspBlock.DSP58)
    assert found.shared_with == ("xczu3cg", "xczu3eg")  # equal facts, whatever the name
    with pytest.raises(TargetRefused, match="states its fabric .* and its DSP block"):
        load(data, overlay(tmp_path, {"devices": [dict(device, fabric="versal")]}))


def test_an_overlay_redefines_a_shipped_entry_only_where_it_says_so(
    data: Path, tmp_path: Path
) -> None:
    again = dict(CUSTOM["parts"][0], name="XCZU3EG-SBVA484-1-E", device="xczu7ev")
    with pytest.raises(
        TargetRefused,
        match="part XCZU3EG-SBVA484-1-E redefines the catalog's xczu3eg-sbva484-1-e; an "
        'entry that replaces it says "overrides": true',
    ):
        load(data, overlay(tmp_path, {"source": "s", "parts": [again]}))
    replaced = load(
        data, overlay(tmp_path, {"source": "s", "parts": [dict(again, overrides=True)]})
    )
    assert replaced.part("xczu3eg-sbva484-1-e").device.name == "xczu7ev"
    stray = dict(CUSTOM["parts"][0], overrides=True)
    with pytest.raises(TargetRefused, match="overrides nothing the catalog ships"):
        load(data, overlay(tmp_path, CUSTOM | {"parts": [stray]}))


@pytest.mark.parametrize(
    "stated, refused",
    [
        ({"parts": [{"name": "p"}]}, r"parts\[0\]: missing \['device', 'package'"),
        ({"devices": [], "colour": 1}, r"unknown \['colour'\]"),
        (
            {"parts": [dict(CUSTOM["parts"][0], source=None)]},
            "states no source",
        ),
        (
            {
                "source": "s",
                "devices": [dict(CUSTOM["devices"][0], resources={"slrs": [{"lut": -1}]})],
            },
            r"SLR 0: missing",
        ),
        (
            {"source": "s", "resources": [{"slrs": counts(ZU3), "digest": "sha256:0"}]},
            "digest 'sha256:0' is not its facts'",
        ),
        (
            {"source": "s", "parts": [dict(CUSTOM["parts"][0], device="xcnone")]},
            "no device 'xcnone'",
        ),
        ([], "a record is a JSON object"),
    ],
)
def test_an_invalid_overlay_is_refused_on_load(
    data: Path, tmp_path: Path, stated: object, refused: str
) -> None:
    with pytest.raises(TargetRefused, match=f"catalog-overlay-invalid: .*{refused}"):
        load(data, overlay(tmp_path, stated))


GUIDE = Path(__file__).resolve().parents[2] / "docs" / "installation.md"


def guide_overlay() -> dict[str, Any]:
    """The installation guide's worked example of an overlay file."""
    (example,) = (
        block
        for block in re.findall(r"```json\n(.*?)```", GUIDE.read_text(), re.DOTALL)
        if '"overrides"' not in block and "xczu3eg_es1" in block
    )
    found: dict[str, Any] = json.loads(example)
    return found


@pytest.fixture
def machine_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The committed catalog with the guide's overlay named by the machine setting."""
    path = overlay(tmp_path, guide_overlay())
    monkeypatch.setenv(OVERLAY_SETTING, str(path))
    catalog.cache_clear()
    yield path
    monkeypatch.delenv(OVERLAY_SETTING)
    catalog.cache_clear()


def test_the_guides_overlay_resolves_its_custom_part(machine_overlay: Path) -> None:
    """``docs/installation.md``'s example, named by ``FINN_PLATFORM_CATALOG``: its part
    resolves with its device's totals, and the exploration report states their source."""
    target = resolve_target(part="XCZU3EG_ES1-SBVA484-1-E", period_ns=5.0)
    assert target.part == "xczu3eg_es1-sbva484-1-e"
    (stated,) = guide_overlay()["devices"][0]["resources"]["slrs"]
    assert target.platform.resources == Resources(**stated)
    assert target.platform.fabric is Fabric.ULTRASCALE and not target.platform.uram
    reported = part_report(target.part)
    assert reported["source"] == f"{guide_overlay()['source']} (overlay {machine_overlay})"
    assert reported["slrs"] == [stated]


def test_device_and_parts_answer_from_the_process_catalog(machine_overlay: Path) -> None:
    """``device`` and ``parts`` (``finn.platform``), the documented queries: the catalog
    the process resolves by, with the overlay the machine setting names."""
    loaded = catalog()
    (custom,) = parts("xczu3eg_es1-*")
    assert custom == loaded.part("XCZU3EG_ES1-SBVA484-1-E")
    assert device(custom.device.name.upper()) == custom.device
    family = custom.device.family
    assert parts(family=family) == loaded.parts(family=family) and custom in parts(family=family)


def test_a_part_the_catalog_lacks_is_reported_without_a_source() -> None:
    reported = part_report("a part with UltraRAM it initializes")
    assert reported["source"] is None and "unknown-part" in str(reported["refused"])


def test_an_overlay_states_a_record_by_its_totals_and_slrs(data: Path, tmp_path: Path) -> None:
    """A device of one SLR may state its totals alone; one of several states each
    SLR, or its totals and SLR count where the split is not known."""
    totals = asdict(ZU7[0])
    devices: list[dict[str, Any]] = [
        {"name": "xcone", "resources": {"totals": totals}},
        {"name": "xcsplit", "resources": {"slrs": counts(ZU3 * 2)}},
        {"name": "xcwhole", "resources": {"totals": totals, "slr_count": 2}},
    ]
    stated: dict[str, Any] = {
        "source": "s",
        "devices": [dict(each, architecture="zynquplus", family="zynquplus") for each in devices],
    }
    loaded = load(data, overlay(tmp_path, stated))
    one, split, whole = (loaded.device(each["name"]) for each in devices)
    assert one.digest == loaded.device("xczu7ev").digest and one.slrs == tuple(ZU7)
    assert split.slr_count == 2 and split.resources == ZU3[0].times(2)
    assert (whole.slr_count, whole.slrs, whole.resources) == (2, None, ZU7[0])
    bad = dict(stated["devices"][1], resources={"slrs": counts(ZU3), "slr_count": 2})
    with pytest.raises(TargetRefused, match="1 SLRs stated, not slr_count 2"):
        load(data, overlay(tmp_path, {"source": "s", "devices": [bad]}))


def test_an_overlay_states_a_capped_record_with_its_own_source(data: Path, tmp_path: Path) -> None:
    """An overlay's reduced die states each SLR's site capacity, its totals and
    ``capped``; no reviewed row is its, so the report's cap cites the overlay."""
    resources = {"totals": asdict(VU5P), "slrs": counts([VU5P_SLR] * 2), "capped": True}
    device = {"name": "xcvu5p_es", "architecture": "virtexuplus", "family": "virtexuplus"}
    loaded = load(
        data, overlay(tmp_path, {"source": "s", "devices": [device | {"resources": resources}]})
    )
    found = loaded.device("xcvu5p_es")
    assert (found.resources, found.slrs) == (VU5P, (VU5P_SLR,) * 2)
    assert found.capped == RESOURCE_NAMES and found.cap_evidence is None
    uncapped = device | {"resources": dict(resources, capped=False)}
    with pytest.raises(TargetRefused, match="not the totals"):
        load(data, overlay(tmp_path, {"source": "s", "devices": [uncapped]}))


def test_a_part_of_an_unsupported_pair_is_catalogued_and_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pair whose rule says FINN builds nothing there keeps its parts' identity and
    totals; the resolution refuses them, saying why."""
    why = "an architecture FINN has not characterised"
    unsupported = replace(
        RULES[("zynquplus", "zynquplus")], fabric=None, dsp=None, evidence={}, unsupported=why
    )
    monkeypatch.setitem(RULES, ("zynquplus", "zynquplus"), unsupported)
    catalog.cache_clear()
    try:
        found = catalog().part("xczu3eg-sbva484-1-e")
        assert found.device.unsupported == why and found.device.resources.lut == 70_560
        with pytest.raises(
            TargetRefused,
            match="unsupported-architecture: xczu3eg-sbva484-1-e \\(xczu3eg, "
            f"zynquplus/zynquplus\\): {why}",
        ):
            resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)
    finally:
        catalog.cache_clear()


# -- the committed data ------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed() -> Catalog:
    return load(DATA)


def test_the_committed_data_is_one_sorted_record_a_line(committed: Catalog) -> None:
    for name, key in (
        ("resources.jsonl", "digest"),
        ("devices.jsonl", "name"),
        ("parts.jsonl", "name"),
    ):
        lines = (DATA / name).read_text().splitlines()
        keys = [json.loads(line)[key] for line in lines]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), name
    for line in (DATA / "resources.jsonl").read_text().splitlines():
        stored = json.loads(line)
        found = Record(
            totals=Resources(**stored["totals"]),
            slr_count=stored["slr_count"],
            slrs=None
            if stored["slrs"] is None
            else tuple(Resources(**slr) for slr in stored["slrs"]),
            capped=stored.get("capped", False),
        )
        assert stored == found.stored()


def test_the_manifest_counts_the_data_and_names_the_rules_the_probe_checked(
    committed: Catalog,
) -> None:
    manifest = committed.manifest
    assert manifest["tool"]["version"] == "2025.2"
    counted = {key: manifest["counts"][key] for key in ("parts", "devices", "resources", "pairs")}
    assert counted == {
        "parts": len(committed.by_name),
        "devices": len(committed.devices),
        "resources": len({each.digest for each in committed.devices.values()}),
        "pairs": len({(d.architecture, d.family) for d in committed.devices.values()}),
    }
    assert set(manifest["series"]) == {"Zynq-7000", "UltraScale", "UltraScale+", "Versal"}
    # A rule the probe checks changed without a new catalog: run the generator.
    assert manifest["rules"] == rules_digest()
    assert {(d.architecture, d.family) for d in committed.devices.values()} == set(RULES)


def test_the_catalog_has_every_installed_part_of_the_supported_series(committed: Catalog) -> None:
    devices = committed.devices.values()
    assert (len(committed.by_name), len(devices)) == (4_725, 261)
    series = Counter(RULES[(d.architecture, d.family)].series for d in devices)
    assert series == {"Zynq-7000": 24, "UltraScale": 30, "UltraScale+": 137, "Versal": 70}


def test_every_part_resolves_or_is_refused_by_name(committed: Catalog) -> None:
    refused: Counter[str] = Counter()
    for each in committed.by_name.values():
        try:
            target = resolve_target(part=each.name, period_ns=5.0)
        except TargetRefused as error:
            refused[f"{error.name}: {each.device.family}"] += 1
            continue
        platform = target.platform
        assert target.part == each.name and platform.resources == each.device.resources
        assert platform.uram == (each.device.resources.uram > 0)
    assert refused == {}  # every pair of the installed series is confirmed by the probe


CAPPED = ["xcku085", "xcku085_CIV", "xcvu160", "xcvu160_CIV", "xcvu27p", "xcvu5p", "xcvu5p_CIV"]
"""The reduced dies of several SLRs Vivado 2025.2 installs."""


def test_a_devices_totals_are_its_slrs_sum_or_on_a_reduced_die_their_cap(
    committed: Catalog,
) -> None:
    """Every device states each SLR's resources, which sum to its totals, but a device
    of several SLRs sold on a larger die: its SLRs state their site capacity, and its
    totals cap their sum."""
    capped = []
    for each in committed.devices.values():
        assert each.slrs is not None and len(each.slrs) == each.slr_count
        summed = sum(each.slrs, Resources())
        if each.capped:
            capped.append(each.name)
            assert each.slr_count > 1 and each.resources != summed
            assert all(getattr(summed, n) >= getattr(each.resources, n) for n in RESOURCE_NAMES)
            continue
        assert each.resources == summed
    assert sorted(capped) == CAPPED
    assert committed.manifest["counts"]["capped"] == len(CAPPED)
    assert "the device's totals cap their sum" in SLRS_CAPPED


def test_xcvu5p_states_two_slrs_of_site_capacity_under_its_totals(committed: Catalog) -> None:
    vu5p = committed.part("xcvu5p-flva2104-1-e").device
    assert vu5p.slrs == (VU5P_SLR,) * 2
    assert VU5P_SLR == Resources(lut=394_080, ff=788_160, bram18=1_440, uram=320, dsp=2_280)
    assert vu5p.resources == VU5P  # Platform.resources: the device's totals
    assert resolve_target(part="xcvu5p-flva2104-1-e", period_ns=5.0).platform.resources == VU5P
    reported = part_report("xcvu5p-flva2104-1-e")
    assert reported["slrs"] == [asdict(VU5P_SLR)] * 2
    assert reported["cap"] == {
        "slrs": SLRS_CAPPED,
        "totals": asdict(VU5P),
        "proven": ["bram18", "uram", "dsp"],
        "inferred": ["lut", "ff"],
        "evidence": CAP_EVIDENCE["xcvu5p"].evidence,
    }
    assert part_report("xcvu9p-flga2104-2L-e")["cap"] is None


def test_the_cap_evidence_covers_every_capped_device(committed: Catalog) -> None:
    """Each reduced die of several SLRs has a reviewed row: which caps a fill proved
    (the rest inferred), in a sentence naming the device, the tool and the method."""
    assert sorted(CAP_EVIDENCE) == CAPPED
    proven = {name: sorted(each.proven) for name, each in CAP_EVIDENCE.items()}
    assert proven == {
        "xcku085": ["bram18", "dsp"],
        "xcvu160": ["bram18", "dsp"],
        "xcvu5p": ["bram18", "dsp", "uram"],
        "xcvu27p": ["bram18", "uram"],
        "xcku085_CIV": [],
        "xcvu160_CIV": [],
        "xcvu5p_CIV": [],
    }
    for name, each in CAP_EVIDENCE.items():
        device = committed.device(name)
        assert device.cap_evidence is each and each.proven <= set(device.capped)
        assert {"lut", "ff"} <= set(device.capped) - each.proven  # never filled
        assert name in each.evidence and "Vivado 2025.2" in each.evidence
    assert "dsp" not in CAP_EVIDENCE["xcvu27p"].proven  # its DSP fills did not complete


def reduced_vu5p() -> dict[str, Any]:
    """An extraction of one reduced device of two SLRs, as the generator reads it."""
    rule_pair = ("virtexuplus", "virtexuplus")
    part_name = "xcvu5p-flva2104-1-e"
    row = PartRow(
        part_name,
        "xcvu5p",
        "flva2104",
        "-1",
        "E",
        *rule_pair,
        ("600577", "1201154", "1024", "470", "3474", "2"),
    )
    sites = {
        "SLICEL": 30_000,
        "SLICEM": 19_260,
        "RAMB18_L": 720,
        "RAMB18_U": 720,
        "DSP48E2": 2_280,
        "URAM288": 320,
    }
    bels = frozenset(
        [f"{x}6LUT" for x in "ABCDEFGH"]
        + [f"{x}FF" for x in "ABCDEFGH"]
        + [f"{x}FF2" for x in "ABCDEFGH"]
        + ["F7MUX_AB", "F7MUX_CD", "F7MUX_EF", "F7MUX_GH", "F8MUX_BOT", "F8MUX_TOP", "F9MUX"]
        + ["CARRY8"]
    )
    probe = DeviceProbe("xcvu5p", part_name, (sites, sites), bels)
    return {
        "tool": {"version": "2025.2", "build": "SW Build 1"},
        "parts": [row],
        "probes": {"xcvu5p": probe},
        "rules": {rule_pair: replace(RULES[rule_pair], sample=part_name)},
    }


def test_the_generator_keeps_a_reduced_dies_slrs_under_its_totals() -> None:
    """Over several SLRs, a reduced die's record states each SLR's sites and
    ``capped``; it needs a reviewed row, and a row needs a capped device."""
    generated = derive(**reduced_vu5p(), caps={"xcvu5p": CAP_EVIDENCE["xcvu5p"]})
    (stored,) = generated.resources
    assert stored["capped"] is True and stored["slrs"] == [asdict(VU5P_SLR)] * 2
    assert stored["totals"] == asdict(VU5P)
    assert generated.manifest["counts"]["capped"] == 1  # type: ignore[index]
    assert any(line.endswith("|CAPPED") for line in generated.probe_log)
    with pytest.raises(GenerationError, match="unreviewed-cap: xcvu5p: no row"):
        derive(**reduced_vu5p(), caps={})
    with pytest.raises(GenerationError, match="rows for devices no capped record has"):
        derive(**reduced_vu5p(), caps=CAP_EVIDENCE)


def test_a_multi_slr_device_states_each_slr(committed: Catalog) -> None:
    vu9p = committed.part("xcvu9p-flga2104-2L-e").device
    assert vu9p.slrs == (Resources(lut=394_080, ff=788_160, bram18=1_440, uram=320, dsp=2_280),) * 3
    u250 = committed.part("xcu250-figd2104-2L-e").device
    assert u250.slrs is not None and len(u250.slrs) == 4
    assert u250.resources.lut == 1_728_000


def test_every_boards_part_resolves_with_its_totals() -> None:
    for board in BOARDS.values():
        assert resolve_target(part=board.part, period_ns=5.0).platform.resources is not None


# -- the generator, with Vivado ----------------------------------------------------------


@requires_vivado
def test_the_generator_reproduces_a_devices_record(tmp_path: Path) -> None:
    """The extraction on the smallest Zynq-7000 device, linked as the generator does:
    its SLRs' resources are the committed record's and confirm its rule."""
    _wait([_vivado(tmp_path, "devices-0", ["devices", "xc7z010clg400-1"])])
    (probe,) = parse_devices([(tmp_path / "devices-0.log").read_text()]).values()
    shipped = load(DATA).device("xc7z010")
    assert slr_resources(probe) == shipped.slrs
    assert confirm(RULES[("zynq", "zynq")], probe) is None
