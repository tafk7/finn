# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The part catalog (``finn.platform.catalog``): its loader, queries and refusals on
synthetic data, the overlay a user adds devices with, the committed data's
invariants, and (with Vivado) the generator against the installed part database."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from finn.kernels.target import DspBlock, Fabric
from finn.kernels.utilization import RESOURCE_NAMES, Resources
from finn.platform import BOARDS, TargetRefused, resolve_target
from finn.platform.architectures import RULES, rules_digest
from finn.platform.catalog import (
    DATA,
    OVERLAY_SETTING,
    SLRS_UNSTATED,
    Catalog,
    Record,
    catalog,
    load,
)
from finn.platform.generate import _vivado, _wait, confirm, parse_devices, slr_resources
from finn.transformation.kernels.choose import part_report

ZU3 = [Resources(lut=70_560, ff=141_120, bram18=432, uram=0, dsp=360)]
ZU7 = [Resources(lut=230_400, ff=460_800, bram18=624, uram=96, dsp=1_728)]


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
    assert found.device.name == "xczu3eg" and list(found.device.slrs) == ZU3
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


def test_a_part_the_catalog_lacks_is_reported_without_a_source() -> None:
    reported = part_report("a part with UltraRAM it initializes")
    assert reported["source"] is None and "unknown-part" in str(reported["refused"])


def test_an_overlay_states_a_record_by_its_totals_and_slrs(data: Path, tmp_path: Path) -> None:
    """A device of one SLR may state its totals alone; one of several states each
    SLR, or its totals and SLR count where the split is not known."""
    totals = asdict(ZU7[0])
    devices = [
        {"name": "xcone", "resources": {"totals": totals}},
        {"name": "xcsplit", "resources": {"slrs": counts(ZU3 * 2)}},
        {"name": "xcwhole", "resources": {"totals": totals, "slr_count": 2}},
    ]
    stated = {
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


def test_a_devices_totals_are_its_slrs_sum_except_a_reduced_dies_split(
    committed: Catalog,
) -> None:
    """Every device states its SLRs' resources, which sum to its totals, but a device
    of several SLRs sold on a larger die: Vivado states its totals, not their split."""
    unstated = []
    for each in committed.devices.values():
        if each.slrs is None:
            unstated.append(each.name)
            assert each.slr_count > 1
            continue
        assert len(each.slrs) == each.slr_count
        assert each.resources == sum(each.slrs, Resources())
        assert all(name in RESOURCE_NAMES for name in asdict(each.resources))
    assert sorted(unstated) == [
        "xcku085",
        "xcku085_CIV",
        "xcvu160",
        "xcvu160_CIV",
        "xcvu27p",
        "xcvu5p",
        "xcvu5p_CIV",
    ]
    vu5p = committed.device("xcvu5p")
    assert (vu5p.slr_count, vu5p.resources.lut) == (2, 600_577)  # VU7P's die halves: 394 080
    assert "reduced die" in SLRS_UNSTATED


def test_a_multi_slr_device_states_each_slr(committed: Catalog) -> None:
    vu9p = committed.part("xcvu9p-flga2104-2L-e").device
    assert vu9p.slrs == (Resources(lut=394_080, ff=788_160, bram18=1_440, uram=320, dsp=2_280),) * 3
    u250 = committed.part("xcu250-figd2104-2L-e").device
    assert len(u250.slrs) == 4 and u250.resources.lut == 1_728_000


def test_every_boards_part_resolves_with_its_totals() -> None:
    for board in BOARDS.values():
        assert resolve_target(part=board.part, period_ns=5.0).platform.resources is not None


# -- the generator, with Vivado ----------------------------------------------------------


@pytest.mark.vivado
def test_the_generator_reproduces_a_devices_record(tmp_path: Path) -> None:
    """The extraction on the smallest Zynq-7000 device, linked as the generator does:
    its SLRs' resources are the committed record's and confirm its rule."""
    if shutil.which("vivado") is None:
        pytest.skip("no Vivado on PATH")
    _wait([_vivado(tmp_path, "devices-0", ["devices", "xc7z010clg400-1"])])
    (probe,) = parse_devices([(tmp_path / "devices-0.log").read_text()]).values()
    shipped = load(DATA).device("xc7z010")
    assert slr_resources(probe) == shipped.slrs
    assert confirm(RULES[("zynq", "zynq")], probe) is None
