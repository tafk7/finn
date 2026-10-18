# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""PackagePartition: a partition of KernelOps as the IP the shells read.

The Tcl and the interface names come from the module's ABI
(``finn.kernels.artifacts.ipxact``, checked as text in
``tests/kernels/artifacts/test_ipxact.py``). Here: the partition's module from
its nodes' and channels' saved choices, an open one refused; that the emitted top
is elaborated before Vivado runs (a toolchain double stands for Vivado, so these run
in the fast gate); and one test that packages the Chain (``support.chain``) and reads
the IP back (marker ``vivado``: the fast gate deselects it, the XSim sweep's
``kernel-ops-vivado`` job runs it).
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest
from support.configure import configured, write_raw
from support.models import choose_memories, configure_partition, kernel_model
from support.packaging import NoVivado, reaches_vivado
from support.refusals import refused
from support.xsim import requires_vivado

from finn.core.graph.annotations.outputs import (
    OUTPUT_INTERFACES,
    OUTPUT_IP,
    OUTPUT_IP_KEY,
    OUTPUT_IP_OPTIONS,
    OUTPUT_VLNV,
)
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest
from finn.core.space import inspection
from finn.custom_op.kernels.network import persist
from finn.custom_op.kernels.shell import InShell, configured_network, in_shell
from finn.kernels.artifacts import build
from finn.kernels.configure import undecided
from finn.toolchain.launcher import (
    Selection,
    Toolchain,
    ToolFailed,
    ToolUnavailable,
    host_toolchain,
)
from finn.toolchain.vivado import hierarchical_utilization
from finn.transformation.kernels import PackagePartition, kernel_choices_config
from finn.transformation.kernels.cut import BaselineChoices
from finn.transformation.kernels.package import (
    ElaboratePartition,
    ip_key,
    partition_utilization_report,
)


def test_the_partition_packages_its_nodes_choices() -> None:
    configure_partition(model := kernel_model())
    module = configured_network(model, "sdp_1")[0].module
    reference = configured_network(model, "partition")[0].module
    assert (module.fragment, module.abi) == (reference.fragment, reference.abi)
    assert module.stem == "finn_partition"
    assert [port.name for port in module.abi.pins] == [
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    ]


def test_a_choice_never_created_refuses_packaging_and_is_named() -> None:
    # The adapters' memories are left open, made by hand with no cut: emission completes
    # nothing and refuses each, named; once BaselineChoices has created them (as the cut
    # does), the body packages as configure_partition commits them.
    model = kernel_model()
    with refused(
        "choices-open",
        error=KernelOpError,
        owner="configured_network",
        node="sdp_1",
        message=r"open choices, refused at emission: .*not created: BaselineChoices",
    ) as left:
        configured_network(model, "sdp_1")
    assert left.error.keys and all(key.endswith(".ram_style") for key in left.error.keys)
    assert kernel_choices_config(model) == kernel_choices_config(kernel_model())
    configure_partition(configured := kernel_model())
    reference = configured_network(configured, "partition")[0].module
    module = configured_network(model.transform(BaselineChoices()), "sdp_1")[0].module
    assert (module.fragment, module.abi) == (reference.fragment, reference.abi)


def test_an_unchecked_choice_is_refused_by_name_beside_the_open_choice_it_waits_on() -> None:
    """A choice stored while the choice its check reads is open is kept unchecked (the
    engine's rule: read unresolved, listed by ``inspection.unchecked``): the thresholding's
    ``block_stages`` with its ``ram_style`` cleared. Emission demands a buildable point, so
    it refuses both by name, in one ``KernelOpError``: ``choices-open`` names
    ``activate.ram_style``, and ``choices-unchecked`` names ``activate.block_stages``, its
    stored value and what it waits on. ``undecided`` still counts only the open one."""
    model = kernel_model()
    choose_memories(model)
    write_raw(model, "activate", "ram_style", None)
    bound = in_shell(model, model.graph.node)
    point = bound.member("activate")
    waiting = inspection.Unchecked("block_stages", 0, ("ram_style",))
    assert inspection.unchecked(point) == (waiting,)
    assert undecided(point, "*") == ["ram_style"]  # the open one; the unchecked is not counted
    assert bound.unchecked() == ((("activate", "block_stages"), waiting),)
    with refused(
        "choices-open",
        error=KernelOpError,
        owner="configured_network",
        node="sdp_1",
        keys=("activate.ram_style",),
        message=r"open choices, refused at emission: activate\.ram_style \(not created",
    ) as left:
        configured_network(model, "sdp_1")
    with refused(
        "choices-unchecked",
        error=KernelOpError,
        owner="configured_network",
        node="sdp_1",
        keys=("activate.block_stages",),
        message=r"unchecked choices, refused at emission: activate\.block_stages = 0"
        r" \(waits on activate\.ram_style\)",
    ):
        raise left.error
    assert cast(KernelOpError, left.error).keys == ("activate.ram_style", "activate.block_stages")


def test_an_unchecked_choice_alone_refuses_emission_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emission refuses a network whose only incomplete choice is one kept unchecked, by
    name (``choices-unchecked``), as it refuses an open one. No real kernel keeps a choice
    unchecked without an open choice it waits on, so a double of the network's
    ``undecided`` hides the open ``ram_style`` that ``block_stages`` waits on: what is left
    for emission to refuse is the unchecked choice."""
    model = kernel_model()
    choose_memories(model)
    write_raw(model, "activate", "ram_style", None)
    monkeypatch.setattr(InShell, "undecided", lambda self: ())
    with refused(
        "choices-unchecked",
        error=KernelOpError,
        owner="configured_network",
        node="sdp_1",
        keys=("activate.block_stages",),
        message=r"activate\.block_stages = 0 \(waits on activate\.ram_style\)",
    ) as left:
        configured_network(model, "sdp_1")
    assert [finding.code for finding in left.error.findings] == ["choices-unchecked"]


def test_an_unchecked_choice_is_saved_and_replayed_as_unchecked() -> None:
    """Saving writes a choice kept unchecked by its stored value, as it writes a checked
    one: nothing a user or a strategy committed is dropped. ``block_stages`` stays on its
    node while its ``ram_style`` is open, and the model read again holds it unchecked,
    waiting on the same ``ram_style``. (A channel's choice kept unchecked is saved the same
    way: ``test_channel_choices``.)"""
    model = kernel_model()
    choose_memories(model)
    write_raw(model, "activate", "ram_style", None)
    bound = in_shell(model, model.graph.node)
    waiting = (inspection.Unchecked("block_stages", 0, ("ram_style",)),)
    assert inspection.unchecked(bound.member("activate")) == waiting
    assert persist(model, bound)["activate"]["block_stages"] == 0
    (node,) = [node for node in model.graph.node if node.name == "activate"]
    assert "block_stages" in {attribute.name for attribute in node.attribute}
    again = in_shell(model, model.graph.node)
    assert inspection.unchecked(again.member("activate")) == waiting


def test_an_open_required_choice_refuses_packaging_and_is_named() -> None:
    # A FIFO saved with no depth, its style created (BaselineChoices): the depth is
    # required, and no strategy set it.
    configure_partition(model := kernel_model())
    configured(model, {"levels": {"transport": "fifo"}})
    model = model.transform(BaselineChoices())
    with refused(
        "choices-open",
        error=KernelOpError,
        node="sdp_1",
        keys=("levels.transport.fifo.buffer.depth",),
        message=r"\(required: run step_size_fifos or step_placeholder_choices\)",
    ):
        configured_network(model, "sdp_1")


def test_the_graphs_input_order_is_the_port_order() -> None:
    model = kernel_model(second_weights=False)
    configure_partition(model)
    pins = configured_network(model, "sdp_1")[0].module.abi.pins
    assert {port.name for port in pins} >= {"s_axis_0", "s_axis_1"}
    model.graph.input.reverse()  # w2 first: the shells would feed it to s_axis_0
    with refused(
        "ports-out-of-order",
        error=KernelOpError,
        owner="configured_network",
        node="sdp_1",
        message="not its graph's inputs and outputs in order",
    ):
        configured_network(model, "sdp_1")


def test_the_chains_emitted_top_elaborates_before_vivado(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configure_partition(model := kernel_model())
    logger = "finn.transformation.kernels.package"
    with caplog.at_level(logging.INFO, logger=logger):
        reaches_vivado(model, "sdp_1", tmp_path / "project")
    # The check logs that it ran, what it checked, and its time.
    (line,) = [record.getMessage() for record in caplog.records if record.name == logger]
    assert line.startswith("sdp_1: slang elaborated finn_partition")
    assert "its 4 ports agree with the pins (" in line and line.endswith(" s)")


def test_a_top_that_does_not_elaborate_is_refused_before_vivado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_partition(model := kernel_model())
    netlist = build.netlist

    def broken(module, name: str) -> str:
        # A read of a net nothing declares (an assignment to one would declare it
        # implicitly), just before the module ends.
        text = netlist(module, name)
        cut = text.rindex("endmodule")
        return text[:cut] + "    wire probe = no_such_net;\n" + text[cut:]

    monkeypatch.setattr(build, "netlist", broken)
    toolchain = NoVivado()
    with refused(
        "top-unelaborated",
        error=KernelOpError,
        owner="check_elaborates",
        node="sdp_1",
        message="refused before packaging",
    ) as unelaborated:
        model.transform(
            PackagePartition(
                "sdp_1", directory=tmp_path / "project", toolchain=cast(Toolchain, toolchain)
            )
        )
    assert "use of undeclared identifier 'no_such_net'" in str(unelaborated.error)
    assert toolchain.ran == []


@requires_vivado
def test_the_chain_packages_as_the_shells_ip(tmp_path: Path) -> None:
    """The Chain packaged by Vivado, then into a second directory from the store: the
    same files, linked, and no path of the directory it was packaged in."""
    configure_partition(model := kernel_model())
    project = tmp_path / "vivado_stitch_proj"
    key = ip_key(model, "sdp_1", run_synth=False, toolchain=host_toolchain())
    model = model.transform(PackagePartition("sdp_1", directory=project))
    assert model.get(OUTPUT_IP_KEY) == key
    # What packaging states of the IP, typed; no flat key.
    assert model.get(OUTPUT_IP) == str(project.resolve() / "ip")
    assert model.get(OUTPUT_VLNV) == "xilinx_finn:finn:sdp_1:1.0"
    assert model.get(OUTPUT_INTERFACES) == {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"],
        "s_axis": [["s_axis_0", 8]],
        "m_axis": [["m_axis_0", 16]],
        "aximm": [],
        "axilite": [],
        "ap_none": [],
    }
    assert list(model.model.metadata_props) == []
    # The part and period are the model's target. Packaging states what it made, in
    # finn.outputs, and nothing else: the boundary is the configured network's, not stored.
    assert "-part xczu3eg-sbva484-1-e" in (project / "package.tcl").read_text()
    assert {item.key.split("/")[0] for item in model.graph.metadata_props} == {
        "finn.platform",
        "finn.outputs",
    }
    spirit = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
    root = ET.parse(project / "ip" / "component.xml").getroot()
    assert [root.find(f"{spirit}{tag}").text for tag in ("vendor", "library", "name")] == [
        "xilinx_finn",
        "finn",
        "sdp_1",
    ]
    interfaces = {}
    for bus in root.iter(f"{spirit}busInterface"):
        parameters = {
            item.find(f"{spirit}name").text: item.find(f"{spirit}value").text
            for item in bus.iter(f"{spirit}parameter")
        }
        kind = bus.find(f"{spirit}busType").get(f"{spirit}name")
        mode = "slave" if bus.find(f"{spirit}slave") is not None else "master"
        interfaces[bus.find(f"{spirit}name").text] = (kind, mode, parameters)
    assert sorted(interfaces) == ["ap_clk", "ap_rst_n", "m_axis_0", "s_axis_0"]
    assert interfaces["ap_clk"][2]["ASSOCIATED_BUSIF"] == "s_axis_0:m_axis_0"
    assert interfaces["ap_clk"][2]["ASSOCIATED_RESET"] == "ap_rst_n"
    assert interfaces["ap_rst_n"][2]["POLARITY"] == "ACTIVE_LOW"
    assert interfaces["s_axis_0"][:2] == ("axis", "slave")
    assert interfaces["s_axis_0"][2]["TDATA_NUM_BYTES"] == "1"
    assert interfaces["m_axis_0"][:2] == ("axis", "master")
    assert interfaces["m_axis_0"][2]["TDATA_NUM_BYTES"] == "2"
    # Self-contained: the sources and the memories' contents are the IP's own, the
    # weights' and the thresholds' alike.
    files = {path.name for path in (project / "ip" / "src").iterdir()}
    for memory in ("memstream_", "thresholds_"):
        assert any(name.startswith(memory) and name.endswith(".dat") for name in files)
    assert any(name.startswith("finn_partition__") for name in files)
    # Movable: every path the IP names is its own, relative.
    assert str(tmp_path) not in (project / "ip" / "component.xml").read_text()
    second = tmp_path / "second"
    configure_partition(again := kernel_model())
    again = again.transform(PackagePartition("sdp_1", directory=second))
    assert again.get(OUTPUT_IP_KEY) == key
    component = second / "ip" / "component.xml"
    assert component.samefile(project / "ip" / "component.xml")


# -- what the tools do: ToolUnavailable, ToolFailed (finn.toolchain.launcher) -----------


def _tools(tmp_path: Path, scripts: dict[str, str], *, vivado: bool = True) -> Toolchain:
    """A toolchain whose tools are shell ``scripts`` in a command directory, its Vivado an
    installation holding ``glbl.v`` (none unless ``vivado``)."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for name, body in scripts.items():
        (tools / name).write_text("#!/bin/sh\n" + body)
        (tools / name).chmod(0o755)
    environment = {"PATH": os.defpath}
    if vivado:
        installation = tmp_path / "Vivado"
        (installation / "data/verilog/src").mkdir(parents=True)
        (installation / "data/verilog/src/glbl.v").write_text("module glbl(); endmodule\n")
        environment["XILINX_VIVADO"] = str(installation)
    return Toolchain(Selection(command_dir=str(tools)), environment)


#: What Vivado logs when it finds no licence for the part.
NO_LICENCE = "ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis'"


def test_a_failed_packaging_run_is_vivados_failure_naming_its_transcript(
    tmp_path: Path, ip_store: Path
) -> None:
    """Outside a build the transcript is beside the replay (``package.sh``), in the run
    the store keeps beside its entry; in a build's step, in its log directory."""
    configure_partition(model := kernel_model())
    project = tmp_path / "project"
    toolchain = _tools(tmp_path, {"vivado": f'echo "{NO_LICENCE}" | tee package.log\nexit 1\n'})
    logs = tmp_path / "out" / "logs" / "step_package_partition"
    for tools in (toolchain, toolchain.transcripts_in(logs)):
        with pytest.raises(ToolFailed) as failed:
            model.transform(PackagePartition("sdp_1", directory=project, toolchain=tools))
        (kept,) = ip_store.glob("sdp_1-*.failed")
        transcript = (kept if tools is toolchain else logs) / "package.sh.stdout.log"
        assert (failed.value.tool, failed.value.status) == ("vivado", 1)
        assert failed.value.transcript == transcript
        assert transcript.read_text() == NO_LICENCE + "\n"
        assert (failed.value.code, failed.value.errors) == ("licence-unavailable", (NO_LICENCE,))
    assert not (project / "ip").exists()


def test_packaging_states_its_ips_key_and_a_second_packaging_is_the_stores(
    tmp_path: Path,
) -> None:
    """Beside the IP, the partition states its key and the options it was packaged with:
    the key ``ip_key`` gives the partition (finn.outputs is not the design, so packaging
    leaves it as it was). A second packaging into another directory runs no Vivado: the
    store's entry, copied."""
    configure_partition(model := kernel_model())
    runs = tmp_path / "runs"
    toolchain = _tools(
        tmp_path, {"vivado": f"echo ran >> {runs}\nmkdir -p ip && touch ip/component.xml\n"}
    )
    key = ip_key(model, "sdp_1", run_synth=False, toolchain=toolchain)
    made_from = design_digest(model)
    packaged = model.transform(
        PackagePartition("sdp_1", directory=tmp_path / "project", toolchain=toolchain)
    )
    assert packaged.get(OUTPUT_IP) == str((tmp_path / "project" / "ip").resolve())
    assert packaged.get(OUTPUT_IP_KEY) == key
    assert packaged.get(OUTPUT_IP_OPTIONS) == {"run_synth": False}
    assert design_digest(packaged) == made_from
    again = packaged.transform(
        PackagePartition("sdp_1", directory=tmp_path / "again", toolchain=toolchain)
    )
    assert again.get(OUTPUT_IP_KEY) == key
    assert runs.read_text() == "ran\n"
    for name in ("package.tcl", "ip/component.xml"):
        assert (tmp_path / "again" / name).samefile(tmp_path / "project" / name)
    assert (tmp_path / "again" / "interface.json").is_file()
    # Another Vivado is another key.
    (tmp_path / "other").mkdir()
    other = _tools(tmp_path / "other", {})
    (Path(other.environment["XILINX_VIVADO"]) / "data" / "version.dat").write_text("2099.1\n")
    assert ip_key(model, "sdp_1", run_synth=False, toolchain=other) != key
    assert ip_key(model, "sdp_1", run_synth=True, toolchain=toolchain) != key


def test_a_run_that_packages_no_ip_is_vivados_failure(tmp_path: Path, ip_store: Path) -> None:
    configure_partition(model := kernel_model())
    project = tmp_path / "project"
    toolchain = _tools(tmp_path, {"vivado": "echo done > package.log\n"})
    with pytest.raises(ToolFailed, match="sdp_1: no IP packaged") as failed:
        model.transform(PackagePartition("sdp_1", directory=project, toolchain=toolchain))
    (kept,) = ip_store.glob("sdp_1-*.failed")
    transcript = kept / "package.sh.stdout.log"
    assert (failed.value.status, failed.value.transcript) == (None, transcript)
    # Nor is a missing or empty utilization report a refusal of the partition.
    with pytest.raises(ToolFailed, match="one hierarchical utilization report"):
        partition_utilization_report(project)
    with pytest.raises(ToolFailed, match="no hierarchical utilization table"):
        hierarchical_utilization("")


def test_elaboration_without_vivado_is_unavailable(tmp_path: Path) -> None:
    configure_partition(model := kernel_model())
    toolchain = _tools(tmp_path, {}, vivado=False)
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO") as missing:
        model.transform(ElaboratePartition(directory=tmp_path / "e", toolchain=toolchain))
    assert missing.value.tool == "vivado"


@pytest.mark.parametrize("tool, log", [("xvlog", "xvlog.log"), ("xelab", "xelab.log")])
def test_elaboration_that_fails_is_the_simulators_failure_naming_its_log(
    tmp_path: Path, tool: str, log: str
) -> None:
    configure_partition(model := kernel_model())
    scripts = {name: "exit 0\n" for name in ("xvlog", "xelab")}
    scripts[tool] = f'echo "ERROR: [VRFC 10-2989] bad" > {log}\nexit 2\n'
    directory = tmp_path / "e"
    with pytest.raises(ToolFailed, match="does not") as failed:
        model.transform(
            ElaboratePartition(directory=directory, toolchain=_tools(tmp_path, scripts))
        )
    assert (failed.value.tool, failed.value.status) == (tool, 2)
    assert failed.value.transcript == directory.resolve() / log
    assert failed.value.errors == ("ERROR: [VRFC 10-2989] bad",)
