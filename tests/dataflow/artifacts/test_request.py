# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A8: request and receipt values, and the absence of an executor.

The phase's exit gate has an unusual shape: **nothing runs a tool, and no
module gains a ``run()`` method -- if one appears, the phase has failed.**  So
the first test here is a structural one over the package itself, and the rest
are about a receipt round-tripping and a declared-versus-actual toolchain
mismatch being refused at publication by a test double.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, RequestSchema
from finn.dataflow.artifacts.projection import digest
from finn.dataflow.artifacts.request import (
    ExecutionReceipt,
    FailureCategory,
    LogicalMount,
    PreparedToolRun,
    RequestError,
    ResourceRequirements,
    ToolchainIdentity,
    refuse_publication,
    statement,
)

KEY = "c" * 64
DIGEST = "a" * 64

VIVADO = ToolchainIdentity("vivado", "2024.2", "xilinx-2024.2", "sha256:" + "f" * 64)
OTHER_IMAGE = ToolchainIdentity("vivado", "2024.2", "xilinx-2024.2", "sha256:" + "e" * 64)
OLDER = ToolchainIdentity("vivado", "2023.2", "xilinx-2023.2", "sha256:" + "f" * 64)

SCHEMA = RequestSchema("synth_design -top {top} -part {part}")


def _run(**overrides: object) -> PreparedToolRun:
    defaults: dict[str, object] = {
        "stage_kind": "ooc-synthesis",
        "build_key": KEY,
        "declared_toolchain": VIVADO,
        "request": SCHEMA,
        "mounts": (LogicalMount("sources", ArtifactRef("rtl-module-package", KEY)),),
        "substitutions": (("top", "mvau_decomposed"), ("part", "xcvc1902")),
        "environment_allowlist": ("XILINX_VIVADO", "LM_LICENSE_FILE"),
        "expected_outputs": ("synth.dcp", "utilization.rpt"),
    }
    defaults.update(overrides)
    return PreparedToolRun(**defaults)  # type: ignore[arg-type]


def _receipt(**overrides: object) -> ExecutionReceipt:
    defaults: dict[str, object] = {
        "build_key": KEY,
        "stage_kind": "ooc-synthesis",
        "executor_id": "finn.builder.backends.local",
        "declared_toolchain": VIVADO,
        "actual_toolchain": VIVADO,
        "exit_status": 0,
        "resolved_inputs": (("sources", ArtifactRef("rtl-module-package", KEY)),),
        "produced": (("synth.dcp", DIGEST),),
        "tree_digest": DIGEST,
    }
    defaults.update(overrides)
    return ExecutionReceipt(**defaults)  # type: ignore[arg-type]


# -- the exit gate: no executor ------------------------------------------------


def test_no_module_in_the_package_gains_a_run_method(artifacts_source_root: Path) -> None:
    """If one appears, the phase has failed.  Stated over the package, not a file.

    The trigger for a real ``BuildBackend`` is the first of: a persistent
    store, the first HLS Kernel, or a second execution environment consuming
    one request.  None of those is here, so an empty abstraction would be an
    abstraction with nothing to abstract over.
    """

    offenders: list[str] = []
    for path in sorted(artifacts_source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in ("run", "execute", "invoke", "spawn"):
                    offenders.append(f"{path.name}:{node.lineno}: {node.name}")
    assert not offenders, "artifacts/ grew an executor:\n" + "\n".join(offenders)


# -- the prepared run ----------------------------------------------------------


def test_a_prepared_run_is_a_value_and_round_trips_through_its_key() -> None:
    assert digest(_run()) == digest(_run())


def test_a_mount_is_a_logical_name_and_never_a_host_path() -> None:
    """A request carrying a host path cannot be executed anywhere else."""

    with pytest.raises(RequestError, match="not a logical mount name"):
        LogicalMount("/scratch/build", ArtifactRef("rtl-module-package", KEY))


def test_a_run_must_supply_exactly_what_its_schema_names() -> None:
    with pytest.raises(RequestError, match="does not supply"):
        _run(substitutions=(("top", "x"),))
    with pytest.raises(RequestError, match="does not name"):
        _run(substitutions=(("top", "x"), ("part", "y"), ("extra", "z")))


def test_a_run_declares_its_outputs_rather_than_discovering_them() -> None:
    """Globbing makes a partial run indistinguishable from a complete one."""

    with pytest.raises(RequestError, match="declares its outputs"):
        _run(expected_outputs=())


def test_an_output_outside_the_workspace_is_refused() -> None:
    for name in ("/abs/synth.dcp", "../escape.dcp"):
        with pytest.raises(RequestError):
            _run(expected_outputs=(name,))


def test_the_environment_is_an_allowlist_of_names_and_not_a_map_of_values() -> None:
    """Values would be a second authority, and the one reaching the tool wins."""

    assert _run().environment_allowlist == ("LM_LICENSE_FILE", "XILINX_VIVADO")


def test_a_licence_requirement_is_declarable() -> None:
    """A Versal synthesis needs one, and a run that skipped is not a pass."""

    resources = ResourceRequirements(timeout_seconds=3600, licences=("Synthesis",))
    assert _run(resources=resources).resources.licences == ("Synthesis",)


def test_a_negative_resource_is_refused() -> None:
    with pytest.raises(RequestError):
        ResourceRequirements(timeout_seconds=-1)


# -- the receipt, and what it refuses ------------------------------------------


def test_a_receipt_round_trips() -> None:
    assert digest(_receipt()) == digest(_receipt())


def test_a_matching_run_may_be_published() -> None:
    assert refuse_publication(_receipt()) is None


def test_a_declared_versus_actual_version_mismatch_is_refused_at_publication() -> None:
    """The guard that actually holds.  Identifying before lookup does not."""

    reason = refuse_publication(_receipt(actual_toolchain=OLDER))
    assert reason is not None
    assert "2024.2" in reason and "2023.2" in reason


def test_a_container_image_digest_mismatch_is_refused_at_publication() -> None:
    """A tag may resolve to another digest between the probe and the run."""

    reason = refuse_publication(_receipt(actual_toolchain=OTHER_IMAGE))
    assert reason is not None
    assert "image" in reason


def test_a_failed_run_is_an_attempt_and_never_an_artifact() -> None:
    reason = refuse_publication(
        _receipt(exit_status=1, failure=FailureCategory.LICENCE, tree_digest="")
    )
    assert reason is not None
    assert "attempt" in reason


def test_a_run_with_no_tree_digest_is_unverifiable_and_refused() -> None:
    reason = refuse_publication(_receipt(tree_digest=""))
    assert reason is not None
    assert "unverifiable" in reason


def test_a_failure_must_be_categorized() -> None:
    """An uncategorized failure is the one nobody triages."""

    with pytest.raises(RequestError, match="without a failure category"):
        _receipt(exit_status=1)


def test_a_receipt_cannot_both_succeed_and_name_a_failure() -> None:
    with pytest.raises(RequestError, match="cannot both succeed"):
        _receipt(failure=FailureCategory.DEADLOCK)


# -- the in-toto shape ---------------------------------------------------------


def test_the_receipt_takes_the_in_toto_statement_shape_unsigned() -> None:
    """Conformance to the layout, and deliberately none of the signing."""

    document = statement(_receipt())
    assert document["_type"] == "https://in-toto.io/Statement/v1"
    assert document["predicateType"] == "https://slsa.dev/provenance/v1"
    assert "signatures" not in document
    assert "payload" not in document


def test_the_statement_carries_resolved_dependencies_with_digests() -> None:
    predicate = statement(_receipt())["predicate"]
    assert isinstance(predicate, dict)
    dependencies = predicate["buildDefinition"]["resolvedDependencies"]
    assert dependencies == [{"name": "sources", "digest": {"sha256": KEY}}]


def test_the_statement_names_the_actual_toolchain_and_not_the_declared_one() -> None:
    """A receipt is about what happened.  What was intended is the other field."""

    document = statement(_receipt(actual_toolchain=OLDER, exit_status=0))
    predicate = document["predicate"]
    assert isinstance(predicate, dict)
    builder = predicate["runDetails"]["builder"]
    assert builder["version"] == {"vivado": "2023.2"}


def test_a_content_input_and_an_artifact_input_both_reduce_to_a_digest() -> None:
    receipt = _receipt(
        resolved_inputs=(
            ("blob", ContentRef(DIGEST)),
            ("upstream", ArtifactRef("rtl-module-package", KEY)),
        )
    )
    predicate = statement(receipt)["predicate"]
    assert isinstance(predicate, dict)
    dependencies = predicate["buildDefinition"]["resolvedDependencies"]
    assert [item["digest"]["sha256"] for item in dependencies] == [DIGEST, KEY]
