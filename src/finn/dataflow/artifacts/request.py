# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepared tool requests and execution receipts.  Value types, and no executor.

Nothing under ``finn.dataflow`` runs a tool.  This module says what a tool
would be *asked* to do and what it would *report*; ``finn.builder.backends`` is
where something runs.  **If a ``run()`` method ever appears here, the
separation has failed** -- there is a test next door asserting none does.

The split is not fastidiousness.  It is what lets a remote, containerized or CI
executor arrive without editing anything in this package, and the way to keep a
seam real is to be unable to cross it.

Two guards, and the first revision of the design wrongly thought one was
enough:

``identify before lookup``
    the toolchain is part of the key, so the key is truthful about what built
    the artifact.

``attest after execution, and refuse publication on mismatch``
    identifying first does **not** make a mismatch impossible.  A remote
    scheduler may select another worker, a container tag may resolve to another
    digest, the executable may change between probe and execution, plugins or
    setup may differ, and the invoked command may not be the probed one.  The
    post-execution attestation is the guard that actually holds.

The receipt takes the in-toto Statement shape with a SLSA-Provenance-shaped
predicate, **unsigned**.  The schema already carries builder identity, resolved
dependencies with digests and run details; the DSSE and transparency-log
machinery addresses a different adversary.  Ours is an accidental wrong cache
hit, not a supply-chain attacker.  Conforming to the shape keeps the upgrade
path free if that ever changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, RequestSchema


class RequestError(Exception):
    """A request or receipt is not well formed."""


class FailureCategory(Enum):
    """Named because these are diagnosable and they recur.

    A category is a claim about *why* a run failed, and an uncategorized
    failure is the one nobody triages.
    """

    LICENCE = "licence"
    MISSING_SOURCE = "missing-source"
    ELABORATION = "elaboration"
    SYNTHESIS = "synthesis"
    SIMULATOR_HANG = "simulator-hang"
    DEADLOCK = "deadlock"
    NUMERICAL_MISMATCH = "numerical-mismatch"
    HARNESS = "harness"


@dataclass(frozen=True)
class LogicalMount:
    """An input tree, by the name the command knows it as.

    A **logical** name and never a host path.  A request that carries
    ``/home/someone/build`` cannot be executed anywhere else, and the legacy
    ``code_gen_dir_ipgen`` family -- where identity *is* a path -- is what that
    turns into.
    """

    name: str
    artifact: ArtifactRef
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise RequestError("a mount needs a logical name")
        if self.name.startswith("/") or "\\" in self.name:
            raise RequestError(f"{self.name!r} is a path, not a logical mount name")


@dataclass(frozen=True)
class ToolchainIdentity:
    """Version, install identity, and container image digest.

    The image digest is **injected**, not discovered.  A process cannot
    reliably learn its own image digest from inside the container: the digest
    is a property of the registry manifest, known to the runtime, and not
    exposed in the filesystem or ``/proc``.  Passing it in is not a workaround
    -- it makes container identity an explicit declared-versus-actual field,
    which is the discipline the rest of the executor already follows.
    """

    tool: str
    version: str
    install_id: str = ""
    image_digest: str = ""

    def __post_init__(self) -> None:
        if not self.tool or not self.version:
            raise RequestError("a toolchain identity needs a tool and a version")


@dataclass(frozen=True)
class ResourceRequirements:
    """What the run needs to be given, and when to give up on it."""

    timeout_seconds: int = 0
    cpus: int = 0
    memory_mb: int = 0
    #: Licence features the tool must be able to check out, e.g. ``Synthesis``.
    licences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("cpus", self.cpus),
            ("memory_mb", self.memory_mb),
        ):
            if value < 0:
                raise RequestError(f"{label} cannot be negative")


@dataclass(frozen=True, init=False)
class PreparedToolRun:
    """Everything an executor is handed, and nothing it can reach around.

    ``environment`` is an **allowlist** of variable names rather than a map of
    values.  A request that carried values would be a second authority for the
    execution environment, and the one that reached the tool would win;
    naming what may pass through says the same thing without stating it twice.
    """

    stage_kind: str
    build_key: str
    declared_toolchain: ToolchainIdentity
    request: RequestSchema
    mounts: tuple[LogicalMount, ...] = ()
    #: Argv relative to the workspace, produced from the schema at run time.
    substitutions: tuple[tuple[str, str], ...] = ()
    environment_allowlist: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    def __init__(
        self,
        stage_kind: str,
        build_key: str,
        declared_toolchain: ToolchainIdentity,
        request: RequestSchema,
        mounts: Iterable[LogicalMount] = (),
        substitutions: Iterable[tuple[str, str]] = (),
        environment_allowlist: Iterable[str] = (),
        expected_outputs: Iterable[str] = (),
        resources: ResourceRequirements | None = None,
    ) -> None:
        if not stage_kind or not build_key:
            raise RequestError("a prepared run needs a stage kind and a build key")
        mounted = tuple(mounts)
        names = [mount.name for mount in mounted]
        if len(names) != len(set(names)):
            raise RequestError("a logical mount name is used twice")
        supplied = tuple(substitutions)
        supplied_names = {name for name, _ in supplied}
        missing = [name for name in request.substitutions if name not in supplied_names]
        if missing:
            raise RequestError(
                f"the request schema expects {missing} which the prepared run does not supply"
            )
        extra = sorted(supplied_names - set(request.substitutions))
        if extra:
            raise RequestError(
                f"the prepared run supplies {extra}, which the request schema does not name; "
                "a substitution nothing consumes is a value with no authority"
            )
        outputs = tuple(expected_outputs)
        for name in outputs:
            if name.startswith("/") or ".." in name.split("/"):
                raise RequestError(f"{name!r} is not an output relative to the workspace")
        if not outputs:
            raise RequestError(
                "a prepared run declares its outputs; discovering them makes a partial "
                "run indistinguishable from a complete one"
            )
        object.__setattr__(self, "stage_kind", stage_kind)
        object.__setattr__(self, "build_key", build_key)
        object.__setattr__(self, "declared_toolchain", declared_toolchain)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "mounts", mounted)
        object.__setattr__(self, "substitutions", tuple(sorted(supplied)))
        object.__setattr__(self, "environment_allowlist", tuple(sorted(set(environment_allowlist))))
        object.__setattr__(self, "expected_outputs", outputs)
        object.__setattr__(self, "resources", resources or ResourceRequirements())


@dataclass(frozen=True)
class ExecutionReceipt:
    """What actually happened, in the in-toto/SLSA predicate shape, unsigned.

    Per-attempt and **never canonical**: timestamps, executor identity and
    resource usage are facts about a run, not about an artifact.  An artifact
    manifest that carried them would stop being a function of its key, and two
    stores would disagree about one artifact.
    """

    #: ``subject`` in the in-toto Statement: what this receipt is about.
    build_key: str
    stage_kind: str
    #: ``builder.id``.
    executor_id: str
    declared_toolchain: ToolchainIdentity
    #: Attested from the run.  The whole point of the second guard.
    actual_toolchain: ToolchainIdentity
    exit_status: int
    #: ``resolvedDependencies`` with digests.
    resolved_inputs: tuple[tuple[str, ContentRef | ArtifactRef], ...] = ()
    produced: tuple[tuple[str, str], ...] = ()
    tree_digest: str = ""
    failure: FailureCategory | None = None
    logs: tuple[tuple[str, ContentRef], ...] = ()
    #: Wall time and peak memory belong to the run, so they live only here.
    duration_seconds: float = 0.0
    peak_memory_mb: int = 0

    def __post_init__(self) -> None:
        if not self.build_key or not self.stage_kind or not self.executor_id:
            raise RequestError("a receipt names its build key, its stage, and its executor")
        if self.exit_status != 0 and self.failure is None:
            raise RequestError(
                f"{self.stage_kind} exited {self.exit_status} without a failure category; "
                "an uncategorized failure is the one nobody triages"
            )
        if self.exit_status == 0 and self.failure is not None:
            raise RequestError("a receipt cannot both succeed and name a failure")

    @property
    def succeeded(self) -> bool:
        return self.exit_status == 0 and self.failure is None

    @property
    def toolchain_matches(self) -> bool:
        """Whether the tool that ran is the tool the key claims ran."""

        return self.declared_toolchain == self.actual_toolchain


def refuse_publication(receipt: ExecutionReceipt) -> str | None:
    """Why this run must not become an artifact, or ``None`` if it may.

    A failed or abandoned run is an **attempt**, not an artifact.  Failures
    never reach ``objects/``, so the store cannot hold a partial tree, and the
    material needed to diagnose them is kept and addressable rather than
    discarded.
    """

    if not receipt.succeeded:
        category = receipt.failure.value if receipt.failure else "unknown"
        return f"{receipt.stage_kind} failed ({category}); a failed run is an attempt"
    if not receipt.toolchain_matches:
        return (
            f"{receipt.stage_kind} was keyed for "
            f"{receipt.declared_toolchain.tool} {receipt.declared_toolchain.version} "
            f"(image {receipt.declared_toolchain.image_digest or 'unspecified'}) but ran under "
            f"{receipt.actual_toolchain.tool} {receipt.actual_toolchain.version} "
            f"(image {receipt.actual_toolchain.image_digest or 'unspecified'}); "
            "publishing it would make the key a lie about what built the artifact"
        )
    if not receipt.tree_digest:
        return f"{receipt.stage_kind} produced no tree digest, so its output is unverifiable"
    return None


def statement(receipt: ExecutionReceipt) -> Mapping[str, object]:
    """The receipt in in-toto Statement shape.

    Conformance to the shape and nothing more: no DSSE envelope, no signature,
    no transparency log.  Those solve a problem we do not have, and adopting
    the field layout now is what keeps the upgrade free if we ever do.
    """

    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": receipt.stage_kind, "digest": {"sha256": receipt.build_key}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": receipt.stage_kind,
                "resolvedDependencies": [
                    {
                        "name": name,
                        "digest": {
                            "sha256": (
                                reference.digest
                                if isinstance(reference, ContentRef)
                                else reference.key
                            )
                        },
                    }
                    for name, reference in receipt.resolved_inputs
                ],
            },
            "runDetails": {
                "builder": {
                    "id": receipt.executor_id,
                    "version": {
                        receipt.actual_toolchain.tool: receipt.actual_toolchain.version,
                    },
                },
                "metadata": {
                    "invocationId": receipt.build_key,
                    "exitStatus": receipt.exit_status,
                },
            },
        },
    }


__all__ = [
    "ExecutionReceipt",
    "FailureCategory",
    "LogicalMount",
    "PreparedToolRun",
    "RequestError",
    "ResourceRequirements",
    "ToolchainIdentity",
    "refuse_publication",
    "statement",
]
