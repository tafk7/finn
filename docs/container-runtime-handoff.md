# Container/runtime implementation handoff

Prepared: 2026-09-20.
Purpose: start the approved implementation in a new session without losing the
prototype or mistaking its behavior for the approved final architecture.

**Start here**

- Workspace: `/home/tkeller787/finn`.
- Implementation branch: `refactor/container-runtime-implementation`.
- Approved scope and acceptance gates:
  [container-runtime-implementation-plan.md](container-runtime-implementation-plan.md).
- Design rationale and detailed Docker/sbx development workflow:
  [container-runtime-refactor-proposal.md](container-runtime-refactor-proposal.md).
- Build-engine cleanup belongs to the user's separate private branch. Its code/API
  is not available here; work independently and integrate it later.
- No merge or push is authorized as part of this handoff. The archive is a local
  recovery checkpoint, not a release-ready change.

**Preserved baseline**

The previous committed container/native-sbx baseline is
`8a05d50a61e2ab4fa7aa28ee3a70f86ea7b8b10d`. Both
`experiment/container-runtime-boundaries` and `refactor/container-stack` were left
at that commit.

All subsequent tracked and untracked prototype work was preserved in:

- Branch: `archive/container-runtime-prototype`.
- Commit: `f20a2a8ea3c280a93a8f04ac958ddd1e48a09b50`.
- Tree: `1232435873ae1c09a643082d46f0c9a5e9acea1d`.
- Inventory: 65 previously unstaged tracked files (61 modified, 4 deleted), plus
  27 previously untracked files; 92 changed paths in the checkpoint.

The implementation branch starts from that checkpoint. This handoff is a separate
commit so the archive continues to represent the exact preserved prototype.
Ignored build products, local environments and temporary validation artifacts were
not force-added. Git had no configured author identity; these local checkpoint and
handoff commits use `Codex <codex@localhost>` through per-command settings without
changing repository or global Git configuration.

Inspect the preserved inventory with:

```bash
git show --stat archive/container-runtime-prototype
git diff --name-status 8a05d50a6 archive/container-runtime-prototype
git diff 8a05d50a6 archive/container-runtime-prototype -- path/to/file
```

No reset or checkout of the earlier baseline is needed to start implementation.
Keep the archive as a comparison/recovery point while making reviewed changes on
the implementation branch.

**Initial triage for P0**

| Prototype area | Initial disposition |
| --- | --- |
| Resource helper and resource consumer migrations | Retain useful behavior/tests; review conventional package layout in P1. |
| Wheel/sdist machinery, version/provenance, entry points and installation inspection | Retain the foundation; validate/revise against the approved packaging contract. |
| `_toolchain.py`, process helpers and representative HLS/Vivado callers | Retain/review; complete concrete caller migrations through P4. |
| Dependency/application image handling | Revise: distinct artifact identities and an offline development wheelhouse are still required. |
| Editable preparation using inherited packages and strict mode | Replace as the default with isolated venv preparation; keep overlays only as an explicit tested alternative. |
| Whole-build Python subprocess and broad compatibility preparation | Integration-sensitive; remove/reconcile when the private build-engine branch is integrated. Do not expand or depend on it. |
| Remaining bare-tool/startup/global-loader mechanisms | Retire only after their actual consumers and required execution gates are covered. |
| XSI source packaging and loader experiments | Useful foundation/evidence; an actual simulation-session worker is not implemented or vendor-validated. |
| Earlier documentation and validation reports | Historical evidence. Update current instructions as new work lands; do not treat old architectural claims as approved requirements. |

Complete the P0 inventory and validation baseline before broad changes. This table
is initial triage, not a claim that every prototype file has been reviewed for
release.

Two concrete early corrections are already known:

1. `setup-local.sh` uses `python -m finn.xsi.setup --check` as evidence that the
   extension exists. That command checks prerequisites, so setup can incorrectly
   skip building XSI. Correct the decision using actual setup/build semantics.
2. `.devcontainer/compose.yaml` still sets `FINN_DEPS: frozen`; remove the stale
   import-mode configuration as part of development-environment cleanup.

Do not reproduce private-branch work on build paths, configuration, allocation or
logging. Treat `src/finn/builder/` and related shared plumbing as integration-
sensitive. Do not invent that branch's API, add a build-workspace framework, or
make tests require the prototype's whole-build worker.

**Independent work that can begin immediately**

- P1: conventional resource packaging and installed/editable correctness.
- P2: dependency wheelhouse, dependency/application artifacts and identities,
  and explicit `--dependencies` image selection.
- P3: persistent Docker venv mounts, sandbox-private sbx venv preparation,
  Dev Container interpreter selection, native/offline instructions.
- P4: scoped tool/process execution and non-overlapping concrete callers.
- P5: independently testable XSI session implementation and its validation.

Development environments are prepared once with ordinary package tools. Docker's
disposable `run --rm` containers reuse a separately mounted venv. Native sbx exec
sessions reuse the sandbox's writable venv. Native runtime configuration selects
the interpreter/PATH; no startup hook installs or repairs packages.

Build-engine integration is P6, once the actual private branch is available. It
may proceed incrementally without waiting for every vendor-validation gate.
Unvalidated simulation paths remain explicitly open. Brainsmith is reference
material only and creates no compatibility requirement.

**Validation state**

The [earlier validation record](runtime-validation.md) reports 157 tests and
Docker/native-sbx resource and editable-install journeys for the earlier prototype.
The [XSI investigation](xsi-process-boundary-investigation.md) records synthetic
loader experiments, not actual AMD simulation validation.

No runtime tests were rerun merely to create the checkpoint/handoff. The exact
checkpoint tree, clean Git state and whitespace were checked. Re-establish the
relevant test baseline during P0; do not present the historical results as proof
of the approved refactor.

Earlier disposable test environments were at `/tmp/finn-runtime-venv` and
`/tmp/finn-sbx-test-venv`. They may be useful if still present, but are not required
or reproducible artifacts. Recreate/provision test environments explicitly when
needed. Check current availability of Docker, native sbx, AMD tools/licences and
Apptainer/Singularity; the prior report's unavailable vendor/HPC coverage remains
unresolved until tested.

**Suggested new-session instruction**

> Implement `docs/container-runtime-implementation-plan.md` on the current
> `refactor/container-runtime-implementation` branch, starting with P0. Read
> `docs/container-runtime-handoff.md` first. The archived prototype is material
> to retain, revise or remove, not the final architecture. Build-engine cleanup
> belongs to a separate private branch: make independent progress and defer
> overlapping integration. Preserve the archive and historical design inputs.
> Do not merge or push.
