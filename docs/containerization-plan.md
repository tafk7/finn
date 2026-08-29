# FINN containerization: execution plan

Date: 2026-08-28
Implements: `containerization-decisions.md` (revision 2)
Branch: `feature/sbx`
Status: **all eight stages implemented, plus an unplanned stage 9.** See "Outcome" at the bottom for what
was verified, what changed from the plan, and what is still open.

Eight stages. Each ends at a green tree that could ship on its own — no stage
depends on a later one to be correct. Stages 1-3 are strictly additive to
current behavior; the first user-visible change lands in stage 6.

Verification commands assume `FINN_XILINX_PATH` and a licence are configured, as
they are on the development host.

---

## Stage 1 — Defects

Independent of the restructuring. Defect 1 is a live containment hole; do this
first regardless of whether the rest proceeds.

**Files:** `run-docker.sh`, `docker/Dockerfile.finn`, `docs/containerization.md`,
`docker/finn.kit/spec.yaml`

1. **`:ro` on the docker Xilinx mount** (`run-docker.sh:547`). Add `:ro`. Run a
   real synthesis afterward — if any Xilinx-side write was silently load-bearing,
   this is where it surfaces, and it must surface now rather than under an agent.
2. **Mount the whole Xilinx root `:ro` on sbx too** (`run-docker.sh:719-720`).
   Replaces the version-directory guard, so the pre-2024.2 layout stops silently
   mounting nothing. This is also the D1a simplification landing early — it
   removes the layout branch from one of the two paths before that path is
   rewritten.
3. **Mount `PLATFORM_REPO_PATHS` on sbx** (`run-docker.sh:722-726`), matching the
   docker path.
4. **`USER agent`** in `Dockerfile.finn` at the `dev` stage. Docker `--user`
   still overrides it; this fixes the default for every consumer that does not
   pass one.
5. **Doc fixes:** delete the `sandbox-persistent.sh` claim at
   `containerization.md:701`; correct the schema claim at line 685 to state that
   v0.39.0 accepts `setup:` under `schemaVersion: "2"`; correct the kit's
   "requires open posture" comment (`spec.yaml:17-20`) to record that a bare
   hostname grant is sufficient and was verified.
6. **Test node-locked licensing** with the licence directory `:ro` (defect 8).
   Whichever way it resolves, fix the code or the comment — do not edit the prose
   on inference.

**Verify**

```bash
./run-docker.sh build          # docker, :ro toolchain
./run-docker.sh sbx build      # sbx, full root, platform repo present
# then a real synth_design on xczu28dr in each
```

**Stop if:** the `:ro` mount breaks synthesis. That would mean the tools write
into the install, which changes the containment story materially and needs its
own decision.

---

## Stage 2 — `finn-env`

The single resolver. Written and tested before anything consumes it.

**New:** `docker/finn-env`

Three subcommands, one resolution path:

- `inspect --format json` — runs **on the host**, emits tier, profile, status,
  workspace policy, mounts, egress hosts, env, platform.
- `print --format=sh|json|env0` — runs **in the container**, emits the resolved
  toolchain environment.
- `exec <cmd>` — sources, then execs. Idempotent.

Host and container views share one implementation so they cannot disagree.

Language: Python. It must parse `deps.env` and profile files and emit JSON;
`finn_paths.py` already establishes Python as available at both ends.

**Verify:** unit tests over synthetic host layouts — both Xilinx layouts, both
licence forms, missing toolchain, `dev` tier. `inspect` output diffed against
what `run-docker.sh` currently computes for the same inputs; any divergence is
either a bug in the new code or an undocumented behavior in the old, and both
need explaining before proceeding.

Nothing consumes it yet. Fully revertable.

---

## Stage 3 — Transparent exec

**New:** `docker/toolchain-shim`
**Modified:** `docker/Dockerfile.finn`, `docker/finn_entrypoint.sh`

1. One shim script; the Dockerfile symlinks it as `vivado`, `vitis_hls`, `v++`,
   `xelab`, `xsim`, `lmutil` into a directory ahead of the mounted toolchain on
   `PATH`. Each sources via `finn-env` and `exec`s the real binary.
2. `BASH_ENV` plus a profile snippet for interactive and `bash -c` sessions.
3. Shrink the entrypoint to first-boot HOME/identity repair only. If nothing
   correctness-critical remains, delete it and drop `ENTRYPOINT` to `tini` alone.

**Verify** — the bare forms, which are the whole point:

```bash
docker exec <c> python -c 'import finn'
docker exec <c> vivado -version
sbx exec <s> python -c 'import finn'
sbx exec <s> vivado -version
```

All four must pass with no wrapper and no shell. Test 4 is currently expected to
fail on `vivado` — that is the latent docker defect this stage closes.

**Watch for:** shim recursion if the shim directory ends up on `PATH` twice.
`exec` the resolved absolute path, never re-resolve through `PATH`.

---

## Stage 4 — Identity, sudo, workspace policy

**Modified:** `docker/Dockerfile.finn`, `run-docker.sh`

1. Add `FROM dev AS sbx-dev` (and `sbx-build`, `sbx-build-xrt` as needed) adding
   only the sbx contract: NOPASSWD sudo, populated `/home/agent`, proxy
   `env_keep`. Remove those from the generic stages.
2. Point the sbx path at the `sbx-*` targets.
3. Teach `finn-env inspect` the workspace policy: `mirror` for FPGA tiers and
   all sbx, `fixed` for `dev` on docker. Not yet the default — plumbed, dormant.

**Verify:** `sbx-dev` and `dev` images built from one lineage; confirm shared
layers with `docker history`. Run the full sbx suite against `sbx-dev` and
confirm no behavior change. Confirm `dev` has no sudo and `sbx-dev` does.

**Note:** this is where "no privileges" becomes two separate properties. The
conformance test in stage 8 must assert both readings; write the assertion now
while the distinction is fresh.

---

## Stage 5 — Dependency modes

**Modified:** `docker/finn_paths.py`, `run-docker.sh`, `docker/Dockerfile.finn`

`frozen` / `live` / `auto`, replacing `finn_paths.py:123`'s
`!= "frozen"` test. `live` fails loudly on a missing checkout. `auto` is today's
behavior, named honestly. Remove the automatic `fetch-repos.sh`
(`run-docker.sh:319-321`) in favor of an explicit command.

Default stays `auto` in this stage. Flipping it to `frozen` is stage 7, with CI.

**Verify:** each mode with checkouts present and absent; `live` must fail with a
clear message naming the missing dependency, not fall back.

---

## Stage 6 — Declarative artifacts

The bulk of the work, and the first user-visible change.

**New:** `docker-bake.hcl`, `compose.yaml`, `.devcontainer/devcontainer.json`
**Modified:** `run-docker.sh` (→ wrapper), `docker/finn.kit/spec.yaml`

1. `docker-bake.hcl`: tier × profile matrix, `status` per target, args, OCI
   labels, `platform`. Tags derived here and nowhere else.
2. `compose.yaml`: services `dev`, `build`, `notebook` with Compose `profiles:`.
   Consumes `finn-env inspect` output via a generated `.env`.
3. `.devcontainer/devcontainer.json` → compose service `dev`.
4. Kit consumes `finn-env inspect` instead of computing anything.
5. `run-docker.sh` reduced to translation: old CLI → bake/compose. Target ~80
   lines. Keep `FINN_DOCKER_EXTRA` as a site escape hatch.

**Verify:** every old invocation still works through the wrapper. Both suites
green. Tag output from bake byte-identical to what `finn_compute_tag` produced,
so no image rebuild is triggered by the migration itself.

**Risk:** the largest stage. If it needs splitting, land bake first (build side
only, no runtime change), then compose.

---

## Stage 7 — Defaults and CI

**Modified:** `docker-bake.hcl`, `compose.yaml`, `ci/`, `docker/profiles/README.md`

Four changes that must land together, because each is a breaking default that
the others compensate for:

1. Default tier `build-xrt` → `dev`.
2. Default `dev` workspace → `/workspace/finn`. FPGA and sbx keep mirroring.
3. Default `FINN_DEPS` → `frozen`.
4. CI: `buildx bake`, publish by digest, shards receive the digest plus the
   provenance tuple (source commit, digest, profile, tier, resolved dep
   commits). Stage matrix gains a capability field.

**Verify:** CI green end to end. Two shards on one digest with `frozen` execute
identical code — assert this explicitly, it is the property the digest exists
for.

**Migration note required** in the release notes: three defaults changed, and a
user doing Vivado work will notice all three.

---

## Stage 8 — Matrix status and conformance

**Modified:** `docker-bake.hcl`, `docs/`
**New:** conformance suite

Status on every bake target (`supported` / `experimental` / `deprecated`);
py312 is `experimental` per `docker/profiles/README.md:15`. Record the supported
sbx version range — the contract is reverse-engineered, so the tested version is
part of it.

The nine conformance tests from the decisions doc, notably:

- bare `docker exec` and bare `sbx exec`, both `python` and `vivado`
- Xilinx and platform mounts present **and read-only**
- both privilege readings (generic: no host privilege, no in-container root;
  `sbx-*`: no host privilege, in-container root)
- a checkout path with spaces and unusual characters, under both workspace
  policies
- node-locked licence with the directory `:ro`

Split `containerization.md` into stable ADRs plus short user docs derived from
the live artifacts.

---

## Sequencing rationale

Stages 1-3 are additive: they fix defects and add capability without changing
any default, so each can ship alone and be reverted alone. Stage 4 introduces
the target split but keeps behavior identical. Stage 5 adds a mode without
changing the default.

Everything user-visible is concentrated in stages 6 and 7, which is deliberate —
one migration note, not five.

Stage 3 before stage 6 matters: the shims must work before `run-docker.sh` stops
being the thing that sets up the environment. Reversing them leaves a window
where neither mechanism is fully responsible.

Stage 5 before stage 7 matters: `frozen` must exist and be tested before it
becomes the default that makes the CI digest meaningful.

## Not in scope

`LIMITATION(finn-root-absolute)` is worked around, not fixed. The real fix is
IP-XACT packaging via finnlib (see that repo's design note), which would remove
`FINN_ROOT` from generated projects and make D4's per-tier policy unnecessary.
That is a separate project.

---

# Outcome

Eight stages, eight commits, all on `feature/sbx`.

| Stage | Commit | What landed |
| --- | --- | --- |
| 1 | `a4f09fbf7` | Defects 1-8, plus two found while fixing them |
| 2 | `71db8850a` | `docker/finn-env`, the single resolver, + 22 tests |
| 3 | `d064d02e8` | Transparent exec: shims, BASH_ENV, entrypoint shrunk |
| 4 | `5530bc006` | `sbx-*` targets, `sbx-contract.sh`, `FINN_TIER` |
| 5 | `731a9d65c` | `frozen` / `live` / `auto`, + 13 tests |
| 6 | `39c0ab9d7`, `84d4a1797` | bake; then compose, devcontainer, kit delegation |
| 7 | `e5fbfe75f` | Three default flips + CI provenance |
| 8 | `924603882` | Conformance suite + supported matrix |

## Verified on real hardware

- `synth_design` on `xczu28dr` with the toolchain mounted `:ro`: licence
  checked out and released, `SYNTH_OK cells=114`, 0 errors. The stage 1 stop
  condition did not trigger — the tools do not write into their installation.
- The same synthesis through a **bare `docker exec`**, no shell and no wrapper.
- Conformance suite: **20 pass, 0 fail, 3 skip.**
- Full `quicktest.sh` in the `dev` tier: **2479 passed**, 16 skipped, 5 xfailed,
  1 xpassed, 0 failed.
- Unit tests: **39 pass** (`tests/util/test_finn_env.py`,
  `test_finn_deps_modes.py`).
- All six bake tags byte-identical to `finn_compute_tag`, so the migration
  triggered no rebuild.
- `dev` 26 layers, `sbx-dev` 28 — the sbx targets really are two thin layers on
  a shared base, not a second image.

## Defects found that were not in the plan

Two, both while fixing the four that were:

- **The node-locked licence mount loop ran in a subshell.** It ended in
  `| while read`, so every `SBX_ARGS+=` was discarded and licence directories
  were never mounted. Same subshell-counter bug class the live-test scripts
  had, and the conformance suite was written to avoid.
- **`yecho` was called and never defined**, so a missing licence path printed
  "command not found" and lost the warning it was trying to give.

And two more found by running things rather than reading them:

- **`${UID}` in compose is a trap.** bash marks it readonly and exports neither
  `UID` nor `GID`, so compose silently takes the default and every file written
  to a bind mount ends up owned by uid 1000.
- **PATH had four copies of the full Xilinx PATH** — about 3 kB — in a live
  sandbox, because `settings64.sh` prepends unconditionally and the toolchain
  was applied by the entrypoint and again by the kit.

## Changed from the plan

- **The profile matrix moved into `docker-bake.hcl`.** The plan assumed bake
  could read the existing shell files. It cannot: HCL has no `file()` and does
  not read `.env` — both verified, not assumed. The alternatives were codegen
  (goes stale silently) or a mandatory wrapper (so a bare
  `docker buildx bake dev-py312` would build with py310's XRT package).
- **`sbx-contract.sh` exists.** The plan implied three inline `RUN` blocks;
  writing them made the duplication obvious given the argument the plan itself
  makes about duplicated paths drifting.
- **`FINN_TIER` was not in the plan** but is needed once targets can be
  sbx-qualified, so that a tier means the same thing on both backends.

## Still open

- **The Jenkins wiring is reviewed, not run.** There is no Jenkins here.
  `ci/scripts/build-images.sh` is tested directly and produces a correct
  provenance record; the Jenkinsfile changes around it are not exercised.
- **The node-locked licence question is unresolved.** Conformance test 9 skips
  for want of a node-locked licence, and says so. The `:ro`-versus-sibling-
  writes contradiction in `docker/finn.kit/spec.yaml` stands.
- **`run-docker.sh` still cannot survive a space in a path.** It assembles
  docker arguments by string concatenation. Conformance test 8 checks the
  workspace policies, not the launcher, and the script says so.
- **`dev-py312` does not build.** Marked `experimental` because it tracks
  unfinished upstream work; reported, does not gate.
- **`run-docker.sh` is ~870 lines, not the ~80 the plan projected.** Bake took
  the build and the tag rule; compose has not yet taken the runtime
  composition, so the launcher still owns the `docker run` assembly and the sbx
  create-or-attach flow. Finishing that is a further step, not a blocker — the
  declarative artifacts work today and the launcher agrees with them.


---

# Stage 9 (unplanned): `sbx env` and the launcher reduction

Added after reading the cloned sbx documentation, which revealed that sbx
0.39.0 — the version already pinned — ships declarative environment files.

| Commit | What landed |
| --- | --- |
| `e53d2ce91` | `docker/sbxenv/*.sbxenv.yaml` + `docker/finn-sbx`; -189 lines |
| `a42aeb238` | compose takes the docker runtime; `run-docker.sh` 900 → 256 |

## What the docs changed

The plan called for hand-writing `docker/finn-sbx` as a ~130-line peer of
`compose.yaml`. That was wrong: **`sbx env run` IS create-or-attach**, so most
of what `run-docker.sh` did for sbx — name derivation, existence check, a
timeout wrapper around a wedged-daemon `sbx ls`, positional mount assembly,
licence-directory mounts, an attach fallback for a racing create — was
reimplementing something sbx now owns. `finn-sbx` is 181 lines and does only
the three things the format cannot express.

## Facts established by testing, not by reading

The docs are silent or wrong on all of these:

- **Host `${VAR}` interpolation works in `name`, `workspace`, `kits[]` and
  `sandboxOptions.template`**, not just `env`. The release notes mention host
  interpolation; the field reference documents no syntax. This is what lets the
  committed environment files stay static and machine-independent.
- **`WORKSPACE_DIR` is a real environment variable** in the sandbox (= the
  primary workspace). **`WORKDIR` is not** — it is a kit-render placeholder for
  `files.content` only. Our kit uses the former and is correct; the two are
  easy to conflate because the docs only ever mention the latter.
- **`pullPolicy: never` genuinely refuses**, with an accurate message. The
  default `always` tries to pull `xilinx/finn:...` from Docker Hub and reports
  what looks like an authentication failure.

## What `.sbxenv.yaml` cannot express

Both confirmed by testing:

1. **Loading a locally built template.** sbx keeps its own image store and
   cannot see the host daemon's, so `docker save` + `sbx template load` must
   precede create. No field for it.
2. **Network egress.** No `network:` field, and no allow-at-create flag — only
   `--deny-network` (sbx ≥ 0.38). `sbx policy allow network --sandbox` stays a
   post-create step.

If a later sbx gains either, delete the corresponding block in `finn-sbx`
rather than keeping both.

## The placement constraint

sbx's docs are explicit that an environment file must live outside every
mounted workspace: with direct mount the agent can edit the file that governs
its own next sandbox. For a repo whose purpose is sandboxed agent work that is
not hypothetical. The committed artifacts under `docker/sbxenv/` are therefore
never what `sbx env` reads — `finn-sbx` materialises them into
`$XDG_STATE_HOME` at launch, and removes a stale `fpga` overlay when switching
to `dev` so the narrow profile cannot inherit a toolchain mount from a previous
build-tier run.

## Resolved from the earlier "still open" list

- **`run-docker.sh` is now 256 lines**, close to the ~80–120 projected once the
  licence header and legacy verb surface are counted. It derives no host facts.
- **The spaces-in-paths gap is closed** for the docker path: compose takes a
  YAML list, not a concatenated argument string.

## Newly open

- **`FINN_SINGULARITY` was dropped.** It worked by string-substituting the
  docker argument list (`-v` → `-B`), which belongs to compose now. It exits
  with a message pointing at `finn-env inspect --format json`. This is a real
  capability removal, not an oversight; if anyone uses it, it needs its own
  small script.
- **`sbx env` is experimental** and its file format may change. That is a
  deliberate bet — the tested version is recorded in
  `docker/profiles/README.md`, and `ci/scripts/conformance.sh 5` is the check
  that catches a break.
