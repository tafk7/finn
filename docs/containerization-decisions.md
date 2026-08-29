# FINN containerization: decisions and justification

Date: 2026-08-28
Revision: 3
  r2 — review feedback; D3, D4 and D5 materially changed
  r3 — `containerization-assessment.md` folded in and deleted; stage 9 added
Status: **implemented.** All nine decisions are in the tree. The text below keeps
its original proposal voice; `containerization-plan.md` records what was built
and where it departed.

This document records what to change about FINN's container story, what to leave
alone, and why. It stands on its own. You do not have to read
`containerization.md` first.

Two names changed during implementation. `docker/finn-container-config` in D1
and D1a became **`docker/finn-env`**, because the same program serves the host
side and the container side. `docker/finn-sbx` in stage 9 is much smaller than
planned, because sbx 0.39.0 added declarative environment files.

## The problem

FINN's container environment must serve three consumers at once:

1. a human who clones the repo and wants a working environment;
2. automated agent sandboxes (docker containers or sbx microVMs) that must
   launch the image, run unattended, and do so **without knowing anything
   FINN-specific**;
3. CI.

Today all three go through `run-docker.sh` — an ~800-line launcher that is
simultaneously the build system, the runtime specification, the CI entry point,
and the sbx adapter. It is the sole authority on what any tier requires. That
concentration is the root cause of the defects below.

## Confirmed defects

Fix these regardless of any restructuring. Each was verified against the code,
not inferred.

### 1. The Xilinx install is mounted read-write under docker

`run-docker.sh:547`

```sh
DOCKER_EXEC+="-v $FINN_XILINX_PATH:$FINN_XILINX_PATH "
```

No `:ro`. An agent in a `build` or `build-xrt` container has write access to the
entire host Xilinx installation — 267 GB of proprietary, versioned,
hard-to-restore state. The sbx path already gets this right
(`run-docker.sh:720`), so this is also a parity gap.

### 2. sbx mounts no toolchain at all under the pre-2024.2 Xilinx layout

`run-docker.sh:719-720`

```sh
[ -d "$FINN_XILINX_PATH/$FINN_XILINX_VERSION" ] && \
  SBX_ARGS+=("$FINN_XILINX_PATH/$FINN_XILINX_VERSION:ro")
```

Xilinx changed its directory layout after 2024.2. The docker path handles both
(`run-docker.sh:536-542`) and sidesteps the question by mounting the whole
`$FINN_XILINX_PATH` root. The sbx path assumes the *new* layout only.

On a 2022.2 site — FINN's documented default version — that directory does not
exist, the `[ -d ]` guard fails silently, and no toolchain is mounted. The
sandbox is then handed `VIVADO_PATH` and friends (`run-docker.sh:722-726`)
pointing at paths that are not present.

This does not reproduce on a host using the post-2024.2 layout, which is why it
has not been seen.

### 3. `PLATFORM_REPO_PATHS` is passed to sbx but never mounted

`run-docker.sh:722-726` puts it in `SBX_ENV`; the only sbx mounts are the Xilinx
version directory and licence directories. The docker path mounts it
(`run-docker.sh:558-561`). Vitis/Alveo flows under sbx receive a path to a
directory that is not there.

### 4. The generic image has no non-root default user

`docker/Dockerfile.finn:415` creates the `agent` user, but there is no `USER`
directive anywhere in the file. The default user is root. Any statement
elsewhere about a "safe non-root default" is aspirational, not current.

### 5. `docs/containerization.md` contradicts itself on `sandbox-persistent.sh`

Line 678 states `/etc/sandbox-persistent.sh` is sbx-managed and the image must
not write it. Line 701 states the entrypoint writes what it resolved to it. The
entrypoint says it deliberately does not
(`docker/finn_entrypoint.sh:252`). Line 678 and the entrypoint are correct.

### 6. `docs/containerization.md` contradicts the working kit on schema

Line 685 states that sbx v0.39.0 *rejects* a `setup:` key and requires
`commands:`. The checked-in kit uses `schemaVersion: "2"` with `setup:`
(`docker/finn.kit/spec.yaml:43,49`) and demonstrably works on v0.39.0. The doc
is wrong.

### 7. The kit claims a floating licence requires open network posture

`docker/finn.kit/spec.yaml:17-20` says a `PORT@HOST` licence "requires open
posture" because a domain-oriented egress policy cannot express `host:port`.
The launcher grants only the licence hostname (`run-docker.sh:776`), and that
was verified sufficient by a real `synth_design` licence checkout. The kit
comment is stale, from before we established that `sbx policy allow network`
accepts bare hosts.

### 8. The kit claims Vivado writes beside the licence, but both paths mount `:ro`

`docker/finn.kit/spec.yaml:27-28` justifies mounting the licence *directory*
rather than the file because "some licences have sibling lock/state files that
Vivado writes next to them." Both the docker and sbx paths mount that directory
read-only. Either the write claim is wrong or the mount policy is.

**Unresolved.** Our licence testing used the floating (`PORT@HOST`) form, which
mounts nothing, so this path is untested. Resolve by testing node-locked
checkout against a named sbx version — not by editing the prose.

### The pattern

Defects 1-3 are one defect wearing three hats: **two code paths independently
deriving the same host fact, and drifting.** Defects 5-8 are the documentation
equivalent. This is the empirical argument for everything below.

## Decisions

### D1. Split `run-docker.sh` into standard declarative artifacts

| Artifact | Owns |
| --- | --- |
| `docker/Dockerfile.finn` | image contents; one lineage, see D3 |
| `docker-bake.hcl` | build matrix (tier x profile x support status), args, tags, labels, platform |
| `compose.yaml` | human-facing runtime services: mounts, env, workdir, user, ports |
| `.devcontainer/devcontainer.json` | portable default contract for agent and IDE tooling |
| `docker/finn-container-config` | **host discovery** — the imperative remainder; see D1a |
| `docker/finn.kit/spec.yaml` | sbx adapter only, consuming D1a output |
| `run-docker.sh` | thin compatibility wrapper translating the old CLI |

**Justification.** `run-docker.sh` performs four separable jobs — build
definition, runtime composition, host discovery, command dispatch — and three
have a well-established declarative format. The tier x profile matrix *is* a
bake matrix; expressing it as shell tag-computation forces CI to re-derive tags
rather than reference them. Declarative artifacts are machine-readable by
construction, which is what consumer (2) needs.

The launcher is not deleted. It has been the interface for close to a decade;
it becomes a wrapper that no longer owns anything.

### D1a. Host discovery gets a named artifact

`docker/finn-container-config inspect --format json` emits normalized,
host-resolved facts:

- selected tier and profile, and the profile's support status
- workspace and build-directory policy (see D4)
- exact Xilinx mount source and target paths
- platform repository mounts
- licence file mounts
- licence server hosts requiring egress
- runtime environment variables
- required platform (`linux/amd64`)

Compose wrapper, sbx adapter, and CI all consume this one output.

**Justification.** D1 identifies host discovery as the irreducible imperative
part but, in revision 1, left it unowned — which would have let the Xilinx
layout logic be reimplemented independently in the Compose and sbx paths, the
exact mechanism that produced defects 1-3.

This is **runtime resolution of host inputs**, not static codegen of the bake,
compose, and kit files. That distinction matters: codegen goes stale silently,
whereas a resolver runs at launch and cannot be out of date with itself.

**Additionally, mount the complete Xilinx root read-only on both backends.**
This is a cheap, independent simplification: it eliminates defect 2 outright and
removes almost all version-layout branching, at the cost of exposing more
read-only filesystem surface. Take both — the wide read-only mount kills the
largest duplication now, and the config component owns what remains (licence
classification, platform repository, egress hosts, environment).

### D2. `dev` becomes the universal default, and requires nothing

The default tier changes from `build-xrt` (`run-docker.sh:116`) to `dev`, held
to a hard contract.

**Precise statement of the contract**, since the loose version is ambiguous:

- No host devices, no added capabilities, no privileged mode, no Docker socket.
- No host toolchain mount, no licence, no secrets.
- **No FINN-specific network egress.** This is distinct from "networking is
  disabled": an agent harness may still need its own control-plane traffic. The
  claim is that *FINN* requires no allowlist entry.
- Only the workspace is mounted **from the host**. The container still requires
  writable in-container state: `$HOME`, `/tmp`, and the build scratch directory.
  These are container-local and imply no host exposure.
- Whether root is available *inside* the container is governed by D3, and is
  deliberately not covered by the word "privileges" here.

**Justification.** Two reasons; the second matters more.

First, `build-xrt` as the default is Jenkins history. It is the largest tier
with the widest host exposure, and most work does not need it. (It adds no
Docker capabilities or privileged mode — the exposure is mount surface and
licence reach, not container privilege.)

Second: if the default agent environment requires nothing, the "how does
automation discover what this image needs" problem largely *dissolves* rather
than needing a mechanism. A profile that needs nothing needs no discovery
protocol. Requirements attach only to explicit escalation, where a human has
opted in. That is stronger than any requirement metadata, because it removes the
requirement instead of describing it.

Partly true already: the sbx path enforces exactly this boundary
(`run-docker.sh:735-737` — `dev` gets no Xilinx mount, no licence, no egress,
even when the caller's shell has those variables set). The decision is to make
it the default and hold the docker path to it.

### D3. One lineage, explicit targets — not one final image, not a second Dockerfile

Revision 1 said "keep one image." That was too strong, and it contained a
contradiction. Corrected:

```dockerfile
FROM dev AS sbx-dev
# adds only the sbx contract: USER agent, NOPASSWD sudo, populated home
```

- **One Dockerfile, one shared lineage.** The sbx target inherits the generic
  image and adds one small final layer.
- **Explicit named targets**, `sbx-dev` and only the further sbx tier variants
  actually needed — not the full cross product.
- **No security-relevant build argument.** A build arg that toggles sudo
  produces a hidden image variant: two images with materially different
  privilege, indistinguishable by tag.

**Justification.** Revision 1's cost argument was wrong. BuildKit shares all
lower layers, so an inherited target does not double build time or storage —
the marginal cost is one thin layer, not a second image. And inheritance
*enforces* the behavioral equivalence that revision 1 worried about losing,
rather than merely hoping for it.

The real problem revision 1 half-saw was the contradiction between D2 ("no
privileges") and gating NOPASSWD sudo behind a build argument. Both are resolved
by making the privilege difference a named target: the tag says which one you
have.

Consequence for D2 and conformance test 7: "no privileges" means no *host*
privilege — no devices, no capabilities, no privileged mode, no sensitive
writable mounts. NOPASSWD root *inside* an sbx microVM is a deliberate property
of the `sbx-*` targets, not a violation. Both readings must appear in the
conformance test or it will be interpreted wrongly.

### D4. Workspace path policy is per-tier, not global

Revision 1 said "mirror the host path everywhere, it costs nothing." The cost
claim was false. Corrected policy:

| Environment | Workspace path |
| --- | --- |
| `dev`, ordinary CI | fixed `/workspace/finn` |
| FPGA / GUI / LSF profile | host-path mirroring |
| sbx (any tier) | host-path mirroring — the backend has no remapping |

**Justification for mirroring where it applies.** `LIMITATION(finn-root-absolute)`:
generated Vivado projects embed `$::env(FINN_ROOT)` as an absolute path via
`add_files`, so the path is written into the `.xpr`. Under a fixed path, such a
project cannot be opened in the host's Vivado GUI — a failure that surfaces at
debug time, and opening the project is routine debugging here.

**Justification for not mirroring everywhere.** That argument applies to tiers
that *generate Vivado projects*. `dev` is Python-only and generates none, so
mirroring buys it nothing while costing:

- compatibility with tooling that assumes a conventional workspace directory;
- remote Docker daemons, where the client's checkout path does not exist on the
  daemon host;
- reproducible diagnostic paths and artifact contents across machines;
- Dev Container configuration simplicity;
- portability across host OSes and path layouts.

**Accepted consequence.** `FINN_ROOT` now differs between `dev` and the FPGA
tiers on the same host, so switching tiers changes the workspace path.
`finn_paths.py` already resolves this correctly, but it will surprise people and
must be documented prominently, not buried.

**Required regardless:** a conformance test using a checkout path containing
spaces and unusual characters. The launcher's string-assembly style
(`DOCKER_EXEC+=` concatenation, `run-docker.sh:487` onward) is specifically
vulnerable, and both path policies traverse it.

The long-term fix is to stop embedding paths at all — see `finnlib`'s IP-XACT
packaging note, which removes `FINN_ROOT` from generated projects entirely.

### D5. Generic exec must work without knowing FINN's conventions

Revision 1 proposed `finn-env exec <command>`. That is necessary but not
sufficient: a generic agent issues `docker exec <c> pytest` or
`sbx exec <s> vivado -version`. It will never know to prepend a FINN wrapper.
An opt-in wrapper does not solve transparent execution; it relocates the
special-casing to the caller.

Three interfaces, together:

1. **`finn-env exec <command>`** — explicit use, CI, scripts.
2. **`finn-env print --format=sh|json|env0`** — for adapters importing the
   resolved environment (the sbx kit, Compose env files, CI).
3. **Transparent command shims** on `PATH` for the mounted vendor tools —
   `vivado`, `vitis_hls`, `v++`, `xelab`, `xsim`, `lmutil`. Each sources the
   toolchain, then `exec`s the real executable.

Shell initialization (`BASH_ENV` plus an immutable profile snippet) covers
interactive sessions and `bash -c`. The shims cover bare `exec` with no shell.
Between them the image behaves correctly under every invocation style with no
caller cooperation.

**Justification.** Not every consumer runs the entrypoint. `sbx exec` bypasses
it — this cost us a Vivado `SIGABRT` when `LD_PRELOAD` lived only in the
entrypoint. Critically, **plain `docker exec` bypasses it too**, and FINN's own
documentation recommends `docker exec` for a second terminal. This is a latent
defect on the docker path that has not yet been tripped, not an sbx quirk.

Today the gap is patched twice and incompletely: `finn_paths.py` repairs Python
after a bypass, and the kit reproduces enough environment for sbx. A
long-running docker container followed by `docker exec` gets neither.

### D6. CI: digest plus a full provenance tuple

Replace the current flow — invoke `run-docker.sh` to run an `echo`, delete the
machine-wide image cache, `docker save` to NFS, shards reload by tag — with
`docker buildx bake`, publication by digest (or an OCI archive plus its digest
where NFS transport is mandatory), and shards receiving the digest.

**The digest identifies the environment only.** FINN source is mounted, not
baked, so the digest does not identify the code under test. The CI provenance
record must be the full tuple:

```
FINN source commit
image digest
profile
tier
resolved dependency commits
```

CI sets `FINN_DEPS=frozen` (see D9) unless a job specifically tests coordinated
changes to qonnx, brevitas, or finn-experimental. Otherwise mounted dependency
checkouts shadow the baked wheels and **two shards on the same image digest can
execute different code** — the exact reproducibility failure the digest was
adopted to prevent.

`deps.env` branch or tag references resolve to commits, and the commits are
recorded in image provenance or CI metadata.

Add a capability field to the CI stage matrix: Python-only tests on `dev`,
Vivado/HLS on `build`, Vitis/Alveo on `build-xrt`. Today everything pays for the
largest tier.

### D7. Define the supported matrix; conformance tests follow from it

Conformance test 1 cannot be "every tier and profile builds" while
`docker/profiles/README.md:15` marks py312 "Tracks upstream PR 1603, which is
WIP. **Not yet validated here.**"

Every bake target carries a status:

- **supported** — must build and must pass its declared tests.
- **experimental** — smoke-built; test failures do not gate.
- **deprecated** — retained, excluded from normal CI.

The same applies to supported Xilinx versions and to **the supported sbx version
or range**. The sbx contract is reverse-engineered from observed behavior, so
the tested version is part of the compatibility contract, not an incidental
detail. Defect 6 is what happens when it is left implicit.

### D8. Conformance tests

1. Every **supported** tier and profile builds; experimental ones smoke-build.
2. The default `dev` environment runs with only the workspace mounted from the
   host, and requires no FINN-specific egress, secret, device, or host
   privilege.
3. A fresh `docker run` works.
4. A long-running container followed by bare `docker exec`:
   ```
   docker exec <c> python -c 'import finn'
   docker exec <c> vivado -version
   ```
5. A plain `sbx create` + bare `sbx exec`, no launcher involved:
   ```
   sbx exec <s> python -c 'import finn'
   sbx exec <s> vivado -version
   ```
6. Xilinx and platform mounts present **and read-only** in FPGA profiles.
7. Privilege boundaries, both readings: generic targets have no host privilege
   *and* no in-container root; `sbx-*` targets have no host privilege *and*
   do have in-container root.
8. A checkout path containing spaces and unusual characters, under both
   workspace policies.
9. Node-locked licence checkout with the licence directory mounted `:ro`
   (resolves defect 8).

Tests 4 and 5 in their **bare** form are the load-bearing ones — they are what
distinguishes D5 from revision 1's wrapper-only proposal.

### D9. Three explicit dependency modes

Once automatic `fetch-repos.sh` is removed, today's `FINN_DEPS=live` default is
misleading. `docker/finn_paths.py:123` treats anything not equal to `"frozen"`
as live, and live silently falls back to baked wheels when a checkout is absent
— so the documented default is really auto-behavior under a name that promises
determinism.

- **`frozen`** — always baked dependencies. Default for CI and agents.
- **`live`** — require all declared source checkouts; **fail clearly** if any
  is missing.
- **`auto`** — live where present, baked otherwise. Today's actual behavior,
  now named honestly.

**Justification.** Unattended runs become reproducible without making
co-development cumbersome. This is also a precondition for D6: `frozen` is what
makes the image digest meaningful.

## Deliberately not adopted

**Image labels declaring runtime requirements.** Tempting — an automation could
`docker inspect` and learn a tier needs a licence host and a toolchain mount.
Rejected as the primary mechanism: there is no OCI convention for egress
allowlists, and FINN should not invent labels implying a standard exists.

The residual gap is real: for `build` and `build-xrt`, requirements live in
per-backend adapters, which is how defects 1-3 arose. D1a narrows this
considerably by giving those adapters one resolver to consume. D2 closes it for
the default profile. If the escalated profiles drift again despite D1a, revisit
labels as an explicitly FINN-namespaced, explicitly non-standard convention.

**Compose as the sbx driver.** sbx's create-or-attach lifecycle does not fit
Compose's model; forcing it yields something neither tool's users recognize.

**Generated single source of truth across bake, compose, and the kit.** They
have different jobs, and codegen buys consistency at the cost of a generation
step that goes stale. D1a is deliberately a *runtime* resolver instead.

## Launcher hygiene

- Stop running `fetch-repos.sh` on nearly every invocation
  (`run-docker.sh:319-321`); make dependency sync explicit, per D9.
- Stop mounting `ssh_keys` automatically (`run-docker.sh:66`); make it opt-in.
- Drop `docker run --init` (`run-docker.sh:475`) — the image has `tini` as PID 1,
  so this is a redundant second init.
- Remove unconditional TTY allocation from non-interactive runs.
- Keep `FINN_DOCKER_EXTRA` as a site escape hatch; stop using it in documented
  workflows.

## Sequencing

1. **Defects 1-8.** Independent of everything else; defect 1 is a live
   containment hole. Defect 8 needs a test, not an edit.
2. **Identity, sudo, and workspace policy** (D3, D4): set the generic default
   user, add the `sbx-*` targets, add the fixed-path profile. These are not
   free — revision 1 wrongly called them no-ops.
3. **D5** (`finn-env` plus shims) and conformance tests 3-5. These pin current
   behavior and make the rest safe to change.
4. **D9** dependency modes — precondition for D6.
5. **D1/D1a** artifacts, with `run-docker.sh` reduced to a wrapper.
6. **D2 and D6 together** — the CI matrix must gain its capability field in the
   same step the default drops to `dev`.
7. **D7** matrix status, then the full D8 suite.


---

# Appendix: the review that produced revision 2

`containerization-assessment.md` was an independent review of the container
story, done before any of this was built. It has been deleted rather than kept,
because every finding is now either fixed or recorded here, and a document that
states resolved problems in the present tense misleads a later reader.

What it got right, and where each item went:

| Finding | Outcome |
|---|---|
| Xilinx mounted read-write under docker | Fixed, defect 1 |
| `run-docker.sh` is the authoritative build, runtime, CI and sandbox spec | D1 |
| `build-xrt` as the default is Jenkins history | D2 |
| Docker and sbx are not equivalent, and the differences are accidental | D1a |
| `docker exec` bypasses the entrypoint, so the entrypoint cannot be load-bearing | D5 |
| CI passes a mutable tag where it needs a digest | D6 |
| py312 is documented as available but is not validated | D7 |
| `containerization.md` has drifted from the implementation | Defects 5-8 |

Where revision 2 disagreed with it, and why:

- **A separate `Dockerfile.sbx`.** Rejected. BuildKit shares lower layers, so an
  inherited target costs one thin layer, not a second image — measured, 26
  layers against 28. Inheritance also *enforces* the equivalence that a second
  Dockerfile could only hope for. See D3.
- **A fixed `/workspace/finn` everywhere.** Adopted for `dev` only. Generated
  Vivado projects embed `$::env(FINN_ROOT)`, so a project built under a fixed
  path cannot be opened in the host GUI. See D4.
- **Rejecting image labels outright.** Adopted in part: labels describe what the
  image *is*, and `finn-env` resolves what it *needs*. See the "Deliberately not
  adopted" section.

Two things the review did not anticipate, both found by building it:

- **`sbx env`.** sbx 0.39.0 ships declarative environment files, so most of the
  sandbox launcher was reimplementing something sbx owns. See stage 9.
- **The defect count.** The review named four; nine were found, and four of the
  nine were the same fact derived in a fourth, fifth and sixth place.
