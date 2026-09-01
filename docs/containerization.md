# FINN containerization: design record

Reference for contributors. To *run* FINN, read
[Getting Started](finn/getting_started.rst).

This replaces three earlier documents — an analysis of the previous design, a
decision record and an implementation plan. They described work that is now
done, in the present tense, which misleads a later reader.

The reasoning for each change is in its commit message. This document holds only
what spans several commits, and the findings that a later reader would otherwise
have to rediscover.

## The one idea

**Every fact about the host is found in exactly one place: `docker/finn-env`.**

This is a correctness requirement, not tidiness. Nine defects were found while
building this. Four were the same defect — two or more programs found the same
fact and then disagreed:

| Defect | Consequence |
|---|---|
| Xilinx mounted read-write on docker, read-only on sbx | An agent could destroy a 267 GB installation |
| The sbx path probed only the post-2024.2 directory layout | On a 2022.2 site it mounted **no toolchain** and still exported `VIVADO_PATH` |
| `PLATFORM_REPO_PATHS` given to sbx as a variable with no mount | Vitis and Alveo flows pointed at an absent directory |
| `setup-local.sh` held a third copy of the layout logic | "Vivado not found" on every recent installation |

None of these reproduced on the development machine, because it has one Xilinx
version in one layout. A program that finds nothing cannot disagree with
anything.

`run-docker.sh` went from **407 lines to 306** against `dev` — a 25% cut, bought
with roughly +5,600 lines across the rest of the stack. An earlier version of
this document said "900 lines to 262". That was wrong twice over: 900 was the
launcher's peak *on this branch*, reached partway through when the sbx logic
was still inline, so the figure measured a detour this work created and then
removed. The value of the change is that the launcher now derives no host
facts, not that it is shorter.

The other five were documentation contradicting itself or the code. Those are
listed under "Corrections" below, because a wrong comment costs as much as a
wrong line.

## Who owns what

| Subject | File |
|---|---|
| Which images exist, their names, their build inputs | `docker-bake.hcl` |
| Host facts: toolchain, licence, mounts, network | `docker/finn-env` |
| Applying the toolchain to a shell | `docker/finn-toolchain.sh` |
| Docker runtime | `compose.yaml` |
| sbx runtime | `docker/sbxenv/*.sbxenv.yaml` |
| Host-system setup | `setup-local.sh` |
| Dependency versions | `deps.env` |
| Accelerator runtime packages | `docker/runtimes/*.env` |
| Legacy command | `run-docker.sh` — owns nothing |

---

## Image structure

```
ubuntu:${UBUNTU_TAG}
 └─ system      OS packages, locale, ncurses 5 ABI, LSB loader, tini
    └─ python   interpreter, torch, requirements, tool pins
       └─ base  FINN wheels, finn-hlslib, board files, finn-env, shims,
          │     ENV LD_PRELOAD, USER agent          <- the publishable image
          └─ runtime   + the stacks in FINN_RUNTIMES (xrt, slash, v80pp)
             └─ sbx    + the sbx template contract
```

**One image, plus two orthogonal layers.**

| axis | values | in the tag |
|---|---|---|
| base | one | `<git>` |
| runtime targets | a set | `.xrt`, `.slash.xrt` (sorted, dot-joined) |
| sbx contract | boolean | `sbx-` prefix |

Four targets: `finn`, `finn-xrt`, `finn-sbx`, `finn-sbx-xrt`. Enumerated in
`docker-bake.hcl`, not generated — three runtime targets and a boolean would be
sixteen images, almost all unwanted. Build any other combination with `--set`.

Use `.` and not `+` to join runtime names: a Docker tag accepts
`[\w][\w.-]{0,127}`, and `+` gives `invalid reference format`.

### Why there is no tier axis, and no profile axis

There were three tiers and a profile. Both are gone, for the same reason: each
encoded a fact something else already owned.

**The profile axis had one member.** It threaded `FINN_PROFILE` through ten
files and four lanes to select from a set of size one. A second Ubuntu/Python
combination is a new value in one place, not an axis that must exist between
migrations.

**The tier axis encoded a host fact.** `dev` and `build` differed by 530 MB of
HLS headers and board definition files (measured: 2.91 GB against 3.44 GB) —
that is, by whether the user has Vivado. `finn-env` already resolves that at
launch, so the fact had two encodings and they could disagree. They did:
Apptainer mounts `$HOME`, so on a host that keeps its tools there, `vivado` ran
out of the *dev* image.

What made `dev` narrow was never which image you pulled. It was the absence of
a toolchain mount, a licence and egress — runtime grants, still enforced, now in
exactly one place. **`tier` still exists and still means something: it selects
grants.** There are two, `dev` and `build`. `build-xrt` was not a third set of
grants; it was XRT.

The cost is that a toolchain-less user carries 530 MB of headers nothing will
read. Without a Vivado mount there is nothing to include them, so they are
inert.

### The runtime seam

A runtime target is a set of `.deb` files installed on top of the base. Add one
by writing `docker/runtimes/<name>.env`; nothing else changes.

Two sourcing modes. `fetch` downloads a pinned, checksummed URL — XRT only,
because AMD publishes one, FINN's own code hard-depends on XRT
(`alveo_build.py:64`), and CI publishes that image unattended. `supply`
requires the user to put the file in `docker/packages/`, and a missing file
**fails the build naming the path**. It is never silently inert.

The admission test: a runtime target belongs only if code running inside the
image links or execs against it. XRT passes (`v++` links it, `make_driver.py`
execs `xclbinutil`). SLASH passes (`MakeCPPDriver` builds `finn-vrt-driver`
against VRT, and `ShellFlowType.SLASH_ALVEO` is a supported flow). PyNQ fails —
`from pynq import ...` appears only in a driver template copied to the board.

Every runtime target has a **host half** — `xocl`/`xclmgmt` for XRT, a kernel
module and `vrtd` for SLASH. The image carries userspace only.

FINN does not build third-party runtime stacks from source. It fetches
published artifacts and accepts supplied packages. A V80 user has already built
and installed SLASH on the host, because the kernel module has to be there, so
the `.deb` is a by-product they already hold — and FINN has no V80 to test a
build against.

### Why the sbx contract stays baked

Folding `sbx-contract.sh` into `docker/finn.kit` would delete the sbx axis and
leave literally one image. It was tried against sbx v0.39.0 and it does not
work. From the base image, with no contract baked:

| check | base image | sbx image |
|---|---|---|
| `id -nG` | `agent` | `agent sudo` |
| `/etc/sandbox-persistent.sh` | missing | present |
| `sudo -n true` | fails | succeeds |

`sbx create` itself now succeeds on the base image, so this fails quietly: the
sandbox comes up and is merely wrong. It also cannot be fixed from a kit for a
reason no ordering change would help — a kit runs as `agent`, and the kit would
have to `sudo` to grant itself the sudo it is granting.

### Why the sbx variant is a separate target

**In-container root is absorbed by a microVM under sbx, and under plain docker
the container is the only boundary there is.** That is the argument that holds,
and it is why the capability belongs to the variant whose runtime can take it.
Merged into the base, every `docker run` would grant passwordless root to uid
1000: an agent could install anything, disable the shims, rewrite `finn-env`, and
— with no userns remap — create root-owned files in the workspace bind mount.
Under sbx that mutability is *intended*; kits install software.

Two mitigations worth knowing: `run-docker.sh` always passes `--user $UID:$GID`,
so the sudoers entry (which is for `agent`, uid 1000) is inert for anyone going
through the launcher; and in-container root cannot write through a `:ro` mount.

A secondary argument: a build argument would produce two images with different
privilege and the same name. The same reasoning makes the runtime set part of
the tag — the predecessor of that layer was a bare `COPY v80pp.de[b]`, which
baked a package when the file happened to be present and produced an identical
tag either way. This one is weaker, though: it only applies *if* you keep two
images, so it cannot be the reason for keeping them.

It is not a separate Dockerfile. BuildKit shares every lower layer, so an
inherited target costs one thin layer — measured, 26 layers against 28 — and
inheritance *enforces* the equivalence that a second Dockerfile could only hope
for. Given that four defects came from duplicated paths drifting, adding a
second axis of duplication would have been the wrong lesson.

The contract itself is `docker/sbx-contract.sh` rather than an inline `RUN`
block. That was originally to stop a change reaching two of three tiers; with
one base image it survives because it is still the only place that knows the
contract, and because it now carries the evidence for why it cannot move.

### Build-time assertions

The build fails rather than shipping something subtly wrong:

1. `finn_paths.DEP_SRC_DIRS` must agree with `deps.env`.
2. A runtime target's `EXPECT_OS` must match the base image. A 24.04 package on
   a jammy base is something apt *resolves* rather than refuses, giving a subtly
   wrong runtime instead of a clean failure. This was an XRT-specific check
   before the runtime seam; every target gets it now.
3. A `SOURCE=supply` target's `.deb` must be present, and the error names the
   path.
4. `libudev.so.1` must be where the baked `ENV LD_PRELOAD` says.
5. The LSB loader symlink must resolve. Without it `lmutil` cannot exec, and the
   error names a file that plainly exists — the classic missing-ELF-interpreter
   confusion.

---

## Decisions worth keeping

### `dev` requires no host state, and that is the point

No toolchain mount, no licence, no secrets, no FINN network access. Only the
workspace.

This is about **host state**, not about setup steps. Plain Compose still needs
`.env` generated once, because Compose cannot run `id -u` and would otherwise
run as uid 1000 and fail to write a workspace owned by anyone else. An earlier
version of this document said `dev` needed no configuration at all. That was
wrong, and it was wrong in the primary documented command.

If the default environment requires nothing, the question "how does automation
discover what this image needs" largely *disappears* rather than needing a
mechanism. Requirements attach only to explicit escalation, where a person has
opted in. That is stronger than any requirement metadata, because it removes the
requirement instead of describing it.

The tier is defined by **absence**, and the absence is structural: in sbx, `dev`
does not read the file that adds the toolchain. It is not that the variables
happen to be empty. The original defect class was a narrow profile widening
because something was set in the caller's shell.

### Tier is a grant profile, and only sbx enforces it

`dev` withholds the toolchain mount, the licence and egress. Only sbx can
actually withhold the third:

| lane | tier | egress |
|---|---|---|
| sbx | `dev` / `build`, explicit | **enforced** — `sbx policy allow network` |
| docker | `auto` by default, `dev` opt-in | declared only |
| apptainer | none | the host's network namespace |
| bare host | n/a | n/a |

`finn-env` reports which through `egress_enforcement`. That field exists because
the dev tier used to report `egress: false` everywhere, and it was measurably
untrue:

```
$ docker run --rm xilinx/finn:<git> wget https://pypi.org/simple/
EGRESS: OPEN
```

Compose adds no network restriction, so a docker `dev` container has the open
internet. Conformance test 2 asserted the declaration and passed. Claiming a
property you do not enforce is worse than not claiming it, because a reader
cannot tell the difference.

**Apptainer has no tier at all**, and that is the honest version of what was
previously a documented defect: `$HOME` and `$PWD` are mounted automatically and
the host environment is inherited wholesale, so `vivado` ran out of the narrow
image on a host that keeps its tools under `$HOME`. A `dev` tier there was a
claim the runtime cannot keep. The positional argument is still accepted and
warns.

**`--tier auto`** resolves to `build` when the host has a toolchain and `dev`
when it does not. That degrade used to live in `run-docker.sh`, which made a
launcher the place a host fact was interpreted — the shape of the four defects
this design exists to prevent. `dev` and `build` stay explicit requests: an
explicit `--tier build` with no toolchain is still a hard error, because
silently narrowing a request is how a CI shard passes without testing anything.

### Licence egress narrows only when the vendor port is pinned

FLEXlm needs two connections: lmgrd on the advertised port hands the checkout to
a vendor daemon on a second port that is **ephemeral by default**. So the grant
is host-wide unless the site pinned it:

```
DAEMON xilinxd /opt/xilinx/xilinxd port=2101
```

`finn-env` reads that from a readable licence file, or takes
`FINN_LICENSE_VENDOR_PORT`. When known, the grant becomes `host:2100,host:2101`;
when not, the whole host. Both cases announce themselves in the launcher output.

`lmutil lmstat` is **not** a test of the narrowed rule — it only talks to lmgrd,
so it succeeds against a port-scoped grant while every real checkout fails with
"A valid license was not found", which reads as a licensing fault rather than a
firewall one.

### Three mechanisms deliver the toolchain, because one cannot

| Invocation | Mechanism |
|---|---|
| Interactive shell, `bash -c` | `BASH_ENV` → `/etc/finn-bashenv.sh` |
| Bare exec, no shell | `docker/toolchain-shim`, symlinked per tool on `PATH` |
| Python | `docker/finn_paths.py`, at interpreter startup |

**Neither `docker exec` nor `sbx exec` runs the ENTRYPOINT.** Docker's exec API
takes an argv array and `execve`s it; no shell is involved. FINN's own
documentation recommends `docker exec` for a second terminal, so this was a live
defect on the docker path and not an sbx quirk — `docker exec <c> vivado
-version` did not work.

An explicit wrapper does not fix it. A generic agent or CI system issues the bare
command and has no reason to know FINN's calling convention; requiring it moves
the special case to the caller instead of removing it.

**Most values do not need these mechanisms.** `XILINX_VIVADO`, `VIVADO_PATH`,
`XILINXD_LICENSE_FILE`, `FINN_ROOT` and `LD_PRELOAD` are all resolved on the host
and passed as container environment, which a bare exec inherits. Only three
values remain: `PATH`, `PYTHONPATH` and `LD_LIBRARY_PATH`, about 1.4 kB, produced
by AMD's `settings64.sh`. Those could in principle be computed host-side too, but
XRT forces the mechanism regardless — XRT is installed *in the image*, so the
host may have none.

**All three apply the same file: `docker/finn-toolchain.sh`.** It finds the
settings scripts from the environment, sources them, adds the two library paths
`settings64.sh` omits (`lib/lnx64.o` for `finn_xsi`'s simulation kernel,
`lnx64/tools/fpo_v7_1` for the floating-point operator libraries HLS-generated
code links), and de-duplicates the path variables.

It is **shell, with no Python in it**, and that is a correction. It used to be
`finn-env print --format sh`: spawn Python, spawn a bash inside it, source the
scripts, diff the environment against an eleven-name allowlist, print
assignments — and cache the result, because a comment claimed "sourcing
settings64.sh is the expensive part, about a second". Measured in the container
against a read-only mount:

| | |
|---|---|
| source Vivado + Vitis `settings64.sh` | **7 ms** |
| `finn-env print --format sh` | **124 ms** (85 ms of it Python startup) |

The extraction cost 18× the thing it existed to avoid, and the cache was storing
the extraction. All of it is gone — about 150 lines from `finn-env` and 35 from
`finn-bashenv.sh` — along with the fork bomb the two created between them: with
no Python in the path there is nothing to recurse into, so the `BASH_ENV`-clearing
handshake is gone too. What is deliberately lost is the allowlist; direct
sourcing applies whatever AMD's script sets.

**It requires bash, and says so loudly.** `settings64.sh` calls the `source`
builtin, which dash lacks. Under `/bin/sh` each source fails with
`source: not found` and — because every source is `|| true`, since these scripts
read unset variables on several releases — it fails *silently*, leaving `PATH`
untouched and every vendor tool reporting 127. This was found by running
`docker exec <c> vivado -version` against a container where the other two
mechanisms worked. The shim now has a bash shebang and the file refuses,
audibly, in a non-bash shell.

**Most values do not need these mechanisms.** `XILINX_VIVADO`, `VIVADO_PATH`,
`XILINXD_LICENSE_FILE`, `FINN_ROOT` and `LD_PRELOAD` are all resolved on the host
and passed as container environment, which a bare exec inherits. Only `PATH`,
`PYTHONPATH` and `LD_LIBRARY_PATH` come from `settings64.sh`.

**The layout probe stays host-side, and only there.** `finn-toolchain.sh` is
handed `XILINX_VIVADO` and friends and probes nothing, so there is still exactly
one resolver for where the tools are. `finn-env` is now a host-side program with
one subcommand.

### Workspace path policy is per-tier

| Environment | Path |
|---|---|
| `dev`, ordinary CI | fixed `/workspace/finn` |
| `build` | the host path |
| sbx, any tier | the host path — sbx cannot remap |

Generated Vivado projects embed `$::env(FINN_ROOT)` as an absolute path via
`add_files`. A project built under a fixed path cannot be opened in the host's
Vivado GUI, which is routine debugging here.

`dev` generates no Vivado projects, so mirroring buys it nothing while costing
remote-daemon support, reproducible diagnostic paths and simple Dev Container
configuration.

**Accepted consequence:** `FINN_ROOT` differs between `dev` and `build` on
one host. `finn_paths.py` resolves either correctly, but the path moving when you
change tier surprises people.

### `frozen` is the dependency default, and it is load-bearing

`FINN_DEPS` was a two-way test whose default was called `live` but fell back to
the wheels without a message when a checkout was absent — `auto` behaviour under
a name that promises determinism.

Three modes now: `frozen` (wheels always), `live` (checkouts, and a clear failure
naming what is missing), `auto` (whichever is there).

`frozen` is a precondition for CI provenance. An image digest identifies the
*environment*; FINN's source is mounted, not baked. With shadowing on, two shards
on one digest can execute different code — exactly the property the digest is
adopted to guarantee. The CI record is therefore the full tuple: source commit,
image digest, profile, tier, resolved dependency commits, deps mode.

### `sbx env` owns the sandbox

sbx 0.39.0 added declarative environment files, and `sbx env run` *is*
create-or-attach. Most of the previous sandbox launcher was reimplementing it.
`docker/finn-sbx` now does only the three things the format cannot express:

1. Load a locally built image into sbx's own image store.
2. Write the environment files outside every mounted workspace.
3. Permit network access to the licence server.

Item 2 is a safety requirement. With a direct mount the agent can write every
file it can read, so an environment file inside the workspace is a file the agent
can edit to widen its own next sandbox.

If a later sbx gains fields for 1 or 3, delete the block rather than keeping
both.

---

## Deliberately not adopted

**Image labels as a requirements protocol.** There is no OCI convention for
network allowlists or mount requirements, and FINN should not invent labels that
imply one exists. Labels describe what the image *is*; `finn-env` resolves what it
*needs*.

**Compose as the sbx driver.** sbx's create-or-attach lifecycle does not fit
Compose's model.

**Generated files as a single source of truth** across bake, compose and the kit.
Generated files go stale silently when someone edits the output. `finn-env` is a
*runtime* resolver instead, so it cannot disagree with itself.

**A separate `Dockerfile.sbx`.** See "Why the sbx variants are separate targets".

**`FINN_SINGULARITY`** is removed, not deprecated. It worked by string-replacing
the docker argument list, which belongs to Compose now.

---

## The `FINN_ROOT` limitation

FINN has no fixed workspace path. sbx cannot remap a mount at all. So the
workspace lives at a different absolute path for every developer, knowable only
at run time.

This is a **known structural flaw, accepted rather than fixed here.** It shapes
the design, so state precisely what it forbids.

**It forbids baked editable installs.** An editable install records an *absolute*
path — a `.pth` line, or a MAPPING inside a PEP 660 `__editable___*_finder.py`.
One developer's home directory, in an image shared by everyone. Standardising on
`/workspace` does not help, because sbx offers no container-side remapping. The
only options that work are resolving the path at import time, or installing at
container start — and the latter is the 33 s, needs-network,
mutates-the-workspace behaviour this work removed.

Hence `docker/finn_paths.py`.

**Five sites**, all tagged `LIMITATION(finn-root-absolute)` and greppable as one
set:

| Site | Role |
|---|---|
| `docker/finn_paths.py` | `workspace_root()` — the seam |
| `docker/finn_entrypoint.sh` | derives `FINN_ROOT` from `$PWD` |
| `run-docker.sh` | the host-path-mirroring mount |
| `src/finn/util/basic.py` | `FINN_HLSLIB_PATH` / `FINN_BOARD_FILES_PATH` defaults |
| `src/finn/xsi/paths.py` | `finn_xsi` source location |

**The extension point.** If the workspace ever gets a fixed path, this design
does not need reworking — it needs deleting:

1. Bake `pip install -e <fixed>/deps/<dep>` into the dev stage.
2. Delete `docker/finn_paths.py` and `docker/finn-live.pth`.
3. Keep `FINN_DEPS` as the switch, selecting between the editable install and the
   wheel rather than by prepending `sys.path`.

That also removes this approach's one wart: in live mode the *code* comes from
the workspace while the *declared version* comes from the baked wheel, so
`importlib.metadata` can report a commit that is not running. Nothing in FINN
reads that metadata today.

The real fix is to stop embedding paths at all. See the IP-XACT packaging note in
the finnlib repository, which would remove `FINN_ROOT` from generated projects
entirely.

---

## The sbx template contract

Reverse-engineered by bisecting real `sbx create` failures against a stock
template. A stock create against an unprepared image fails with nothing but
`ERROR: failed to run sandbox container`.

| Requirement | Symptom when absent |
|---|---|
| `agent` uid 1000, NOPASSWD sudo | Kits run as `agent` and cannot write |
| `/home/agent` **755**, with `.claude/ .local/bin .npm .cache workspace/` agent-owned | `cannot create .../settings.json: Permission denied` |
| `tini` as PID 1 | No zombie reaping |
| A `CMD` that does not exit | The container dies before setup |
| `NO_PROXY` / `no_proxy` | Loopback traffic goes through the egress proxy |
| `NPM_CONFIG_PREFIX` **and its bin directory on `PATH`** | `npm install -g` lands off `PATH`; a CLI installs "successfully", then `command -v` fails with 127 |

The 755-plus-populated-home requirement is in no documentation. It came from
diffing a stock template after the permission failure.

**`/etc/sandbox-persistent.sh` is sbx-managed.** The image must not write its
content — sbx replaces the file after the entrypoint runs, so anything written
there is dead code. It must however *exist*, so the first shell's `BASH_ENV` hook
can source it. A kit's startup command may add a line to it.

**sbx substitutes its own CMD.** PID 1 in a sandbox is
`tini -- finn_entrypoint.sh sh -c 'trap ...; sleep infinity & wait'`. The image's
`CMD` is irrelevant there. What kills a sandbox is the ENTRYPOINT *exiting*.

**Kit schema versions**, verified against v0.39.0: `schemaVersion: "1"` uses
`commands:`, `"2"` uses `setup:`. v2 is available from sbx 0.36. The grammars must
not be mixed.

**`.sbxenv.yaml` facts not in the documentation**, all established by testing:

- Host `${VAR}` interpolation works in `name`, `workspace`, `kits[]` and
  `sandboxOptions.template`, not only in `env`.
- `WORKSPACE_DIR` is a real environment variable. `WORKDIR` is **not** — it is a
  kit-render placeholder for `files.content` only.
- `pullPolicy: never` genuinely refuses. The default `always` tries to pull
  `xilinx/finn:...` from Docker Hub and reports what looks like an
  authentication failure.
- A **symlink is resolved and mounted at the link's path**. Path normalisation is
  therefore possible under sbx, using a host-side symlink.
- There is no network field, and no allow-at-create flag — only `--deny-network`
  (sbx 0.38 and later).

**Why not extend `docker/sandbox-templates`.** That is Docker's supported route
and it is closed to FINN: those images are Ubuntu 26.04 on Python 3.14, and
`torch 2.8.0` publishes no `cp314` wheel. The build tiers are pinned to an LTS by
XRT and Vivado regardless.

**This is reverse-engineered and an sbx release can break it.** The check that
matters is `ci/scripts/conformance.sh 5`: create a sandbox with plain `sbx`, no
wrapper, and run a bare `sbx exec`.

---

### Licensing, and what the kit cannot express

`XILINXD_LICENSE_FILE` takes two forms, and only one implies a mount:

| Form | Mount | Network |
|---|---|---|
| `PORT@HOST` | none | TCP to the **host** |
| `/path/to.lic` | the containing directory, `:ro` | none |

**Grant the host, never `host:port`.** FLEXlm uses two connections: `lmgrd` on
the advertised port is a directory service that hands back a second, usually
ephemeral, port for the vendor daemon where the checkout happens. A port-scoped
rule lets `lmutil lmstat` succeed while every real checkout fails with "A valid
license was not found" — a thoroughly misleading error for a firewall problem.
Verified by a real `synth_design`.

The directory rather than the file, because a single-file bind cannot carry
sibling state. **Unresolved:** the kit claims some FLEXlm setups write beside
the licence, and both backends mount that directory `:ro`. Both cannot be true.
Our licence testing used the floating form, which mounts nothing, so this is
untested — conformance test 9.

`.sbxenv.yaml` has **no network field**, and there is no allow-at-create flag
(only `--deny-network`, sbx 0.38+). So the grant is a post-create step in
`finn-sbx`, and it is the reason that script exists at all alongside the
declarative files.

### Why a kit and not the entrypoint

`sbx exec` does not run the ENTRYPOINT, so nothing `finn_entrypoint.sh` exports
reaches an exec session. The kit's `environment.variables` becomes **real
process env**, which does:

| Mechanism | `bash -c` | `sh -c` | bare exec |
|---|---|---|---|
| kit `environment.variables` | yes | yes | yes |
| `/etc/sandbox-persistent.sh` | yes | no | no |

The image still works with no kit at all: `finn_paths.py` resolves the same
values from `WORKSPACE_DIR` at interpreter startup.

## Corrections

Claims that were wrong, kept because the reasoning that produced them is a
pattern worth recognising.

| Claim | Reality |
|---|---|
| The libudev `LD_PRELOAD` workaround is microVM-specific | It is not. `realloc(): invalid pointer` reproduces on plain `docker run`. |
| sbx v0.39.0 rejects `setup:` and requires `commands:` | It does not. That was a v1 spec read by a v2 parser, and the diagnosis stopped at the first error message. |
| The entrypoint writes `/etc/sandbox-persistent.sh` | It does not, and must not. Stated four paragraphs after the opposite. |
| A floating licence requires an open network posture | A bare hostname grant is sufficient. Verified by a real checkout. |
| The Docker container gives no isolation | It gives separate namespaces, a reduced capability set and a seccomp filter. What it lacks is a separate kernel and any network allowlist. |

Every one came from reasoning where a measurement was available and cheap.

**Still unresolved.** `docker/finn.kit/spec.yaml` says some FLEXlm setups write
beside the licence file, and both backends mount that directory read-only. Both
cannot be true. Licence testing used the floating form, which mounts nothing, so
the path is untested. Conformance test 9 covers it and currently skips.

---

## Verification

`ci/scripts/conformance.sh` — thirteen checks, each corresponding to a defect
that happened or a contract that would erode silently. Tests 4 and 5 matter
most, and only in their **bare** form:

```
docker exec <c> vivado -version
sbx exec <s> vivado -version
```

Running them through a wrapper would pass while the property they exist to check
is broken. Test 4 did not pass before the transparent shims.

Test 7 asserts **both** readings of privilege, because "no privileges" is
ambiguous and the natural reading is the wrong one: the generic and sbx images
differ only in in-container root, and neither has host privilege.

Tests 12 and 13 guard the runtime seam. 12 checks that the tag names the
runtime set and that the two producers of that suffix — `tag()` in
`docker-bake.hcl` and `runtime_tag()` in `finn-env` — agree; if they drift,
compose names an image bake never built. 13 checks that a `SOURCE=supply`
target with no `.deb` fails the build with the path in the message, because the
build is the only place that is checked.

Unit tests: **45** in `tests/util/test_finn_env.py`, including the path-dedup
and idempotence checks, which now drive `finn-toolchain.sh` through bash rather
than calling a Python function.

Last measured: **27 pass, 0 fail, 4 skip.** Skips are the sbx sandbox, the
node-locked licence (still unresolved, see above) and the Apptainer `.sif`.

`quicktest.sh`: **2494 passed, 16 skipped, 5 xfailed, 1 xpassed, 32 errors.**
The 32 are one file, `tests/util/test_config.py`, where onnxscript's `@script`
decorator calls `inspect.getsource` on a function pytest's assertion rewriting
compiled. It is unrelated to containerization and reproduces identically in the
pre-restructure image — verified by running the same suite in both. An earlier
revision of this document reported "2487 passed, 0 failed", which was wrong.
