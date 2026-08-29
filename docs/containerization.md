# FINN containerization redesign — analysis and design

Branch: `feature/sbx`. Status: **implemented and verified on docker and plain sbx.** §1–§3 are the original analysis
and are unchanged. §4 records what was built and where it departs from the
proposal; §5 answers the open questions with measurements.

Goal: rebuild FINN's container story around standard Docker practice, make the
image usable as an `sbx` base, and make it safe to hand a coding agent a
fully sandboxed FINN working environment.

---

## 1. What exists today

Three files carry the whole flow:

| File | Lines | Role |
|---|---|---|
| `run-docker.sh` | 407 | env defaults, tag computation, `docker build`, `docker run`, mode dispatch |
| `docker/Dockerfile.finn` | 171 | single-stage image: OS deps → XRT → ~30 `pip install` layers → user creation |
| `docker/finn_entrypoint.sh` | 171 | per-start: 4 editable installs, tool sourcing, `finn_xsi` build, tcl copying |

Jenkins (`ci/Jenkinsfile`, 1157 lines) drives it through a small contract that
any redesign **must preserve**: `./run-docker.sh print-tag`, `./run-docker.sh
<cmd>` passthrough, and the env vars `FINN_DOCKER_PREBUILT`,
`FINN_DOCKER_SHARED_IMAGE_DIR`, `FINN_DOCKER_CACHE_DIR`, `FINN_DOCKER_EXTRA`.

---

## 2. Findings

### 2.1 Identity is baked into the image — the central problem

```dockerfile
ARG USERNAME / USER_UID / GROUP_ID / GROUPNAME
RUN useradd $USERNAME -u $USER_UID
USER USERNAME
```

The image is built per-user, but `FINN_DOCKER_TAG` encodes only `git describe`
+ XRT version. **Two developers produce materially different images under an
identical tag.** Every tag-keyed path — the Jenkins publish step,
`FINN_DOCKER_SHARED_IMAGE_DIR`, any registry push — is therefore unsound.

Note also that line 169 reads `USER USERNAME`, not `USER $USERNAME` — a literal
username that does not exist in the image. It is masked at runtime because
`run-docker.sh` passes `--user $DOCKER_UID:$DOCKER_GID`, which overrides it. The
bug is latent, not dormant: anything running the image without that flag breaks.

Docker's convention is that identity is a **runtime** concern, and sandbox
tooling built on sbx follows it: a shim provisions the passwd entry and a
writable HOME as root, then drops privileges and execs the original entrypoint.
Build-time identity fights that machinery for no gain.

### 2.2 The entrypoint does build work on every start

`finn_entrypoint.sh` runs on **every container start**:

- four `pip install --user -e` calls (`qonnx`, `finn-experimental`, `brevitas`, `finn`)
- a `mv`/`trap` dance on `deps/qonnx/pyproject.toml`
- a `finn_xsi` compile writing `xsi.so` **into the source tree**
- copying `.Xilinx/*.tcl` from the repo into `$HOME`

Three distinct problems:

1. **Latency.** Work that belongs in a cached image layer is paid per start. For
   an agent workflow — where sandboxes are created and destroyed freely — this
   is the dominant cost.
2. **Workspace mutation.** The `mv` renames a file in the *mounted repo*. If the
   container dies between the `mv` and the `trap`, the tree is left damaged. Two
   containers starting concurrently race it outright. Same for `xsi.so`, which
   is a build artifact deposited in source control's path.
3. **Non-idempotence.** Nothing is guarded by a sentinel, so every start redoes
   everything.

### 2.3 `HOME` is overwritten unconditionally

```bash
export HOME=/tmp/home_dir
```

Under any runtime that provisions a HOME before exec, this discards it — taking
with it whatever state was seeded there. Result: an agent CLI loses its
configuration, and `pip install --user` lands somewhere unexpected. The
same line makes the container hostile to any orchestrator with an opinion about
HOME. `SHELL` and `PS1` are overwritten the same way.

The trailing `exec "$@"` is commented "execute the provided command(s) as root",
which has been false since the `USER` line was added.

### 2.4 Environment variables have no tiering

45 distinct variables are in play with no separation between build inputs,
host bindings, and tunables. Classified:

**Build-time (correctly ARGs today):** `XRT_DEB_VERSION`, `SKIP_XRT`,
`LOCAL_XRT`, `V80PP_DEB_PACKAGE`.

**Static — should be `ENV`, currently passed as `-e` on every run:**
`LD_PRELOAD`, `XILINX_LOCAL_USER_DATA`, `LOCALHOST_URL`, `NUM_DEFAULT_WORKERS`,
`SHELL`, plus `TZ`/`LANG`/`LC_ALL` which the entrypoint re-exports despite the
Dockerfile already setting the locale. `VIVADO_IP_CACHE` is already `ENV` and is
the model to follow.

**Genuine host bindings — must stay runtime:** `FINN_XILINX_PATH`,
`VIVADO_PATH`, `VITIS_PATH`, `HLS_PATH`, `PLATFORM_REPO_PATHS`, `FINN_ROOT`,
`FINN_BUILD_DIR`, `IMAGENET_VAL_PATH`, `FINN_EXAMPLES_ROOT`, `VERIFICATION_IO`,
`XILINXD_LICENSE_FILE`.

**Runtime tunables:** `JUPYTER_PORT`, `NETRON_PORT`, `VERIFICATION_EN`,
`LIVENESS_THRESHOLD`, `RTLSIM_TRACE_DEPTH`.

**Cache locations:** `TORCH_HOME`, `HF_HOME`, `FINN_DOCKER_CACHE_DIR`.

This matters concretely for `sbx`: **the sbx backend has no create-time env
injection seam.** Every variable that stays in the `-e` tier is a variable that
cannot be delivered to a microVM. Moving the static tier to `ENV` is what makes
FINN usable under sbx at all.

### 2.5 No `.dockerignore`

The build context is **1.3 GB**, of which `deps/` is 1.2 GB and `.git` is 97 MB.
All of it is shipped to the daemon on every build, and none of it is used by the
Dockerfile except `requirements.txt` and `docker/`. This is pure latency.

### 2.6 Layer ordering defeats the cache

~30 sequential `RUN pip install` lines, each its own layer, with
`pip install -r /tmp/requirements.txt` in the middle. Editing `requirements.txt`
invalidates everything after it. The heaviest single install (torch + cu126,
several GB) sits mid-chain rather than early where it would be stable.

### 2.7 Host-path mirroring

`-v $SCRIPTPATH:$SCRIPTPATH` and `-v $FINN_XILINX_PATH:$FINN_XILINX_PATH` mount
host paths at *identical* container paths, which is why `FINN_ROOT` must be
passed explicitly and cannot be baked. Docker convention would be a fixed
`/workspace`. Note this one happens to align with sbx, which also mounts the
workspace at its host path — so the fix is not to hardcode either, but to make
`FINN_ROOT` *derived* (see §3.4).

### 2.8 Base image and XRT version are pinned independently — and can disagree

The XRT package name encodes three coordinates:

```
xrt_202420.2.18.179_22.04-amd64-xrt
    └─ version ──┘ └OS┘ └arch┘
```

The base image encodes the OS a second time, with nothing tying them together:

```dockerfile
FROM ubuntu:jammy-20230126          # 22.04
ARG XRT_DEB_VERSION="xrt_..._22.04-amd64-xrt"
```

Setting `XRT_DEB_VERSION` to a `24.04` build against a Jammy base installs a
Noble package on the wrong OS. `apt install` will generally resolve it rather
than refuse, yielding a subtly wrong runtime instead of a clean failure. The
architecture has the same shape of problem: `amd64` is a literal inside a string
while the actual build platform is whatever the builder happens to be.

This is not theoretical drift — this checkout pins `xrt_202220.2.14.354`
(2022.2-era) while other checkouts have moved to `202420`.

Two adjacent weaknesses in the same `RUN`:

- the download is an unauthenticated `wget` with a spoofed browser user-agent
  and **no checksum verification** — a supply-chain soft spot
- it targets `xilinx.com/bin/public/openDownload?filename=`, the older redirect
  endpoint; current releases are served from `download.amd.com/opendownload/xrt/`

**Why not build `FROM` an XRT image.** AMD publishes no XRT base image on
ghcr.io. The nearest equivalent is
[Xilinx/Xilinx_Base_Runtime](https://github.com/Xilinx/Xilinx_Base_Runtime) on
Docker Hub, which is unsuitable here for three reasons: it targets **Alveo
deployment**, pairing a container runtime with a matching host driver for device
access that FINN's build-time usage does not need; it would invert control over
the Ubuntu and Python versions, both of which FINN is opinionated about; and its
published tags trail current XRT releases substantially.

### 2.9 Vivado, licensing, and the agent threat model

The Xilinx install is **267 GB** (127 GB for 2025.2, 135 GB for 2026.1) and is
mounted from the host — it cannot be baked, and mounting both versions when one
is needed doubles the exposure for nothing.

Licensing takes one of two forms, and `XILINXD_LICENSE_FILE` accepts either:

- **node-locked** — a path to a `.lic` file, which must be readable *inside* the
  sandbox;
- **floating** — `PORT@HOST`, a FLEXlm server spoken to over **raw TCP, not
  HTTP**. In a corporate deployment that host is typically on internal RFC1918
  space.

A floating licence needs raw TCP egress, and **sbx grants that narrowly — open
posture is not required.** `sbx policy allow network` accepts hostnames,
domains, IP addresses and an optional port suffix, scoped per sandbox:

```
sbx policy allow network --sandbox <name> licence-server.internal:27034
```

Measured on a live sandbox, with a Python socket rather than the CLI so the
result is not an artifact of the HTTP proxy:

- before the rule, the licence host fails to resolve (`gaierror`) — enforcement
  is at **DNS**;
- after it, a raw TCP connection to the licence port is established;
- an allowed host reached over a plain socket completes a full TLS handshake
  (`pypi.org:443`), confirming raw TCP genuinely flows and is not merely
  proxied HTTP.

**Caveat on the port suffix.** Whether it narrows *below* host granularity could
not be determined. sbx's user-mode network stack accepts connections
optimistically: a port with certainly nothing listening still reports a
successful `connect()`, so connect-based probes cannot distinguish "port
allowed" from "port filtered". Only data flow is a reliable signal, and FLEXlm
returns nothing unprompted. Treat the grant as **host-level** until proven
otherwise, and size the blast radius accordingly: one internal host, not the
whole routable range.

The conclusion drives the design: **an agent editing FINN's Python does not
need Vivado, XRT, or the license.** `quicktest.sh` is already defined as the
non-Vivado, non-slow test subset — that is precisely an agent's inner loop.

---

## 3. Proposed design

### 3.1 Multi-stage build with two publishable targets

```
ubuntu:jammy
  └── system      OS packages, locale, TZ            (changes ~never)
       └── python  requirements.txt, torch, jupyter  (changes with requirements)
            ├── dev   + FINN & deps/ editable installs, NO XRT   ← agent target
            └── full  = dev + XRT + board files                   ← build/CI target
```

`dev` is the agent image: no XRT, no Xilinx mount, no license, no open network.
It runs `quicktest.sh`, pytest's non-Vivado subset, linting, and every
import-level task. `full` is what CI and synthesis use, and is the only image
that ever sees the license server.

Both are user-agnostic and content-addressed by their tag.

### 3.2 Identity moves to runtime

Delete `USERNAME`, `USER_UID`, `GROUP_ID`, `GROUPNAME`, the `useradd`/`groupadd`
lines, and `USER`. The image ships as root; the runtime decides who it is:

- `run-docker.sh` keeps passing `--user $UID:$GID` (already does)
- sbx-based sandbox tooling provisions identity and drops privileges at start

One image, all users, tag means what it says.

### 3.3 A thin, idempotent entrypoint

Everything that can be a layer becomes one. What remains at runtime is only what
genuinely depends on mounted state:

```bash
#!/bin/bash
set -e
export HOME="${HOME:-/tmp/home_dir}"          # respect, don't clobber
export FINN_ROOT="${FINN_ROOT:-$PWD}"          # derive, don't require
source_xilinx_tools                            # depends on the ro mount
exec "$@"
```

Specifically removed:

- the four editable installs → baked into the `dev` stage
- the `qonnx` `pyproject.toml` `mv`/`trap` → fix the root cause in `deps/qonnx`
  or in how it is installed; do not mutate a mounted tree at runtime
- the `finn_xsi` compile → cached under `$FINN_BUILD_DIR`, never written into
  the source tree; built on demand rather than per start
- `.Xilinx/*.tcl` copying → guarded by a sentinel, or handled by the caller

### 3.4 Environment tiering

- **`ENV` in the image:** `TZ`, `LANG`, `LC_ALL`, `LANGUAGE`, `SHELL`,
  `LD_PRELOAD`, `XILINX_LOCAL_USER_DATA=no`, `VIVADO_IP_CACHE`,
  `NUM_DEFAULT_WORKERS`, `LOCALHOST_URL`, `TORCH_HOME`, `HF_HOME`
- **`ARG` at build:** XRT and optional-package inputs, unchanged
- **Runtime only:** the host-binding tier from §2.4, plus ports and tunables
- **`FINN_ROOT` derived from `$PWD`** when unset, so both host-path mirroring
  (sbx) and a fixed `/workspace` (plain docker) work with no configuration

The test of this design: `docker run <dev-image> quicktest.sh` should work with
**zero `-e` flags**.

### 3.5 XRT as the single source of truth for base image and architecture

Make the mismatch in §2.8 unrepresentable: one variable is authored, everything
else is derived from it.

```dockerfile
ARG XRT_DEB_VERSION="xrt_202420.2.18.179_24.04-amd64-xrt"
ARG UBUNTU_CODENAME=noble      # derived from XRT_DEB_VERSION, never hand-set
FROM ubuntu:${UBUNTU_CODENAME}
ARG TARGETARCH                 # supplied automatically by BuildKit
```

`run-docker.sh` parses the package name into `(version, os, arch)`, maps the OS
version to a codename (`20.04→focal`, `22.04→jammy`, `24.04→noble`), and passes
the codename as a build arg. A single assertion inside the image compares the
running OS against the package's target OS and **fails the build** on
disagreement, so a mismatched pair cannot yield a working-looking image.

Architecture comes from BuildKit's automatic `TARGETARCH` rather than a literal,
which makes the package name *computed*. That is what turns this into genuine
multi-arch support instead of an `amd64` string that is currently correct by
coincidence.

Harden the fetch at the same time: pin by **checksum** and move to
`download.amd.com/opendownload/xrt/`.

This lives entirely in the `full` stage. The `dev` stage — the agent image —
selects its Ubuntu directly and never touches an XRT package, so agent builds
inherit none of this coupling and none of its failure modes.

### 3.6 Build-context hygiene

Add `.dockerignore` excluding `deps/`, `.git/`, `notebooks/`, `docs/`,
`tutorials/`, `tests/`, build outputs, and `.venv`. Reorder the Dockerfile so
torch installs early and `requirements.txt` is copied immediately before its own
`pip install`, collapsing the ~30 `RUN` lines into a handful of grouped layers.

### 3.7 Tag stability

Drop `--dirty` from the dev tag. Under an agent workflow the tree is dirty by
definition, so a dirty-tagged image rebuilds on the first edit. Keep the full
`git describe --dirty` for the Jenkins publish path where provenance matters.

### 3.8 Agent sandbox profile

With `dev` in place, a sandbox profile reduces to: mount the repo read-write, no
Xilinx mount, no license variable, network closed except PyPI and GitHub. No
image overlay is needed to carry configuration, because §3.4 put the static
environment in the image — which is also what makes the profile expressible on
sbx, whose create path has no env-injection seam.

A separate opt-in `build` agent uses `full`, mounts only
`$FINN_XILINX_PATH/<version>` read-only, sets `XILINXD_LICENSE_FILE`, and
accepts open posture as a documented, deliberate widening. Run one at a time:
floating licenses are seat-counted, and two agents on one working tree race the
git index.

---

## 4. What was built

All seven migration steps landed together. Three places depart from §3, each
because measurement contradicted an assumption in the proposal.

### 4.1 Three tiers, not two

`dev` → `build` → `build-xrt`, selected by `FINN_DOCKER_TARGET` (default
`build-xrt`, which preserves the Jenkins contract).

§3.1 proposed `dev` + `full`. The split moved because XRT and Vivado are not the
same boundary: HLS synthesis and rtlsim need Vivado but never XRT, so folding
them together would have forced ~1–2 GB of XRT onto every RTL developer.

| Tier | Adds | Xilinx mount | Licence | Egress |
|---|---|---|---|---|
| `dev` | Python dependency closure | none | none | **closed** |
| `build` | finn-hlslib, board files | 1 version, ro | yes | open |
| `build-xrt` | XRT, v80++ | 1 version, ro | yes | open |

`build` and `build-xrt` share a security posture — both need the FLEXlm server
over raw TCP, which sbx's domain-oriented egress policy cannot express — so the only real
containment boundary is `dev` vs. the rest. The `build`/`build-xrt` split is a
resource boundary, not an isolation one.

### 4.2 Dependency resolution: baked wheels, shadowed by the workspace

§3.1 said `dev` bakes "FINN & `deps/` editable installs" while §3.6 said
`.dockerignore` excludes `deps/`. Those cannot both hold. Measuring what the
editable installs actually did settled it:

- **33 s** per container start, producing **668 KB**
- the entire path mechanism was a four-line `easy-install.pth` naming the four
  `src` directories — nothing more
- **three packages fetched from PyPI on every start** (`importlib-metadata`,
  `unfoldNd`, `zipp`), which is a hard blocker for §3.8's closed-egress sandbox,
  not merely a latency cost
- plus the `mv`/`trap` on the mounted qonnx tree

So the image installs qonnx, brevitas and finn-experimental as **ordinary
wheels** at the `deps.env` pins — carrying the dependency closure, metadata and
19 console scripts — and `docker/finn_paths.py` prepends the mounted
`$FINN_ROOT/deps/*/src` ahead of them at interpreter startup. Live editing and
branch switching work with no reinstall, which matters because qonnx and
brevitas are co-developed regularly. `FINN_DEPS=frozen` skips the shadowing.

FINN's own `src` is never baked and is always resolved from the workspace.

Verified equivalent: all modules import, `brevitas.__version__` resolves, 2462
quicktest tests collect, and `tests/brevitas` passes 264 / skips 7 / xfails 1
with zero editable installs.

### 4.3 Build data split out of `deps/`

`deps/` was doing two unrelated jobs: 235 MB of co-developed Python packages,
and 923 MB of HLS headers and board files that are never edited and never
imported — reached through hardcoded `$FINN_ROOT/deps/...` paths in generated
Tcl and `-I` flags.

Those paths are now `FINN_HLSLIB_PATH` and `FINN_BOARD_FILES_PATH`, defaulting to
the historical locations so nothing breaks. The `build` tiers bake their own
copies, so `dev` never fetches them: `./fetch-repos.sh python` is what a dev-tier
launch runs.

### 4.4 Deviations from §3.4 and §3.7

- **`TORCH_HOME` / `HF_HOME` were left out of `ENV`.** §3.4 grouped them with the
  static tier, but they point into `/finn_cache`, which exists only when
  `FINN_DOCKER_CACHE_DIR` is mounted. Baking them makes torch try to create
  `/finn_cache` as a non-root user and fail whenever the cache is absent.
  `run-docker.sh` sets them alongside the mount; without it torch falls back to
  `~/.cache`.
- **`LD_PRELOAD` moved into the entrypoint rather than `ENV`.** It is a FlexLM
  workaround; setting it image-wide makes every process in the dev tier pay for
  it. The entrypoint applies it only when it has found Xilinx tools to source.
- **`--no-install-recommends` was not adopted** in the system stage. The size win
  comes from the tier split, and trimming the base OS is a change to the
  Vivado/XRT runtime that cannot be validated without a full synthesis run.

### 4.5 Two things §3.2 and §3.3 did not anticipate

Removing baked identity and respecting `HOME` both had second-order effects that
only testing surfaced. Both are fixed in the entrypoint.

- **`useradd` was also providing a passwd entry.** §3.2 treats it purely as the
  thing feeding the broken `USER` line, but it additionally gave `getpwuid()`
  something to resolve. Without it, `docker run --user 1234:1234` maps to no
  entry and `getpass.getuser()` raises `uid not found` — which breaks pytest's
  `tmp_path` fixture, and with it every test that uses one. Ten
  `test_loop_rolling` tests failed this way. The entrypoint now exports
  `USER`/`LOGNAME`, which `getpass` consults before the passwd database, so no
  root and no `useradd` is needed.

- **"Respect `HOME`" has to mean "if it is usable".** Docker sets `HOME=/` for a
  `--user` uid with no passwd entry: set, but not writable. A `${HOME:-...}`
  default never fires, and anything falling back to `~/.cache` — torch.hub, HF,
  pip — dies with `PermissionError: '/.cache'`. Thirteen brevitas tests failed
  this way. The check is now writability, not definedness.

Both were caught by diffing against the reference image rather than by reading
the code, which is the argument for keeping that baseline comparison in CI.

### 4.6 XRT endpoint: fall back, do not switch

§2.8 recommends moving to `download.amd.com/opendownload/xrt/`. Measured: that
endpoint **404s for the currently pinned 2022.2-era package**, which is still
served by the older `openDownload` redirect. Switching unconditionally would
have broken `build-xrt` outright, so the build tries the modern endpoint and
falls back. The spoofed browser user-agent is dropped — the endpoint serves a
plain `wget` fine.

The checksum is now pinned by default for the pinned package
(`00f62fbf…ccdbe0`), scoped so that overriding `XRT_DEB_VERSION` without a
matching `FINN_XRT_SHA256` degrades to a warning rather than a spurious failure.

### 4.7 The base stays date-pinned

`run-docker.sh` maps the derived codename to a **snapshot** tag
(`22.04 → jammy → jammy-20230126`), not the rolling `ubuntu:jammy`. Canonical
republishes rolling tags regularly, so building from one would let image contents
drift under an unchanged FINN tag — the same class of unsoundness as the per-user
images §2.1 describes, only slower moving. The default pin is byte-identical to
what the previous Dockerfile hardcoded.

### 4.8 24.04 is blocked on the Python upgrade, not on the container

**jammy is not a lag — it is how FINN gets Python 3.10**, which is what its pin
set targets. So "move the container to 24.04" is really "move FINN to Python
3.12", and that is a dependency migration, not a packaging change.

The OS-level blockers have been removed, so the container is no longer the
constraint:

- **ncurses 5 ABI.** Vivado links against it; noble dropped it. The system stage
  now asks apt what it has and backports the jammy `libtinfo5` and `libncurses5`
  packages when it has nothing — the same workaround used to run Vivado on a
  noble host, where it is empirically validated. Checksummed, taken from the
  archive pool rather than a PPA, ~200 KB. On focal and jammy this is inert:
  apt has the packages and the distro path is taken.
- **PEP 668.** noble marks the system Python externally managed, so a
  system-wide `pip install` refuses. `PIP_BREAK_SYSTEM_PACKAGES=1` is set: in a
  single-purpose disposable container the Python environment *is* the
  deliverable, so there is no system to protect. jammy has no
  `EXTERNALLY-MANAGED` marker and its pip 22.0.2 accepts the variable as a
  no-op, so this too is inert on the validated base.

**What remains is purely the Python 3.10 → 3.12 pin migration.** Measured
against `requirements.txt` on Python 3.12, only four pins lack cp312 wheels, and
two of those are pure-Python sdists:

| Pin | Status on 3.12 |
|---|---|
| `numpy==1.24.1` | no cp312 wheel; source build fails (predates 3.12, uses removed `distutils`) |
| `scipy==1.10.1` | no cp312 wheel |
| `vcdvcd==1.0.5` | sdist only, pure Python |
| `wget==3.2` | sdist only, pure Python |

The larger risk is the test stack rather than the numerics: `pytest==6.2.5` and
its pinned plugins **install** on 3.12 but **do not run**. So the migration is
gated on modernising pytest and its plugin set, which is exactly the work §2.6's
`--ignore-installed` tangle and the anyio conflict already point at.

When that upgrade happens, the container side needs no work: set
`XRT_DEB_VERSION` to a 24.04 package and the base follows automatically.

Note this was never a regression. Previously the base stayed hardcoded at jammy,
so a Noble `.deb` was force-installed on Jammy; apt resolves that rather than
refusing, producing a subtly wrong runtime that looks fine. Nobody was getting a
correct 24.04 image — the difference is that the failure is now loud.

The `XRT_DEB_VERSION` → codename derivation works — a `24.04` package correctly
selects a noble base, so the §2.8 mismatch is genuinely unrepresentable now. But
the apt list is jammy-only: noble ships `libtinfo6` and has no `libtinfo5`, so
the build fails there. Not swapped blindly, because older Vivado links against
`libtinfo5` specifically and establishing which releases still need it requires a
synthesis run. Supporting 24.04 means making the package list codename-conditional
and validating Vivado against it.

### 4.9 Also fixed, because the workflow needed it

`fetch_repo` hard-checked-out the pinned SHA on every `run-docker.sh`, so working
on a qonnx branch meant remembering `FINN_SKIP_DEP_REPOS=1` or being silently
yanked back. It now:

- reads pins from `deps.env`, which the Dockerfile also reads, so the host
  checkout and the baked wheels cannot disagree
- resolves refs with `rev-parse <ref>^{commit}`, so **branches and tags work**,
  not just raw SHAs
- **refuses to move a dependency whose tree is dirty**

Build context: **1.3 GB → 64 KB**.

### 4.10 Measured outcome

| | Before | After |
|---|---|---|
| Build context | 1.3 GB | 64 KB |
| Agent image | 26.4 GB (single tier) | 8.57 GB (`dev`) |
| Container start | 33 s of `pip install` | 0.67 s |
| Network at start | required (3 PyPI fetches) | none |
| Workspace mutation | qonnx `mv`/`trap`, `xsi.so` | none |
| Runs as arbitrary uid | fails | works |

Verification against the reference image, all in the `dev` tier:

- `tests/util` + `tests/transformation` + `tests/brevitas`, non-Vivado markers:
  **1970 passed, 16 skipped, 5 xfailed, 1 xpassed, 0 failed**
- quicktest collection: **2462 tests**, matching the reference exactly
- `docker run <dev>` with **zero `-e` flags**
- negative tests: a wrong `FINN_XRT_SHA256` fails the build; a `24.04` package
  correctly selects a noble base rather than silently installing on jammy

Two build gates were added so the failure modes found here cannot return
silently: `pip check`, and a real pytest run. The second is not redundant — the
anyio/pytest breakage has no metadata conflict, so `pip check` passes while every
pytest invocation dies at plugin load.

## 5. The `FINN_ROOT` limitation, and its extension point

FINN has no fixed workspace path. `run-docker.sh` mirrors the host path
(`-v $SCRIPTPATH:$SCRIPTPATH`), and sbx cannot remap a mount at all — it attaches
every mount at its host path by design. So the workspace lives at a different
absolute path for every developer, and that path is only knowable at run time.

This is a **known structural flaw in the repo, accepted rather than fixed here.**
It shapes the design, so it is worth stating precisely what it forbids.

**What it forbids: baked editable installs.** An editable install — from pip or
uv — records an *absolute* path, either as a `.pth` line or as a MAPPING inside a
PEP 660 `__editable___*_finder.py`. The current image's own `easy-install.pth`
shows it plainly:

```
/home/tkeller/prj-clara/finn/deps/qonnx/src
```

One developer's home directory, in an image shared by everyone — the same
unsoundness as the per-user images §2.1 describes. Standardising on `/workspace`
does not help, because sbx offers no container-side remapping. So the only two
options that work are computing the path at import time from an environment
variable, or running the install at container start — and the latter is the
33 s, needs-network, mutates-the-workspace behaviour this redesign removed.

Hence `docker/finn_paths.py`: path resolution at import time, from `FINN_ROOT`.

**It is confined to five sites**, all tagged `LIMITATION(finn-root-absolute)` and
greppable as one set:

| Site | Role |
|---|---|
| `docker/finn_paths.py` | `workspace_root()` — the seam |
| `docker/finn_entrypoint.sh` | derives `FINN_ROOT` from `$PWD` |
| `run-docker.sh` | the host-path-mirroring mount |
| `src/finn/util/basic.py` | `FINN_HLSLIB_PATH` / `FINN_BOARD_FILES_PATH` defaults |
| `src/finn/xsi/paths.py` | finn_xsi source location |

**The extension point.** If the workspace ever gets a fixed path, this design
does not need reworking — it needs deleting:

1. Bake `pip install -e <fixed>/deps/<dep>` into the dev stage, alongside the
   existing wheels.
2. Delete `docker/finn_paths.py` and `docker/finn-live.pth`, and drop the COPY
   and install of both.
3. Keep `FINN_DEPS` as the live/frozen switch, selecting between the editable
   install and the wheel rather than by prepending `sys.path`.

That also removes this approach's one wart: in live mode the *code* comes from
the workspace while the *declared version* comes from the baked wheel, so
`importlib.metadata` can report a commit that is not the one running. Nothing in
FINN reads that metadata today, which is why the trade is currently worth making.

Note that `deps.env` remains the single place a Python dependency is declared
either way: a build-time assertion fails if `finn_paths.DEP_SRC_DIRS` and
`deps.env`'s `FINN_DEP_GROUP_PYTHON` disagree, so adding a dep to one and not the
other cannot become a silent unresolvable import.

## 6. Open questions, answered

- **Does anything require the `qonnx` `pyproject.toml` rename?** No. `pip wheel`
  builds qonnx with `pyproject.toml` in place; the failure was specific to
  `pip install -e`. Since nothing pip-installs the mounted tree any more, the
  `mv`/`trap` is deleted rather than fixed.

- **Should `dev` carry XRT?** No, and the premise was wrong. XRT appears in four
  non-test source lines, all Vitis/Alveo-gated. `pyxsi`/rtlsim needs
  `XILINX_VIVADO`, `vivado` on `PATH` and the xsim headers — never XRT. Excluding
  it costs zero runnable coverage, because every XRT consumer also needs Vitis,
  platform repos and a licence.

- **Is `--init` still needed when the sandbox runtime supplies its own PID 1?**
  Answered for sbx: the image now runs `tini` as PID 1, which is what stock sbx
  templates do and what `--init` provides, so the reaping the Vivado hang needs
  is present without the flag. `run-docker.sh` keeps passing `--init` on the
  docker path, where the flag is free and the behaviour is long-established.

## 7. The sbx template contract

Making the image *sbx-shaped* (§3.4's environment tiering) was necessary but not
sufficient. A stock `sbx create` against it failed with nothing but
`ERROR: failed to run sandbox container`. The requirements below were each
established by bisecting a real failure against a stock template.

| Requirement | Symptom when absent |
|---|---|
| `agent` uid 1000, NOPASSWD sudo | kits run as `agent` and cannot write |
| `/home/agent` **755**, with `.claude/ .local/bin .npm .cache workspace/` agent-owned | `cannot create .../settings.json: Permission denied` |
| `tini` as PID 1 | no zombie reaping |
| `CMD` that does not exit | container dies before setup; sbx reports only the opaque message above |
| `NO_PROXY` / `no_proxy` | loopback pushed through the egress proxy |
| `NPM_CONFIG_PREFIX` **and its bin dir on `PATH`** | `npm install -g` lands off `PATH`; a CLI installs "successfully" then `command -v` fails with 127 |

The 755-plus-populated-home requirement is in no documentation; it came from
diffing a stock template after the permission failure.

**Environment is delivered by a kit, not the entrypoint.** `docker/finn.kit`
sets `FINN_ROOT: "${WORKSPACE_DIR}"` (sbx interpolates it at create) and
`FINN_BUILD_DIR`. `environment.variables` becomes *real process env*, so unlike
`/etc/sandbox-persistent.sh` it reaches every invocation style:

| Mechanism | `bash -c` | `sh -c` | bare exec |
|---|---|---|---|
| kit `environment.variables` | yes | yes | yes |
| `/etc/sandbox-persistent.sh` | yes | no | no |

`/etc/sandbox-persistent.sh` is **sbx-managed** — the image must not write it.
An earlier revision had the entrypoint do so and it was dead code: the file
reads 0 bytes in a live sandbox because sbx replaces it after the entrypoint
runs. Anything needing to be computed at start belongs in the kit's
`commands.startup`, which may write that file.

Kit schema versions, verified against sbx v0.39.0: `schemaVersion: "1"` uses
`commands:`, `schemaVersion: "2"` uses `setup:`. v2 is available from sbx 0.36
and is what `docker/finn.kit/spec.yaml` uses. The two grammars must not be
mixed — v1 fields inside a v2 spec are rejected at decode.

An earlier revision of this document claimed v0.39.0 *rejects* `setup:` and
requires `commands:`. That was wrong: it was a v1 spec being read by a v2 parser,
and the diagnosis stopped at the error message. The checked-in kit uses `setup:`
and works. Validate against the binary — but validate the whole hypothesis, not
just the first failure.

**sbx substitutes its own CMD.** PID 1 in a sandbox is
`tini -- finn_entrypoint.sh sh -c 'trap ...; sleep infinity & wait'`, so the
image's `CMD` is irrelevant there and stays `["bash"]` for docker's interactive
default. What kills a sandbox is the ENTRYPOINT *exiting*, not the CMD.

**`sbx exec` does not run the ENTRYPOINT.** This is the subtle one. Everything
`finn_entrypoint.sh` exports is absent from exec sessions, so path resolution
cannot live in the shell. `finn_paths.workspace_root()` therefore falls back
`FINN_ROOT` → `WORKSPACE_DIR` (set by sbx) → a cwd that actually contains
`src/finn`, and `FINN_BUILD_DIR` is defaulted the same way.

The entrypoint does **not** write `/etc/sandbox-persistent.sh` — see above, that
file is sbx-managed and the image must not touch it. The kit's startup command
writes `/etc/finn-toolchain.sh` and appends a single guarded source line to it
instead.

**Why not extend `docker/sandbox-templates`.** That is Docker's supported route
and it is closed to FINN: the images are Ubuntu 26.04 on Python 3.14, and
`torch 2.8.0` publishes no `cp314` wheel. The build tiers are pinned to an LTS
by XRT and Vivado regardless. If XRT ever ships a 26.04 package, a new profile
can track it and this section can be reconsidered.

**This is reverse-engineered and an sbx release can break it.** The check that
matters is creating a sandbox from the image with plain `sbx` — no wrapper —
and running a test inside it.

**Licensing: both forms are supported.** `XILINXD_LICENSE_FILE` (and the older
`LM_LICENSE_FILE`) take a colon-separated list whose entries are either
`PORT@HOST` or a path to a `.lic`. `run-docker.sh` passes the variable through
verbatim and mounts read-only, at its own host path, the *directory* containing
each path-form entry — the directory rather than the file, because Vivado writes
sibling lock/state files next to some licences. Mounting at the host path means
the variable's value never needs rewriting.

On sbx the same split applies, with the mount expressed positionally:

```
sbx create claude <repo> <xilinx-version-dir>:ro <licence-dir>:ro \
    --kit docker/finn.kit -e XILINXD_LICENSE_FILE=/path/to.lic
```

Only the floating form needs egress, and it does **not** force open posture — a
per-sandbox `sbx policy allow network` rule for the licence host is enough (see
above). A node-locked licence still needs no network reach at all, so it remains
the tighter option where it is available, but both forms are supported and
neither requires giving up a closed posture.

**Egress is wider than the manifest asks for.** sbx's per-sandbox rules are
*additive to a machine-level preset*, so a "closed" posture is not closed.
Measured on a live sandbox: `pypi.org` (requested) 200, `example.com`
(unrequested) blocked by policy — but `api.openai.com`, which no profile
requested, also returns 401 rather than being blocked, because the machine
preset grants ~195 hosts to every sandbox. Tightening that is a host-level
decision (`sbx policy`), not something a profile can express.
