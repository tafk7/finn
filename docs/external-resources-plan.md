# Implementation plan: external resource management

Written 2026-09-26. Scope: how FINN **declares, downloads, verifies, updates and
supplies** non-Python resources (hlslib, Vivado board files, users' RTL/HLS
libraries and boards). How the builder *uses* them (op and board registration) is
out of scope; the builder branch owns that. This work keeps today's two call
sites (HLS include path, Zynq board path) working through the new API.

## Goals

1. One mechanism for every external resource: FINN's own, users', and those from
   installed packages.
2. Works for `pip install finn` users: the pins ship in the wheel, and resources
   are fetched on first use.
3. Reproducible and safe: pinned sources, content digests, atomic, locked,
   digest-keyed caches.
4. Straightforward to update (one command) and to override locally (one
   environment variable).
5. Same behaviour natively, in containers, in CI and offline.
6. The resolver is stdlib-only, so the image build can run it before FINN is
   installed.

## Design summary

### Terms

* **External resource:** a named directory tree FINN doesn't contain, delivered
  from a pinned source and verified by a content digest.
* **Kind:** a free-form tag such as `hls-include` or `vivado-boards`. Consumers ask
  for resources by kind, so they never need to know names.

### Declaration format (TOML)

```toml
[external.hlslib]
description = "finn-hlslib HLS C++ library"
kind = ["hls-include"]
redistributable = true
git = "https://github.com/Xilinx/finn-hlslib.git"
commit = "8d979e2bdced486dd25d26607d1ff5ae327ed6a8"
digest = "sha256:<tree digest>"

[external.xilinx-rfsoc2x2-boards]
kind = ["vivado-boards"]
redistributable = false
git = "https://github.com/Xilinx/XilinxBoardStore.git"
commit = "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"
subdir = "boards/Xilinx/rfsoc2x2"   # sparse: only this path is fetched
into = "rfsoc2x2"                   # placed here inside the resource root
digest = "sha256:<tree digest>"

[external.acme-rtl]                 # archive source
url = "https://example.com/acme-rtl-1.2.tar.gz"
sha256 = "<archive checksum>"
subdir = "rtl"
digest = "sha256:<tree digest>"
kind = ["rtl"]
```

**Fields:**

| Field | Meaning |
|---|---|
| exactly one of `git`+`commit`, `url`+`sha256`, `package` | The source |
| `subdir` | Sparse path within the source |
| `into` | Placement inside the resource root |
| `digest` | Tree digest of the delivered root |
| `kind` | List of tags |
| `redistributable` | Whether FINN may bake it into published images |
| `mirrors` | Alternative URLs (optional) |
| `description` | Free text |

**`package` source:** `package = "module.name:subdir"`, for data already shipped
in a Python distribution. It's resolved with `importlib.resources`, with no
fetching.

The **tree digest** is today's algorithm: sha256 over sorted relative paths and
file hashes, excluding the marker file.

### Where declarations come from (merged in this order)

1. **FINN's own file:** `src/finn/_data/external.toml`, shipped in the wheel.
   Contains hlslib and the board repositories.
2. **Installed packages:** each advertises a declarations file through the
   `finn.external` entry-point group, whose value is a module containing
   `external.toml`. These may add names, but may not redefine FINN's; doing so
   is an error.
3. **The project:** `[tool.finn.external]` in the nearest `pyproject.toml`, found
   by walking up from the working directory, or files listed in
   `FINN_EXTERNAL_FILES`. The project may redefine any name, for example to try
   a newer hlslib commit; this is logged.

### Supplying (Python API, `finn.external`)

* `path(name)` returns the directory, fetching it if needed.
* `paths(kind=...)` returns every resource of a kind, in declaration order.
* `declarations()` returns the merged declarations; `status(name)` reports
  whether a resource is cached, overridden or missing, and where.
* **Override:** `FINN_EXTERNAL_<NAME>=/dir` (the name upper-cased, `-` becomes
  `_`) uses a local directory with no digest check. It's the equivalent of a uv
  path source, for co-development.
* **Compatibility:**
  * `FINN_HLSLIB_PATH` is an alias for the hlslib override.
  * `FINN_BOARD_FILES_PATH`, if set, replaces the whole `vivado-boards` set, as it
    does today. Deprecated.

### Caches (searched in order; the first complete copy wins)

1. `FINN_EXTERNAL_CACHE`: a writable cache for site-managed or offline use.
2. The system cache, `FINN_EXTERNAL_SYSTEM_CACHE` (default `/opt/finn/external`
   if it exists): read-only, filled at image build.
3. The user cache: `${XDG_CACHE_HOME:-~/.cache}/finn/external`.

Each entry is stored as `<name>-<digest16>/`, with a marker file recording the
full digest. Fetches go into the first writable root, keeping today's file lock,
temporary directory, verification and atomic rename. A new pin means a new
directory, so staleness can't go unnoticed.

### Fetchers (stdlib only)

* **git:** today's sparse single-commit fetch (`--depth 1 --filter=blob:none`,
  sparse checkout of `subdir`), followed by the copy into `into`.
* **archive:** `urllib` download (proxy settings honoured), sha256 check, and safe
  extraction (`tarfile` data filter, zip path checks).
* **git-less fallback (phase 6):** for github.com sources, if `git` is missing,
  download the commit's tarball archive and apply `subdir`.
* **mirrors:** tried in order after the primary source;
  `FINN_EXTERNAL_<NAME>_URL` overrides the source URL for internal mirrors.

### Command line: `finn-external` (also `python -m finn.external`)

| Command | Does |
|---|---|
| `list [--kind K] [--boards]` | Declarations, source, cache or override state; `--boards` lists board names by parsing `board.xml` (inspired by Brainsmith) |
| `path NAME`, `paths --kind K` | Print locations, fetching if needed |
| `fetch [NAME…\|--kind K\|--all] [--dest DIR] [--redistributable-only]` | Pre-fetch for images, CI and offline sites |
| `verify [NAME…]` | Re-check the digests of cached copies |
| `update NAME [--ref REF] [--file F]` | Resolve the ref to a commit, fetch, recompute the digest, and rewrite `commit`/`digest` in the declaring file |
| `digest DIR` | Print a tree digest |
| `clean [--unused]` | Remove cache entries whose digest no current declaration references (inspired by Brainsmith's `remove`) |
| `check` | Missing tools (`git`), unreachable caches, and override paths that don't exist |

`update` edits only the `commit` and `digest` lines inside that resource's table,
then re-parses the file to confirm the result; the stdlib can read TOML but not
write it. It refuses packaged declarations and prints the new values instead.

## hlslib and board files under the new model

* **hlslib** becomes the `hlslib` resource (`kind = ["hls-include"]`,
  `redistributable = true`). This removes:
  * the submodule and `packages/finn-hlslib/`
  * `.gitmodules`
  * the uv workspace
  * the `finn[hw]` extra and the dev-group entry
  * the finn-hlslib wheel build (Package workflow, release stage)
  * the entrypoint's `--no-build-isolation-package finn-hlslib`
  * the Dockerfile's `--no-install-workspace`, which goes back to
    `--no-install-project`
  * submodule initialization in CI and `setup-local.sh`

  Co-development: `FINN_EXTERNAL_HLSLIB=../finn-hlslib`.
* **Board files** become **five resources, one per repository** (Avnet,
  Xilinx-rfsoc2x2, RFSoC4x2, KV260, AUP-ZU3), each `kind = ["vivado-boards"]`,
  `redistributable = false`, placed exactly as today (the Avnet repository root,
  the others in their named subdirectories).
  * **Check:** the union of the five trees equals today's assembled tree
    (258 files).
  * **One unverified assumption:** each resource root now becomes a separate
    Vivado board path instead of one directory holding them all. The board
    directories sit at the same depth as today, so Vivado should find them, but
    that needs a real Vivado run.
* **The two call sites:**
  * `hlsbackend.py`: `hlslib_path()` becomes `external.path("hlslib")`.
  * `make_zynq_proj.py`: the single `$BOARD_FILES$` path becomes every
    `paths(kind="vivado-boards")` entry, Tcl-quoted, since `lappend` accepts
    several arguments.

  Tests using `hlslib_path` are updated. `finn/util/external.py` is removed.

## Images, CI and native

* **Images:** a new `dev` stage after `runtime` separates what the images may
  contain:

  ```text
  runtime ──► dev ──► sbx
     └──► release
  ```

  * `python` stage: `fetch --redistributable-only` into the system cache
    (hlslib).
  * `dev` stage (the default target): `fetch --kind vivado-boards` into the
    system cache. Local-only images may contain third-party board files.
  * `release`: from `runtime`, so it carries only redistributable resources;
    everything else is fetched on first use.
  * The build runs `PYTHONPATH=/src python -m finn.external …` from a bind mount
    of `src/`. `finn` is a namespace package and `finn.external` imports only the
    stdlib (a test enforces this).
  * `ENV FINN_BOARD_FILES_PATH` goes; `FINN_EXTERNAL_SYSTEM_CACHE` is added.
  * Image inputs: `src/finn/external/**` and `src/finn/_data/external.toml`.
* **CI:**
  * The Package workflow checks that a wheel-only install knows the declarations
    (`finn-external list`) and can fetch hlslib.
  * Workflow submodule checkouts and the Jenkins `git submodule update` step are
    removed.
  * Native CI caches `~/.cache/finn/external`, keyed by `external.toml`.
* **Native:** `setup-local.sh` drops the submodule step. With Vivado present, it
  runs `finn-external fetch --kind vivado-boards --kind hls-include` in place of
  today's `fetch-boards`.

## Branch

This is an exploration, so it happens on its own branch and
`refactor/container-runtime-implementation` stays untouched until the approach
is accepted.

* **Create:** `feature/external-resources`, branched from
  `refactor/container-runtime-implementation` at its current head (`09fb86df7`,
  which includes the upstream `dev` merge). Its first commit is this plan, moved
  to `docs/external-resources-plan.md`.
* **Work:** one or more commits per phase below, each leaving the tests passing,
  so the exploration can stop or be abandoned at any phase boundary.
* **Keeping up:** if the base branch moves (for example another upstream merge),
  merge it into the feature branch rather than rebasing, so the per-phase
  history stays reviewable.
* **Outcome:**
  * *Accepted:* merge into the base branch (or fold into the planned commit
    reorganization).
  * *Abandoned:* the branch remains as a record, and the base branch is
    unaffected.
* Nothing is pushed or merged without explicit approval, as for the rest of this
  work.

## Phases (one or more commits each; tests pass after each)

**0. Branch:** create `feature/external-resources` and commit this plan (see
above).

**1. Core module** (`src/finn/external/`, stdlib only): declaration parsing and
validation, merging (FINN's file, then the project file for now), the digest,
cache roots and lookup, git and archive fetchers, overrides, and the Python API.
FINN's `external.toml` gets the six declarations with computed digests.
Nothing is switched over yet.

*Tests (no network):*
* local git repos and archives (`file://`)
* sparse `subdir` and `into`
* digest mismatch leaves nothing published
* cache order, including a read-only system cache
* override and compatibility variables
* parallel first use fetches once
* project redefinition is logged; conflicting definitions are rejected
* a subprocess check that `finn.external` imports only the stdlib

*One networked check (recorded):* the five board resources reproduce today's
assembled tree.

**2. Switch over and remove the hlslib package.** Update the two call sites and
their tests, remove `finn/util/external.py`, then remove the submodule,
`packages/`, the workspace, the extra and every flag listed above. Relock (only
`finn-hlslib` disappears).

**3. Command line:** `list/path/paths/fetch/verify/update/digest/clean/check`,
`--boards`, the console script in `[project.scripts]`, and tests, including
`update` against a local repo with a new commit.

**4. Entry-point declarations** (the `finn.external` group) with their
redefinition rules, tested with a sample installed package in a temporary venv.

**5. Images, CI, native setup:** the `dev` stage and pre-fetch policy, image
inputs, workflow and Jenkins changes, and `setup-local.sh`.

*Validation:*
* image build (24.04)
* conformance suite
* runtime tests in the container
* `--network none` container using the system cache
* wheel-only install fetching hlslib
* offline: `fetch --dest`, then `FINN_EXTERNAL_CACHE`

**6. Robustness:** the git-less GitHub archive fallback, `mirrors`, and
`FINN_EXTERNAL_<NAME>_URL`.

**7. Docs:**
* `docs/installation.md`: external resources, offline use, overrides, adding
  your own (a worked example: custom boards and an RTL library declared in a
  project's `pyproject.toml`).
* `docs/environment.md`: replace the `packages/` and workspace material.
* `docker/README.md`, `developers.rst`, the validation record, and the status
  file.

## Risks and checks

| Risk | Handling |
|---|---|
| Vivado doesn't find boards across several board paths | Same directory depth as today; confirm in a real Vivado run (P8). Fallback: supply one directory of links to the resource roots |
| An upstream commit disappears or the repo moves | Mirrors (phase 6); digests prevent silent substitution |
| `update` corrupts a hand-edited TOML | Edits limited to two lines of one table, then re-parsed; the old file is kept if the result doesn't parse or differs anywhere else |
| Name clashes between packages | Error, naming both sources; only the project may redefine |
| Hidden network access during builds | Fetches log once to stderr; `FINN_EXTERNAL_OFFLINE=1` turns a missing resource into an error with the `fetch` command to run |
| The standalone import requirement erodes | A test imports `finn.external` in a clean interpreter and asserts no non-stdlib modules were loaded |

## Decisions to confirm before phase 1

1. **Naming:**
   * module `finn.external`
   * command `finn-external`
   * FINN's file `external.toml`
   * project table `[tool.finn.external]`
   * variables `FINN_EXTERNAL_*`

   The alternative is the word "resources", which already names `finn._data`
   files (`resource_path`) and would be ambiguous.
2. **Project discovery:** walk up from the working directory to the nearest
   `pyproject.toml` (the uv/ruff convention), or only explicit
   `FINN_EXTERNAL_FILES`?
3. **Deprecation:** keep `FINN_BOARD_FILES_PATH` with its all-or-nothing meaning
   for one release, then remove it?
4. **Git-less fallback:** now (phase 6), or wait until someone needs it?

## Records

Decisions confirmed before phase 1 (2026-09-26):

1. **Naming:** "resources", not "external": module `finn.resources`, command
   `finn-resources`, FINN's file `src/finn/_data/resources.toml` (table
   `[resources.NAME]`), project table `[tool.finn.resources]`, variables
   `FINN_RESOURCES_*` (override `FINN_RESOURCES_<NAME>`), entry-point group
   `finn.resources`. `finn.util.resources` (`resource_path`, FINN's own package
   data) is unrelated and keeps its name. Names whose override variable would be
   a setting (`cache`, `system-cache`, `files`, `offline`, anything ending in
   `-url`) are rejected.
2. **Project discovery:** the nearest `pyproject.toml`, walking up from the
   working directory, then `FINN_RESOURCES_FILES`. Found once per process.
3. **`FINN_BOARD_FILES_PATH`:** removed now; if set, FINN warns that it is no
   longer used.
4. **Git-less fallback:** phase 6, as planned.

Networked check (phase 1, 2026-09-26): the six sources fetched from GitHub.

| Resource | Files | Digest |
|---|---|---|
| hlslib | 187 | `sha256:eaeda81f…c989` |
| avnet-boards | 227 | `sha256:69673676…f215` |
| rfsoc2x2-boards | 6 | `sha256:c6329368…0aff` |
| rfsoc4x2-boards | 6 | `sha256:892a9389…e42e1b` |
| kv260-som-boards | 14 | `sha256:366c8adc…27fc` |
| aup-zu3-boards | 5 | `sha256:a14a0213…797d` |

The five board trees copied into one directory hold 258 files with digest
`89ebe8049a4f…6960e`, identical to `BOARD_FILES_DIGEST` of the previous
assembled tree, under both the previous and the new digest implementation.
