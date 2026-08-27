# Dependency profiles

A profile is one coherent `(Ubuntu release, Python version, pin set, XRT package)`
combination. Selecting one selects the whole stack:

```bash
FINN_PROFILE=py310 ./run-docker.sh quicktest   # default
FINN_PROFILE=py312 ./run-docker.sh quicktest
```

| Profile | Ubuntu | Python | Status |
|---|---|---|---|
| `py310` | 22.04 jammy | 3.10 | Current. Validated: 1970 tests pass. |
| `py312` | 24.04 noble | 3.12 | Tracks upstream PR 1603 (`upgrade/python_version`), which is WIP. **Not yet validated here.** |

This exists because 24.04 / Python 3.12 is a near-term requirement but the
upstream work landing it is unfinished. Rather than fork or wait, both stacks
are carried side by side so switching is one variable rather than a migration.

## Why the Ubuntu release is not in the profile

It is derived from the profile's XRT package by `run-docker.sh`, because the
package name already encodes the OS it targets:

```
xrt_202420.2.18.179_24.04-amd64-xrt
    └ version ──┘ └ OS ┘ └arch┘
```

Pinning both independently is what let a Noble package be installed on a Jammy
base — apt resolves that rather than refusing, yielding a subtly wrong runtime.
The build re-checks the derivation from inside the image and fails on
disagreement. So a profile sets the XRT package and the OS follows; do not set
`UBUNTU_TAG` by hand.

## Files

| File | Purpose |
|---|---|
| `profile.env` | XRT package + checksum, base requirements path, description |
| `requirements.txt` | Base FINN pins. Optional — `py310` points at the repo-root file instead, so the profile in active use has no duplicate to drift from. |
| `tools.txt` | Interaction and test stack, plus the dep repos' third-party requirements |
| `tools-force.txt` | Installed with `--ignore-installed`, for packages that fight distro-provided copies. Which ones those are differs by release. |
| `constraints.txt` | Bounds every install in the profile must respect |

## Why `constraints.txt` is load-bearing

Not defensive — it prevents two failures that both produce a working-*looking*
image:

- **numpy.** Installs using `--ignore-installed` re-resolve their transitive
  closure from scratch and will pull a major version the existing pins cannot
  work with. Without the bound, the jupyter step upgrades numpy to 2.x, silently
  breaking scipy and onnxruntime. pip reports that as a warning and exits 0.
- **anyio.** jupyter pulls `anyio >= 4.13`, whose pytest plugin imports
  `_pytest.scope` — added in pytest 7.0.0. Under `py310`'s pytest 6.2.5 that
  makes *every* pytest invocation die at plugin load. `pip check` does not catch
  it: there is no metadata conflict, only a runtime one. The Dockerfile therefore
  runs a real pytest as a build gate.

## Adding or changing pins

The dep repos' third-party requirements (`deap`, `mip`, `networkx`,
`future-annotations`, `dependencies`, `tokenize-rt`) are pinned here rather than
left to the dep wheels' unpinned `install_requires`, which resolve to latest.
Dropping them once silently moved `mip` 1.13→2.0 and `networkx` 2.8→3.4.

Dependency *repo* commits are not part of a profile — they live in `deps.env`
and are currently identical across both profiles.

## Validating a profile

```bash
FINN_PROFILE=py312 ./run-docker.sh build dev
FINN_PROFILE=py312 ./run-docker.sh quicktest
```

The build itself gates on `pip check`, a real pytest run, and agreement between
`finn_paths.DEP_SRC_DIRS` and `deps.env`. Those catch environment breakage but
not behavioural regressions, so compare a test run against the other profile.
