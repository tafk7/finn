# FINN Test Guidelines

Help keep FINN's testing suite fast, deterministic, and parallelisable when writing or modifying tests.

The trees the code gates run (the Space engine, dataflow, kernel and KernelOp tests) differ from the rest in their conftest, seeding and markers: see [5. Trees and gates](#5-trees-and-gates) and [6. `tests/support`](#6-testssupport-the-test-side-library).

### 1. Scratch paths

Never write scratch files into the repo root or CWD. Collisions are guaranteed when tests run in parallel. Instead:

- Use pytest's `tmp_path` fixture (or `tmp_path_factory` for a fixture that several tests share): a unique directory per test, which pytest manages.
- Never allocate a build directory a test leaves behind (`make_build_dir()`, `finn.resources.scratch()`, `FINN_BUILD_DIR`): nothing removes it, and runs accumulate them.
- If the test's outputs are useful for diagnosis after a failure, pytest keeps the most recent `tmp_path` directories (`--basetemp` names another root).

```python
def test_my_feature(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
```

### 2. Process state & environment

Don't mutate env variables or global process state. Use pytest's `monkeypatch` fixture to override env variables or system attributes.

```python
def test_sim_behaviour(monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", "/custom/path")
    # original FINN_BUILD_DIR restored automatically after the test
```

### 3. Markers

`.pytest.ini` declares the markers, and the gates run with `--strict-markers`: an undeclared marker is an error. The gates act on these:

- `xsim` and `vivado`: deselected by every `check-kernels.sh` step but the Space, dataflow and XSim-sweep-tooling ones (`scripts/xsim-sweep.sh` runs them).
- `vitis`: deselected in `tests/kernels` only; an HLS top's C simulation is a check to run by hand.
- `slow`: deselected by a gate's `--fast`, and in `tests/util`, `tests/toolchain` and `tests/docker_scripts` in both modes, so no gate runs the `slow` tests there.
- `container`: no gate acts on it; `tests/container` is run by hand.

The upstream category markers (`util`, `transform`, `streamline`, `brevitas_export`, ...) are applied inconsistently, and no gate selects by them: select tests by tree (`pytest tests/util`), not by those markers.

### 4. Randomness

Every test under `tests/conftest.py` gets a stable `finn_test_seed` through it (not the trees the code gates run with their own `--confcutdir`: [5. Trees and gates](#5-trees-and-gates)). An autouse fixture seeds Python `random`, `numpy.random`, and the PyTorch CPU generator when installed.

For example, the below test receives identical weights and input on every run.

```python
def test_my_op():
    weights = np.random.rand(64, 64).astype(np.float32)
    inp = np.random.randn(1, 64).astype(np.float32)
```

Seed independent generators from `finn_test_seed`:

```python
def test_my_op(finn_test_seed):
    rng = np.random.default_rng(finn_test_seed)
    weights = rng.random((64, 64), dtype=np.float32)
```

- Some tests will fail at the given seed. This should be treated as a deficiency in the test.
- If coverage across a range of random inputs is required, build it into the test as a parametrisation.

### 5. Trees and gates

The code gates (`scripts/check-*.sh`) run each tree with `--confcutdir` (`gate_pytest` in `scripts/_gate-common.sh`): no conftest above it loads, and `tests/conftest.py` (the seed of §4, `tests/support/rng_seed.py`) only where the root is `tests`. `check-kernels.sh` runs `check-space.sh` and `check-dataflow-design.sh` first.

| Tree | Tests | Gate | `--confcutdir` |
|---|---|---|---|
| `tests/core/reporting` | `finn.core.reporting` | `check-space.sh` | `tests/core/reporting` |
| `tests/core/space` | `finn.core.space`, the engine | `check-space.sh` | `tests/core/space` |
| `tests/dataflow` | `finn.dataflow` | `check-dataflow-design.sh` | `tests/dataflow` |
| `tests/kernels` | `finn.kernels`; `sweeps/` holds the XSim sweep's job modules (`python -m kernels.sweeps.<module>`) and their transport | `check-kernels.sh` | `tests/kernels` |
| `tests/kernel_ops` | the KernelOps, conversion, the partition cut, `finn.transformation.kernels`, the graph annotations | `check-kernels.sh` | `tests/kernel_ops` |
| `tests/builder` | `finn.builder`: the steps, the plan, the manifest, the resources report | `check-kernels.sh` | `tests/builder` |
| `tests/platforms` | `finn.platform` | `check-kernels.sh` | `tests/platforms` |
| `tests/analysis` | `finn.analysis` | `check-kernels.sh` | `tests/analysis` |
| `tests/shells` | `finn.shells`: the pynq runner and driver | `check-kernels.sh` | `tests/shells` |
| `tests/networks` | whole networks: TFC's build, its measured cycles, FINN's captured ones | `check-kernels.sh` | `tests/networks` |
| `tests/layers` | FINN's layer table (`support/layering.py`) | `check-kernels.sh` | `tests/layers` |
| `tests/xsim_sweep` | the sweep's tooling (`scripts/emitted_text.py`) | `check-kernels.sh` | `tests/xsim_sweep` |
| `tests/transformation`, `tests/brevitas` | graph preparation and the Brevitas export | `check-kernels.sh` | `tests` |
| `tests/util`, `tests/toolchain`, `tests/docker_scripts` | `finn.util`; `finn.toolchain`; the `docker/` scripts (the `slow` tests deselected in both modes) | `check-kernels.sh` | `tests` |

A tree takes its name from the package it tests, so it mirrors `src`; each of the trees under its own `--confcutdir` has a `conftest.py` that imports the fixtures it wants from `tests/support` (`fixtures.py`, `tools.py`). A tree's name must not shadow a module the gates import: `tests` is on `sys.path` ahead of the standard library, so `tests/platform` would replace `platform`, and `tests/docker` the repository's `docker/`; hence `tests/platforms` and `tests/docker_scripts`.

`tests/container` (container runtimes and host toolchains, marker `container`) runs under `tests/conftest.py`; no code gate runs it.

The trees under their own `--confcutdir` get no `finn_test_seed`: a test seeds the generators it uses itself (for example `np.random.default_rng(<seed>)`). They mark Vivado work with `requires_xsim`, `requires_hls` and `requires_vivado` (`tests/support/xsim.py`), which mark and skip, and the markers of [3. Markers](#3-markers).

### 6. `tests/support`: the test-side library

What two test modules share lives in `tests/support`, one package. The gates put `tests` on `PYTHONPATH`, so it imports by package name (`from support.models import ...`), and `check-kernels.sh` lints and types the package as a whole (ruff, strict mypy), as it does the typed trees: a helper is a module of the library, not a file one test imports from another. No test module imports another, in any tree: `tests/layers/test_test_modules.py` fails on an import of a module named `test_*`. A helper two modules need moves to `tests/support`, and a fixture several trees need goes to `support.fixtures` (or `support.tools`), which each tree's `conftest.py` imports from, so a tree registers the fixtures it wants and no more.

What it holds:

- `support/rng_seed.py`: the seed of a test's random state, from its node id (`tests/conftest.py`, §4).
- `support/layering.py`: FINN's layers, what each may import, and the import walker. Each layer names the tree whose `test_layering.py` checks it; `tests/layers` tests the table and the walker.
- `support/refusals.py`: `refused(code, ...)`, which a test asserts a `Refusal` by: its finding's code (and its owner and details), the message of a code pinned once, in the test of the module that writes it; it imports `finn.core.reporting` and pytest only, so every tree may import it.
- `support/value_classes.py`: the check that a frozen value class holds immutable values, used by the engine, dataflow and kernel trees.
- `support/trained_networks.py`: the pretrained Brevitas example networks (`get_test_model_trained`), which the Brevitas, transformation and KernelOp tests export.
- `support/conformance.py`, `support/helpers.py`, `support/xsim.py`, `support/chain.py`, `support/adapted.py`, `support/kernel_specs/`: the kernel tests' harness (the conformance case, the construction helpers, the XSim markers, the Chain, the adapted-channel cases) and each kernel's spec.
- `support/models.py`, `support/tfc.py`, `support/cnv.py`, `support/packaging.py`, `support/measure_cycles.py`, `support/op_specs/`: the KernelOps' side (small models, TFC_W2A2 and CNV_W2A2 from the trained network to a partition, packaging without Vivado, the cycles measured in XSim) and the specs of the KernelOps' graphs.
- `support/fixtures.py`, `support/tools.py`: the session fixtures of TFC and CNV and the per-test `ip_store`; the fake Vivado installation and fake vendor tools.
- `support/capture.py`: `SEAMS`, the functions on the tests' side that `scripts/emitted_text.py` replaces to capture what an XSim job simulates without Vivado (`conformance.conformance`, `conformance.stream_through`, `xsim.vivado_simulator`, and the sweeps' transport). Renaming one is a change here: `tests/xsim_sweep` resolves every seam, and the tool reaches them through `support.capture.replace`, never by naming them (a checkout from before `support.capture` is still read by its old names).

`tests/oracle`, beside the library, holds the finn-dev oracle's captures: values only the HWCustomOp flow computed (it is deleted), which the kernel tests compare with; the gates never run the oracle.
