# FINN Test Guidelines

Help keep FINN's testing suite fast, deterministic, and parallelisable when writing or modifying tests.

The trees the code gates run (the Space engine, dataflow, kernel and KernelOp tests) differ from the rest in their conftest, seeding and markers: see [5. Trees and gates](#5-trees-and-gates).

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
- `slow`: deselected by a gate's `--fast`, and in `tests/util` in both modes, so no gate runs the `slow` tests there.
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

The code gates (`scripts/check-*.sh`) run each tree with `--confcutdir` (`gate_pytest` in `scripts/_gate-common.sh`): no conftest above it loads, and `tests/conftest.py` (the seed of §4, `tests/support/rng_seed.py`, and the autouse member cache of `tests/support/member_cache.py`: each test runs in one `finn.kernels.members` cache, dropped at its end) only where the root is `tests`. `check-kernels.sh` runs `check-space.sh` and `check-dataflow-design.sh` first.

| Tree | Gate | `--confcutdir` |
|---|---|---|
| `tests/core/reporting` | `scripts/check-space.sh` | `tests/core/reporting` |
| `tests/core/space` | `scripts/check-space.sh` | `tests/core/space` |
| `tests/dataflow` | `scripts/check-dataflow-design.sh` | `tests/dataflow` |
| `tests/kernels` | `scripts/check-kernels.sh` | `tests/kernels` |
| `tests/kernel_ops` | `scripts/check-kernels.sh` | `tests/kernel_ops` |
| `tests/xsim_sweep` | `scripts/check-kernels.sh` | `tests/xsim_sweep` |
| `tests/transformation`, `tests/brevitas` | `scripts/check-kernels.sh` | `tests` |
| `tests/util` | `scripts/check-kernels.sh` (its `slow` tests deselected in both modes) | `tests` |

`tests/container` (container runtimes and host toolchains, marker `container`) runs under `tests/conftest.py`; no code gate runs it.

The trees under their own `--confcutdir` get no `finn_test_seed`: a test seeds the generators it uses itself (for example `np.random.default_rng(<seed>)`). They mark Vivado work with `requires_xsim`, `requires_hls` and `requires_vivado` (`tests/support/xsim.py`), which mark and skip, and the markers of [3. Markers](#3-markers).

The gates put `tests` on `PYTHONPATH`, so its helpers import by package name; these are in `tests/support`:

- `support/rng_seed.py`: the seed of a test's random state, from its node id (`tests/conftest.py`, §4).
- `support/layering.py`: FINN's layers, what each may import, and the import walker. Each layer names the tree whose `test_layering.py` checks it; `tests/core/space/test_layering.py` also tests the table and the walker.
- `support/value_classes.py`: the check that a frozen value class holds immutable values, used by the engine, dataflow and kernel trees.
- `support/trained_networks.py`: the pretrained Brevitas example networks (`get_test_model_trained`), which the Brevitas, transformation and KernelOp tests export.
- `support/conformance.py`, `support/helpers.py`, `support/xsim.py`, `support/chain.py`, `support/adapted.py`, `support/kernel_specs/`: the kernel tests' harness (the conformance case, the construction helpers, the XSim markers, the Chain, the adapted-channel cases) and each kernel's spec. `check-kernels.sh` lints and types `tests/support` as a whole.
- `oracle/`: the finn-dev oracle's captures, values only the HWCustomOp flow computed (it is deleted), which the kernel tests compare with; the gates never run the oracle.
