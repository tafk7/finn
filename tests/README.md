# FINN Test Guidelines

Help keep FINN's testing suite fast, deterministic, and parallelisable when writing or modifying tests.

The trees the code gates run (the Space engine, dataflow, kernel and KernelOp tests) differ from the rest in their conftest, seeding and markers: see [6. Trees and gates](#6-trees-and-gates).

### 1. Scratch paths

Never write scratch files into the repo root or CWD. Collisions are guaranteed when tests run in parallel. Instead:

- Use `make_build_dir()` to allocate a unique directory in the build directory,
  `finn.resources.scratch()` (`FINN_BUILD_DIR`, by default `$FINN_HOME/build`).
- Use `robust_rmtree()` to tear down the test.
- If the test's outputs are useful for diagnosis, you may keep the outputs if the test failed.

```python
from finn.util.basic import make_build_dir, robust_rmtree

test_dir = make_build_dir("test_my_feature_")

# later on
if not failed:
    robust_rmtree(test_dir)
```

### 2. Process state & environment

Don't mutate env variables or global process state. Use pytest's `monkeypatch` fixture to override env variables or system attributes.

```python
def test_sim_behaviour(monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", "/custom/path")
    # original FINN_BUILD_DIR restored automatically after the test
```

### 3. Parallel Scheduling (`xdist_group`)

If a chain of tests must run on the same worker process, group them with the
`xdist_group` marker and run with `--dist loadgroup` when running with multiple
workers (i.e. `-n <N>`):

```python
@pytest.mark.xdist_group(name="my_feature_chain")
def test_step_1(): ...

@pytest.mark.xdist_group(name="my_feature_chain")
def test_step_2(): ...
```


### 4. Markers

Decorate tests with the existing markers (`.pytest.ini`). For example, `@pytest.mark.slow`. The kernel trees' markers are in [6. Trees and gates](#6-trees-and-gates).

### 5. Randomness

Every test under `tests/conftest.py` gets a stable `finn_test_seed` through it (not the trees the code gates run with their own `--confcutdir`: [6. Trees and gates](#6-trees-and-gates)). An autouse fixture seeds Python `random`, `numpy.random`, and the PyTorch CPU generator when installed.

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

### 6. Trees and gates

The code gates (`scripts/check-*.sh`) run each tree with `--confcutdir` (`gate_pytest` in `scripts/_gate-common.sh`): no conftest above it loads, and `tests/conftest.py` (the seed of §5, `rng_seed.py`) only where the root is `tests`. `check-kernels.sh` runs `check-space.sh` and `check-dataflow-design.sh` first.

| Tree | Gate | `--confcutdir` |
|---|---|---|
| `tests/core/space` | `scripts/check-space.sh` | `tests/core/space` |
| `tests/dataflow` | `scripts/check-dataflow-design.sh` | `tests/dataflow` |
| `tests/kernels` | `scripts/check-kernels.sh` | `tests/kernels` |
| `tests/kernel_ops` | `scripts/check-kernels.sh` | `tests/kernel_ops` |
| `tests/xsim_sweep` | `scripts/check-kernels.sh` | `tests/xsim_sweep` |
| `tests/transformation`, `tests/brevitas` (less two files the script names) | `scripts/check-kernels.sh` | `tests` |
| `tests/util` | `scripts/check-kernels.sh` | `tests` |

`tests/container` (container runtimes and host toolchains, marker `container`) runs under `tests/conftest.py`; no code gate runs it.

The trees under their own `--confcutdir` get no `finn_test_seed`: a test seeds the generators it uses itself (for example `np.random.default_rng(<seed>)`). They mark Vivado work with `requires_xsim`, `requires_hls` and `requires_vivado` (`tests/kernels/xsim.py`), which mark and skip, and the markers `xsim`, `vivado` and `slow`: `check-kernels.sh` deselects `xsim` and `vivado` (`scripts/xsim-sweep.sh` runs them), and a gate's `--fast` deselects `slow`.

The gates put `tests` on `PYTHONPATH`, so its helpers import by bare name:

- `rng_seed.py`: the seed of a test's random state, from its node id (`tests/conftest.py`, §5).
- `layering.py`: FINN's layers, what each may import, and the import walker. Each layer names the tree whose `test_layering.py` checks it; `tests/core/space/test_layering.py` also tests the table and the walker.
- `value_classes.py`: the check that a frozen value class holds immutable values, used by the engine, dataflow and kernel trees.
- `oracle/`: the finn-dev oracle's captures, values only the HWCustomOp flow computed (it is deleted), which the kernel tests compare with; the gates never run the oracle.
