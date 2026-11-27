# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's checkpoint-linked chains hold together under ``--dist loadgroup``.

The end-to-end families (BNN, cybsec, MobileNet, external weights), the notebooks
and the benchmark builds are chains: each step reads what the step before it saved,
and skips when it is missing. Jenkins runs them with ``-n N --dist loadgroup`` and
relies on ``xdist_group`` to keep a chain on one worker. If the installed pytest and
pytest-xdist stop honouring the mark (pytest 9.2 with pytest-xdist 3.8 does), the
steps are dealt round-robin, every step after the first skips, and the run still
exits 0. These tests run a small chain in each form FINN uses, under xdist, and
fail if it is split.
"""

import pytest

pytestmark = pytest.mark.util
pytest_plugins = ["pytester"]

# Each step records the worker it ran on, then checks the previous step's
# checkpoint. A missing checkpoint fails here (FINN's own chains skip, which is
# what let the split go unnoticed).
_STEPS = """
import os


def _step(tmp_root, chain, step, worker_log):
    with open(worker_log, "a") as f:
        f.write("%s %s %s\\n" % (chain, step, os.environ["PYTEST_XDIST_WORKER"]))
    chain_dir = os.path.join(tmp_root, chain)
    os.makedirs(chain_dir, exist_ok=True)
    if step > 0:
        prev = os.path.join(chain_dir, "step%d.ckpt" % (step - 1))
        assert os.path.isfile(prev), "%s not found from previous test step" % prev
    open(os.path.join(chain_dir, "step%d.ckpt" % step), "w").close()
"""

# the form of test_end2end_bnn_pynq.py: scenarios as pytest.param marks from
# pytest_generate_tests, class-scoped
_PARAM_CHAIN = (
    _STEPS
    + """
import pytest

ROOT = os.environ["CHAIN_ROOT"]
LOG = os.environ["CHAIN_LOG"]


def pytest_generate_tests(metafunc):
    names = ["s%d" % i for i in range(3)]
    metafunc.parametrize(
        "scenario",
        [pytest.param(n, marks=pytest.mark.xdist_group(name="param_" + n)) for n in names],
        ids=names,
        scope="class",
    )


class TestChain:
    def test_s0(self, scenario):
        _step(ROOT, scenario, 0, LOG)

    def test_s1(self, scenario):
        _step(ROOT, scenario, 1, LOG)

    def test_s2(self, scenario):
        _step(ROOT, scenario, 2, LOG)

    def test_s3(self, scenario):
        _step(ROOT, scenario, 3, LOG)
"""
)

# the form of test_end2end_cybsec_mlp.py, test_end2end_mobilenet_v1.py and
# test_ext_weights.py: a decorator mark on each step function
_DECORATED_CHAIN = (
    _STEPS
    + """
import pytest

ROOT = os.environ["CHAIN_ROOT"]
LOG = os.environ["CHAIN_LOG"]

"""
    + "".join(
        '@pytest.mark.xdist_group(name="decorated")\n'
        "def test_d%d():\n"
        '    _step(ROOT, "decorated", %d, LOG)\n\n\n' % (i, i)
        for i in range(6)
    )
)


def test_checkpoint_chains_stay_on_one_worker_under_loadgroup(pytester, monkeypatch):
    pytest.importorskip("xdist")
    pytester.makeini("[pytest]\n")
    pytester.makepyfile(test_param_chain=_PARAM_CHAIN, test_decorated_chain=_DECORATED_CHAIN)
    log = pytester.path / "workers.log"
    monkeypatch.setenv("CHAIN_ROOT", str(pytester.path / "ckpt"))
    monkeypatch.setenv("CHAIN_LOG", str(log))

    result = pytester.runpytest_subprocess(
        "-n", "2", "--dist", "loadgroup", "-v", "-p", "no:cacheprovider"
    )

    # judged by where each step ran, not by the "@<group>" node id suffix: that
    # suffix is how pytest-xdist 3.8 marks a group, and later releases drop it.
    # Ungrouped, loadgroup deals consecutive steps to alternating workers, so a
    # lost group splits at least the six-step chain.
    workers = {}
    for line in log.read_text().splitlines():
        chain, _, worker = line.split()
        workers.setdefault(chain, set()).add(worker)
    assert sorted(workers) == ["decorated", "s0", "s1", "s2"]
    split = {chain: sorted(w) for chain, w in workers.items() if len(w) > 1}
    assert not split, "chains split across xdist workers: %s" % split
    result.assert_outcomes(passed=18)
