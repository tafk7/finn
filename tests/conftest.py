# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Each test runs with its random state seeded from its node id (support.rng_seed, in
# tests/support).
import pytest

import numpy as np
import random
import sys
from support.checked_cache import DIFFERING_READS, CheckedCache
from support.member_cache import _member_cache, pytest_sessionfinish  # noqa: F401
from support.rng_seed import seed_from_nodeid

from finn.core.space import _runtime

# Every Space snapshot's cache, in every suite, refuses what the engine's rule forbids
# (support.checked_cache); production keeps exact dicts.
_runtime.cache_factory = CheckedCache

try:
    import torch
except ImportError:
    torch = None


@pytest.fixture
def finn_test_seed(request):
    return seed_from_nodeid(request.node.nodeid)


@pytest.fixture(autouse=True)
def _deterministic_random_state(request, finn_test_seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state() if torch is not None else None

    try:
        random.seed(finn_test_seed)
        np.random.seed(finn_test_seed)
        if torch is not None:
            torch.random.default_generator.manual_seed(finn_test_seed)
        request.node.user_properties.append(("finn_test_rng_seed", str(finn_test_seed)))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)


@pytest.fixture(params=["index", "scan"])
def index_path(request, monkeypatch):
    """Run a test with the Space's readers index used for every cache, then for none: the
    answers are the same either way (``_runtime.INDEX_FROM``)."""
    monkeypatch.setattr(_runtime, "INDEX_FROM", 0 if request.param == "index" else sys.maxsize)
    return request.param


def pytest_terminal_summary(terminalreporter):
    if DIFFERING_READS:
        terminalreporter.section(
            "cache setdefault: kept and dropped evaluations differing in reads"
        )
        for (site, kind), count in sorted(DIFFERING_READS.items()):
            terminalreporter.write_line(f"{count:6d}  {site}: {kind}")
