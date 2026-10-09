# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""xelab's thread count, one rule for every caller: FINN's XSI compilation and the
kernel harness (finn.core.executors.xsim.rtl.simulate) both elaborate with it, so the
XSim sweep's FINN_XELAB_MT bounds each of its jobs."""

import pytest

from finn.util.toolchain import xelab_threads


@pytest.mark.parametrize(
    ("environment", "threads"),
    [
        ({"FINN_XELAB_MT": "2", "NUM_DEFAULT_WORKERS": "16"}, "2"),
        ({"NUM_DEFAULT_WORKERS": "16"}, "16"),
        ({}, "8"),
        ({"FINN_XELAB_MT": "1"}, "off"),
    ],
)
def test_xelab_threads_follow_the_environment_and_stay_bounded(environment, threads):
    assert xelab_threads(environment) == threads
