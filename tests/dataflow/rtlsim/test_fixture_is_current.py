# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the RTL harness compiling against the design space it drives.

``composed_mvau_equiv.py`` only runs inside the container, behind Vivado, so
nothing in the ordinary suite touched it -- and a pool rename left it importing
a symbol that no longer existed for two commits without anything noticing.

Everything up to the point where XSI is needed is plain Python, so that much
can be a normal test: import the module, resolve every configuration's design
point, and read the parameters it will drive.  A rename that breaks the harness
now breaks this.
"""

from __future__ import annotations

import pytest

from dataflow.rtlsim import composed_mvau_equiv as fixture
from finn.dataflow.mvau.compute_kernels import DECOMPOSED_MVAU_KERNELS


@pytest.mark.parametrize("config", fixture.CONFIGS, ids=lambda item: item.label)
def test_every_configuration_resolves_the_parameters_it_will_drive(
    config: fixture.Config,
) -> None:
    values = fixture.declared_parameters(config)
    declared = {item.name for item in DECOMPOSED_MVAU_KERNELS.provider_parameters()}
    assert set(values) == declared
