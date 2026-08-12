############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""The op-model: how a hardware kernel is modeled on top of the resolve engine.

Holds the :class:`~finn.kernels.model.kernel.DataflowKernel` / :class:`~finn.kernels.model.backend.Backend`
/ :class:`~finn.kernels.model.parameter_source.ParameterSource` vocabulary, the tiling/fold-depth
projections, ``ports``, and the op-agnostic parameter-source CONTRACT (``demand``,
``param_names``, ``param_contract``) plus the ``source_backend`` abstraction (the
realizations live in ``dataflow/parameters/``). Also holds ``registry`` (the op-agnostic
implementation-registry factory) and ``artifacts`` (the typed emit-output vocabulary
consumed by ``emit/``).
"""
