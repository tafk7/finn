############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""SUPERSEDED SPIKE — the first FINN-native dataflow-kernel backend attempt.

STATUS (2026-07-17): this package is a superseded exploratory spike. The TRUNK of
the dataflow-kernel work is ``finn/src/finn/design_space/`` (the resolve engine +
per-op fixture packages + hermetic emit). Do NOT extend this package; add new work
under ``design_space/``.

This spike is retained for two reasons — one LIVE, one reference:

1. LIVE DEPENDENCY — ``kernels/derivation/`` is still imported by the trunk
   (``design_space/primitives/spec_helpers.py`` pulls ``derive_dim``,
   ``ShapeHierarchy``, ``VALUE_OPTIMIZED``, ``constant_datatype`` from it). This
   sub-package is the vendored Brainsmith derivation layer and is NOT dead — do not
   delete it without first giving the trunk its own copy and rewiring those imports.
   (Note: ``design_space/primitives/spec_helpers.py`` is currently a byte-identical
   copy of ``kernels/derivation/spec_helpers.py`` that re-imports from here — a known
   duplication to resolve when the derivation layer is properly extracted.)

2. REFERENCE ONLY — the rest of this package (``adapter.py``, ``core.py``,
   ``implementation.py``, ``kernel_op.py``, ``registry.py``, ``kernels/thresholding/``,
   ``tests/``) is ORPHANED relative to the trunk: nothing in ``design_space/`` imports
   it. It proved the mechanisms the trunk later rebuilt flat — the typed
   ``Template``/``Artifacts`` codegen (harvested into ``design_space/space/artifacts.py``)
   and the hermetic ``emit(design_point, params, config)`` contract. Keep as an
   educational reference for the emit + adapter shape; a future "retire the spike" task
   would extract ``derivation/`` and delete everything else.

See memory notes ``emit-phase`` / ``impl-bundles`` and
``kernel-design/kernel-final-design/`` for the trunk design.
"""
