# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis passes over a graph, in qonnx's form: ``model.analysis(fn)`` returns
``fn(model)``, a dict.

They are generic: no build step runs one, and what a finding stops is the caller's (a
test's, or a step's that runs one). Each returns its findings (``finn.core.space.Finding``)
under ``findings``. An analysis that needs more than the model is a function of what it
needs that returns the pass.

``finn.analysis.equivalence`` checks a prepared graph against its export on draws from
its input annotation, up to preparation's declared deviations, and its integer
annotations against the values it computes; ``finn.analysis.initializers`` checks each
integer-annotated initializer's values against its annotation.

The package re-exports nothing: each name is imported from the module that owns it.
"""
