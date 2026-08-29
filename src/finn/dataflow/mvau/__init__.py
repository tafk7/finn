# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete MVAU Kernel authoring support.

Deliberately empty of re-exports.  Eagerly importing the submodules here made
the package unable to hold two modules that need each other -- the compute pool
and the Kernels it contains -- because reaching any submodule ran this file
first.  Import the submodule you want; ``finn.dataflow.mvau_design`` is the
facade for callers that want one name.
"""
