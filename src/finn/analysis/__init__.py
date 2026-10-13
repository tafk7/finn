# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's analysis passes: functions that read the ONNX graph and return what they find,
without changing it (Transformations change it). In qonnx's form, ``model.analysis(fn)``
returns ``fn(model)``, a dict.

The package re-exports nothing: each name is imported from the module that owns it.
"""
