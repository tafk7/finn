# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The values kernels hold and their ports admit.

``semantics``: the Space value semantics of datatypes (over :mod:`finn.dataflow.datatypes`
values), integer vectors and tensors, and threshold tables. ``domains``: the ``Integer``
policy a port's datatype must satisfy, ``admit_element`` (an element, or the refusal of
one its datatype cannot hold), and the set selector's datatype.
"""
