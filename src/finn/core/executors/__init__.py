# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Executors: what runs a model's nodes under ``finn.core.onnx_exec.execute_onnx``.

The caller chooses them, as ONNX Runtime's execution providers are chosen:
``execute_onnx(model, inputs, executors=(...))``. For each node, the first executor
that claims it runs it; qonnx's execution runs the nodes none claims (the host's ONNX
ops and the custom ops no executor knows). The graph holds no execution state.

``Python`` is the default: it runs a KernelOp's ``execute_node``, the op's reference,
and a partition node's body under the executors its caller chose. ``XSim``
(``finn.core.executors.xsim.executor``, not loaded here: it loads the kernel stack)
runs a partition of KernelOps as its hardware, in XSim. An executor states whether it
is ``hardware``; a run that requires hardware refuses to run a hardware node
(``hardware_node``: a KernelOp, a partition node) any other way.
"""

from finn.core.executors.base import Context, Executor, hardware_node, is_partition
from finn.core.executors.python import Python

__all__ = ["Context", "Executor", "Python", "hardware_node", "is_partition"]
