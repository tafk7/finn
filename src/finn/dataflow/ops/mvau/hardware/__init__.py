# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The hardware that realizes the decomposed MVAU.

- :mod:`inputs` -- what each Kernel is allowed to read.
- :mod:`replay_buffer` -- ``ReplayBufferKernel``, covering the replay Region.
- :mod:`dotp_axi` -- ``DotpAxiKernel``, covering the dot-product Region.
- :mod:`binding` -- which node fills which Kernel role, and where sources live.
- :mod:`composition` -- wiring two bound Kernels into one buildable artifact.

Nothing here reads a ``ModelWrapper``, an ONNX node, or a build configuration.
Everything a Kernel knows arrives as a typed handle wired in by
:mod:`finn.dataflow.mvau.decomposed`, which declares the Regions these cover.

**No re-exports on purpose.**  ``decomposed`` declares the Kernels, so it
imports from here; ``binding`` needs the assembled pool, so it imports from
``decomposed``.  A facade in this file would close that loop at import time.
Consumers name the submodule they want.
"""
