# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The host driver every runtime shares: plain Python, shipped as it is beside a build's
device image and its runtime file (``finn.shells.integration.runtime_file``,
``integration.json``), importing the standard library and numpy alone.

- ``description``: the runtime file, read (``read``), a schema it does not list refused;
- ``packing``: an edge's values packed into the bytes its mover moves, and back;
- ``device``: the one seam a runtime adds, ``Device``, over its platform's own library;
- ``movers``: each mover kind's protocol, driven through its endpoint's named registers;
- ``driver``: ``Driver``, which runs each core by its name, writes a node's run-time
  weights at its endpoint, and runs the parent graph (``run_graph``: its host nodes by ONNX
  execution, onnx and qonnx imported only then).
- ``cli`` and ``validate``: the command line a runtime's ``driver.py`` runs and the top-1
  validation its ``validate.py`` runs, each over the device the runtime opens.

Every refusal is a ``Refused``, its code and its message.
"""

from .description import SCHEMAS, Description, read
from .device import Buffer, Device
from .driver import Driver
from .errors import Refused

__all__ = ["SCHEMAS", "Buffer", "Description", "Device", "Driver", "Refused", "read"]
