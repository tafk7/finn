# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""List every decision key and every linked node key of MVAU (the family compiled alone).

Run it against two revisions and diff the output, for example (from the FINN checkout):

    git archive 35e59b442 src | tar -x -C /tmp/base
    PYTHONPATH=/tmp/base/src:tests:deps/qonnx/src python mvau_keys.py > before.txt
    PYTHONPATH=src:tests:deps/qonnx/src python mvau_keys.py > after.txt
"""

from finn.core.space import inspection
from finn.kernels.mvau import MVAU

linked = inspection.model(MVAU).linked
print("# decisions")
for info in inspection.decisions(MVAU):
    print(info.key)
print("# nodes")
for node in sorted(linked.nodes, key=lambda n: n.key):
    print(node.kind, node.key)
