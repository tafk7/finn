# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's deployment package: what is copied to the board to run the build's
bitfile there.

``MakePynqDeployment`` writes ``deploy/`` under the build's output directory, afresh:
the bitfile (``runner.BITFILE``'s directory), the driver (``driver.DRIVER_DIR``) and its
description (``driver.json``, from ``driver.DRIVER_REPORT``), and the host's model
(``model/``, ``host_model``). It ships only what that output directory's own builds
made, from the partition's design as it is now, and refuses anything else by name
before it copies (``PynqRefused``).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.core.graph.annotations.outputs import OUTPUT_BITFILE, OUTPUT_HWH, OUTPUT_IP
from finn.core.graph.partition import design_digest, partition_body
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT
from finn.shells.pynq.runner import BITFILE, PynqRefused, pynq_row, refuse_stale_bitfile

#: The deployment package's directory, under the output directory.
DEPLOY_DIR = Path("deploy")

#: The finn.outputs keys a deployed model drops: they hold paths on the build machine.
MACHINE_PATHS = (OUTPUT_IP, OUTPUT_BITFILE, OUTPUT_HWH)

#: The output the deployment makes, which owns its refusals.
_OUTPUT = "deployment_package"


def host_model(model: ModelWrapper, directory: Path) -> Path:
    """Write the parent graph ``model`` into ``directory`` as the deployment ships it:
    parent.onnx (the host's nodes and the partition node) and partition.onnx beside it.
    The node refers to its body by a relative path, and neither model keeps a path on
    the build machine (``MACHINE_PATHS``). Returns the path of parent.onnx."""
    directory.mkdir(parents=True, exist_ok=True)
    node, body, _ = partition_body(model)
    shipped = ModelWrapper(copy.deepcopy(model.model))
    shipped_body = ModelWrapper(copy.deepcopy(body.model))
    for each in (shipped, shipped_body):
        for key in MACHINE_PATHS:
            each.delete(key)
    (shipped_node,) = [each for each in shipped.graph.node if each.name == node.name]
    getCustomOp(shipped_node).set_nodeattr("model", "partition.onnx")
    shipped_body.save(str(directory / "partition.onnx"))
    shipped.save(str(directory / "parent.onnx"))
    return directory / "parent.onnx"


class MakePynqDeployment(Transformation):
    """The deployment package of the build's parent graph, into ``deploy/`` under
    the output ``directory``, written afresh: the bitfile, the driver, the host's model
    (``deploy/model/``, ``host_model``) and the driver's description
    (``deploy/driver.json``). Its shell is the target the partition's body states
    (``pynq_row``). It ships the bitfile and the driver this output directory's own
    builds made: a parent graph that states no bitfile, or another directory's, an
    output directory without the driver's description, and a bitfile or driver made from
    another design than the partition's now are refused, by name. The parent graph is
    unchanged."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        pynq_row(model, _OUTPUT)
        output = self.directory
        stated = model.get(OUTPUT_BITFILE)
        if stated is None:
            raise PynqRefused(
                _OUTPUT,
                "bitfile-absent",
                "the model states no bitfile (finn.outputs): the deployment ships the "
                "bitfile and the driver its build made; list step_build_bitfile and "
                "step_make_driver before it",
            )
        if Path(stated) != output / BITFILE:
            raise PynqRefused(
                _OUTPUT,
                "bitfile-foreign",
                f"the model's bitfile is {stated}, not this output directory's "
                f"({output / BITFILE}); build the bitfile into this output directory "
                "(step_build_bitfile)",
            )
        if not (output / DRIVER_REPORT).is_file():
            raise PynqRefused(
                _OUTPUT,
                "driver-absent",
                f"no driver in this output directory ({output / DRIVER_REPORT}): the "
                "deployment ships the bitfile and the driver its build made; list "
                "step_make_driver before it",
            )
        refuse_stale_bitfile(model, _OUTPUT)
        _, body, _ = partition_body(model)
        made_from = json.loads((output / DRIVER_REPORT).read_text()).get("made_from")
        now = design_digest(body)
        if made_from != now:
            raise PynqRefused(
                _OUTPUT,
                "driver-stale",
                f"the driver in this output directory ({output / DRIVER_DIR}) was made from "
                f"the design {made_from}, and the partition's design is now {now}; make the "
                "driver again (step_make_driver)",
            )
        deploy = output / DEPLOY_DIR
        shutil.rmtree(deploy, ignore_errors=True)
        shutil.copytree(output / BITFILE.parent, deploy / BITFILE.parent)
        shutil.copytree(output / DRIVER_DIR, deploy / DRIVER_DIR, copy_function=shutil.copyfile)
        host_model(model, deploy / "model")
        shutil.copy(output / DRIVER_REPORT, deploy / "driver.json")
        return model, False


__all__ = ["DEPLOY_DIR", "MACHINE_PATHS", "MakePynqDeployment", "host_model"]
