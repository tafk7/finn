# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build's top-1 accuracy on its board over a dataset's test set (``main``): the shared
driver over a runtime's device, by the runtime file beside it, as ``cli.main`` runs it.

A build of one core with one input and one output is validated (another is refused,
``validate-unsupported``), with the run-time weights in ``--runtime_weight_dir`` written
first, as ``cli.main`` writes them: every test image, ``--batchsize`` at a time and the last
batch what remains, each its input's frame, run through the core; a frame's label is its
output where that is one value, else the place of its largest value; the accuracy is over
the whole test set. A batch of no frame (``batch-invalid``) and an empty test set
(``dataset-empty``) are refused. The datasets, MNIST and CIFAR-10, are loaded by
``dataset_loading``, imported only here (its absence refused, ``dataset-loading-absent``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import cli
from .cli import RUNTIME_FILE, OpenDevice
from .description import read
from .driver import Driver
from .errors import Refused


def arguments(image: Path) -> argparse.ArgumentParser:
    """The command line's arguments, the device image ``image`` the default of
    ``--bitfile``."""
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=100
    )
    parser.add_argument(
        "--dataset", help="dataset to use", choices=("mnist", "cifar10"), required=True
    )
    parser.add_argument(
        "--bitfile", help="the bitfile to run (default: %(default)s)", default=str(image)
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="/tmp"
    )
    parser.add_argument("--device", help="the index of the device to use", type=int, default=0)
    parser.add_argument(
        "--runtime_weight_dir",
        help="a directory of run-time weights, a <node>.dat file a node",
        default="runtime_weights/",
    )
    return parser


def dataset_of(dataset: str, root: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The test images and labels of ``dataset``, downloaded into ``root`` if absent."""
    try:
        if dataset == "mnist":
            from dataset_loading import mnist  # noqa: PLC0415

            _, _, images, labels, _, _ = mnist.load_mnist_data(root, download=True, one_hot=False)
        else:
            from dataset_loading import cifar  # noqa: PLC0415

            _, _, images, labels, _, _ = cifar.load_cifar_data(root, download=True, one_hot=False)
    except ImportError as error:
        raise Refused(
            "dataset-loading-absent",
            f"loading the {dataset} dataset takes dataset_loading ({error.name} is not "
            "installed): install it",
        ) from error
    return np.asarray(images), np.asarray(labels)


def labels_of(output: np.ndarray[Any, Any], frames: int) -> np.ndarray[Any, Any]:
    """Each of ``frames`` frames' label in ``output``: its value where a frame holds one,
    else the place of its largest value."""
    each = output.reshape(frames, -1)
    return each[:, 0] if each.shape[1] == 1 else np.argmax(each, axis=1)


def validate(
    driver: Driver, images: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], batch: int
) -> float:
    """The top-1 accuracy of the build's one core on every one of ``images`` against
    ``labels``, ``batch`` at a time and the last batch what remains, each batch's count
    printed as it runs: in percent. A batch below one and no image are refused."""
    cores = driver.description.cores
    if len(cores) != 1 or len(cores[0].inputs) != 1 or len(cores[0].outputs) != 1:
        ports = [(core.name, len(core.inputs), len(core.outputs)) for core in cores]
        raise Refused(
            "validate-unsupported",
            "validation runs a build of one core with one input and one output, and this "
            f"build's cores and their inputs and outputs are {ports}",
        )
    (core,) = cores
    (given,), (taken,) = core.inputs, core.outputs
    first = given.shape[0] if given.shape else 1
    size, total = cli.batch(batch), len(images)
    if total == 0:
        raise Refused("dataset-empty", "the test set holds no image")
    batches = -(-total // size)
    right = 0
    for index in range(batches):
        chosen = slice(index * size, min((index + 1) * size, total))
        count = chosen.stop - chosen.start
        frames = images[chosen].reshape(count * first, *given.shape[1:])
        found = driver.run_core(core.name, {given.tensor: frames})[taken.tensor]
        right += int(np.sum(labels_of(found, count) == labels[chosen].reshape(-1)))
        sys.stdout.write(
            f"batch {index + 1} / {batches} : total OK {right} NOK {chosen.stop - right}\n"
        )
    return 100.0 * right / total


def main(argv: Sequence[str] | None, open_device: OpenDevice, here: Path, image: Path) -> int:
    """Validate the build (the module docstring) on ``argv``: the runtime file in ``here``,
    the device ``open_device`` opens, ``image`` the device image it runs unless told
    another. Its exit status."""
    args = arguments(image).parse_args(argv)
    try:
        description = read(here / RUNTIME_FILE)
        with Driver(open_device(description, args.device), description) as driver:
            driver.load(Path(args.bitfile))
            driver.write_weight_files(Path(args.runtime_weight_dir))
            images, labels = dataset_of(args.dataset, args.dataset_root)
            accuracy = validate(driver, images, labels, args.batchsize)
    except Refused as refused:
        sys.stderr.write(f"refused: {refused}\n")
        return 1
    sys.stdout.write(f"Final accuracy: {accuracy:f}\n")
    return 0


__all__ = ["arguments", "labels_of", "main", "dataset_of", "validate"]
