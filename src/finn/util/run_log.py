# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One run's log: FINN's records and Python's warnings, to a log file and the console.

``log_run`` sets it up for a ``with`` block and restores the logging it found,
``running_step`` names the step stamped on each record, and ``FILE_ONLY`` keeps a record
off the console. The standard library only, so any entry point can use it.
"""

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Iterator

#: Pass as ``extra`` to send a record to the log file only, unless ``verbose`` (used for
#: a refusal's traceback).
FILE_ONLY = {"finn_console": False}

#: The step stamped on every record, ``-`` outside a step. A module global, not a context
#: variable, because a step's thread pool (HLS synthesis) does not inherit context.
_step = "-"


@contextlib.contextmanager
def running_step(name: str) -> Iterator[None]:
    """Stamp the records of the ``with`` block with the step ``name``."""
    global _step
    outer, _step = _step, name
    try:
        yield
    finally:
        _step = outer


class _StepStamp(logging.Filter):
    """Stamps each record with the running step (``step``), and strips the trailing
    newline warnings.formatwarning adds to a captured warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.step = _step
        if record.name == "py.warnings" and isinstance(record.msg, str):
            record.msg = record.msg.rstrip("\n")
        return True


class _Console(logging.Filter):
    """One console stream's share of the records: below WARNING for stdout, WARNING and
    above for stderr (``errors``). With ``verbose``, everything the log file has;
    otherwise the INFO lines of the logger named ``runner``, and every logger's warnings
    and errors, less the ``FILE_ONLY`` records."""

    def __init__(self, runner: str, verbose: bool, errors: bool):
        super().__init__()
        self.runner = runner
        self.verbose = verbose
        self.errors = errors

    def filter(self, record: logging.LogRecord) -> bool:
        if (record.levelno >= logging.WARNING) != self.errors:
            return False
        if self.verbose:
            return True
        if not getattr(record, "finn_console", True):
            return False
        return record.levelno >= logging.WARNING or record.name == self.runner


class _ConsoleFormat(logging.Formatter):
    """The message alone for INFO and below; ``LEVEL: message`` above."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text if record.levelno < logging.WARNING else f"{record.levelname}: {text}"


@contextlib.contextmanager
def log_run(path: str, *, runner: str, verbose: bool = False) -> Iterator[None]:
    """Log the ``with`` block to the file ``path`` (appended to, its directory made) and
    to the console, then restore the logging found.

    FINN's loggers (``finn``) and Python's warnings (``py.warnings``, captured through
    ``logging.captureWarnings``) go to the file, each line with its level, step
    (running_step) and logger, and to the console (_Console): with ``verbose`` all of
    it, otherwise the INFO lines of the logger named ``runner``, warnings and errors.
    ``finn`` logs from INFO unless the caller set it higher; a level the caller set above
    INFO is the one the file and the console show, for every logger below ``finn``. No stream
    is redirected, so a third party's print reaches the console unchanged. The handlers,
    levels and warnings hook are restored afterwards, leaving a notebook's own logging as
    it was."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stamp = _StepStamp()
    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(step)s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [file_handler]
    for stream, errors in ((sys.stdout, False), (sys.stderr, True)):
        console = logging.StreamHandler(stream)
        console.addFilter(_Console(runner, verbose, errors))
        console.setFormatter(_ConsoleFormat("%(message)s"))
        handlers.append(console)
    loggers = [logging.getLogger("finn"), logging.getLogger("py.warnings")]
    saved = [(logger, logger.level, logger.propagate) for logger in loggers]
    for handler in handlers:
        handler.addFilter(stamp)
        for logger in loggers:
            logger.addHandler(handler)
    for logger in loggers:
        # Not also to the root's handlers (a notebook's), which would print each line twice.
        logger.propagate = False
    shown = loggers[0].level or logging.INFO
    if shown > logging.INFO:
        for handler in handlers:
            handler.addFilter(
                lambda record: record.levelno >= shown or record.name == "py.warnings"
            )
    loggers[0].setLevel(shown)
    # Leave the warnings hook alone if the caller already captures warnings.
    capture = warnings.showwarning is not getattr(logging, "_showwarning", None)
    if capture:
        logging.captureWarnings(True)
    try:
        yield
    finally:
        if capture:
            logging.captureWarnings(False)
        for logger, level, propagate in saved:
            logger.setLevel(level)
            logger.propagate = propagate
            for handler in handlers:
                logger.removeHandler(handler)
        for handler in handlers:
            handler.close()
