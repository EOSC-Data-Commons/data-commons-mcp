"""Small ANSI-colored logging formatter used by logging.yml.

This avoids external dependencies by injecting ANSI color codes into
the levelname for terminal output. The formatter preserves the original
record attribute values and only changes the displayed levelname.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import Literal

COLOR_MAP: Mapping[int, str] = {
    logging.CRITICAL: "\x1b[1;35m",  # bright magenta
    logging.ERROR: "\x1b[1;31m",  # bright red
    logging.WARNING: "\x1b[1;33m",  # bright yellow
    logging.INFO: "\x1b[1;32m",  # bright green
    logging.DEBUG: "\x1b[1;34m",  # bright blue
}
GREY = "\x1b[90m"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
BLUE = "\x1b[34m"
YELLOW = "\x1b[33m"


class ColoredFormatter(logging.Formatter):
    """Formatter that colors the levelname using ANSI escape sequences.

    Usage in dictConfig / YAML config: use the special `()` key to point
    at this class, e.g. `() : data_commons_search.logging.ColoredFormatter`.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: Literal["%", "{", "$"] = "%"):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        """Return the formatted time for a record, wrapped in GREY when on a TTY."""
        ts = super().formatTime(record, datefmt=datefmt)
        # Only color when the output is a TTY so log files remain plain.
        if sys.stdout.isatty():
            return f"{GREY}{ts}{RESET}"
        return ts

    def format(self, record: logging.LogRecord) -> str:
        # Save original attributes in case other handlers rely on them
        original_levelname = record.levelname
        original_name = getattr(record, "name", None)
        try:
            color = COLOR_MAP.get(record.levelno, "")
            if color and sys.stdout.isatty():
                record.levelname = f"{color}{original_levelname}{RESET}"
            # color the logger name in grey for TTY output
            if original_name is not None and sys.stdout.isatty():
                record.name = f"{GREY}{original_name}{RESET}"
            return super().format(record)
        finally:
            # restore to original to avoid side effects
            record.levelname = original_levelname
            if original_name is not None:
                record.name = original_name


__all__ = ["ColoredFormatter"]
