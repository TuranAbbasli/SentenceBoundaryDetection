# -*- coding: utf-8 -*-
"""
Colorful console logging and rotating file logs with UTC timestamps,
context tags (prompt_id, event), and extra key=value pairs.

To adjust:
- LOG_LEVEL: DEBUG/INFO/WARNING/ERROR/CRITICAL
- LOG_TS_FMT: timestamp format (default "%Y-%m-%d %H:%M:%S.%f", trimmed to ms)
- LOG_COLOR: "0" to disable color
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from datetime import datetime, timezone


try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

# Quiet down underlying HTTP libs if present
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

cv_prompt_id: ContextVar[str | None] = ContextVar("prompt_id", default=None)
cv_event: ContextVar[str | None] = ContextVar("event", default=None)

def get_log_level() -> int:
    return getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

TS_FMT = os.getenv("LOG_TS_FMT", "%Y-%m-%d %H:%M:%S.%f")
USE_COLOR = os.getenv("LOG_COLOR", "1") not in ("0", "false", "False", "no", "No")


class ContextTagFilter(logging.Filter):
    """Injects prompt_id and event contextvars into log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.prompt_id = cv_prompt_id.get() or "-"
        record.event = cv_event.get() or "-"
        return True


class ColumnFormatter(logging.Formatter):
    """Plain (no-color) formatter used for file logs."""
    _STD_ATTRS = {
        "name","msg","args","levelname","levelno","pathname","filename","module",
        "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
        "relativeCreated","thread","threadName","processName","process",
        "message","asctime","taskName"
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime(TS_FMT)[:-3]  # trim to ms
        level = record.levelname
        logger_name = record.name
        task = getattr(record, "taskName", record.threadName)
        prefix = f"{ts} | {level} | {logger_name} | {task} | "

        msg = record.getMessage()

        # tags
        tags = []
        if getattr(record, "prompt_id", "-") != "-":
            tags.append(f"pid={record.prompt_id}")
        if getattr(record, "event", "-") != "-":
            tags.append(f"event={record.event}")
        if tags:
            msg = f"{msg} [{' '.join(tags)}]"

        # extras
        extras = []
        for k, v in record.__dict__.items():
            if k in self._STD_ATTRS or k in ("prompt_id", "event"):
                continue
            if k.startswith("_") or v is None:
                continue
            extras.append((k, v))
        if extras:
            extras.sort(key=lambda kv: kv[0])
            kvs = " ".join(f"{k}={v!r}" for k, v in extras)
            msg = f"{msg} {{{kvs}}}"

        line = prefix + msg
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class ColorColumnFormatter(ColumnFormatter):
    """Colorized formatter for console."""
    RESET = "\x1b[0m"
    COLORS = {
        "DEBUG": "\x1b[36m",
        "INFO":  "\x1b[32m",
        "WARNING":"\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL":"\x1b[35m",
    }
    DIM = "\x1b[2m"

    def format(self, record: logging.LogRecord) -> str:
        if not USE_COLOR or not _is_tty():
            return super().format(record)

        ts = datetime.now(timezone.utc).strftime(TS_FMT)[:-3]
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        logger_name = record.name
        task = getattr(record, "taskName", record.threadName)
        level_colored = f"{color}{record.levelname}{reset}"
        prefix = f"{self.DIM}{ts}{reset} | {level_colored} | {logger_name} | {task} | "

        # rebuild message-only portion to avoid duplicating ColumnFormatter prefix
        message_only = record.getMessage()
        tags = []
        if getattr(record, "prompt_id", "-") != "-":
            tags.append(f"pid={record.prompt_id}")
        if getattr(record, "event", "-") != "-":
            tags.append(f"event={record.event}")
        if tags:
            message_only = f"{message_only} [{' '.join(tags)}]"

        extras = []
        for k, v in record.__dict__.items():
            if k in self._STD_ATTRS or k in ("prompt_id", "event"):
                continue
            if k.startswith("_") or v is None:
                continue
            extras.append((k, v))
        if extras:
            extras.sort(key=lambda kv: kv[0])
            kvs = " ".join(f"{k}={v!r}" for k, v in extras)
            message_only = f"{message_only} {{{kvs}}}"

        line = prefix + message_only
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def setup_logging(log_file: str = "app.log") -> None:
    """
    Configure root logger with color console + rotating file handlers.
    Safe to call multiple times.
    """
    root = logging.getLogger()
    if getattr(root, "_ultrachat_logging_configured", False):
        return

    # Populate record.taskName with current asyncio task name (fallback to thread)
    orig_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        rec = orig_factory(*args, **kwargs)
        try:
            task = asyncio.current_task()
            rec.taskName = task.get_name() if task else None
        except RuntimeError:
            rec.taskName = None
        if rec.taskName is None:
            rec.taskName = rec.threadName
        return rec

    logging.setLogRecordFactory(record_factory)

    tag_filter = ContextTagFilter()

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(ColorColumnFormatter())
    stream_handler.addFilter(tag_filter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(ColumnFormatter())
    file_handler.addFilter(tag_filter)

    root.handlers[:] = [stream_handler, file_handler]
    root.setLevel(get_log_level())
    root._ultrachat_logging_configured = True


def with_prompt_id(pid: str | None):
    """Context manager to tag logs with a prompt_id."""
    class _Ctx:
        def __enter__(self):
            self._tok = cv_prompt_id.set(pid)
            return self
        def __exit__(self, exc_type, exc, tb):
            cv_prompt_id.reset(self._tok)
    return _Ctx()


def with_event(name: str | None):
    """Context manager to tag logs with an event name (e.g., 'produce', 'consume')."""
    class _Ctx:
        def __enter__(self):
            self._tok = cv_event.set(name)
            return self
        def __exit__(self, exc_type, exc, tb):
            cv_event.reset(self._tok)
    return _Ctx()