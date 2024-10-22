from __future__ import annotations

import logging
import os
import sys


def print_environment_info(log: logging.Logger | None = None):
    if log is None:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)

    log_message_lines: list[str] = []
    log_message_lines.append("Python executable: " + sys.executable)
    log_message_lines.append("Python version: " + sys.version)
    log_message_lines.append("Python prefix: " + sys.prefix)
    log_message_lines.append("Python path:")
    for path in sys.path:
        log_message_lines.append(f"  {path}")

    log_message_lines.append("Environment variables:")
    for key, value in os.environ.items():
        log_message_lines.append(f"  {key}={value}")

    log_message_lines.append("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        log_message_lines.append(f"  {i}: {arg}")

    log.critical("\n".join(log_message_lines))


if __name__ == "__main__":
    print_environment_info()
