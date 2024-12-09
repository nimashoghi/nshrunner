from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import execute


def main() -> int:
    """Main entry point for picklerunner command line tool.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Execute pickled Python functions")
    parser.add_argument("fn_path", type=Path, help="Path to pickled function")
    parser.add_argument("args_path", type=Path, help="Path to pickled arguments")

    args = parser.parse_args()

    try:
        result = execute(args.fn_path, args.args_path)
        print(f"Result: {result}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
