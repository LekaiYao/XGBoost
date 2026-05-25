#!/usr/bin/env python3
"""Compatibility entrypoint; use scripts/cleanup/cleanup_selected_events.py."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "cleanup" / "cleanup_selected_events.py"
    runpy.run_path(str(target), run_name="__main__")
