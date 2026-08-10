#!/usr/bin/env python
"""CLI entry: python scripts/run.py [--config PATH]."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install when src/ is on the path.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvar_psha.run_experiment import main

if __name__ == "__main__":
    main()
