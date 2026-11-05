#!/usr/bin/env python3
"""Merge LeRobot datasets (toyblocks.vecgen.recording.merge; replaces TAMP/factory/merge_success.py).

  python scripts/datagen/merge_datasets.py --input_dirs <dir> [<dir> ...] --output_dir <merged> \
      [--cameras all] [--num_workers 16] [--union_schema] [--validate [--validate_mode full]]
  python scripts/datagen/merge_datasets.py --recompute-stats <merged dataset>
"""

import sys
from pathlib import Path

# This checkout's package even when the venv's ToyBlocks is another checkout (TAMP's rerender
# orchestrator runs this script from its --code_dir). Spawned merge workers inherit sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toyblocks.vecgen.recording.merge import run_cli  # noqa: E402

if __name__ == "__main__":
    run_cli()
