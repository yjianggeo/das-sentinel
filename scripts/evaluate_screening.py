"""Run from the worktree; outputs an independent report and optional review sidecars."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from das_sentinel.screening_evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--measurements-only", action="store_true")
    parser.add_argument("--background-limit", type=int, default=4)
    args = parser.parse_args()
    result = run_evaluation(args.database, args.output, measurement_dir=args.measurement_dir,
                            event_limit=args.limit, measurements_only=args.measurements_only,
                            background_limit=args.background_limit)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
