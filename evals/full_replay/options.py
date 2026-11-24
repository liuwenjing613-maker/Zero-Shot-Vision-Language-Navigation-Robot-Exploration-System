"""Paths shared by the standalone replay exporter."""
import argparse
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]


def options():
    parser = argparse.ArgumentParser(
        description="Export interactive navigation episodes"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing completed episode folders",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="MP3D scenes directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Dedicated export directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional ordered JSON list with id, label, goal, setting and audit paths",
    )
    parser.add_argument(
        "--memory-trial", type=Path, help="Matched recall report.json to embed"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    for key in ("runs_root", "dataset", "output", "manifest", "memory_trial"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    return args


def cases(args):
    if args.manifest:
        return json.loads(args.manifest.read_text())
    return [{"id": p.parent.name} for p in sorted(args.runs_root.glob("*/trace.json"))]
