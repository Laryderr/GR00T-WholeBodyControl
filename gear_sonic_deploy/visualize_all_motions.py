#!/usr/bin/env python3
"""
Batch launcher for visualize_motion.py.

This script finds motion subdirectories and launches visualize_motion.py
one-by-one. Close each MuJoCo window to continue to the next motion.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def is_motion_dir(path: Path) -> bool:
    required = ("joint_pos.csv", "body_pos.csv", "body_quat.csv")
    return path.is_dir() and all((path / name).is_file() for name in required)


def discover_motion_dirs(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        # Include any directory in the tree that looks like a motion folder.
        candidates = [p for p in root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    return sorted([p for p in candidates if is_motion_dir(p)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop over motion folders and launch visualize_motion.py for each."
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        required=True,
        help="Root directory containing one or more motion subdirectories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for motion subdirectories.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Play in reverse alphabetical order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of motions to play (0 means all).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this index in the discovered motion list (0-based).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run visualize_motion.py.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next motion even if one launch fails.",
    )
    parser.add_argument(
        "--wait-key",
        action="store_true",
        help="After each motion, wait for terminal input: Enter=next, q=quit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned motion order without launching.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    motion_root = args.motion_root.resolve()
    script_dir = Path(__file__).resolve().parent
    visualizer = script_dir / "visualize_motion.py"

    if not motion_root.is_dir():
        print(f"[ERROR] motion root not found: {motion_root}")
        return 2
    if not visualizer.is_file():
        print(f"[ERROR] visualize_motion.py not found next to this script: {visualizer}")
        return 2

    motions = discover_motion_dirs(motion_root, recursive=args.recursive)
    if args.reverse:
        motions = list(reversed(motions))
    if args.start_index > 0:
        motions = motions[args.start_index :]
    if args.limit > 0:
        motions = motions[: args.limit]

    if not motions:
        print("[ERROR] no motion directories found.")
        print("        Expected each motion directory to contain:")
        print("        joint_pos.csv, body_pos.csv, body_quat.csv")
        return 2

    print(f"[INFO] Found {len(motions)} motion(s).")
    for i, motion_dir in enumerate(motions, start=1):
        print(f"  {i:03d}. {motion_dir}")

    if args.dry_run:
        return 0

    for i, motion_dir in enumerate(motions, start=1):
        cmd = [args.python, str(visualizer), "--motion_dir", str(motion_dir)]
        print(f"\n[RUN {i}/{len(motions)}] {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(script_dir)).returncode
        if rc != 0:
            print(f"[ERROR] visualizer exited with code {rc} for: {motion_dir}")
            if not args.continue_on_error:
                return rc

        if args.wait_key and i < len(motions):
            key = input("[NEXT] Press Enter for next motion, or type 'q' to quit: ").strip().lower()
            if key == "q":
                print("[INFO] Stopped by user.")
                return 0

    print("\n[INFO] All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
