"""Standalone recount script.

Usage:
    python utils/recount.py --run-dir data/artifacts/detect_file_20240119_120000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mx_tracker.recount import recount  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recount laps from events.jsonl → results.csv")
    parser.add_argument("--run-dir", required=True, help="Run directory produced by detect")
    parser.add_argument("--race-start-sec", type=float, default=0.0, help="Seconds from video start to race start (default: 0)")
    parser.add_argument("--race-start-at", default=None, help="Wall-clock race start ISO time (e.g. 10:31:00); requires run_info.json")
    args = parser.parse_args()
    recount(args.run_dir, race_start_sec=args.race_start_sec, race_start_at=args.race_start_at)
