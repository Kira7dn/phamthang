#!/usr/bin/env python3
"""
Clear all __pycache__ folders and *.pyc/*.pyo files under a target path (default: repo root).

Usage:
  venv/bin/python scripts/clear_pycache.py              # clean current repo
  venv/bin/python scripts/clear_pycache.py --path backend  # clean only backend/
  venv/bin/python scripts/clear_pycache.py --dry-run     # show what would be removed
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def iter_bytecode_paths(root: Path):
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            yield p
    for ext in ("*.pyc", "*.pyo"):
        for p in root.rglob(ext):
            if p.is_file():
                yield p


def remove_path(p: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"DRY-RUN would remove: {p}")
        return
    try:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        print(f"Removed: {p}")
    except Exception as e:
        print(f"Failed to remove {p}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clear Python bytecode caches")
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Root path to clean (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete, only print what would be removed",
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    print(f"Cleaning bytecode under: {root}")

    for p in iter_bytecode_paths(root):
        remove_path(p, dry_run=args.dry_run)

    if args.dry_run:
        print("Done (dry-run). Nothing was deleted.")
    else:
        print("Done. Bytecode caches removed.")


if __name__ == "__main__":
    main()
