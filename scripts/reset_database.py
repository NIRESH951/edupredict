"""
Delete the database and trained models, then re-create the demo data.

Usage:  python scripts/reset_database.py [--yes] [--no-train]
"""
import argparse
import shutil

import _bootstrap  # noqa: F401

import config
from core.seed import seed_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--no-train", action="store_true", help="skip model training")
    args = parser.parse_args()
    if not args.yes:
        answer = input(f"This deletes {config.DB_PATH} and all trained models. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Cancelled.")
            return
    for suffix in ("", "-wal", "-shm"):
        path = config.DB_PATH.with_name(config.DB_PATH.name + suffix)
        if path.exists():
            path.unlink()
    for folder in config.MODEL_DIR.glob("v*"):
        shutil.rmtree(folder, ignore_errors=True)
    summary = seed_database(progress=lambda f, m: print(f"[{f:4.0%}] {m}"), train=not args.no_train)
    print("Done:", summary)


if __name__ == "__main__":
    main()
