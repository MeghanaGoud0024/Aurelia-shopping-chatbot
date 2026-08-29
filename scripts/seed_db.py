#!/usr/bin/env python3
"""Create the schema and load the synthetic dataset.

Usage:
    python scripts/seed_db.py            # seed if empty
    python scripts/seed_db.py --force    # wipe and regenerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                      # noqa: E402
from app.db.seed import seed_database                # noqa: E402
from app.db.session import init_db, session_scope    # noqa: E402
from app.logging_setup import configure_logging      # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the Aurelia database.")
    parser.add_argument("--force", action="store_true", help="Delete existing rows and regenerate.")
    args = parser.parse_args()

    configure_logging(settings.log_level)
    init_db()
    with session_scope() as session:
        counts = seed_database(session, force=args.force)

    width = max(len(k) for k in counts)
    print("\n  Aurelia database ready\n  " + "-" * (width + 12))
    for key, value in counts.items():
        print(f"  {key.replace('_', ' '):<{width}}  {value:>7,}")
    print(f"\n  DSN: {settings.database_url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
