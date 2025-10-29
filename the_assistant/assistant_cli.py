"""Command line interface for the memory containment vessel."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

from .memory_vessel import MemoryContainmentVessel, MemoryFragment


def _parse_metadata(metadata: str | None) -> dict[str, object]:
    if not metadata:
        return {}
    try:
        return json.loads(metadata)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid metadata JSON: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path(".assistant_memory/memories.db"),
        help="Path to the SQLite file used to persist assistant memories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Store command
    store_parser = subparsers.add_parser("store", help="Persist a memory fragment")
    store_parser.add_argument("content", help="Textual content of the fragment")
    store_parser.add_argument(
        "--importance",
        type=float,
        default=0.5,
        help="Relative importance in the range [0, 1]",
    )
    store_parser.add_argument(
        "--tags",
        nargs="*",
        default=(),
        help="Optional tags to associate with the fragment",
    )
    store_parser.add_argument(
        "--metadata",
        help="Optional JSON blob with arbitrary metadata for the fragment",
    )

    # Recent command
    recent_parser = subparsers.add_parser("recent", help="List the most recent memories")
    recent_parser.add_argument("--limit", type=int, default=10)
    recent_parser.add_argument("--tags", nargs="*", default=())

    # Search command
    search_parser = subparsers.add_parser("search", help="Search stored memories")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--tags", nargs="*", default=())
    search_parser.add_argument("--min-importance", type=float)

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Show a summary")
    summarize_parser.add_argument("--limit", type=int, default=5)

    # Purge command
    purge_parser = subparsers.add_parser("purge", help="Remove old memories")
    purge_parser.add_argument("--before", help="ISO timestamp; delete memories before this time")
    purge_parser.add_argument(
        "--older-than",
        type=str,
        help="Duration like '7d', '12h', '30m' representing an age threshold.",
    )
    purge_parser.add_argument("--max-records", type=int)
    purge_parser.add_argument("--tags", nargs="*", default=())

    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories to JSONL")
    export_parser.add_argument("destination", type=Path)
    export_parser.add_argument("--tags", nargs="*", default=())

    return parser


def parse_duration(value: str) -> timedelta:
    units = {"m": 60, "h": 3600, "d": 86400}
    if len(value) < 2:
        raise ValueError("Duration must include a numeric value and unit suffix")
    suffix = value[-1].lower()
    if suffix not in units:
        raise ValueError("Duration must end with m, h, or d")
    try:
        amount = float(value[:-1])
    except ValueError as exc:
        raise ValueError("Duration must start with a numeric value") from exc
    if amount < 0:
        raise ValueError("Duration value must be non-negative")
    return timedelta(seconds=amount * units[suffix])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    vessel = MemoryContainmentVessel(args.storage)

    if args.command == "store":
        metadata = _parse_metadata(args.metadata)
        fragment = MemoryFragment(
            content=args.content,
            importance=args.importance,
            metadata=metadata,
        )
        row_id = vessel.store(fragment, tags=args.tags)
        print(row_id)
        return 0

    if args.command == "recent":
        for row in vessel.iter_recent(limit=args.limit, tags=args.tags):
            print(f"[{row['created_at']}] ({row['importance']:.2f}) {row['content']}")
        return 0

    if args.command == "search":
        rows = vessel.search(
            args.query,
            limit=args.limit,
            tags=args.tags,
            min_importance=args.min_importance,
        )
        for row in rows:
            print(f"[{row['created_at']}] ({row['importance']:.2f}) {row['content']}")
        return 0

    if args.command == "summarize":
        print(vessel.summarize(limit=args.limit))
        return 0

    if args.command == "purge":
        try:
            before = datetime.fromisoformat(args.before) if args.before else None
        except ValueError as exc:  # pragma: no cover - CLI guard
            parser.error(f"Invalid --before timestamp: {exc}")
        try:
            older_than = parse_duration(args.older_than) if args.older_than else None
        except ValueError as exc:  # pragma: no cover - CLI guard
            parser.error(f"Invalid --older-than duration: {exc}")
        deleted = vessel.purge(
            before=before,
            older_than=older_than,
            max_records=args.max_records,
            tags=args.tags,
        )
        print(deleted)
        return 0

    if args.command == "export":
        path = vessel.export(args.destination, tags=args.tags)
        print(path)
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
