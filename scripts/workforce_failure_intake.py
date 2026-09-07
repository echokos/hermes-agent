#!/usr/bin/env python3
"""Append one sanitized, ownership-declared host-job failure to Hermes intake.

This is an opt-in adapter for deterministic host jobs. It does not send chat,
edit a crontab, or create Kanban work; the existing workforce health monitor
does those idempotently on its next timer tick.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cron.operational_failures import append_host_failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None,
                        help="JSON object path; defaults to stdin")
    args = parser.parse_args(argv)
    raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"input must be one JSON object: {exc.msg}")
    if not isinstance(payload, dict):
        parser.error("input must be one JSON object")
    event = append_host_failure(args.hermes_home, payload)
    print(json.dumps({"event_id": event["event_id"], "status": event["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
