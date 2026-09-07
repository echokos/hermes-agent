#!/usr/bin/env python3
"""Stage Aurora's compact asynchronous-request intake contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "workforce/templates/aurora-coordination-intake.md"
BEGIN = "<!-- BEGIN MANAGED AURORA ASYNC INTAKE -->"
END = "<!-- END MANAGED AURORA ASYNC INTAKE -->"
EXTERNALIZED = "<!-- MANAGED WORKFORCE CONTRACT: EXTERNALIZED -->"
ROUTING_ANCHOR = "## Routing and privacy"
BLOCK_RE = re.compile(rf"{re.escape(BEGIN)}.*?{re.escape(END)}", re.DOTALL)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: bytes, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_block(path: Path) -> str:
    block = path.read_text(encoding="utf-8").strip()
    if block.count(BEGIN) != 1 or block.count(END) != 1:
        raise ValueError("Aurora intake template must contain one managed block")
    required = (
        "explicitly accept a clear Elliott request",
        "`report_to_origin: true`",
        "`coordination: {}`",
        "before any delegation",
        "synchronous answers, exploration or discovery",
    )
    if any(value not in block for value in required):
        raise ValueError("Aurora intake template is missing a required boundary")
    return block


def render_candidate(original: str, block: str) -> tuple[str, str, str, str]:
    """Return candidate, operation, and untouched prefix/suffix."""
    matches = list(BLOCK_RE.finditer(original))
    if len(matches) > 1:
        raise ValueError("Aurora profile contains duplicate intake blocks")
    if matches:
        match = matches[0]
        prefix = original[: match.start()]
        suffix = original[match.end() :]
        candidate = prefix + block + suffix
        operation = "replace"
    else:
        anchor = original.find(ROUTING_ANCHOR)
        if anchor < 0:
            raise ValueError("Aurora profile is missing the routing section anchor")
        prefix = original[:anchor]
        suffix = original[anchor:]
        candidate = prefix + block + "\n\n" + suffix
        operation = "insert"
    if candidate.count(BEGIN) != 1 or candidate.count(END) != 1:
        raise ValueError("staged Aurora profile does not contain one intake block")
    return candidate, operation, prefix, suffix


def stage_profile(
    *,
    profile: Path,
    output: Path,
    template: Path = DEFAULT_TEMPLATE,
) -> dict[str, object]:
    source = profile.expanduser().resolve() / "AGENTS.md"
    if not source.is_file():
        raise FileNotFoundError(f"Aurora instruction file is missing: {source}")
    original_bytes = source.read_bytes()
    original = original_bytes.decode("utf-8-sig")
    if not original.startswith("# Aurora") or EXTERNALIZED not in original:
        raise ValueError(
            "target must be Aurora's compact externalized instruction file"
        )
    block = _load_block(template.expanduser().resolve())
    candidate, operation, prefix, suffix = render_candidate(original, block)
    repeated, _, _, _ = render_candidate(candidate, block)
    if repeated != candidate:
        raise RuntimeError("Aurora intake staging is not idempotent")

    output_dir = output.expanduser().absolute()
    if output_dir.is_symlink() or (
        output_dir.exists() and not output_dir.is_dir()
    ):
        raise ValueError("staging output must be a real directory")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    candidate_path = output_dir / "AGENTS.md"
    manifest_path = output_dir / "manifest.json"
    if candidate_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("staging output files must not be symbolic links")
    if candidate_path.resolve() == source:
        raise ValueError("staging output must not be the live Aurora profile")
    candidate_bytes = candidate.encode("utf-8")
    _atomic_write(candidate_path, candidate_bytes, 0o600)
    manifest = {
        "profile": "aurora",
        "source": str(source),
        "source_sha256": _sha256(original_bytes),
        "candidate": str(candidate_path),
        "candidate_sha256": _sha256(candidate_bytes),
        "operation": operation,
        "managed_block_count": 1,
        "untouched_prefix_sha256": _sha256(prefix.encode("utf-8")),
        "untouched_suffix_sha256": _sha256(suffix.encode("utf-8")),
        "source_unchanged": source.read_bytes() == original_bytes,
        "idempotent": True,
    }
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        0o600,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    print(json.dumps(stage_profile(
        profile=args.profile,
        output=args.output,
        template=args.template,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
