from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "workforce_aurora_coordination_stage",
    ROOT / "scripts/workforce_aurora_coordination_stage.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _profile(tmp_path: Path) -> tuple[Path, str]:
    profile = tmp_path / "aurora"
    profile.mkdir()
    original = (
        "# Aurora Operating Core\n"
        f"{module.EXTERNALIZED}\n\n"
        "Private voice and authority remain unchanged.\n\n"
        "## Routing and privacy\n\n"
        "Existing routing rules.\n"
    )
    (profile / "AGENTS.md").write_text(original, encoding="utf-8")
    return profile, original


def test_stages_only_compact_aurora_block_and_preserves_source(tmp_path):
    profile, original = _profile(tmp_path)
    output = tmp_path / "stage"

    manifest = module.stage_profile(profile=profile, output=output)
    candidate = (output / "AGENTS.md").read_text(encoding="utf-8")

    assert (profile / "AGENTS.md").read_text(encoding="utf-8") == original
    assert manifest["source_unchanged"] is True
    assert manifest["operation"] == "insert"
    assert candidate.count(module.BEGIN) == 1
    assert candidate.index(module.BEGIN) < candidate.index(module.ROUTING_ANCHOR)
    assert "`report_to_origin: true`" in candidate
    assert "`coordination: {}`" in candidate
    assert "Contract version:" not in candidate
    assert candidate.startswith(original.split(module.ROUTING_ANCHOR)[0])
    assert candidate.endswith(
        module.ROUTING_ANCHOR + original.split(module.ROUTING_ANCHOR, 1)[1]
    )


def test_restage_is_idempotent_and_replaces_only_managed_block(tmp_path):
    profile, _ = _profile(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    module.stage_profile(profile=profile, output=first)

    staged_profile = tmp_path / "staged-aurora"
    staged_profile.mkdir()
    (staged_profile / "AGENTS.md").write_bytes((first / "AGENTS.md").read_bytes())
    manifest = module.stage_profile(profile=staged_profile, output=second)

    assert manifest["operation"] == "replace"
    assert (second / "AGENTS.md").read_bytes() == (first / "AGENTS.md").read_bytes()


def test_rejects_noncompact_or_non_aurora_target(tmp_path):
    profile = tmp_path / "not-aurora"
    profile.mkdir()
    (profile / "AGENTS.md").write_text(
        "# Another Agent\n\n## Routing and privacy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compact externalized"):
        module.stage_profile(profile=profile, output=tmp_path / "stage")
