import importlib.util
from pathlib import Path

import yaml

from tests.workforce_test_helpers import materialize_test_organization


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "workforce_compile", ROOT / "scripts" / "workforce_compile.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_insert_is_idempotent_and_preserves_unmanaged_content():
    original = "# Agent\n\nPrivate voice stays here.\n"
    block = f"{module.BEGIN}\ncontract\n{module.END}"
    first, operation = module.insert_block(original, block)
    second, replacement = module.insert_block(first, block)
    assert operation == "insert"
    assert replacement == "replace"
    assert first == second
    assert "Private voice stays here." in first
    assert first.endswith(original)
    assert first.count(module.BEGIN) == 1


def test_insert_preserves_externalized_contract():
    original = "# Agent\n" + module.EXTERNALIZED + "\n"
    candidate, operation = module.insert_block(original, "generated")
    assert candidate == original
    assert operation == "preserve-externalized"


def test_compile_reports_externalized_contract_reconciliation(tmp_path):
    organization = materialize_test_organization(
        ROOT / "workforce" / "organization.yaml", tmp_path
    )
    source = tmp_path / "workforce-profiles" / "aurora" / "AGENTS.md"
    source.write_text("# Aurora\n" + module.EXTERNALIZED + "\n")
    manifest = module.compile_profiles(
        organization, ROOT / "workforce" / "templates" / "workforce-contract.md", tmp_path / "out"
    )
    entry = next(item for item in manifest["profiles"] if item["agent"] == "aurora")
    assert entry["operation"] == "preserve-externalized"
    assert entry["managed_contract_mode"] == "externalized-reference"
    assert entry["template_reconciliation_required"] is True
    assert (tmp_path / "out" / "aurora" / "AGENTS.md").read_text() == source.read_text()


def test_canonical_compile_includes_active_chloe_and_emma(tmp_path):
    organization = materialize_test_organization(
        ROOT / "workforce" / "organization.yaml", tmp_path
    )
    manifest = module.compile_profiles(
        organization,
        ROOT / "workforce" / "templates" / "workforce-contract.md",
        tmp_path,
    )
    assert len(manifest["profiles"]) == 22
    assert [item["agent"] for item in manifest["profiles"][:2]] == ["aurora", "grace"]
    assert all(item["original_instruction_preserved_as_exact_suffix"] for item in manifest["profiles"])
    chloe = next(item for item in manifest["profiles"] if item["agent"] == "chloe")
    assert chloe["status"] == "active"
    assert chloe["source_kind"] == "live-profile"
    expected_chloe = tmp_path / "workforce-profiles" / "chloe" / "AGENTS.md"
    assert chloe["source"] == str(expected_chloe)
    assert chloe["target"] == str(expected_chloe)
    text = (tmp_path / "chloe" / "AGENTS.md").read_text()
    assert "may not interpret, rank, recommend" in text
    assert module.BEGIN in text
    emma = next(item for item in manifest["profiles"] if item["agent"] == "emma")
    assert emma["status"] == "active"
    assert emma["source_kind"] == "live-profile"

    aurora_text = (tmp_path / "aurora" / "AGENTS.md").read_text()
    assert "translate that intent into execution" in aurora_text
    assert "delegate the rest to the right owner" in aurora_text
    assert "follow through until the outcome is completed" in aurora_text
    assert "I do not send a signal to myself" in aurora_text
    assert "Proactivity begins with understanding" in aurora_text
    assert "requirements conversation" in aurora_text
    assert "do not launch a production graph or downstream task chain" in aurora_text
    assert "submit one concrete workforce signal for Aurora" not in aurora_text
    assert "Root is a configured, active worker under the `main` profile" in aurora_text
    assert "I assign DigitalOcean" in aurora_text
    assert "Close every accepted commitment" in aurora_text
    assert "report_to_origin: true" in aurora_text
    assert "coordination: {}` before any delegation" in aurora_text
    assert "synchronous answers, exploration or discovery" in aurora_text
    assert "exact DM, room thread, or conversation" in aurora_text

    alina_text = (tmp_path / "alina" / "AGENTS.md").read_text()
    assert "I do not own DigitalOcean" in alina_text
    assert "Those route to Root (`main`)" in alina_text

    root_text = (tmp_path / "root" / "AGENTS.md").read_text()
    assert "active Hermes worker under profile directory `main`" in root_text
    assert "I own DigitalOcean" in root_text
    assert "coordination: {}` before any delegation" not in root_text


def test_planned_profile_can_use_owner_only_private_source(tmp_path):
    organization_fixture = materialize_test_organization(
        ROOT / "workforce" / "organization.yaml", tmp_path
    )
    organization = yaml.safe_load(
        organization_fixture.read_text(encoding="utf-8")
    )
    chloe = next(item for item in organization["agents"] if item["agent"] == "chloe")
    chloe["status"] = "planned"
    chloe["profile_path"] = str(tmp_path / "missing-live-profile")
    organization_path = tmp_path / "organization.yaml"
    organization_path.write_text(
        yaml.safe_dump(organization, sort_keys=False), encoding="utf-8"
    )
    planned = tmp_path / "planned"
    source = planned / "chloe" / "AGENTS.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Chloe private source\n\nHer established voice.\n")
    output = tmp_path / "output"
    manifest = module.compile_profiles(
        organization_path,
        ROOT / "workforce" / "templates" / "workforce-contract.md",
        output,
        planned_source_root=planned,
    )
    entry = next(item for item in manifest["profiles"] if item["agent"] == "chloe")
    assert entry["source_kind"] == "planned-private-source"
    assert entry["target"] == str(tmp_path / "missing-live-profile" / "AGENTS.md")
    assert entry["source_sha256"]
    assert (output / "chloe" / "AGENTS.md").read_text().endswith(
        source.read_text()
    )


def test_generated_reference_excludes_friends_from_dispatch():
    rendered = module.render_organization_reference(
        ROOT / "workforce" / "organization.yaml"
    )
    assert "The workforce is proactive" in rendered
    assert "Amy (`amy`) | friend" in rendered
    assert "friends, not operational assignees" in rendered
