from pathlib import Path

import pytest
import yaml

from hermes_cli.workforce_org import (
    WorkforceOrganizationAbsentError,
    WorkforceOrganizationError,
    load_organization,
    validate_workflow_profiles,
)


SOURCE = Path(__file__).parents[2] / "workforce" / "organization.yaml"


def test_canonical_organization_is_reciprocal_and_excludes_friends():
    org = load_organization(SOURCE)
    assert len(org.operational_agents()) == 22
    assert org.get("root").profile_path.endswith("/profiles/main")
    assert org.resolve_profile("main").agent == "root"
    assert org.validate_execution_profile("main").agent == "root"
    assert org.technical_ownership["external_cloud_server_app_operations"] == "root"
    assert org.technical_ownership["domains_dns_ssl"] == "root"
    assert org.technical_ownership["local_agent_host_operations"] == "alina"
    assert org.technical_ownership["local_host_install_service_activation"] == "alina"
    assert "host_install_service_activation" not in org.technical_ownership
    assert org.get("chloe").manager == "aurora"
    assert org.get("amy").operational is False
    with pytest.raises(WorkforceOrganizationError):
        org.validate_execution_profile("amy")
    with pytest.raises(WorkforceOrganizationError):
        org.validate_execution_profile("default")


def test_unknown_workflow_owner_fails_closed():
    org = load_organization(SOURCE)
    with pytest.raises(WorkforceOrganizationError):
        validate_workflow_profiles(org, "stranger", [])


def test_organization_load_distinguishes_absent_from_present_invalid(tmp_path):
    path = tmp_path / "organization.yaml"
    with pytest.raises(WorkforceOrganizationAbsentError):
        load_organization(path)

    path.write_text("not: [valid", encoding="utf-8")
    with pytest.raises(WorkforceOrganizationError) as invalid:
        load_organization(path)
    assert not isinstance(invalid.value, WorkforceOrganizationAbsentError)


def test_manager_cycle_is_rejected(tmp_path):
    data = yaml.safe_load(SOURCE.read_text())
    by_id = {item["agent"]: item for item in data["agents"]}
    by_id["aurora"]["manager"] = "chloe"
    by_id["chloe"]["direct_reports"] = ["aurora"]
    path = tmp_path / "organization.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(WorkforceOrganizationError, match="cycle"):
        load_organization(path)


def test_active_profile_existence_can_be_enforced(tmp_path):
    data = yaml.safe_load(SOURCE.read_text())
    by_id = {item["agent"]: item for item in data["agents"]}
    by_id["aurora"]["profile_path"] = str(tmp_path / "missing")
    path = tmp_path / "organization.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(WorkforceOrganizationError, match="active profile path"):
        load_organization(path, validate_profiles=True)
