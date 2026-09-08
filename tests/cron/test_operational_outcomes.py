"""Operational returns stay bound to their actual routine destinations."""

from copy import deepcopy
import json

import pytest

from cron.operational_outcomes import capture_outcome_notice, validate_outcome_notice


@pytest.fixture
def job():
    return {
        "id": "daily-note", "deliver": "telegram:12345",
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
            "return_outcome_to_origin": True,
        },
    }


def test_capture_uses_real_cron_route_resolution_without_mutating_job(job):
    before = deepcopy(job)
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="execution-one")

    assert job == before
    assert notice["status"] == "bound"
    assert notice["source_profile"] == "grace"
    assert notice["routes"][0]["platform"] == "telegram"
    assert notice["routes"][0]["chat_id"] == "12345"
    assert validate_outcome_notice(notice, job, source_profile="grace") == notice


@pytest.mark.parametrize("opt_in", [None, False, "true", 1])
def test_outcome_notice_requires_explicit_operator_opt_in(job, opt_in):
    job["failure_ownership"]["return_outcome_to_origin"] = opt_in
    assert capture_outcome_notice(job, source_profile="grace", execution_id="one") is None


@pytest.mark.parametrize("field,value", [("deliver", "telegram:67890"), ("id", "another-job")])
def test_source_change_does_not_silently_reroute(job, field, value):
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="one")
    job[field] = value
    with pytest.raises(ValueError, match="source route changed"):
        validate_outcome_notice(notice, job, source_profile="grace")


def test_profile_and_ownership_changes_invalidate_notice(job):
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="one")
    with pytest.raises(ValueError, match="source route changed"):
        validate_outcome_notice(notice, job, source_profile="emily")
    job["failure_ownership"]["director"] = "another-director"
    with pytest.raises(ValueError, match="source route changed"):
        validate_outcome_notice(notice, job, source_profile="grace")


def test_only_route_metadata_is_retained(job):
    job["prompt"] = "private task contents"
    job["token"] = "never-retain-this"
    job["origin"] = {"platform": "telegram", "chat_id": "12345", "private": "hidden"}
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="one")
    encoded = json.dumps(notice)
    for private in ("private task contents", "never-retain-this", "hidden"):
        assert private not in encoded


def test_multiple_explicit_destinations_have_distinct_stable_route_keys(job):
    job["deliver"] = "telegram:12345,telegram:67890,telegram:12345"
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="one")
    assert len(notice["routes"]) == 2
    assert len({route["route_key"] for route in notice["routes"]}) == 2


@pytest.mark.parametrize("profile", ["../grace", "grace\nother", ""])
def test_malformed_profile_is_unroutable_evidence_not_an_intake_exception(job, profile):
    notice = capture_outcome_notice(job, source_profile=profile, execution_id="one")
    assert notice == {"version": 1, "status": "unroutable", "reason": "invalid_or_missing_source_route"}


def test_local_delivery_has_no_external_outcome_authority(job):
    job["deliver"] = "local"
    notice = capture_outcome_notice(job, source_profile="grace", execution_id="one")
    assert notice["status"] == "unroutable"
    with pytest.raises(ValueError, match="no bound source route"):
        validate_outcome_notice(notice, job, source_profile="grace")
