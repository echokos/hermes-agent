import json

from cron.operational_failures import (
    append_host_failure,
    append_profile_failure,
    host_failure_event,
)


def test_profile_intake_is_sanitized_and_host_adapter_is_durable(tmp_path):
    profile_job = {
        "id": "github-job",
        "workflow_id": "grace-github",
        "failure_streak": 2,
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    }
    event = append_profile_failure(
        tmp_path,
        profile_job,
        "access_token=very-secret timed out",
        execution_id="profile-run-1",
    )
    assert event["source_kind"] == "profile_cron"
    assert "very-secret" not in event["sanitized_error"]

    host = append_host_failure(tmp_path, {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
        "execution_id": "host-run-1",
    })
    assert host["source_kind"] == "host_job"
    persisted = (tmp_path / "state" / "operational-failures.jsonl").read_text().splitlines()
    assert json.loads(persisted[0])["event_id"] == host["event_id"]


def test_host_intake_identity_is_stable_per_execution_and_unique_across_runs(tmp_path):
    payload = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
        "execution_id": "host-run-1",
    }
    first = append_host_failure(tmp_path, payload)
    second = append_host_failure(tmp_path, payload)
    assert first["event_id"] == second["event_id"]
    third = append_host_failure(tmp_path, {**payload, "execution_id": "host-run-2"})
    assert third["event_id"] != first["event_id"]


def test_host_recovery_is_an_explicit_event(tmp_path):
    failure = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
        "execution_id": "host-run-1",
    }
    recovered = append_host_failure(tmp_path, {
        **failure, "execution_id": "host-run-2", "outcome": "recovered",
    })
    assert recovered["status"] == "recovered"
    assert recovered["recovery_successes_required"] == 2


def test_execution_identity_is_required(tmp_path):
    payload = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
    }
    try:
        append_host_failure(tmp_path, payload)
    except ValueError as exc:
        assert "execution_id" in str(exc)
    else:
        raise AssertionError("host intake accepted an event without execution identity")


def test_host_millisecond_timestamp_produces_current_deadlines():
    occurred_at_ms = 1_800_000_000_123
    event = host_failure_event({
        "workflow_id": "host-check", "source_id": "host-check",
        "technical_owner": "root", "director": "aurora",
        "execution_id": "host-run-ms", "error": "failed",
        "occurred_at": occurred_at_ms,
    })
    assert event["recorded_at"] == occurred_at_ms // 1000
    assert event["ack_deadline"] == occurred_at_ms // 1000 + 15 * 60
