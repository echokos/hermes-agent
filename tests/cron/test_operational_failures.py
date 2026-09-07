import json

from cron.operational_failures import append_host_failure, append_profile_failure


def test_profile_intake_is_sanitized_and_host_adapter_is_durable(tmp_path):
    profile_job = {
        "id": "github-job",
        "workflow_id": "grace-github",
        "failure_streak": 2,
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    }
    event = append_profile_failure(tmp_path, profile_job, "access_token=very-secret timed out")
    assert event["source_kind"] == "profile_cron"
    assert "very-secret" not in event["sanitized_error"]

    host = append_host_failure(tmp_path, {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
    })
    assert host["source_kind"] == "host_job"
    persisted = (tmp_path / "state" / "operational-failures.jsonl").read_text().splitlines()
    assert json.loads(persisted[0])["event_id"] == host["event_id"]


def test_repeated_host_intake_has_stable_event_identity(tmp_path):
    payload = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
    }
    first = append_host_failure(tmp_path, payload)
    second = append_host_failure(tmp_path, payload)
    assert first["event_id"] == second["event_id"]


def test_host_recovery_is_an_explicit_event(tmp_path):
    failure = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "root", "director": "aurora", "error": "nonzero exit",
    }
    recovered = append_host_failure(tmp_path, {**failure, "outcome": "recovered"})
    assert recovered["status"] == "recovered"
