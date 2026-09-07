"""Durable delivery-ledger behavior for ambiguity-safe cron sends."""

from __future__ import annotations


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_delivery_ledger_preserves_three_terminal_states_and_blocks_replay(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    for index, terminal_state in enumerate(
        ("confirmed_sent", "confirmed_absent", "unknown")
    ):
        identity = f"delivery-{index}"
        claimed = executions.claim_delivery(
            identity,
            execution_id="execution-1",
            job_id="job-1",
            platform="photon",
            target_fingerprint="safe-target-fingerprint",
        )
        assert claimed["claimed"] is True
        assert claimed["state"] == "pending"

        resolved = executions.resolve_delivery(identity, terminal_state)
        assert resolved["state"] == terminal_state

        replay = executions.claim_delivery(
            identity,
            execution_id="execution-1",
            job_id="job-1",
            platform="photon",
            target_fingerprint="safe-target-fingerprint",
        )
        assert replay == {"claimed": False, "state": terminal_state}


def test_delivery_ledger_allows_only_one_fallback_after_confirmed_absence(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    identity = "delivery-fallback"

    assert executions.claim_delivery(
        identity,
        execution_id="execution-2",
        job_id="job-2",
        platform="photon",
        target_fingerprint="safe-target-fingerprint",
    ) == {"claimed": True, "state": "pending"}
    assert executions.resolve_delivery(identity, "confirmed_absent")["state"] == "confirmed_absent"

    fallback = executions.claim_delivery(
        identity,
        execution_id="execution-2",
        job_id="job-2",
        platform="photon",
        target_fingerprint="safe-target-fingerprint",
        fallback=True,
    )
    assert fallback == {"claimed": True, "state": "pending"}
    assert executions.resolve_delivery(identity, "confirmed_sent")["state"] == "confirmed_sent"

    assert executions.claim_delivery(
        identity,
        execution_id="execution-2",
        job_id="job-2",
        platform="photon",
        target_fingerprint="safe-target-fingerprint",
        fallback=True,
    ) == {"claimed": False, "state": "confirmed_sent"}
