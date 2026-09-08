"""Reserved decisions use source review without pretending an incident recovered."""

import json
import time
from datetime import datetime, timezone

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.workforce_handoffs import acknowledge_handoff, create_handoff
from tests.hermes_cli.test_coordination_requests import kanban_home, organization


def owner_run(conn, organization):
    now = int(time.time())
    iso = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
    handoff = create_handoff(
        conn, source_agent="aurora", target_agent="builder",
        expected_outcome="Investigate required integration authorization",
        acceptance_test="Repair verified or exact reserved action source-reviewed",
        evidence_references=["execution:failure-one"],
        acknowledgment_deadline=iso(now + 60), checkpoint_at=iso(now + 600),
        organization=organization, requires_source_acceptance=True,
        context={"kind": "owned_operational_failure", "technical_owner": "builder",
                 "director": "aurora", "workflow_id": "daily-triage", "event_id": "failure-one"},
    )
    task_id = handoff["task_id"]
    request = kb.create_owned_failure_coordination_request(
        conn, root_task_id=task_id, organization=organization, now=now,
    )
    acknowledge_handoff(conn, task_id, actor="builder", organization=organization, now=now)
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "workforce_handoff_recovery_required", {
            "failure_event_id": "failure-one", "failure_order": 10, "required_successes": 2,
        })
    task, _ = kb.claim_task_for_dispatch(conn, task_id, organization=organization, now=now)
    assert task is not None
    return task, request


def proposal(**changes):
    return {
        "failure_event_id": "failure-one", "action_kind": "user_reauthentication",
        "integration": "Nirvana", "account": "Elliott's Nirvana account",
        "action": "Reconnect the existing Nirvana application in account settings.",
        "evidence_references": ["execution:failure-one", "artifact:auth-diagnosis"],
        **changes,
    }


def send_to_review(conn, owner):
    ok, reason = kb.request_review(
        conn, owner.id, summary="Task reads require a reserved user action, not a claimed repair.",
        reserved_decision=proposal(), expected_run_id=owner.current_run_id, with_reason=True,
    )
    assert ok, reason
    snapshot = kb.owned_failure_decision_review_snapshot(conn, owner.id)
    assert snapshot is not None
    return snapshot


@pytest.mark.parametrize("outcome", ["accepted", "rejected"])
def test_current_owner_to_aurora_review_stops_unrepaired_without_budget_reset(organization, monkeypatch, outcome):
    from agent.coordination_budget import scoped_coordination_budget
    from agent.conversation_loop import _terminal_review_tool_round_completed

    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        work_limit = request.max_model_calls - request.final_model_call_reserve
        for _ in range(work_limit):
            kb.charge_coordination_model_call(conn, request.id, task_id=owner.id)
        with pytest.raises(kb.CoordinationBudgetExceeded):
            kb.charge_coordination_model_call(conn, request.id, task_id=owner.id)
        original = kb.get_coordination_request(conn, request.id)
        snapshot = send_to_review(conn, owner)
        assert kb.coordination_launch_deferral_reason(conn, owner.id) is None
        reviewer, _ = kb.claim_task_for_dispatch(
            conn, owner.id, review=True, organization=organization, now=request.checkpoint_at + 1,
        )
        assert reviewer is not None and reviewer.assignee == "aurora"
        reviewer.coordination_purpose = "terminal_review"
        context = kb.terminal_review_context_snapshot(
            conn, owner.id, request_root_id=request.id, run_id=reviewer.current_run_id,
        )
        assert context["recovery"] is None
        assert context["reserved_decision"]["proposal_event_id"] == snapshot["proposal_event_id"]
        assert "Never use kanban_complete" in kb.build_terminal_review_worker_prompt(conn, reviewer)
        assert not kb.complete_task(conn, owner.id, summary="not repaired", expected_run_id=reviewer.current_run_id)
        assert not kb.request_changes(conn, owner.id, reason="retry automatically", expected_run_id=reviewer.current_run_id)[0]
        kb.charge_coordination_model_call(conn, request.id, task_id=owner.id, purpose="terminal_review")
        assert kb.block_task(
            conn, owner.id, reason="Reviewed exact action and evidence.",
            expected_run_id=reviewer.current_run_id,
            decision_review={"proposal_event_id": snapshot["proposal_event_id"], "outcome": outcome},
        )
        task = kb.get_task(conn, owner.id)
        assert task.status == "blocked" and task.current_run_id is None
        assert task.block_kind == ("needs_input" if outcome == "accepted" else "capability")
        current = kb.get_coordination_request(conn, request.id)
        assert current.status == "active"
        assert current.model_calls_used == original.model_calls_used + 1
        assert current.checkpoint_at == original.checkpoint_at
        assert current.max_model_calls == original.max_model_calls
        rows = conn.execute("SELECT kind, payload FROM task_events WHERE task_id = ?", (owner.id,)).fetchall()
        assert not any(row["kind"] in {"completed", "workforce_handoff_recovery_verified", "coordination_internal_request_completed"} for row in rows)
        verdicts = [json.loads(row["payload"]) for row in rows if row["kind"] == f"workforce_handoff_decision_{outcome}"]
        assert len(verdicts) == 1
        assert verdicts[0]["reserved_decision"] == proposal()
        assert verdicts[0]["incident_repaired"] is False
        assert verdicts[0]["user_authorization_granted"] is False
        notice = kb.owned_failure_decision_outcome_snapshot(conn, owner.id)
        assert notice is not None
        assert notice["outcome"] == outcome
        assert notice["reserved_decision"] == proposal()
        assert notice["proposal_event_id"] == snapshot["proposal_event_id"]
        assert notice["review_run_id"] == reviewer.current_run_id
        assert not kb.unblock_task(conn, owner.id)
        assert kb.get_coordination_request(conn, request.id) == current
        assert not kb.block_task(conn, owner.id, reason="duplicate", expected_run_id=reviewer.current_run_id,
                                 decision_review={"proposal_event_id": snapshot["proposal_event_id"], "outcome": outcome})
        assert kb.list_notify_subs(conn, owner.id) == []
        monkeypatch.setenv("HERMES_KANBAN_TASK", owner.id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
        with scoped_coordination_budget(request_root_id=request.id, task_id=owner.id, purpose="terminal_review", db_path=kb.kanban_db_path()):
            assert _terminal_review_tool_round_completed([{
                "role": "tool", "name": "kanban_block", "content": json.dumps({"ok": True, "task_id": owner.id,
                    "run_id": reviewer.current_run_id, "status": "blocked"}),
            }])


@pytest.mark.parametrize("changes", [
    {"failure_event_id": "old-failure"}, {"account": "unknown"},
    {"integration": ""}, {"action": ""}, {"evidence_references": []},
    {"evidence_references": ["x"] * 9}, {"action": "x" * 1201},
    {"action": "Reconnect https://service.example/callback?code=private-code"},
    {"action_kind": "spend_now"},
])
def test_malformed_or_stale_proposal_has_no_review_transition(organization, changes):
    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        assert not kb.request_review(conn, owner.id, summary="decision", reserved_decision=proposal(**changes),
                                     expected_run_id=owner.current_run_id)
        assert kb.get_task(conn, owner.id).status == "running"


def test_body_actor_spoof_and_missing_active_run_cannot_propose(organization):
    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        assert not kb.request_review(conn, owner.id, summary="decision", reserved_decision=proposal(), force=True)
        body = json.loads(owner.body)
        body["source_agent"] = "builder"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (json.dumps(body), owner.id))
        assert not kb.request_review(conn, owner.id, summary="decision", reserved_decision=proposal(),
                                     expected_run_id=owner.current_run_id)


def test_new_failure_revokes_pending_decision_and_wrong_reviewer_cannot_accept(organization):
    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        snapshot = send_to_review(conn, owner)
        assert not kb.block_task(conn, owner.id, reason="owner self-acceptance", expected_run_id=owner.current_run_id,
                                 decision_review={"proposal_event_id": snapshot["proposal_event_id"], "outcome": "accepted"})
        with kb.write_txn(conn):
            kb._append_event(conn, owner.id, "workforce_handoff_recovery_required", {
                "failure_event_id": "failure-two", "failure_order": 20, "required_successes": 2,
            })
        assert kb.owned_failure_decision_review_snapshot(conn, owner.id) is None
        assert "awaits durable recovery" in kb.coordination_launch_deferral_reason(conn, owner.id)


def test_plain_needs_input_or_exhaustion_does_not_fabricate_reviewed_user_action(organization):
    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        assert kb.block_task(conn, owner.id, kind="needs_input", reason="uninvestigated", expected_run_id=owner.current_run_id)
        assert kb.owned_failure_decision_review_snapshot(conn, owner.id) is None
        kb.mark_coordination_guardrail(conn, request.id, task_id=owner.id, reason="checkpoint reached")
        assert kb.owned_failure_decision_review_snapshot(conn, owner.id) is None
        assert not kb.prepare_coordination_final_return_deliveries(conn)


def test_real_tool_handlers_bind_proposal_and_verdict_to_worker_runs(organization, monkeypatch):
    from tools import kanban_tools as kt

    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        monkeypatch.setenv("HERMES_KANBAN_TASK", owner.id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(owner.current_run_id))
        monkeypatch.setenv("HERMES_PROFILE", "builder")
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
        result = json.loads(kt._handle_request_review({
            "summary": "Review reserved action", "reserved_decision": proposal(),
        }))
        assert result["ok"] and result["status"] == "review"
        snapshot = kb.owned_failure_decision_review_snapshot(conn, owner.id)
        reviewer, _ = kb.claim_task_for_dispatch(conn, owner.id, review=True, organization=organization)
        assert reviewer is not None
        args = {"reason": "Exact action and evidence accepted", "decision_review": {
            "proposal_event_id": snapshot["proposal_event_id"], "outcome": "accepted",
        }}
        assert json.loads(kt._handle_block(args)).get("error")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
        monkeypatch.setenv("HERMES_PROFILE", "aurora")
        result = json.loads(kt._handle_block(args))
        assert result["ok"] and result["status"] == "blocked"
        notice = kb.owned_failure_decision_outcome_snapshot(conn, owner.id)
        assert notice["outcome"] == "accepted"
        assert notice["reserved_decision"] == proposal()
        with kb.write_txn(conn):
            kb._append_event(conn, owner.id, "workforce_handoff_recovery_required", {
                "failure_event_id": "failure-two", "failure_order": 20, "required_successes": 2,
            })
        assert kb.owned_failure_decision_outcome_snapshot(conn, owner.id) is None


def recovered_completion(conn, organization):
    owner, request = owner_run(conn, organization)
    with kb.write_txn(conn):
        kb._append_event(conn, owner.id, "workforce_handoff_recovery_verified", {
            "failure_event_id": "failure-one", "failure_order": 10,
            "success_event_ids": ["success-one", "success-two"], "success_orders": [11, 12],
            "required_successes": 2,
        })
    assert kb.request_review(conn, owner.id, summary="Verified recovery", expected_run_id=owner.current_run_id)
    reviewer, _ = kb.claim_task_for_dispatch(conn, owner.id, review=True, organization=organization)
    assert reviewer is not None
    assert kb.owned_failure_recovery_outcome_snapshot(conn, owner.id) is None
    assert kb.complete_task(conn, owner.id, summary="Source accepted verified recovery", expected_run_id=reviewer.current_run_id)
    return owner, request, reviewer


def test_recovered_snapshot_proves_completed_source_review_and_contains_only_metadata(organization):
    with kb.connect_closing() as conn:
        owner, request, reviewer = recovered_completion(conn, organization)
        snapshot = kb.owned_failure_recovery_outcome_snapshot(conn, owner.id)
        assert snapshot is not None
        assert snapshot["request_root_id"] == request.id
        assert snapshot["technical_owner"] == "builder"
        assert snapshot["director"] == "aurora"
        assert snapshot["failure_event_id"] == "failure-one"
        assert snapshot["review_run_id"] == reviewer.current_run_id
        assert snapshot["incident_repaired"] is True
        assert snapshot["user_authorization_granted"] is False
        assert snapshot["recovery"]["success_event_ids"] == ["success-one", "success-two"]
        event = conn.execute("SELECT kind FROM task_events WHERE id = ?", (snapshot["event_id"],)).fetchone()
        assert event["kind"] == "completed"
        assert "summary" not in json.dumps(snapshot)


@pytest.mark.parametrize("damage", [
    "new_failure", "invalid_recovery", "missing_required", "missing_verified",
    "missing_internal_completion", "wrong_reviewer", "wrong_owner",
    "wrong_claim", "wrong_review_event", "generic_completed", "body_actor_spoof",
    "missing_origin", "unfinished_review_run", "unfinished_request",
])
def test_recovered_snapshot_rejects_missing_forged_or_stale_proof(organization, damage):
    with kb.connect_closing() as conn:
        owner, request, reviewer = recovered_completion(conn, organization)
        assert kb.owned_failure_recovery_outcome_snapshot(conn, owner.id) is not None
        with kb.write_txn(conn):
            if damage == "new_failure":
                kb._append_event(conn, owner.id, "workforce_handoff_recovery_required", {
                    "failure_event_id": "failure-two", "failure_order": 20, "required_successes": 2,
                })
            elif damage == "invalid_recovery":
                kb._append_event(conn, owner.id, "workforce_handoff_recovery_verified", {
                    "failure_event_id": "failure-one", "success_event_ids": ["duplicate", "duplicate"],
                    "success_orders": [11, 12],
                })
            elif damage in {"missing_required", "missing_verified", "missing_internal_completion"}:
                kind = {"missing_required": "workforce_handoff_recovery_required",
                        "missing_verified": "workforce_handoff_recovery_verified",
                        "missing_internal_completion": "coordination_internal_request_completed"}[damage]
                conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = ?", (owner.id, kind))
            elif damage in {"wrong_reviewer", "wrong_owner"}:
                run_id = reviewer.current_run_id if damage == "wrong_reviewer" else owner.current_run_id
                conn.execute("UPDATE task_runs SET profile = 'qa' WHERE id = ?", (run_id,))
            elif damage == "wrong_claim":
                conn.execute("UPDATE task_events SET payload = ? WHERE task_id = ? AND run_id = ? AND kind = 'claimed'",
                             (json.dumps({"source_status": "ready"}), owner.id, reviewer.current_run_id))
            elif damage == "wrong_review_event":
                conn.execute("UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'review_requested'",
                             (json.dumps({"implementer": "builder", "reviewer": "qa"}), owner.id))
            elif damage == "generic_completed":
                kb._append_event(conn, owner.id, "completed", {}, run_id=owner.current_run_id)
            elif damage == "missing_origin":
                conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'coordination_internal_request_accepted'", (owner.id,))
            elif damage == "unfinished_review_run":
                conn.execute("UPDATE task_runs SET status = 'running', outcome = NULL WHERE id = ?", (reviewer.current_run_id,))
            elif damage == "unfinished_request":
                conn.execute("UPDATE coordination_requests SET status = 'active' WHERE id = ?", (request.id,))
            else:
                body = json.loads(kb.get_task(conn, owner.id).body)
                body["source_agent"] = "qa"
                conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (json.dumps(body), owner.id))
        assert kb.owned_failure_recovery_outcome_snapshot(conn, owner.id) is None


def test_generic_done_task_has_no_recovery_snapshot(organization):
    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (owner.id,))
            kb._append_event(conn, owner.id, "completed", {})
        assert kb.owned_failure_recovery_outcome_snapshot(conn, owner.id) is None


@pytest.mark.parametrize("boundary", ["checkpoint", "budget"])
@pytest.mark.parametrize("owner_state", ["running", "blocked"])
def test_host_incomplete_review_uses_existing_reserve_and_stops_without_user_ask(
    organization, monkeypatch, boundary, owner_state,
):
    from tools import kanban_tools as kt
    from agent.coordination_budget import scoped_coordination_budget
    from agent.conversation_loop import _terminal_review_tool_round_completed

    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        if boundary == "budget":
            for _ in range(request.max_model_calls - request.final_model_call_reserve):
                kb.charge_coordination_model_call(conn, request.id, task_id=owner.id)
        if owner_state == "blocked":
            assert kb.block_task(conn, owner.id, reason="No supported diagnosis", expected_run_id=owner.current_run_id)
        original = kb.get_coordination_request(conn, request.id)
        timestamp = request.checkpoint_at + 1 if boundary == "checkpoint" else int(time.time())
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=timestamp) == [owner.id]
        snapshot = kb.owned_failure_incomplete_review_snapshot(conn, owner.id)
        assert snapshot is not None
        assert snapshot["disposition"] == "investigation_incomplete"
        assert snapshot["incident_repaired"] is False and snapshot["user_action_required"] is False
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=timestamp) == []
        assert kb.get_task(conn, owner.id).status == "review"
        assert not kb.block_task(conn, owner.id, reason="Stale owner", expected_run_id=owner.current_run_id)
        reviewer, _ = kb.claim_task_for_dispatch(conn, owner.id, review=True, organization=organization, now=timestamp)
        assert reviewer is not None and reviewer.assignee == "aurora"
        reviewer.coordination_purpose = "terminal_review"
        context = kb.terminal_review_context_snapshot(conn, owner.id, request_root_id=request.id, run_id=reviewer.current_run_id)
        assert context["investigation_incomplete"]["event_id"] == snapshot["event_id"]
        assert context["recovery"] is None
        assert "Do not fabricate a repair" in kb.build_terminal_review_worker_prompt(conn, reviewer)
        assert not kb.complete_task(conn, owner.id, summary="done", expected_run_id=reviewer.current_run_id)
        assert not kb.request_changes(conn, owner.id, reason="retry", expected_run_id=reviewer.current_run_id)[0]
        kb.charge_coordination_model_call(conn, request.id, task_id=owner.id, purpose="terminal_review", now=timestamp)
        monkeypatch.setenv("HERMES_KANBAN_TASK", owner.id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
        monkeypatch.setenv("HERMES_PROFILE", "aurora")
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
        result = kt._handle_block({"reason": "Investigation remains incomplete; evidence does not support a reserved action."})
        assert json.loads(result)["ok"]
        with scoped_coordination_budget(request_root_id=request.id, task_id=owner.id,
                                        purpose="terminal_review", db_path=kb.kanban_db_path()):
            assert _terminal_review_tool_round_completed([{"role": "tool", "name": "kanban_block", "content": result}])
        notice = kb.owned_failure_incomplete_outcome_snapshot(conn, owner.id)
        assert notice is not None and notice["disposition"] == "investigation_incomplete"
        assert not notice["incident_repaired"] and not notice["user_action_required"]
        assert not notice["user_authorization_granted"]
        assert kb.owned_failure_incomplete_review_snapshot(conn, owner.id) is None
        assert not kb.unblock_task(conn, owner.id)
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=timestamp) == []
        current = kb.get_coordination_request(conn, request.id)
        assert current.status == "return_pending"
        assert current.model_calls_used == original.model_calls_used + 1
        assert current.checkpoint_at == original.checkpoint_at
        assert current.max_model_calls == original.max_model_calls
        assert current.leaf_launches_used == original.leaf_launches_used
        assert not kb.prepare_coordination_final_return_deliveries(conn, now=timestamp)
        assert kb.owned_failure_decision_outcome_snapshot(conn, owner.id) is None
        assert kb.owned_failure_recovery_outcome_snapshot(conn, owner.id) is None


def test_incomplete_review_is_host_generated_only_at_real_boundary_and_episode_bound(organization):
    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=request.checkpoint_at - 1) == []
        assert kb.owned_failure_incomplete_review_snapshot(conn, owner.id) is None
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=request.checkpoint_at) == [owner.id]
        with kb.write_txn(conn):
            kb._append_event(conn, owner.id, "workforce_handoff_recovery_required", {
                "failure_event_id": "failure-two", "failure_order": 20, "required_successes": 2,
            })
        assert kb.owned_failure_incomplete_review_snapshot(conn, owner.id) is None


@pytest.mark.parametrize("evidence", ["reserved_decision", "recovery"])
def test_incomplete_review_never_replaces_an_existing_supported_review(organization, evidence):
    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        if evidence == "reserved_decision":
            send_to_review(conn, owner)
        else:
            with kb.write_txn(conn):
                kb._append_event(conn, owner.id, "workforce_handoff_recovery_verified", {
                    "failure_event_id": "failure-one", "success_event_ids": ["one", "two"],
                    "success_orders": [11, 12],
                })
            assert kb.request_review(conn, owner.id, summary="Recovered", expected_run_id=owner.current_run_id)
        assert kb.prepare_owned_failure_incomplete_reviews(conn, now=request.checkpoint_at) == []
        assert kb.owned_failure_incomplete_review_snapshot(conn, owner.id) is None
        assert kb.coordination_launch_deferral_reason(conn, owner.id) is None


def test_dispatch_sweep_finds_blocked_investigation_without_ready_admission(organization, monkeypatch):
    with kb.connect_closing() as conn:
        owner, request = owner_run(conn, organization)
        assert kb.block_task(conn, owner.id, reason="No supported diagnosis", expected_run_id=owner.current_run_id)
        monkeypatch.setattr(kb.time, "time", lambda: request.checkpoint_at + 1)
        kb._dispatch_once_locked(conn, max_spawn=0, spawn_fn=lambda *_args: pytest.fail("capacity is zero"))
        assert kb.get_task(conn, owner.id).status == "review"
        assert kb.owned_failure_incomplete_review_snapshot(conn, owner.id) is not None
