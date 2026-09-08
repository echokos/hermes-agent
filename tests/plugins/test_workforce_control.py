from __future__ import annotations

from pathlib import Path
import time

import pytest
import json

from hermes_cli import kanban_db
from hermes_cli.workforce_org import load_organization
from plugins.workforce_control.store import (
    apply_reconciliation,
    dashboard_snapshot,
    materialize_plan,
    observe_dispatch_tick,
    propose_reconciliation,
    record_correction,
    record_plan,
    record_signal,
    runtime_state,
    set_runtime_mode,
    complete_vision_review,
    current_goal_snapshot,
    list_vision_reviews,
    publish_goal_snapshot,
    request_vision_review,
)
from plugins.workforce_control import tools as workforce_tools


ROOT = Path(__file__).parents[2]


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    organization_dir = home / "organization"
    organization_dir.mkdir(parents=True)
    (organization_dir / "organization.yaml").write_text(
        (ROOT / "workforce" / "organization.yaml").read_text()
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kanban_db.connect(tmp_path / "kanban.db")
    runtime_state(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def organization():
    return load_organization(ROOT / "workforce/organization.yaml")


def plan_payload(*, nodes=None, unresolved=None):
    return {
        "title": "Ship the controlled workforce observer",
        "goal_ref": "evernote:goal/proactive-workforce",
        "goal_evidence_at": int(time.time()),
        "desired_outcome": "The workforce finds and closes useful work without speculative fan-out",
        "acceptance_test": "Named proactive scenarios pass with no unauthorized external action",
        "priority_rationale": "This is Elliott's active operating-system priority",
        "checkpoint": "After the first isolated whole-workforce simulation",
        "capacity_assessment": "One bounded implementation node; no competing production work",
        "deadline_dependencies": "No external deadline; depends on isolated test state",
        "displaced_work": "None; implementation remains isolated",
        "unresolved_decisions": list(unresolved or []),
        "defer_or_stop": "Stop if current-state evidence is stale or acceptance fails",
        "evidence_references": ["file://authoritative-plan"],
        "nodes": nodes or [
            {
                "key": "implementation",
                "title": "Implement the bounded observer",
                "assignee": "sloane",
                "responsibility": "implementation",
                "action_class": "software_implementation",
                "acceptance_test": "Focused tests pass",
                "authority_class": "routine",
                "parents": [],
            }
        ],
    }


def test_runtime_is_paused_and_killed_by_default(board):
    state = runtime_state(board)
    assert state["mode"] == "paused"
    assert state["kill_switch"] == 1
    assert state["daily_model_cost_ceiling_usd"] == 0


def test_semantic_signal_identity_deduplicates_new_evidence(board):
    first = record_signal(
        board,
        source_agent="chloe",
        expected_outcome="Stop presenting completed work as new",
        goal_ref="evernote:goal/proactive-workforce",
        observation="The board card is already complete",
        evidence_references=["kanban:event/1"],
        action_class="already_complete",
        target_ref="task-123",
    )
    second = record_signal(
        board,
        source_agent="brenna",
        expected_outcome="Stop presenting completed work as new",
        goal_ref="evernote:goal/proactive-workforce",
        observation="A later observation found the same completed card",
        evidence_references=["kanban:event/2"],
        action_class="already_complete",
        target_ref="task-123",
    )
    assert first["created"] is True
    assert first["status"] == "blocked"
    assert board.execute(
        "SELECT status,block_kind,block_recurrences FROM tasks WHERE id = ?", (first["task_id"],)
    ).fetchone()[:] == ("blocked", "needs_input", 1)
    assert kanban_db.recompute_ready(board) == 0
    assert board.execute(
        "SELECT status FROM tasks WHERE id = ?", (first["task_id"],)
    ).fetchone()[0] == "blocked"
    assert second["created"] is False
    assert first["task_id"] == second["task_id"]
    assert board.execute("SELECT COUNT(*) FROM wc_items WHERE item_kind='signal'").fetchone()[0] == 1


def test_goal_projection_is_aurora_owned_bounded_and_reports_freshness(board):
    with pytest.raises(PermissionError, match="only Aurora"):
        publish_goal_snapshot(
            board, actor="emily", source_guid="guid", source_title="Goals",
            source_updated_at="2026-08-20T09:00:00-05:00",
            goals=[{"goal_id": "g1", "title": "Return time", "desired_outcome": "Less supervision"}],
        )
    published = publish_goal_snapshot(
        board, actor="aurora", source_guid="guid", source_title="Goals",
        source_updated_at="2026-08-20T09:00:00-05:00",
        goals=[{
            "goal_id": "g1", "title": "Return time to Elliott",
            "desired_outcome": "The workforce handles routine work without supervision",
            "priority": "highest", "status": "active", "departments": ["Operations", "Product"],
        }],
    )
    snapshot = current_goal_snapshot(board, max_age_hours=36)
    assert snapshot is not None
    assert snapshot["snapshot_id"] == published["snapshot_id"]
    assert snapshot["stale"] is False
    assert snapshot["goals"][0]["goal_id"] == "g1"
    assert "private_notes" not in snapshot["goals"][0]
    first_capture = snapshot["captured_at"]
    published_again = publish_goal_snapshot(
        board, actor="aurora", source_guid="guid", source_title="Goals",
        source_updated_at="2026-08-20T09:00:00-05:00",
        goals=[{
            "goal_id": "g1", "title": "Return time to Elliott",
            "desired_outcome": "The workforce handles routine work without supervision",
            "priority": "highest", "status": "active", "departments": ["Product", "Operations"],
        }],
    )
    assert published_again["snapshot_id"] == published["snapshot_id"]
    assert current_goal_snapshot(board)["captured_at"] >= first_capture
    board.execute(
        "UPDATE wc_goal_snapshots SET captured_at = captured_at - ? WHERE snapshot_id = ?",
        (37 * 3600, published["snapshot_id"]),
    )
    assert current_goal_snapshot(board, max_age_hours=36)["stale"] is True
    with pytest.raises(ValueError, match="older than"):
        publish_goal_snapshot(
            board, actor="aurora", source_guid="guid", source_title="Goals",
            source_updated_at="2026-08-19T09:00:00-05:00",
            goals=[{"goal_id": "old", "title": "Old", "desired_outcome": "Old state"}],
        )


def test_vision_end_layer_requires_aurora_request_and_mel_response(board):
    with pytest.raises(PermissionError, match="only Aurora"):
        request_vision_review(
            board, actor="chloe", source_ref="task:t1", goal_ref="g1",
            brief="Challenge this outcome", evidence_references=[],
        )
    requested = request_vision_review(
        board, actor="aurora", source_ref="task:t1", goal_ref="g1",
        brief="Ask how this could create ten times more value", evidence_references=["kanban:t1"],
    )
    duplicate = request_vision_review(
        board, actor="aurora", source_ref="task:t1", goal_ref="g1",
        brief="Ask how this could create ten times more value", evidence_references=["kanban:t1"],
    )
    assert requested["created"] is True
    assert duplicate == {"review_id": requested["review_id"], "status": "pending", "created": False}
    assert list_vision_reviews(board)[0]["review_id"] == requested["review_id"]
    response = {
        "reframe": "Treat the output as a reusable system",
        "ten_x_option": "Build the factory behind the recurring result",
        "assumptions": ["The need recurs"],
        "value_case": "Future cycles become faster and more reliable",
        "risks": ["Premature abstraction"],
        "smallest_test": "Reuse one primitive in the next two cycles",
    }
    with pytest.raises(PermissionError, match="only Mel"):
        complete_vision_review(board, actor="aurora", review_id=requested["review_id"], response=response)
    completed = complete_vision_review(
        board, actor="mel", review_id=requested["review_id"], response=response
    )
    assert completed["status"] == "completed"
    assert list_vision_reviews(board, status="completed")[0]["response"]["ten_x_option"].startswith("Build")


def test_buzz_observer_is_bounded_and_role_restricted(monkeypatch):
    monkeypatch.setattr(workforce_tools, "_actor", lambda: "chloe")
    monkeypatch.setattr(
        workforce_tools,
        "_buzz_events",
        lambda **kwargs: {
            "since": 1, "rooms_checked": 2,
            "events": [{"room": "admin", "content": "A commitment changed"}],
            "errors": [], "requested": kwargs,
        },
    )
    result = json.loads(workforce_tools._observe_buzz({"lookback_minutes": 90, "per_room_limit": 4}))
    assert result["success"] is True
    assert result["requested"] == {
        "lookback_minutes": 90, "per_room_limit": 4, "max_events": 20,
    }
    monkeypatch.setattr(workforce_tools, "_actor", lambda: "milena")
    assert json.loads(workforce_tools._observe_buzz({}))["success"] is True
    monkeypatch.setattr(workforce_tools, "_actor", lambda: "emily")
    denied = json.loads(workforce_tools._observe_buzz({}))
    assert "success" not in denied
    assert "restricted" in denied["error"]


def test_only_aurora_can_plan_and_draft_creates_no_execution(board, organization):
    with pytest.raises(PermissionError, match="only Aurora"):
        record_plan(board, actor="emily", payload=plan_payload(), organization=organization)
    drafted = record_plan(board, actor="aurora", payload=plan_payload(), organization=organization)
    assert drafted["state"] == "draft"
    assert drafted["execution_cards_created"] == 0
    assert board.execute("SELECT COUNT(*) FROM wc_items WHERE item_kind='execution'").fetchone()[0] == 0


def test_technical_ownership_and_reserved_authority_are_enforced(board, organization):
    wrong_owner = plan_payload(nodes=[{
        "key": "implementation", "title": "Implement it", "assignee": "sage",
        "responsibility": "implementation", "action_class": "software_implementation",
        "acceptance_test": "Tests pass", "authority_class": "routine", "parents": [],
    }])
    with pytest.raises(ValueError, match="owned by sloane"):
        record_plan(board, actor="aurora", payload=wrong_owner, organization=organization)

    reserved = plan_payload(nodes=[{
        "key": "activation", "title": "Activate production", "assignee": "alina",
        "responsibility": "local_host_install_service_activation", "action_class": "activation",
        "acceptance_test": "Service is live", "authority_class": "reserved", "parents": [],
    }])
    plan = record_plan(board, actor="aurora", payload=reserved, organization=organization)
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    with pytest.raises(PermissionError, match="reserved-authority"):
        materialize_plan(
            board, actor="aurora", plan_id=plan["plan_id"],
            current_state_evidence=["copy://kanban/current"],
            current_state_evidence_at=int(time.time()), confirmed_execution_ready=True,
            organization=organization,
        )

    wrong_external_owner = plan_payload(nodes=[{
        "key": "cloud", "title": "Operate external cloud resource", "assignee": "alina",
        "responsibility": "external_cloud_server_app_operations", "action_class": "provider_operation",
        "acceptance_test": "Provider state is verified", "authority_class": "routine", "parents": [],
    }])
    with pytest.raises(ValueError, match="owned by root"):
        record_plan(
            board, actor="aurora", payload=wrong_external_owner,
            organization=organization,
        )

    correct_external_owner = plan_payload(nodes=[{
        "key": "cloud", "title": "Operate external cloud resource", "assignee": "main",
        "responsibility": "external_cloud_server_app_operations", "action_class": "provider_operation",
        "acceptance_test": "Provider state is verified", "authority_class": "routine", "parents": [],
    }])
    record_plan(
        board, actor="aurora", payload=correct_external_owner,
        organization=organization,
    )


def test_materialization_requires_activation_fresh_state_and_resolved_intake(board, organization):
    plan = record_plan(board, actor="aurora", payload=plan_payload(unresolved=["Elliott taste decision"]), organization=organization)
    with pytest.raises(RuntimeError, match="paused"):
        materialize_plan(
            board, actor="aurora", plan_id=plan["plan_id"],
            current_state_evidence=["copy://kanban/current"],
            current_state_evidence_at=int(time.time()), confirmed_execution_ready=True,
            organization=organization,
        )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    with pytest.raises(ValueError, match="unresolved decisions"):
        materialize_plan(
            board, actor="aurora", plan_id=plan["plan_id"],
            current_state_evidence=["copy://kanban/current"],
            current_state_evidence_at=int(time.time()), confirmed_execution_ready=True,
            organization=organization,
        )


def test_bounded_graph_materializes_atomically_and_idempotently(board, organization):
    payload = plan_payload()
    payload["desired_outcome"] += " in the isolated fixture"
    plan = record_plan(board, actor="aurora", payload=payload, organization=organization)
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    first = materialize_plan(
        board, actor="aurora", plan_id=plan["plan_id"],
        current_state_evidence=["copy://kanban/current"],
        current_state_evidence_at=int(time.time()), confirmed_execution_ready=True,
        organization=organization,
    )
    second = materialize_plan(
        board, actor="aurora", plan_id=plan["plan_id"],
        current_state_evidence=["copy://kanban/current"],
        current_state_evidence_at=int(time.time()), confirmed_execution_ready=True,
        organization=organization,
    )
    assert first["created"] is True
    assert second == {"plan_id": plan["plan_id"], "root_task_id": first["root_task_id"], "created": False}
    assert len(first["execution_tasks"]) == 1
    root = kanban_db.get_task(board, first["root_task_id"])
    assert root is not None and root.status == "todo"


def test_materialization_inherits_active_coordination_budget_and_origin(board, organization):
    payload = plan_payload()
    payload["desired_outcome"] += " inside one accepted request"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")

    request_root_id = kanban_db.create_task(
        board,
        title="Return the verified workforce outcome",
        assignee="aurora",
        session_id="aurora-buzz-session",
    )
    kanban_db.add_notify_sub(
        board,
        task_id=request_root_id,
        platform="buzz",
        chat_id="private-origin",
        notifier_profile="aurora",
        delivery_mode="wake",
    )
    request = kanban_db.create_coordination_request(
        board,
        root_task_id=request_root_id,
        origin_session_id="aurora-buzz-session",
        origin_message_id="root-request-message",
        max_leaf_launches=2,
        max_concurrent_leaf=1,
        max_model_calls=8,
        organization=organization,
    )

    materialized = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_context=(request.id, request_root_id, "work"),
    )

    assert materialized["request_root_id"] == request.id
    task_ids = [*materialized["execution_tasks"].values(), materialized["root_task_id"]]
    for task_id in task_ids:
        task = kanban_db.get_task(board, task_id)
        assert task is not None
        assert task.request_root_id == request.id
        assert task.session_id == request.origin_session_id
        created = board.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id=? AND kind='created' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert json.loads(created["payload"])["coordination_origin_message_id"] == request.origin_message_id
    assert kanban_db.get_coordination_request(board, request.id).max_model_calls == 8


@pytest.mark.parametrize("request_status", [None, "completed"])
def test_invalid_coordination_context_rejects_before_materializing_tasks(
    board, organization, request_status,
):
    payload = plan_payload()
    payload["desired_outcome"] += f" with invalid request {request_status}"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    source_id = kanban_db.create_task(
        board,
        title="Claimed coordination source",
        assignee="aurora",
        session_id="aurora-buzz-session",
    )
    request_id = "cr_missing"
    if request_status is not None:
        kanban_db.add_notify_sub(
            board,
            task_id=source_id,
            platform="buzz",
            chat_id="private-origin",
            notifier_profile="aurora",
            delivery_mode="wake",
        )
        request = kanban_db.create_coordination_request(
            board,
            root_task_id=source_id,
            origin_session_id="aurora-buzz-session",
            origin_message_id="closed-request-message",
            organization=organization,
        )
        request_id = request.id
        board.execute(
            "UPDATE coordination_requests SET status=? WHERE id=?",
            (request_status, request.id),
        )

    with pytest.raises(ValueError, match="active accepted coordination request"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(request_id, source_id, "work"),
        )

    assert board.execute(
        "SELECT state FROM wc_plans WHERE plan_id=?", (plan["plan_id"],)
    ).fetchone()[0] == "draft"
    assert board.execute(
        "SELECT COUNT(*) FROM wc_items WHERE item_kind IN ('execution','outcome')"
    ).fetchone()[0] == 0


def test_materialize_tool_forwards_only_the_trusted_runtime_coordination(monkeypatch):
    expected = ("cr_current", "t_current", "work")
    captured = {}

    class ConnectionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(workforce_tools, "current_coordination_execution", lambda: expected)
    monkeypatch.setattr(workforce_tools.kanban_db, "connect_closing", ConnectionContext)
    monkeypatch.setattr(workforce_tools, "_actor", lambda: "aurora")

    def fake_materialize(_conn, **kwargs):
        captured.update(kwargs)
        return {"plan_id": kwargs["plan_id"], "created": False}

    monkeypatch.setattr(workforce_tools, "materialize_plan", fake_materialize)
    result = json.loads(workforce_tools._materialize({
        "plan_id": "plan_current",
        "current_state_evidence": ["kanban:current"],
        "current_state_evidence_at": "2026-09-08T13:00:00-05:00",
        "confirmed_execution_ready": True,
    }))

    assert result["success"] is True
    assert captured["coordination_context"] == expected


def test_plan_rejects_more_than_eight_execution_nodes(board, organization):
    nodes = [
        {
            "key": f"node-{index}",
            "title": f"Bounded node {index}",
            "assignee": "sloane",
            "responsibility": "backend_development",
            "acceptance_test": "A bounded acceptance check passes",
            "parents": [f"node-{index - 1}"] if index else [],
        }
        for index in range(9)
    ]
    with pytest.raises(ValueError, match="8-node safety limit"):
        record_plan(
            board,
            actor="aurora",
            payload=plan_payload(nodes=nodes),
            organization=organization,
        )


def test_failed_verification_reopens_outcome_and_creates_one_remediation(board, organization):
    outcome_id = kanban_db.create_task(board, title="Outcome under verification", assignee="aurora")
    verification_id = kanban_db.create_task(board, title="Verify outcome", assignee="reese")
    now = int(time.time())
    board.execute(
        "INSERT INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,acceptance_test,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (outcome_id, "outcome", "fixture-outcome", "goal", "Verified result", "Tests pass", "pending", "open", now, now),
    )
    board.execute(
        "INSERT INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,acceptance_test,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (verification_id, "verification", "fixture-verification", "goal", "Verify result", "Reproduce failure", "failed", "complete", now, now),
    )
    actions = propose_reconciliation(
        board, actor="reese", mode="proposed", organization=organization,
        observations=[{
            "task_id": verification_id, "target_task_id": outcome_id,
            "classification": "failed_verification", "confidence": "high",
            "rationale": "The acceptance test failed reproducibly",
            "evidence_references": ["test://failure/1"], "evidence_at": now,
        }],
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    applied = apply_reconciliation(board, actor="aurora", action_ids=[actions[0]["action_id"]], organization=organization)
    assert applied[0]["state"] == "applied"
    assert kanban_db.get_task(board, outcome_id).status == "triage"
    remediation = board.execute("SELECT source_task_id FROM wc_relations WHERE relation='remediates' AND target_task_id=?", (outcome_id,)).fetchall()
    assert len(remediation) == 1


def test_unverified_outcome_is_quarantined_and_external_blockers_stay_blocked(board, organization):
    outcome_id = kanban_db.create_task(board, title="Unverified complete claim", assignee="aurora")
    blocked_id = kanban_db.create_task(board, title="Needs Elliott input", assignee="aurora")
    now = int(time.time())
    board.execute(
        "INSERT INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (outcome_id, "outcome", "unverified-outcome", "goal", "Claimed result", "pending", "open", now, now),
    )
    actions = propose_reconciliation(
        board, actor="chloe", mode="proposed", organization=organization,
        observations=[
            {"task_id": outcome_id, "classification": "already_complete", "confidence": "high", "rationale": "A completion was claimed", "evidence_references": ["kanban://claim"], "evidence_at": now},
            {"task_id": blocked_id, "classification": "external_blocker", "confidence": "high", "rationale": "A retained decision is required", "evidence_references": ["decision://elliott"], "evidence_at": now},
        ],
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    results = apply_reconciliation(board, actor="aurora", action_ids=[a["action_id"] for a in actions], organization=organization)
    assert results[0]["state"] == "quarantined"
    assert results[1]["state"] == "applied"
    blocked = kanban_db.get_task(board, blocked_id)
    assert blocked.status == "blocked" and blocked.block_kind == "needs_input"


def test_corrections_preserve_privacy_scope_and_dashboard_exposes_exceptions(board, organization):
    with pytest.raises(PermissionError, match="private relationship context"):
        record_correction(
            board, actor="aurora", classification="quality_standard", scope="workforce",
            description="Private relationship preference", provenance_ref="private://conversation",
            privacy_class="relationship_private", organization=organization,
        )
    correction = record_correction(
        board, actor="root", classification="workflow_defect", scope="system",
        description="Semantic identity must ignore changing observation prose",
        provenance_ref="test://semantic-dedupe", privacy_class="organizational",
        rule_target="plugins/workforce_control/store.py",
        regression_ref="tests/plugins/test_workforce_control.py",
        organization=organization,
    )
    assert correction["status"] == "implemented"
    snapshot = dashboard_snapshot(board)
    assert snapshot["runtime"]["mode"] in {"paused", "apply"}
    assert snapshot["corrections"]
    assert "exceptions" in snapshot


def test_dashboard_snapshot_is_read_only(board):
    before = board.execute(
        "SELECT updated_at FROM wc_schema WHERE singleton=1"
    ).fetchone()["updated_at"]
    snapshot = dashboard_snapshot(board)
    after = board.execute(
        "SELECT updated_at FROM wc_schema WHERE singleton=1"
    ).fetchone()["updated_at"]
    assert snapshot["available"] is True
    assert after == before


def test_observer_is_inert_while_paused_then_quarantines_unverified_completion(board, organization):
    outcome_id = kanban_db.create_task(board, title="Observer outcome", assignee="aurora")
    now = int(time.time())
    board.execute(
        "INSERT INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (outcome_id, "outcome", "observer-outcome", "goal", "Observer result", "pending", "open", now, now),
    )
    with kanban_db.write_txn(board):
        board.execute("UPDATE tasks SET status='done',completed_at=? WHERE id=?", (now, outcome_id))
        kanban_db._append_event(board, outcome_id, "completed", {"fixture": True})
    assert observe_dispatch_tick(board, organization=organization)["paused"] is True
    assert board.execute("SELECT COUNT(*) FROM wc_reconcile_actions").fetchone()[0] == 0

    set_runtime_mode(board, mode="shadow", kill_switch=False, reason="isolated shadow test")
    observed = observe_dispatch_tick(board, organization=organization)
    assert observed["proposed"] == 1
    action = board.execute("SELECT state,classification FROM wc_reconcile_actions").fetchone()
    assert dict(action) == {"state": "shadow", "classification": "already_complete"}
