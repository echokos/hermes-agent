from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import threading
import time

import pytest
import json
from unittest.mock import MagicMock

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
from plugins.workforce_control import store as workforce_store


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


def accepted_coordination_request(
    board,
    organization,
    *,
    suffix: str,
    assignee: str = "aurora",
):
    session_id = f"{assignee}-buzz-session-{suffix}"
    root_task_id = kanban_db.create_task(
        board,
        title=f"Return the verified workforce outcome {suffix}",
        assignee=assignee,
        session_id=session_id,
    )
    kanban_db.add_notify_sub(
        board,
        task_id=root_task_id,
        platform="buzz",
        chat_id=f"private-origin-{suffix}",
        notifier_profile=assignee,
        delivery_mode="wake",
    )
    request = kanban_db.create_coordination_request(
        board,
        root_task_id=root_task_id,
        origin_session_id=session_id,
        origin_message_id=f"request-message-{suffix}",
        max_leaf_launches=8,
        max_concurrent_leaf=2,
        max_model_calls=16,
        organization=organization,
    )
    return request, root_task_id


def assert_plan_remains_draft_without_materialization(board, plan_id, task_count):
    assert board.execute(
        "SELECT state FROM wc_plans WHERE plan_id=?", (plan_id,)
    ).fetchone()[0] == "draft"
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == task_count
    assert board.execute(
        "SELECT COUNT(*) FROM wc_items WHERE item_kind IN ('execution','outcome')"
    ).fetchone()[0] == 0


def concurrent_executor_stub(monkeypatch, invoke):
    """Minimal agent using the production concurrent tool executor."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as run_agent_module

    class Stub:
        _interrupt_requested = False
        _interrupt_message = None
        _execution_thread_id = threading.current_thread().ident
        _interrupt_thread_signal_pending = False
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        _context_engine_tool_names = set()
        _memory_manager = None
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0
        _print_fn = print
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""
        _active_children: list = []

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, _description):
            self._last_activity = time.time()

        def _vprint(self, _message, force=False):
            pass

        def _safe_print(self, _message):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, _name, result):
            return result

        def _record_file_mutation_result(self, *_args, **_kwargs):
            pass

    stub = Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *_args, **_kwargs: None
    stub._tool_guardrails = MagicMock()
    stub._tool_guardrails.before_call = (
        lambda *_args, **_kwargs: MagicMock(allows_execution=True)
    )
    stub._flush_messages_to_session_db = lambda *_args, **_kwargs: None
    stub._append_guardrail_observation = (
        lambda _name, _function_args, result, *_args, **_kwargs: result
    )
    stub._execute_tool_calls_concurrent = (
        run_agent_module.AIAgent._execute_tool_calls_concurrent.__get__(stub)
    )
    stub._execute_tool_calls_sequential = (
        run_agent_module.AIAgent._execute_tool_calls_sequential.__get__(stub)
    )
    stub._execute_tool_calls = run_agent_module.AIAgent._execute_tool_calls.__get__(stub)
    stub._apply_pending_steer_to_tool_results = lambda *_args, **_kwargs: None
    stub._guardrail_block_result = lambda _decision: json.dumps({"error": "blocked"})
    stub._invoke_tool = invoke
    monkeypatch.setattr(
        run_agent_module,
        "handle_function_call",
        lambda name, args, *_positional, **_kwargs: invoke(name, args),
    )
    return stub


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


def test_materialization_resolves_a_committed_same_origin_request_after_runtime_miss(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " after the runtime binding missed a commit"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, _source_id = accepted_coordination_request(
        board, organization, suffix="durable-origin",
    )

    materialized = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_origin=(
            request.origin_session_id,
            request.origin_message_id,
        ),
    )

    assert materialized["request_root_id"] == request.id
    task_ids = [*materialized["execution_tasks"].values(), materialized["root_task_id"]]
    assert {
        kanban_db.get_task(board, task_id).request_root_id for task_id in task_ids
    } == {request.id}


def test_materialized_plan_cannot_claim_pending_adoption_by_another_origin(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " without crossing request origins"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_origin=("original-origin", "original-message"),
        coordination_acceptance_pending=True,
    )

    with pytest.raises(ValueError, match="not pending adoption"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_origin=("different-origin", "different-message"),
        )


def test_uncoordinated_materialized_plan_is_idempotent_across_origins(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " with ordinary cross-turn idempotency"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    first = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_origin=("ordinary-origin-one", "ordinary-message-one"),
    )

    second = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:still-current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_origin=("ordinary-origin-two", "ordinary-message-two"),
    )

    assert second == {
        "plan_id": plan["plan_id"],
        "root_task_id": first["root_task_id"],
        "created": False,
    }
    root = kanban_db.get_task(board, first["root_task_id"])
    assert root.request_root_id is None
    assert "coordination_acceptance_pending" not in json.loads(root.body)


@pytest.mark.parametrize("source_kind", ["unbound", "other_request"])
def test_active_coordination_rejects_a_source_outside_the_request_before_mutation(
    board, organization, source_kind,
):
    payload = plan_payload()
    payload["desired_outcome"] += f" with {source_kind} coordination source"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, _request_root_id = accepted_coordination_request(
        board, organization, suffix="current",
    )
    if source_kind == "unbound":
        source_id = kanban_db.create_task(
            board,
            title="Unbound coordination source",
            assignee="aurora",
            session_id=request.origin_session_id,
        )
    else:
        _other_request, source_id = accepted_coordination_request(
            board, organization, suffix="other",
        )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    with pytest.raises(ValueError, match="not bound to the accepted request"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(request.id, source_id, "work"),
        )

    assert_plan_remains_draft_without_materialization(
        board, plan["plan_id"], task_count,
    )


def test_active_coordination_rejects_the_wrong_responsible_agent_before_mutation(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " with another responsible agent"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, source_id = accepted_coordination_request(
        board, organization, suffix="root-owned", assignee="root",
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    with pytest.raises(PermissionError, match="owned by another manager"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(request.id, source_id, "work"),
        )

    assert_plan_remains_draft_without_materialization(
        board, plan["plan_id"], task_count,
    )


def test_active_coordination_rejects_non_work_purpose_before_mutation(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " with terminal review purpose"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, source_id = accepted_coordination_request(
        board, organization, suffix="terminal-review",
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    with pytest.raises(ValueError, match="current coordination work context"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(request.id, source_id, "terminal_review"),
        )

    assert_plan_remains_draft_without_materialization(
        board, plan["plan_id"], task_count,
    )


def test_coordinated_materialization_is_idempotent_only_within_the_same_request(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " with request-scoped idempotency"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, source_id = accepted_coordination_request(
        board, organization, suffix="first",
    )
    context = (request.id, source_id, "work")

    first = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_context=context,
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    second = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_context=context,
    )

    assert second == {
        "plan_id": plan["plan_id"],
        "root_task_id": first["root_task_id"],
        "created": False,
        "request_root_id": request.id,
    }
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == task_count

    other_request, other_source_id = accepted_coordination_request(
        board, organization, suffix="second",
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    with pytest.raises(ValueError, match="not bound to the current coordination request"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(other_request.id, other_source_id, "work"),
        )
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == task_count


def test_concurrent_materializations_bind_the_plan_to_exactly_one_request(
    board, organization, monkeypatch,
):
    payload = plan_payload()
    payload["desired_outcome"] += " under concurrent accepted requests"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    requests = [
        accepted_coordination_request(
            board, organization, suffix=f"concurrent-{suffix}",
        )
        for suffix in ("one", "two")
    ]
    database_path = Path(
        board.execute("PRAGMA database_list").fetchone()["file"]
    )

    ready = threading.Barrier(2)
    real_write_txn = workforce_store.write_txn

    @contextmanager
    def synchronized_write_txn(conn):
        ready.wait(timeout=5)
        with real_write_txn(conn) as transaction:
            yield transaction

    monkeypatch.setattr(workforce_store, "write_txn", synchronized_write_txn)

    def materialize(request_and_source):
        request, source_id = request_and_source
        conn = kanban_db.connect(database_path)
        try:
            try:
                result = materialize_plan(
                    conn,
                    actor="aurora",
                    plan_id=plan["plan_id"],
                    current_state_evidence=["kanban:current"],
                    current_state_evidence_at=int(time.time()),
                    confirmed_execution_ready=True,
                    organization=organization,
                    coordination_context=(request.id, source_id, "work"),
                )
            except ValueError as exc:
                return request.id, None, str(exc)
            return request.id, result, None
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(materialize, requests))

    successful = [item for item in results if item[1] is not None]
    rejected = [item for item in results if item[2] is not None]
    assert len(successful) == 1
    assert successful[0][1]["created"] is True
    assert successful[0][1]["request_root_id"] == successful[0][0]
    assert len(rejected) == 1
    assert rejected[0][2] == (
        "materialized plan is not bound to the current coordination request"
    )

    winner_request_id = successful[0][0]
    materialized = board.execute(
        "SELECT materialized_root_task_id FROM wc_plans WHERE plan_id=?",
        (plan["plan_id"],),
    ).fetchone()
    root = kanban_db.get_task(board, materialized["materialized_root_task_id"])
    assert root is not None and root.request_root_id == winner_request_id
    tasks = board.execute(
        "SELECT t.request_root_id FROM tasks t "
        "JOIN wc_items w ON w.task_id=t.id "
        "WHERE w.item_kind IN ('execution','outcome')"
    ).fetchall()
    assert len(tasks) == 2
    assert {row["request_root_id"] for row in tasks} == {winner_request_id}


def test_multilevel_materialization_inherits_coordination_on_every_task(
    board, organization,
):
    nodes = [
        {
            "key": key,
            "title": f"Implement stage {key}",
            "assignee": "sloane",
            "responsibility": "implementation",
            "action_class": "software_implementation",
            "acceptance_test": f"Stage {key} passes",
            "authority_class": "routine",
            "parents": parents,
        }
        for key, parents in (
            ("one", []),
            ("two", ["one"]),
            ("three", ["two"]),
        )
    ]
    payload = plan_payload(nodes=nodes)
    payload["desired_outcome"] += " through a three-level graph"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, source_id = accepted_coordination_request(
        board, organization, suffix="multilevel",
    )

    materialized = materialize_plan(
        board,
        actor="aurora",
        plan_id=plan["plan_id"],
        current_state_evidence=["kanban:current"],
        current_state_evidence_at=int(time.time()),
        confirmed_execution_ready=True,
        organization=organization,
        coordination_context=(request.id, source_id, "work"),
    )

    execution = materialized["execution_tasks"]
    assert kanban_db.parent_ids(board, execution["two"]) == [execution["one"]]
    assert kanban_db.parent_ids(board, execution["three"]) == [execution["two"]]
    task_ids = [*execution.values(), materialized["root_task_id"]]
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
        assert (
            json.loads(created["payload"])["coordination_origin_message_id"]
            == request.origin_message_id
        )


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
    expected_origin = ("origin-session", "origin-message")
    captured = {}

    class ConnectionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    @contextmanager
    def binding():
        yield expected, expected_origin, True

    monkeypatch.setattr(workforce_tools, "coordination_materialization_binding", binding)
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
    assert captured["coordination_origin"] == expected_origin
    assert captured["coordination_acceptance_pending"] is True


def test_native_executor_dispatches_ordinary_uncoordinated_materialization(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget
    from agent import tool_dispatch_helpers
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import kanban_tools

    database_path = Path(board.execute("PRAGMA database_list").fetchone()["file"])
    profile_home = tmp_path / "profiles" / "aurora"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(database_path))
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(ROOT / "workforce" / "organization.yaml"),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    payload = plan_payload()
    payload["desired_outcome"] += " in an ordinary uncoordinated Buzz turn"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    materialize_arguments = {
        "plan_id": plan["plan_id"],
        "current_state_evidence": ["kanban:current"],
        "current_state_evidence_at": int(time.time()),
        "confirmed_execution_ready": True,
    }
    report_only_arguments = {
        "title": "Return an ordinary uncoordinated result",
        "assignee": "aurora",
        "report_to_origin": True,
    }

    def invoke(name, args, *_positional, **_kwargs):
        if name == "workforce_materialize":
            return workforce_tools._materialize(args)
        if name == "kanban_create":
            return kanban_tools._handle_create(args)
        raise AssertionError(f"unexpected tool: {name}")

    materialize_function = MagicMock(
        name="workforce_materialize",
        arguments=json.dumps(materialize_arguments),
    )
    materialize_function.name = "workforce_materialize"
    report_only_function = MagicMock(
        name="kanban_create",
        arguments=json.dumps(report_only_arguments),
    )
    report_only_function.name = "kanban_create"
    assistant_message = MagicMock(
        tool_calls=[
            MagicMock(
                function=materialize_function,
                id="call-workforce-materialize",
            ),
            MagicMock(function=report_only_function, id="call-report-only"),
        ],
    )
    agent = concurrent_executor_stub(monkeypatch, invoke)
    monkeypatch.setattr(
        tool_dispatch_helpers,
        "_plan_tool_batch_segments",
        lambda tool_calls, **_kwargs: [("sequential", list(tool_calls))],
    )
    messages = []
    tokens = set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        chat_type="dm",
        user_id="elliott",
        session_id="ordinary-materialization-session",
        message_id="ordinary-materialization-message",
        profile="aurora",
    )
    try:
        with coordination_budget.scoped_coordination_budget():
            agent._execute_tool_calls(
                assistant_message, messages, "origin-task",
            )
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    results = {message["name"]: json.loads(message["content"]) for message in messages}
    assert results["workforce_materialize"]["success"] is True
    assert results["kanban_create"]["ok"] is True
    assert results["kanban_create"]["request_root_id"] is None
    assert board.execute("SELECT COUNT(*) FROM coordination_requests").fetchone()[0] == 0
    rows = board.execute(
        "SELECT t.id,t.body,t.request_root_id,t.session_id,w.item_kind "
        "FROM tasks t JOIN wc_items w ON w.task_id=t.id "
        "WHERE w.item_kind IN ('execution','outcome') ORDER BY w.item_kind"
    ).fetchall()
    assert len(rows) == 2
    assert {row["request_root_id"] for row in rows} == {None}
    assert {row["session_id"] for row in rows} == {
        "ordinary-materialization-session"
    }
    assert all(
        "coordination_acceptance_pending" not in json.loads(row["body"])
        for row in rows
    )

    execution_task_id = next(
        row["id"] for row in rows if row["item_kind"] == "execution"
    )
    monkeypatch.setattr(
        kanban_db, "_resolve_dispatch_profile", lambda assignee: assignee,
    )
    monkeypatch.setattr(kanban_db, "_memory_pressure_level", lambda: "normal")
    spawned = []
    dispatched = kanban_db.dispatch_once(
        board,
        spawn_fn=lambda task, _workspace: spawned.append(task.id),
        reconcile_orphans=False,
    )
    assert execution_task_id in spawned
    assert execution_task_id in {
        task_id for task_id, _assignee, _workspace in dispatched.spawned
    }
    assert dispatched.coordination_deferred == []


def test_sequential_materialization_waits_for_later_same_batch_acceptance(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget
    from agent import tool_dispatch_helpers
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import kanban_tools

    database_path = Path(board.execute("PRAGMA database_list").fetchone()["file"])
    profile_home = tmp_path / "profiles" / "aurora"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(database_path))
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(ROOT / "workforce" / "organization.yaml"),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        kanban_db, "_resolve_dispatch_profile", lambda assignee: assignee,
    )
    monkeypatch.setattr(kanban_db, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(
        tool_dispatch_helpers,
        "_plan_tool_batch_segments",
        lambda tool_calls, **_kwargs: [("sequential", list(tool_calls))],
    )

    payload = plan_payload()
    payload["desired_outcome"] += " before later sequential acceptance"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    session_id = "sequential-same-batch-session"
    message_id = "sequential-same-batch-message"
    gap_dispatch: dict[str, object] = {}
    real_materialize = workforce_tools.materialize_plan

    def materialize_with_gap_dispatch(*args, **kwargs):
        result = real_materialize(*args, **kwargs)
        execution_task_id = result["execution_tasks"]["implementation"]
        spawned: list[str] = []
        with kanban_db.connect_closing(database_path) as dispatch_conn:
            dispatch = kanban_db.dispatch_once(
                dispatch_conn,
                spawn_fn=lambda task, _workspace: spawned.append(task.id),
                reconcile_orphans=False,
            )
            gap_dispatch.update(
                result=dispatch,
                spawned=spawned,
                task_id=execution_task_id,
                status=kanban_db.get_task(dispatch_conn, execution_task_id).status,
                run_count=dispatch_conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id=?",
                    (execution_task_id,),
                ).fetchone()[0],
            )
        return result

    monkeypatch.setattr(
        workforce_tools, "materialize_plan", materialize_with_gap_dispatch,
    )
    arguments = {
        "workforce_materialize": {
            "plan_id": plan["plan_id"],
            "current_state_evidence": ["kanban:current"],
            "current_state_evidence_at": int(time.time()),
            "confirmed_execution_ready": True,
        },
        "kanban_create": {
            "title": "Return the sequential same-batch result",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {
                "max_leaf_launches": 2,
                "max_concurrent_leaf": 1,
                "max_model_calls": 8,
            },
        },
    }

    def invoke(name, args, *_positional, **_kwargs):
        if name == "workforce_materialize":
            return workforce_tools._materialize(args)
        if name == "kanban_create":
            return kanban_tools._handle_create(args)
        raise AssertionError(f"unexpected tool: {name}")

    def tool_call(name):
        function = MagicMock(name=name, arguments=json.dumps(arguments[name]))
        function.name = name
        return MagicMock(function=function, id=f"call-{name}")

    assistant_message = MagicMock(
        tool_calls=[tool_call("workforce_materialize"), tool_call("kanban_create")],
    )
    agent = concurrent_executor_stub(monkeypatch, invoke)
    messages = []
    tokens = set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        chat_type="dm",
        user_id="elliott",
        session_id=session_id,
        message_id=message_id,
        profile="aurora",
    )
    try:
        with coordination_budget.scoped_coordination_budget():
            agent._execute_tool_calls(assistant_message, messages, "origin-task")
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    dispatch = gap_dispatch["result"]
    assert gap_dispatch["spawned"] == []
    assert dispatch.spawned == []
    assert dispatch.coordination_deferred == [
        (
            gap_dispatch["task_id"],
            "workforce task is pending coordination acceptance",
        )
    ]
    assert gap_dispatch["status"] == "ready"
    assert gap_dispatch["run_count"] == 0

    results = {message["name"]: json.loads(message["content"]) for message in messages}
    assert results["workforce_materialize"]["success"] is True
    assert results["kanban_create"]["ok"] is True
    request_root_id = results["kanban_create"]["request_root_id"]
    rows = board.execute(
        "SELECT t.id,t.request_root_id,t.session_id,e.payload "
        "FROM tasks t JOIN wc_items w ON w.task_id=t.id "
        "JOIN task_events e ON e.task_id=t.id AND e.kind='created' "
        "WHERE w.item_kind IN ('execution','outcome') ORDER BY t.id"
    ).fetchall()
    assert len(rows) == 2
    assert {row["request_root_id"] for row in rows} == {request_root_id}
    assert {row["session_id"] for row in rows} == {session_id}
    assert {
        json.loads(row["payload"])["coordination_origin_message_id"]
        for row in rows
    } == {message_id}

    claimed, reservation = kanban_db.claim_task_for_dispatch(
        board, gap_dispatch["task_id"], organization=organization,
    )
    assert claimed is not None
    assert reservation is not None
    assert reservation.request_root_id == request_root_id


@pytest.mark.parametrize("winner", ["acceptance", "materialization"])
def test_concurrent_executor_binds_same_batch_materialization_to_accepted_request(
    board, organization, monkeypatch, tmp_path, winner,
):
    from agent import coordination_budget
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import kanban_tools

    database_path = Path(board.execute("PRAGMA database_list").fetchone()["file"])
    profile_home = tmp_path / "profiles" / "aurora"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(database_path))
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(ROOT / "workforce" / "organization.yaml"),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    payload = plan_payload()
    payload["desired_outcome"] += f" when {winner} wins the same-batch race"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")

    session_id = f"same-batch-{winner}-session"
    message_id = f"same-batch-{winner}-message"
    materialization_started = threading.Event()
    acceptance_started = threading.Event()
    gap_dispatch: dict[str, object] = {}

    if winner == "acceptance":
        real_factory = kanban_db.create_coordination_request
        real_binding = workforce_tools.coordination_materialization_binding

        def controlled_factory(*args, **kwargs):
            acceptance_started.set()
            assert materialization_started.wait(timeout=5)
            return real_factory(*args, **kwargs)

        @contextmanager
        def observed_materialization_binding():
            materialization_started.set()
            with real_binding() as value:
                yield value

        monkeypatch.setattr(
            kanban_db, "create_coordination_request", controlled_factory,
        )
        monkeypatch.setattr(
            workforce_tools,
            "coordination_materialization_binding",
            observed_materialization_binding,
        )
        call_order = ("kanban_create", "workforce_materialize")
    else:
        real_materialize = workforce_tools.materialize_plan
        real_request_lookup = coordination_budget.current_coordination_request_id

        def controlled_materialize(*args, **kwargs):
            materialization_started.set()
            assert acceptance_started.wait(timeout=5)
            result = real_materialize(*args, **kwargs)
            execution_task_id = result["execution_tasks"]["implementation"]
            spawned_in_gap: list[str] = []
            with kanban_db.connect_closing(database_path) as dispatch_conn:
                dispatch = kanban_db.dispatch_once(
                    dispatch_conn,
                    spawn_fn=lambda task, _workspace: spawned_in_gap.append(task.id),
                    reconcile_orphans=False,
                )
                gap_dispatch.update(
                    result=dispatch,
                    spawned=spawned_in_gap,
                    task_id=execution_task_id,
                    status=kanban_db.get_task(dispatch_conn, execution_task_id).status,
                    run_count=dispatch_conn.execute(
                        "SELECT COUNT(*) FROM task_runs WHERE task_id=?",
                        (execution_task_id,),
                    ).fetchone()[0],
                )
            return result

        def observed_request_lookup():
            acceptance_started.set()
            return real_request_lookup()

        monkeypatch.setattr(workforce_tools, "materialize_plan", controlled_materialize)
        monkeypatch.setattr(
            coordination_budget,
            "current_coordination_request_id",
            observed_request_lookup,
        )
        monkeypatch.setattr(
            kanban_db, "_resolve_dispatch_profile", lambda assignee: assignee,
        )
        monkeypatch.setattr(kanban_db, "_memory_pressure_level", lambda: "normal")
        call_order = ("workforce_materialize", "kanban_create")

    arguments = {
        "kanban_create": {
            "title": f"Return the {winner}-first result",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {
                "max_leaf_launches": 2,
                "max_concurrent_leaf": 1,
                "max_model_calls": 8,
            },
        },
        "workforce_materialize": {
            "plan_id": plan["plan_id"],
            "current_state_evidence": ["kanban:current"],
            "current_state_evidence_at": int(time.time()),
            "confirmed_execution_ready": True,
        },
    }

    def invoke(name, args, *_positional, **_kwargs):
        if name == "kanban_create":
            return kanban_tools._handle_create(args)
        if name == "workforce_materialize":
            return workforce_tools._materialize(args)
        raise AssertionError(f"unexpected tool: {name}")

    def tool_call(name):
        function = MagicMock(name=name, arguments=json.dumps(arguments[name]))
        function.name = name
        return MagicMock(function=function, id=f"call-{name}")

    agent = concurrent_executor_stub(monkeypatch, invoke)
    assistant_message = MagicMock(
        tool_calls=[tool_call(name) for name in call_order],
    )
    messages = []
    tokens = set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        chat_type="dm",
        user_id="elliott",
        session_id=session_id,
        message_id=message_id,
        profile="aurora",
    )
    try:
        with coordination_budget.scoped_coordination_budget():
            agent._execute_tool_calls_concurrent(
                assistant_message, messages, "origin-task",
            )
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    results = {message["name"]: json.loads(message["content"]) for message in messages}
    assert results["kanban_create"]["ok"] is True
    assert results["workforce_materialize"]["success"] is True
    request_root_id = results["kanban_create"]["request_root_id"]
    materialized = board.execute(
        "SELECT t.id,t.request_root_id,t.session_id,e.payload "
        "FROM tasks t JOIN wc_items w ON w.task_id=t.id "
        "JOIN task_events e ON e.task_id=t.id AND e.kind='created' "
        "WHERE w.item_kind IN ('execution','outcome') ORDER BY t.id"
    ).fetchall()
    assert len(materialized) == 2
    assert {row["request_root_id"] for row in materialized} == {request_root_id}
    assert {row["session_id"] for row in materialized} == {session_id}
    assert {
        json.loads(row["payload"])["coordination_origin_message_id"]
        for row in materialized
    } == {message_id}
    if winner == "materialization":
        dispatch = gap_dispatch["result"]
        assert gap_dispatch["spawned"] == []
        assert dispatch.spawned == []
        assert dispatch.coordination_deferred == [
            (
                gap_dispatch["task_id"],
                "workforce task is pending coordination acceptance",
            )
        ]
        assert gap_dispatch["status"] == "ready"
        assert gap_dispatch["run_count"] == 0

    execution_task_id = board.execute(
        "SELECT t.id FROM tasks t JOIN wc_items w ON w.task_id=t.id "
        "WHERE w.item_kind='execution'"
    ).fetchone()["id"]
    claimed, reservation = kanban_db.claim_task_for_dispatch(
        board, execution_task_id, organization=organization,
    )
    assert claimed is not None
    assert reservation is not None
    assert reservation.request_root_id == request_root_id


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
