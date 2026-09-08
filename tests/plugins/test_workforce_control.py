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


@contextmanager
def native_coordination_materialization_case(
    board, organization, monkeypatch, tmp_path, *, suffix: str,
):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import kanban_tools

    database_path = Path(board.execute("PRAGMA database_list").fetchone()["file"])
    profile_home = tmp_path / "profiles" / "aurora"
    profile_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(database_path))
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(ROOT / "workforce" / "organization.yaml"),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    payload = plan_payload()
    payload["desired_outcome"] += f" in native acceptance case {suffix}"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    arguments = {
        "workforce_materialize": {
            "plan_id": plan["plan_id"],
            "current_state_evidence": ["kanban:current"],
            "current_state_evidence_at": int(time.time()),
            "confirmed_execution_ready": True,
        },
        "kanban_create": {
            "title": f"Return native acceptance result {suffix}",
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
        if name == "kanban_update":
            return json.dumps({"ok": True, "review_handoff": True})
        raise AssertionError(f"unexpected tool: {name}")

    def tool_call(name, *, args=None, call_id=None):
        call_args = arguments[name] if args is None else args
        function = MagicMock(name=name, arguments=json.dumps(call_args))
        function.name = name
        return MagicMock(
            function=function,
            id=call_id or f"call-{name}",
        )

    agent = concurrent_executor_stub(monkeypatch, invoke)
    session_id = f"native-acceptance-{suffix}-session"
    message_id = f"native-acceptance-{suffix}-message"
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
        yield {
            "agent": agent,
            "arguments": arguments,
            "plan": plan,
            "tool_call": tool_call,
        }
    finally:
        clear_session_vars(tokens)
        reset_session_vars()


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


@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("key_kind", ["execution", "outcome"])
def test_materialization_rejects_preseeded_task_keys_before_mutation(
    board, organization, coordinated, key_kind,
):
    payload = plan_payload()
    payload["desired_outcome"] += (
        f" with a {key_kind} key collision and coordinated={coordinated}"
    )
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    plan_row = board.execute(
        "SELECT stable_key FROM wc_plans WHERE plan_id=?", (plan["plan_id"],)
    ).fetchone()
    if key_kind == "execution":
        collision_key = f"workforce-plan:{plan['plan_id']}:implementation"
    else:
        collision_key = f"workforce-outcome:{plan_row['stable_key']}"
    foreign_id = kanban_db.create_task(
        board,
        title="Foreign task using a reserved workforce key",
        assignee="sloane",
        idempotency_key=collision_key,
    )
    coordination_context = None
    if coordinated:
        request, source_id = accepted_coordination_request(
            board, organization, suffix=f"preseed-{key_kind}",
        )
        coordination_context = (request.id, source_id, "work")
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")

    with pytest.raises(ValueError, match="idempotency key already belongs"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=coordination_context,
        )

    assert_plan_remains_draft_without_materialization(
        board, plan["plan_id"], task_count,
    )
    foreign = kanban_db.get_task(board, foreign_id)
    assert foreign is not None
    assert foreign.request_root_id is None


@pytest.mark.parametrize("returned_kind", ["execution", "outcome"])
def test_materialization_validates_every_returned_task_before_commit(
    board, organization, monkeypatch, returned_kind,
):
    payload = plan_payload()
    payload["desired_outcome"] += f" with an unexpected returned {returned_kind} task"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    foreign_id = kanban_db.create_task(
        board,
        title="Unrelated existing task",
        assignee="sloane",
        idempotency_key=f"foreign-{returned_kind}",
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    real_create_task = kanban_db.create_task

    def mismatched_create(conn, **kwargs):
        is_outcome = str(kwargs.get("idempotency_key") or "").startswith(
            "workforce-outcome:"
        )
        if is_outcome == (returned_kind == "outcome"):
            return foreign_id
        return real_create_task(conn, **kwargs)

    monkeypatch.setattr(kanban_db, "create_task", mismatched_create)

    with pytest.raises(RuntimeError, match="unexpected task identity"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
        )

    assert_plan_remains_draft_without_materialization(
        board, plan["plan_id"], task_count,
    )


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
        coordination_origin=(request.origin_session_id, ""),
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


def test_session_only_origin_stays_uncoordinated_without_request_fallback(
    board, organization,
):
    payload = plan_payload()
    payload["desired_outcome"] += " with classic session-only provenance"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    session_id = "classic-session-only-origin"
    source_id = kanban_db.create_task(
        board,
        title="Unrelated request from the same session",
        assignee="aurora",
        session_id=session_id,
    )
    kanban_db.add_notify_sub(
        board,
        task_id=source_id,
        platform="buzz",
        chat_id="unrelated-request-origin",
        notifier_profile="aurora",
        delivery_mode="wake",
    )
    unrelated_request = kanban_db.create_coordination_request(
        board,
        root_task_id=source_id,
        origin_session_id=session_id,
        origin_message_id="another-message",
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
        coordination_origin=(session_id, ""),
    )

    assert materialized.get("request_root_id") is None
    rows = board.execute(
        "SELECT t.session_id,t.request_root_id,e.payload FROM tasks t "
        "JOIN wc_items w ON w.task_id=t.id "
        "JOIN task_events e ON e.task_id=t.id AND e.kind='created' "
        "WHERE w.item_kind IN ('execution','outcome')"
    ).fetchall()
    assert len(rows) == 2
    assert {row["session_id"] for row in rows} == {session_id}
    assert {row["request_root_id"] for row in rows} == {None}
    assert {
        json.loads(row["payload"])["coordination_origin_message_id"] for row in rows
    } == {None}
    assert kanban_db.get_coordination_request(
        board, unrelated_request.id
    ).status == "active"


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


@pytest.mark.parametrize(
    "coordination_origin",
    [("wrong-session", ""), ("aurora-buzz-session-origin-mismatch", "wrong-message")],
)
def test_active_coordination_rejects_mismatched_partial_origin_before_mutation(
    board, organization, coordination_origin,
):
    payload = plan_payload()
    payload["desired_outcome"] += f" with mismatched origin {coordination_origin}"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")
    request, source_id = accepted_coordination_request(
        board, organization, suffix="origin-mismatch",
    )
    task_count = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    with pytest.raises(ValueError, match="origin does not match"):
        materialize_plan(
            board,
            actor="aurora",
            plan_id=plan["plan_id"],
            current_state_evidence=["kanban:current"],
            current_state_evidence_at=int(time.time()),
            confirmed_execution_ready=True,
            organization=organization,
            coordination_context=(request.id, source_id, "work"),
            coordination_origin=coordination_origin,
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
        yield expected, expected_origin

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


def test_materialize_tool_accepts_classic_session_only_provenance(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget
    from gateway import session_context

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
    monkeypatch.delenv("HERMES_SESSION_MESSAGE_ID", raising=False)
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda _key, default="": default,
    )
    payload = plan_payload()
    payload["desired_outcome"] += " through the classic session-only tool path"
    plan = record_plan(
        board, actor="aurora", payload=payload, organization=organization,
    )
    set_runtime_mode(board, mode="apply", kill_switch=False, reason="isolated test")

    with coordination_budget.scoped_coordination_budget(
        session_id="classic-tool-session"
    ):
        result = json.loads(workforce_tools._materialize({
            "plan_id": plan["plan_id"],
            "current_state_evidence": ["kanban:current"],
            "current_state_evidence_at": int(time.time()),
            "confirmed_execution_ready": True,
        }))

    assert result["success"] is True
    assert result["created"] is True
    assert result.get("request_root_id") is None
    rows = board.execute(
        "SELECT t.session_id,t.request_root_id,e.payload FROM tasks t "
        "JOIN wc_items w ON w.task_id=t.id "
        "JOIN task_events e ON e.task_id=t.id AND e.kind='created' "
        "WHERE w.item_kind IN ('execution','outcome')"
    ).fetchall()
    assert len(rows) == 2
    assert {row["session_id"] for row in rows} == {"classic-tool-session"}
    assert {row["request_root_id"] for row in rows} == {None}
    assert {
        json.loads(row["payload"])["coordination_origin_message_id"] for row in rows
    } == {None}


def test_later_round_acceptance_rejects_prior_uncoordinated_materialization(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget

    with native_coordination_materialization_case(
        board,
        organization,
        monkeypatch,
        tmp_path,
        suffix="later-round",
    ) as case:
        with coordination_budget.scoped_coordination_budget():
            materialize_messages = []
            case["agent"]._execute_tool_calls(
                MagicMock(
                    tool_calls=[case["tool_call"]("workforce_materialize")],
                ),
                materialize_messages,
                "origin-task",
            )
            materialized = json.loads(materialize_messages[0]["content"])
            assert materialized["success"] is True
            assert materialized["created"] is True
            assert materialized.get("request_root_id") is None

            acceptance_messages = []
            case["agent"]._execute_tool_calls(
                MagicMock(tool_calls=[case["tool_call"]("kanban_create")]),
                acceptance_messages,
                "origin-task",
            )
            rejected = json.loads(acceptance_messages[0]["content"])
            assert "after uncoordinated workforce materialization" in rejected["error"]

    assert board.execute("SELECT COUNT(*) FROM coordination_requests").fetchone()[0] == 0
    rows = board.execute(
        "SELECT t.request_root_id FROM tasks t "
        "JOIN wc_items w ON w.task_id=t.id "
        "WHERE w.item_kind IN ('execution','outcome')"
    ).fetchall()
    assert len(rows) == 2
    assert {row["request_root_id"] for row in rows} == {None}


def test_sequential_materialization_before_acceptance_is_recoverable(
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
            first_results = {
                message["name"]: json.loads(message["content"])
                for message in messages
            }
            assert "success" not in first_results["workforce_materialize"]
            assert "requires successful coordination acceptance" in (
                first_results["workforce_materialize"]["error"]
            )
            assert first_results["kanban_create"]["ok"] is True
            assert board.execute(
                "SELECT state FROM wc_plans WHERE plan_id=?", (plan["plan_id"],)
            ).fetchone()[0] == "draft"
            assert board.execute(
                "SELECT COUNT(*) FROM wc_items "
                "WHERE item_kind IN ('execution','outcome')"
            ).fetchone()[0] == 0

            retry_messages = []
            retry_message = MagicMock(
                tool_calls=[tool_call("workforce_materialize")],
            )
            agent._execute_tool_calls(retry_message, retry_messages, "origin-task")
            retry_result = json.loads(retry_messages[0]["content"])
            assert retry_result["success"] is True
            assert retry_result["created"] is True

            idempotent_messages = []
            agent._execute_tool_calls(
                retry_message, idempotent_messages, "origin-task",
            )
            idempotent_result = json.loads(idempotent_messages[0]["content"])
            assert idempotent_result == {
                "success": True,
                "plan_id": plan["plan_id"],
                "root_task_id": retry_result["root_task_id"],
                "created": False,
                "request_root_id": first_results["kanban_create"][
                    "request_root_id"
                ],
            }
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

    request_root_id = first_results["kanban_create"]["request_root_id"]
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


@pytest.mark.parametrize(
    ("failure", "acceptance_error"),
    [
        ("validation", "title is required"),
        ("middleware", "approval denied"),
    ],
)
def test_failed_declared_acceptance_leaves_materialization_recoverable(
    board, organization, monkeypatch, tmp_path, failure, acceptance_error,
):
    from agent import coordination_budget, tool_dispatch_helpers
    from hermes_cli import plugins as plugin_runtime

    with native_coordination_materialization_case(
        board,
        organization,
        monkeypatch,
        tmp_path,
        suffix=failure,
    ) as case:
        if failure == "validation":
            case["arguments"]["kanban_create"].pop("title")
        else:
            monkeypatch.setattr(
                plugin_runtime,
                "_dispatch_pre_tool_call_hooks",
                lambda name, _args, **_kwargs: (
                    ("approval denied", None)
                    if name == "kanban_create"
                    else (None, None)
                ),
            )
        monkeypatch.setattr(
            tool_dispatch_helpers,
            "_plan_tool_batch_segments",
            lambda tool_calls, **_kwargs: [("sequential", list(tool_calls))],
        )
        assistant_message = MagicMock(
            tool_calls=[
                case["tool_call"]("kanban_create"),
                case["tool_call"]("workforce_materialize"),
            ],
        )
        messages = []

        with coordination_budget.scoped_coordination_budget():
            case["agent"]._execute_tool_calls(
                assistant_message, messages, "origin-task",
            )
            results = {
                message["name"]: json.loads(message["content"])
                for message in messages
            }
            assert acceptance_error in results["kanban_create"]["error"]
            assert "requires successful coordination acceptance" in (
                results["workforce_materialize"]["error"]
            )

            retry_messages = []
            case["agent"]._execute_tool_calls(
                MagicMock(
                    tool_calls=[case["tool_call"]("workforce_materialize")],
                ),
                retry_messages,
                "origin-task",
            )
            assert "requires successful coordination acceptance" in json.loads(
                retry_messages[0]["content"]
            )["error"]
            assert_plan_remains_draft_without_materialization(
                board, case["plan"]["plan_id"], 0,
            )
            assert board.execute(
                "SELECT COUNT(*) FROM coordination_requests"
            ).fetchone()[0] == 0

        fresh_turn_messages = []
        with coordination_budget.scoped_coordination_budget():
            case["agent"]._execute_tool_calls(
                MagicMock(
                    tool_calls=[case["tool_call"]("workforce_materialize")],
                ),
                fresh_turn_messages,
                "origin-task",
            )
        fresh_result = json.loads(fresh_turn_messages[0]["content"])
        assert fresh_result["success"] is True
        assert fresh_result["created"] is True
        assert board.execute(
            "SELECT COUNT(*) FROM coordination_requests"
        ).fetchone()[0] == 0


def test_skipped_declared_acceptance_never_materializes(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget, tool_dispatch_helpers, tool_executor

    with native_coordination_materialization_case(
        board,
        organization,
        monkeypatch,
        tmp_path,
        suffix="skipped",
    ) as case:
        monkeypatch.setattr(
            tool_dispatch_helpers,
            "_plan_tool_batch_segments",
            lambda tool_calls, **_kwargs: [("sequential", list(tool_calls))],
        )
        monkeypatch.setattr(
            tool_executor,
            "_work_review_handoff_completed",
            lambda message: message.get("name") == "kanban_update",
        )
        assistant_message = MagicMock(
            tool_calls=[
                case["tool_call"]("workforce_materialize"),
                case["tool_call"](
                    "kanban_update", args={}, call_id="call-review-handoff",
                ),
                case["tool_call"]("kanban_create"),
            ],
        )
        messages = []

        with coordination_budget.scoped_coordination_budget():
            case["agent"]._execute_tool_calls(
                assistant_message, messages, "origin-task",
            )
            assert "requires successful coordination acceptance" in json.loads(
                messages[0]["content"]
            )["error"]
            assert "successful kanban review handoff" in messages[2]["content"]
            assert_plan_remains_draft_without_materialization(
                board, case["plan"]["plan_id"], 0,
            )
            assert board.execute(
                "SELECT COUNT(*) FROM coordination_requests"
            ).fetchone()[0] == 0


def test_partial_executor_shutdown_leaves_materialization_recoverable(
    board, organization, monkeypatch, tmp_path,
):
    from agent import coordination_budget
    from tools import daemon_pool

    class PartialShutdownExecutor:
        def __init__(self, *args, **kwargs):
            self._executor = ThreadPoolExecutor(*args, **kwargs)
            self._submissions = 0

        def submit(self, *args, **kwargs):
            self._submissions += 1
            if self._submissions > 1:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )
            return self._executor.submit(*args, **kwargs)

        def shutdown(self, *args, **kwargs):
            return self._executor.shutdown(*args, **kwargs)

    with native_coordination_materialization_case(
        board,
        organization,
        monkeypatch,
        tmp_path,
        suffix="partial-shutdown",
    ) as case:
        monkeypatch.setattr(
            daemon_pool, "DaemonThreadPoolExecutor", PartialShutdownExecutor,
        )
        messages = []

        with coordination_budget.scoped_coordination_budget():
            case["agent"]._execute_tool_calls_concurrent(
                MagicMock(
                    tool_calls=[
                        case["tool_call"]("workforce_materialize"),
                        case["tool_call"]("kanban_create"),
                    ],
                ),
                messages,
                "origin-task",
            )
            assert "requires successful coordination acceptance" in json.loads(
                messages[0]["content"]
            )["error"]
            assert "Python interpreter is shutting down" in messages[1]["content"]

            retry_messages = []
            case["agent"]._execute_tool_calls(
                MagicMock(
                    tool_calls=[case["tool_call"]("workforce_materialize")],
                ),
                retry_messages,
                "origin-task",
            )
            assert "requires successful coordination acceptance" in json.loads(
                retry_messages[0]["content"]
            )["error"]
            assert_plan_remains_draft_without_materialization(
                board, case["plan"]["plan_id"], 0,
            )
            assert board.execute(
                "SELECT COUNT(*) FROM coordination_requests"
            ).fetchone()[0] == 0


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
    acceptance_finished = threading.Event()
    materialization_finished = threading.Event()
    real_materialization_binding = (
        workforce_tools.coordination_materialization_binding
    )

    if winner == "acceptance":
        @contextmanager
        def delayed_materialization_binding():
            assert acceptance_finished.wait(timeout=5)
            with real_materialization_binding() as value:
                yield value

        monkeypatch.setattr(
            workforce_tools,
            "coordination_materialization_binding",
            delayed_materialization_binding,
        )
        call_order = ("kanban_create", "workforce_materialize")
    else:
        @contextmanager
        def observed_materialization_binding():
            try:
                with real_materialization_binding() as value:
                    yield value
            finally:
                materialization_finished.set()

        monkeypatch.setattr(
            workforce_tools,
            "coordination_materialization_binding",
            observed_materialization_binding,
        )
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
            if winner == "materialization":
                assert materialization_finished.wait(timeout=5)
            result = kanban_tools._handle_create(args)
            acceptance_finished.set()
            return result
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
            results = {
                message["name"]: json.loads(message["content"])
                for message in messages
            }
            assert results["kanban_create"]["ok"] is True
            if winner == "acceptance":
                materialized_result = results["workforce_materialize"]
                assert materialized_result["success"] is True
                assert materialized_result["created"] is True
            else:
                assert "requires successful coordination acceptance" in (
                    results["workforce_materialize"]["error"]
                )
                assert board.execute(
                    "SELECT state FROM wc_plans WHERE plan_id=?",
                    (plan["plan_id"],),
                ).fetchone()[0] == "draft"
                assert board.execute(
                    "SELECT COUNT(*) FROM wc_items "
                    "WHERE item_kind IN ('execution','outcome')"
                ).fetchone()[0] == 0
                retry_messages = []
                retry_message = MagicMock(
                    tool_calls=[tool_call("workforce_materialize")],
                )
                agent._execute_tool_calls(
                    retry_message, retry_messages, "origin-task",
                )
                materialized_result = json.loads(retry_messages[0]["content"])
                assert materialized_result["success"] is True
                assert materialized_result["created"] is True

            idempotent_messages = []
            agent._execute_tool_calls(
                MagicMock(tool_calls=[tool_call("workforce_materialize")]),
                idempotent_messages,
                "origin-task",
            )
            assert json.loads(idempotent_messages[0]["content"])["created"] is False
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

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
