from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.workforce_org import load_organization


ORGANIZATION = """
schema_version: 1
agents:
  - agent: elliott
    display_name: Elliott
    status: artifact
    operational: false
    manager: null
    direct_reports: [aurora]
    mission: Own the system
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: aurora
    display_name: Aurora
    status: active
    operational: true
    function: Chief of Staff
    manager: elliott
    direct_reports: [director]
    mission: Coordinate requests
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: director
    display_name: Director
    status: active
    operational: true
    function: Product Director
    manager: aurora
    direct_reports: [builder, qa]
    mission: Direct delivery
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: builder
    display_name: Builder
    status: active
    operational: true
    function: Software Developer
    manager: director
    direct_reports: []
    mission: Implement work
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: qa
    display_name: QA
    status: active
    operational: true
    function: Code Reviewer and QA Gatekeeper
    manager: director
    direct_reports: []
    mission: Verify work
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
"""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    organization_dir = home / "organization"
    organization_dir.mkdir()
    (organization_dir / "organization.yaml").write_text(ORGANIZATION)
    kb.init_db()
    return home


@pytest.fixture
def organization(kanban_home):
    return load_organization()


def test_legacy_upgrade_is_additive_and_does_not_invent_requests(tmp_path):
    db_path = tmp_path / "legacy-kanban.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
            status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER,
            completed_at INTEGER, workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT, claim_lock TEXT, claim_expires INTEGER
        );
        CREATE TABLE task_links (
            parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id)
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL,
            outcome TEXT
        );
        CREATE TABLE kanban_notify_subs (
            task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '', created_at INTEGER NOT NULL,
            last_event_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (task_id, platform, chat_id, thread_id)
        );
        """
    )
    legacy.executemany(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, priority, created_by, created_at, "
        "workspace_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("legacy-parent", "Parent", "body-a", "aurora", "done", 7, "cli", 10, "scratch"),
            ("legacy-child", "Child", "body-b", "emily", "ready", 3, "cli", 11, "scratch"),
        ],
    )
    legacy.execute(
        "INSERT INTO task_links VALUES ('legacy-parent', 'legacy-child')"
    )
    legacy.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES ('legacy-child', 'created', '{\"legacy\": true}', 12)"
    )
    legacy.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, outcome) "
        "VALUES ('legacy-parent', 'aurora', 'done', 13, 'completed')"
    )
    legacy.execute(
        "INSERT INTO kanban_notify_subs "
        "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
        "VALUES ('legacy-child', 'buzz', 'room-1', '', 14, 1)"
    )
    legacy.commit()
    before = {
        "tasks": legacy.execute(
            "SELECT id, title, body, assignee, status, priority, created_by, "
            "created_at, workspace_kind FROM tasks ORDER BY id"
        ).fetchall(),
        "links": legacy.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        ).fetchall(),
        "events": legacy.execute(
            "SELECT id, task_id, kind, payload, created_at FROM task_events ORDER BY id"
        ).fetchall(),
        "runs": legacy.execute(
            "SELECT id, task_id, profile, status, started_at, outcome "
            "FROM task_runs ORDER BY id"
        ).fetchall(),
        "routes": legacy.execute(
            "SELECT task_id, platform, chat_id, thread_id, created_at, last_event_id "
            "FROM kanban_notify_subs ORDER BY task_id, platform, chat_id, thread_id"
        ).fetchall(),
    }
    legacy.close()

    kb.init_db(db_path)
    kb.init_db(db_path)

    upgraded = sqlite3.connect(db_path)
    assert upgraded.execute(
        "SELECT id, title, body, assignee, status, priority, created_by, "
        "created_at, workspace_kind FROM tasks ORDER BY id"
    ).fetchall() == before["tasks"]
    assert upgraded.execute(
        "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
    ).fetchall() == before["links"]
    assert upgraded.execute(
        "SELECT id, task_id, kind, payload, created_at FROM task_events ORDER BY id"
    ).fetchall() == before["events"]
    assert upgraded.execute(
        "SELECT id, task_id, profile, status, started_at, outcome "
        "FROM task_runs ORDER BY id"
    ).fetchall() == before["runs"]
    assert upgraded.execute(
        "SELECT task_id, platform, chat_id, thread_id, created_at, last_event_id "
        "FROM kanban_notify_subs ORDER BY task_id, platform, chat_id, thread_id"
    ).fetchall() == before["routes"]
    assert upgraded.execute(
        "SELECT COUNT(*) FROM tasks WHERE request_root_id IS NOT NULL"
    ).fetchone()[0] == 0
    assert upgraded.execute(
        "SELECT COUNT(*) FROM coordination_requests"
    ).fetchone()[0] == 0
    assert upgraded.execute(
        "SELECT COUNT(*) FROM coordination_acceptance_debits"
    ).fetchone()[0] == 0
    assert upgraded.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    upgraded.close()


def _accept_request(
    conn,
    organization,
    *,
    session_id: str = "aurora-session",
    message_id: str = "message-1",
    **limits,
):
    root_id = kb.create_task(
        conn,
        title=f"Return verified result for {message_id}",
        assignee="aurora",
        session_id=session_id,
    )
    kb.add_notify_sub(
        conn,
        task_id=root_id,
        platform="buzz",
        chat_id="elliott-dm",
        notifier_profile="aurora",
        delivery_mode="wake",
    )
    request = kb.create_coordination_request(
        conn,
        root_task_id=root_id,
        origin_session_id=session_id,
        origin_message_id=message_id,
        organization=organization,
        now=100,
        **limits,
    )
    return root_id, request


def test_explicit_request_requires_one_origin_route_and_is_idempotent(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id = kb.create_task(
            conn,
            title="final result",
            assignee="aurora",
            session_id="session-1",
        )
        with pytest.raises(ValueError, match="exactly one"):
            kb.create_coordination_request(
                conn,
                root_task_id=root_id,
                origin_session_id="session-1",
                origin_message_id="message-1",
                organization=organization,
            )

        kb.add_notify_sub(
            conn,
            task_id=root_id,
            platform="buzz",
            chat_id="elliott-dm",
            delivery_mode="wake",
        )
        first = kb.create_coordination_request(
            conn,
            root_task_id=root_id,
            origin_session_id="session-1",
            origin_message_id="message-1",
            organization=organization,
            now=100,
            acceptance_scope_id="accepting-turn-1",
            accepting_model_calls=1,
        )
        second = kb.create_coordination_request(
            conn,
            root_task_id=root_id,
            origin_session_id="session-1",
            origin_message_id="message-1",
            organization=organization,
            now=999,
            acceptance_scope_id="accepting-turn-1",
            accepting_model_calls=1,
        )

        assert first == second
        assert first.id == kb.coordination_request_id("session-1", "message-1")
        assert first.checkpoint_at == 1300
        assert first.model_calls_used == 1
        assert kb.get_task(conn, root_id).request_root_id == first.id
        accepted = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'coordination_request_accepted'",
            (root_id,),
        ).fetchone()[0]
        assert accepted == 1


def test_acceptance_call_settlement_is_cumulative_per_scope_and_rootwide(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id = kb.create_task(
            conn,
            title="final result",
            assignee="aurora",
            session_id="session-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=root_id,
            platform="buzz",
            chat_id="elliott-dm",
            delivery_mode="wake",
        )
        request = kb.create_coordination_request(
            conn,
            root_task_id=root_id,
            origin_session_id="session-1",
            origin_message_id="message-1",
            acceptance_scope_id="turn-a",
            accepting_model_calls=1,
            organization=organization,
            now=100,
        )
        assert kb.settle_coordination_acceptance_calls(
            conn, request.id, "turn-a", 1, now=101,
        ) == 1
        assert kb.settle_coordination_acceptance_calls(
            conn, request.id, "turn-a", 2, now=102,
        ) == 2
        assert kb.settle_coordination_acceptance_calls(
            conn, request.id, "turn-b", 1, now=103,
        ) == 3
        assert conn.execute(
            "SELECT COUNT(*) FROM coordination_acceptance_debits "
            "WHERE request_root_id = ?",
            (request.id,),
        ).fetchone()[0] == 2

        assert kb.settle_coordination_acceptance_calls(
            conn, request.id, "turn-c", 36, now=104,
        ) == 39
        current = kb.get_coordination_request(conn, request.id)
        assert current.status == "return_pending"
        assert current.model_calls_used == 39
        guardrails = [
            event
            for event in kb.list_events(conn, root_id)
            if event.kind == "coordination_guardrail_reached"
        ]
        assert len(guardrails) == 1


def test_acceptance_rejects_when_accepting_turn_invades_final_reserve(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id = kb.create_task(
            conn,
            title="final result",
            assignee="aurora",
            session_id="session-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=root_id,
            platform="buzz",
            chat_id="elliott-dm",
            delivery_mode="wake",
        )
        with pytest.raises(ValueError, match="work-call budget"):
            kb.create_coordination_request(
                conn,
                root_task_id=root_id,
                origin_session_id="session-1",
                origin_message_id="message-1",
                max_model_calls=4,
                final_model_call_reserve=2,
                acceptance_scope_id="over-budget-turn",
                accepting_model_calls=3,
                organization=organization,
            )
        assert kb.get_task(conn, root_id).request_root_id is None
        assert conn.execute(
            "SELECT COUNT(*) FROM coordination_requests"
        ).fetchone()[0] == 0


def test_same_origin_adoption_serializes_with_dispatch_claim(
    kanban_home, organization,
):
    def create_root(conn, *, session_id: str, message_id: str) -> str:
        root_id = kb.create_task(
            conn,
            title="return final result",
            assignee="aurora",
            session_id=session_id,
            coordination_origin_message_id=message_id,
        )
        kb.add_notify_sub(
            conn,
            task_id=root_id,
            platform="buzz",
            chat_id="elliott-dm",
            delivery_mode="wake",
        )
        return root_id

    with kb.connect_closing() as conn:
        claimed_first = kb.create_task(
            conn,
            title="already launched",
            assignee="builder",
            session_id="session-claim-first",
            coordination_origin_message_id="message-claim-first",
        )
        claimed = kb.claim_task(conn, claimed_first)
        assert claimed is not None
        root_id = create_root(
            conn,
            session_id="session-claim-first",
            message_id="message-claim-first",
        )
        with pytest.raises(ValueError, match="already launched"):
            kb.create_coordination_request(
                conn,
                root_task_id=root_id,
                origin_session_id="session-claim-first",
                origin_message_id="message-claim-first",
                organization=organization,
            )
        assert kb.get_task(conn, claimed_first).request_root_id is None
        assert kb.get_task(conn, root_id).request_root_id is None

        adopted_first = kb.create_task(
            conn,
            title="unlaunched child",
            assignee="builder",
            session_id="session-accept-first",
            coordination_origin_message_id="message-accept-first",
        )
        accepted_root = create_root(
            conn,
            session_id="session-accept-first",
            message_id="message-accept-first",
        )
        request = kb.create_coordination_request(
            conn,
            root_task_id=accepted_root,
            origin_session_id="session-accept-first",
            origin_message_id="message-accept-first",
            organization=organization,
            now=100,
        )
        assert kb.get_task(conn, adopted_first).request_root_id == request.id
        dispatched, reservation = kb.claim_task_for_dispatch(
            conn, adopted_first, organization=organization, now=101,
        )
        assert dispatched is not None
        assert reservation is not None
        assert reservation.leaf_launch_ordinal == 1


def test_root_inherits_only_from_current_task_or_parents(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(conn, organization)
        from_current = kb.create_task(
            conn,
            title="implementation",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        from_parent = kb.create_task(
            conn,
            title="verification",
            assignee="qa",
            parents=[from_current],
        )
        ordinary = kb.create_task(conn, title="ordinary", assignee="builder")

        assert kb.get_task(conn, from_current).request_root_id == request.id
        assert kb.get_task(conn, from_parent).request_root_id == request.id
        assert kb.get_task(conn, ordinary).request_root_id is None

        other_root, _ = _accept_request(
            conn,
            organization,
            session_id="other-session",
            message_id="other-message",
        )
        with pytest.raises(ValueError, match="multiple coordination"):
            kb.create_task(
                conn,
                title="invalid cross-request child",
                assignee="builder",
                parents=[from_current, other_root],
            )


def test_leaf_launches_are_capped_but_manager_and_qa_are_not_leaf_charges(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(
            conn,
            organization,
            max_leaf_launches=1,
        )
        leaf_one = kb.create_task(
            conn,
            title="implementation one",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        leaf_two = kb.create_task(
            conn,
            title="implementation two",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        manager = kb.create_task(
            conn,
            title="director decision",
            assignee="director",
            coordination_source_task_id=root_id,
        )
        qa = kb.create_task(
            conn,
            title="quality gate",
            assignee="qa",
            coordination_source_task_id=root_id,
        )

        assert kb.reserve_coordination_launch(
            conn, leaf_one, organization=organization, now=101,
        ).leaf_launch_ordinal == 1
        assert kb.reserve_coordination_launch(
            conn, manager, organization=organization, now=101,
        ).role == "manager"
        assert kb.reserve_coordination_launch(
            conn, qa, organization=organization, now=101,
        ).role == "qa"
        with pytest.raises(kb.CoordinationBudgetExceeded, match="launch budget"):
            kb.reserve_coordination_launch(
                conn, leaf_two, organization=organization, now=101,
            )

        current = kb.get_coordination_request(conn, request.id)
        assert current.leaf_launches_used == 1
        assert current.manager_handoffs == 1


def test_concurrent_leaf_limit_uses_live_root_tasks(kanban_home, organization):
    with kb.connect_closing() as conn:
        root_id, _ = _accept_request(
            conn,
            organization,
            max_leaf_launches=4,
            max_concurrent_leaf=1,
        )
        first = kb.create_task(
            conn,
            title="first",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        second = kb.create_task(
            conn,
            title="second",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        kb.reserve_coordination_launch(
            conn, first, organization=organization, now=101,
        )
        assert kb.claim_task(conn, first) is not None

        with pytest.raises(kb.CoordinationLaunchDeferred, match="concurrency"):
            kb.reserve_coordination_launch(
                conn, second, organization=organization, now=101,
            )


def test_atomic_dispatch_never_runs_third_leaf_and_releases_capacity(
    kanban_home, organization,
):
    with kb.connect_closing() as setup:
        root_id, request = _accept_request(
            setup,
            organization,
            max_leaf_launches=4,
            max_concurrent_leaf=2,
        )
        tasks = [
            kb.create_task(
                setup,
                title=f"leaf {index}",
                assignee="builder",
                coordination_source_task_id=root_id,
            )
            for index in range(3)
        ]

    first_conn = kb.connect()
    second_conn = kb.connect()
    try:
        first, _ = kb.claim_task_for_dispatch(
            first_conn, tasks[0], organization=organization, now=101,
        )
        second, _ = kb.claim_task_for_dispatch(
            second_conn, tasks[1], organization=organization, now=101,
        )
        assert first is not None and second is not None
        with pytest.raises(kb.CoordinationLaunchDeferred, match="concurrency"):
            kb.claim_task_for_dispatch(
                second_conn, tasks[2], organization=organization, now=101,
            )
        assert kb.get_task(second_conn, tasks[2]).status == "ready"
        assert second_conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE request_root_id = ? "
            "AND status = 'running'",
            (request.id,),
        ).fetchone()[0] == 2

        assert kb.complete_task(
            first_conn,
            tasks[0],
            summary="first leaf done",
            expected_run_id=first.current_run_id,
        )
        third, _ = kb.claim_task_for_dispatch(
            second_conn, tasks[2], organization=organization, now=102,
        )
        assert third is not None
        assert second_conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE request_root_id = ? "
            "AND status = 'running'",
            (request.id,),
        ).fetchone()[0] == 2
        assert kb.get_coordination_request(
            second_conn, request.id
        ).leaf_launches_used == 3
    finally:
        first_conn.close()
        second_conn.close()


def test_dispatch_caps_worker_runtime_to_remaining_request_checkpoint(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, _ = _accept_request(conn, organization)
        task_id = kb.create_task(
            conn,
            title="bounded process",
            assignee="builder",
            coordination_source_task_id=root_id,
            max_runtime_seconds=9999,
        )
        claimed, _ = kb.claim_task_for_dispatch(
            conn, task_id, organization=organization, now=1250,
        )
        assert claimed is not None
        assert claimed.max_runtime_seconds == 50


def test_root_completion_waits_for_entire_cohort_then_enters_one_return_phase(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(conn, organization)
        with pytest.raises(kb.CoordinationLaunchDeferred, match="final return"):
            kb.claim_task_for_dispatch(
                conn, root_id, organization=organization, now=101,
            )
        assert not kb.begin_coordination_final_return_if_ready(
            conn, request.id, now=101,
        )
        assert not kb.complete_task(conn, root_id, summary="premature")

        hidden_worker = kb.create_task(
            conn,
            title="unlinked cohort work",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        assert kb.complete_task(conn, hidden_worker, summary="verified")
        assert kb.begin_coordination_final_return_if_ready(
            conn, request.id, now=102,
        )
        assert not kb.begin_coordination_final_return_if_ready(
            conn, request.id, now=103,
        )
        pending = [
            event for event in kb.list_events(conn, root_id)
            if event.kind == "coordination_return_pending"
        ][-1]
        assert kb.validate_coordination_final_return_authority(
            conn,
            request_root_id=request.id,
            task_id=root_id,
            event_id=pending.id,
            responsible_agent="aurora",
        ) == kb.get_coordination_request(conn, request.id)
        with pytest.raises(ValueError, match="not terminal"):
            kb.validate_coordination_final_return_authority(
                conn,
                request_root_id=request.id,
                task_id=root_id,
                event_id=pending.id,
                responsible_agent="aurora",
                require_terminal=True,
            )
        with pytest.raises(ValueError, match="responsible agent"):
            kb.validate_coordination_final_return_authority(
                conn,
                request_root_id=request.id,
                task_id=root_id,
                event_id=pending.id,
                responsible_agent="director",
            )
        with pytest.raises(kb.CoordinationLaunchDeferred, match="final return"):
            kb.claim_task_for_dispatch(
                conn, root_id, organization=organization, now=103,
            )
        assert kb.complete_task(conn, root_id, summary="final aggregation")
        assert kb.get_coordination_request(conn, request.id).status == "return_pending"
        assert kb.validate_coordination_final_return_authority(
            conn,
            request_root_id=request.id,
            task_id=root_id,
            event_id=pending.id,
            responsible_agent="aurora",
            require_terminal=True,
        )
        with pytest.raises(ValueError, match="returned_message_id"):
            kb.acknowledge_coordination_return(
                conn,
                request.id,
                event_id=pending.id,
                responsible_agent="aurora",
                returned_message_id="",
                now=104,
            )
        assert kb.acknowledge_coordination_return(
            conn,
            request.id,
            event_id=pending.id,
            responsible_agent="aurora",
            returned_message_id="buzz-message-42",
            now=104,
        )
        assert not kb.acknowledge_coordination_return(
            conn,
            request.id,
            event_id=pending.id,
            responsible_agent="aurora",
            returned_message_id="buzz-message-42",
            now=105,
        )
        assert kb.get_coordination_request(conn, request.id).status == "completed"
        route = kb.list_notify_subs(conn, root_id)[0]
        assert route["last_event_id"] == pending.id
        events = [event.kind for event in kb.list_events(conn, root_id)]
        assert events.count("coordination_return_pending") == 1
        assert events.count("coordination_return_acknowledged") == 1


def test_coordination_tick_probe_and_batch_return_exact_unclaimed_route(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(conn, organization)
        child_id = kb.create_task(
            conn,
            title="verified cohort work",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        assert kb.complete_task(conn, child_id, summary="verified")
        db_path = Path(
            conn.execute("PRAGMA database_list").fetchone()["file"]
        )

        assert not kb.has_coordination_tick_work(
            db_path, notifier_profiles={"director"}
        )
        assert kb.has_coordination_tick_work(
            db_path, notifier_profiles={"aurora"}
        )
        assert kb.prepare_coordination_final_return_deliveries(
            conn, notifier_profiles={"director"}, now=102,
        ) == []

        deliveries = kb.prepare_coordination_final_return_deliveries(
            conn, notifier_profiles={"aurora"}, now=102,
        )
        repeated = kb.prepare_coordination_final_return_deliveries(
            conn, notifier_profiles={"aurora"}, now=103,
        )

        assert len(deliveries) == 1
        delivery = deliveries[0]
        assert repeated == deliveries
        assert delivery["db_path"] == str(db_path.resolve())
        assert delivery["request_root_id"] == request.id
        assert delivery["task_id"] == root_id
        assert delivery["event_kind"] == "coordination_return_pending"
        assert delivery["event_payload"]["request_root_id"] == request.id
        assert delivery["responsible_agent"] == "aurora"
        assert delivery["origin_session_id"] == "aurora-session"
        assert delivery["origin_message_id"] == "message-1"
        assert delivery["root_session_id"] == "aurora-session"
        assert delivery["subscription"]["notifier_profile"] == "aurora"
        assert delivery["subscription"]["delivery_mode"] == "wake"
        assert delivery["old_cursor"] < delivery["event_id"]
        assert (
            kb.list_notify_subs(conn, root_id)[0]["last_event_id"]
            == delivery["old_cursor"]
        )


def test_pending_coordination_route_is_retained_until_exact_ack(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(conn, organization)
        child_id = kb.create_task(
            conn,
            title="cohort work",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        assert kb.complete_task(conn, child_id, summary="verified")
        delivery = kb.prepare_coordination_final_return_deliveries(
            conn, notifier_profiles={"aurora"}, now=102,
        )[0]
        assert kb.complete_task(conn, root_id, summary="final aggregation")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = 1 WHERE task_id = ?",
                (root_id,),
            )

        assert kb.purge_stale_done_notify_subs(conn, max_age_days=1) == 0
        assert len(kb.list_notify_subs(conn, root_id)) == 1
        assert kb.acknowledge_coordination_return(
            conn,
            request.id,
            event_id=delivery["event_id"],
            responsible_agent="aurora",
            returned_message_id="message-receipt-1",
            now=104,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = 1 WHERE task_id = ?",
                (root_id,),
            )
        assert kb.purge_stale_done_notify_subs(conn, max_age_days=1) == 1
        assert kb.list_notify_subs(conn, root_id) == []
        assert not kb.has_coordination_tick_work(
            kanban_home / "kanban.db", notifier_profiles={"aurora"}
        )


def test_model_calls_reserve_two_final_attempts_and_roots_are_independent(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        first_root, first = _accept_request(conn, organization)
        second_root, second = _accept_request(
            conn,
            organization,
            session_id="aurora-session-2",
            message_id="message-2",
        )

        for ordinal in range(1, 39):
            assert kb.charge_coordination_model_call(
                conn, first.id, purpose="work", task_id=first_root, now=101,
            ) == ordinal
        with pytest.raises(kb.CoordinationBudgetExceeded, match="reserve preserved"):
            kb.charge_coordination_model_call(
                conn, first.id, purpose="work", task_id=first_root, now=101,
            )
        manager = kb.create_task(
            conn,
            title="must not spawn after work calls are exhausted",
            assignee="director",
            coordination_source_task_id=first_root,
        )
        with pytest.raises(kb.CoordinationBudgetExceeded, match="reserve preserved"):
            kb.claim_task_for_dispatch(
                conn, manager, organization=organization, now=101,
            )
        assert kb.get_task(conn, manager).status == "ready"
        assert kb.mark_coordination_guardrail(
            conn,
            first.id,
            task_id=first_root,
            reason="work complete",
            now=1300,
        )
        assert kb.charge_coordination_model_call(
            conn,
            first.id,
            purpose="final_return",
            task_id=first_root,
            now=1301,
        ) == 39
        assert kb.charge_coordination_model_call(
            conn,
            first.id,
            purpose="final_return",
            task_id=first_root,
            now=1301,
        ) == 40
        with pytest.raises(kb.CoordinationBudgetExceeded, match="aggregate"):
            kb.charge_coordination_model_call(
                conn,
                first.id,
                purpose="final_return",
                task_id=first_root,
                now=1301,
            )

        assert kb.charge_coordination_model_call(
            conn, second.id, task_id=second_root, now=101,
        ) == 1
        assert kb.charge_coordination_model_call(conn, None) is None
        assert kb.get_coordination_request(conn, first.id).model_calls_used == 40
        assert kb.get_coordination_request(conn, second.id).model_calls_used == 1


def test_retry_and_elapsed_limits_survive_reopen(kanban_home, organization):
    with kb.connect_closing() as conn:
        root_id, request = _accept_request(conn, organization)
        task_id = kb.create_task(
            conn,
            title="retry candidate",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        assert not kb.reserve_coordination_retry(
            conn,
            request.id,
            task_id=task_id,
            classification="deterministic",
            cause_code="authentication",
            now=101,
        )
        assert kb.reserve_coordination_retry(
            conn,
            request.id,
            task_id=task_id,
            classification="transient",
            cause_code="provider_timeout",
            now=101,
        )

    with kb.connect_closing() as conn:
        assert not kb.reserve_coordination_retry(
            conn,
            request.id,
            task_id=task_id,
            classification="transient",
            cause_code="provider_timeout",
            now=101,
        )
        with pytest.raises(kb.CoordinationBudgetExceeded, match="checkpoint"):
            kb.reserve_coordination_launch(
                conn, task_id, organization=organization, now=1300,
            )


@pytest.mark.parametrize(
    ("assignee", "expected_leaf_launches"),
    [("builder", 2), ("director", 0), ("qa", 0)],
)
def test_crashed_worker_any_role_gets_one_root_retry_then_blocks(
    kanban_home, organization, monkeypatch, assignee, expected_leaf_launches,
):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("signaled", 15))
    monkeypatch.setattr(kb.time, "time", lambda: 101)

    with kb.connect_closing() as conn:
        root_id, request = _accept_request(
            conn, organization, max_transient_retries=1,
        )
        task_id = kb.create_task(
            conn,
            title="retry one crashed worker",
            assignee=assignee,
            coordination_source_task_id=root_id,
        )

        first, _ = kb.claim_task_for_dispatch(
            conn, task_id, organization=organization, now=101,
        )
        assert first is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = 0 WHERE id = ?",
            (41001, task_id),
        )
        conn.commit()
        assert kb.detect_crashed_workers(conn) == [task_id]
        assert kb.get_task(conn, task_id).status == "ready"
        after_first = kb.get_coordination_request(conn, request.id)
        assert after_first is not None
        assert after_first.status == "active"
        assert after_first.transient_retries_used == 1

        second, _ = kb.claim_task_for_dispatch(
            conn, task_id, organization=organization, now=102,
        )
        assert second is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = 0 WHERE id = ?",
            (41002, task_id),
        )
        conn.commit()
        assert kb.detect_crashed_workers(conn) == [task_id]

        assert kb.get_task(conn, task_id).status == "blocked"
        exhausted = kb.get_coordination_request(conn, request.id)
        assert exhausted is not None
        assert exhausted.status == "return_pending"
        assert exhausted.transient_retries_used == 1
        assert exhausted.leaf_launches_used == expected_leaf_launches
        kinds = [event.kind for event in kb.list_events(conn, root_id)]
        assert kinds.count("coordination_retry_reserved") == 1
        assert kinds.count("coordination_retry_exhausted") == 1
        assert kinds.count("coordination_guardrail_reached") == 1


def test_coordinated_quota_exit_without_retry_blocks_for_final_return(
    kanban_home, organization, monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("rate_limited", 75))

    with kb.connect_closing() as conn:
        root_id, request = _accept_request(
            conn, organization, max_transient_retries=0,
        )
        task_id = kb.create_task(
            conn,
            title="quota-bound worker",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        claimed, _ = kb.claim_task_for_dispatch(
            conn, task_id, organization=organization, now=101,
        )
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = 0 WHERE id = ?",
            (42001, task_id),
        )
        conn.commit()

        assert kb.detect_crashed_workers(conn) == []
        assert task_id in kb.detect_crashed_workers._last_rate_limited
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.consecutive_failures == 0
        stopped = kb.get_coordination_request(conn, request.id)
        assert stopped is not None
        assert stopped.status == "return_pending"
        assert stopped.transient_retries_used == 0


def test_timed_out_worker_uses_same_root_retry_budget(
    kanban_home, organization, monkeypatch,
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb.time, "time", lambda: 101)

    with kb.connect_closing() as conn:
        root_id, request = _accept_request(
            conn, organization, max_transient_retries=1,
        )
        task_id = kb.create_task(
            conn,
            title="timeout-bound worker",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        for pid in (43001, 43002):
            claimed, _ = kb.claim_task_for_dispatch(
                conn, task_id, organization=organization, now=101,
            )
            assert claimed is not None
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, started_at = 0, "
                "max_runtime_seconds = 1 WHERE id = ?",
                (pid, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = 0 WHERE id = ?",
                (claimed.current_run_id,),
            )
            conn.commit()
            assert task_id in kb.enforce_max_runtime(
                conn, signal_fn=lambda *_args: None,
            )

        assert kb.get_task(conn, task_id).status == "blocked"
        stopped = kb.get_coordination_request(conn, request.id)
        assert stopped is not None
        assert stopped.status == "return_pending"
        assert stopped.transient_retries_used == 1


def test_protocol_violation_is_deterministic_and_not_retried(
    kanban_home, organization, monkeypatch,
):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("clean_exit", 0))
    monkeypatch.setattr(kb.time, "time", lambda: 101)

    with kb.connect_closing() as conn:
        root_id, request = _accept_request(
            conn, organization, max_transient_retries=1,
        )
        task_id = kb.create_task(
            conn,
            title="worker must close its durable task",
            assignee="builder",
            coordination_source_task_id=root_id,
        )
        claimed, _ = kb.claim_task_for_dispatch(
            conn, task_id, organization=organization, now=101,
        )
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = 0 WHERE id = ?",
            (44001, task_id),
        )
        conn.commit()

        assert kb.detect_crashed_workers(conn) == [task_id]
        assert kb.get_task(conn, task_id).status == "blocked"
        stopped = kb.get_coordination_request(conn, request.id)
        assert stopped is not None
        assert stopped.status == "return_pending"
        assert stopped.transient_retries_used == 0


def _owned_failure_task(conn):
    body = {
        "kind": "workforce_handoff",
        "state": "pending_acknowledgment",
        "source_agent": "director",
        "target_agent": "builder",
        "requires_source_acceptance": True,
        "context": {
            "kind": "owned_operational_failure",
            "technical_owner": "builder",
            "director": "director",
            "workflow_id": "scheduled-repair",
            "event_id": "failure-1",
        },
    }
    return kb.create_task(
        conn,
        title="Repair scheduled failure",
        body=json.dumps(body),
        assignee="builder",
        triage=True,
    )


def test_owned_failure_factory_has_fixed_internal_caps_and_no_user_route(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        task_id = _owned_failure_task(conn)
        request = kb.create_owned_failure_coordination_request(
            conn, root_task_id=task_id, organization=organization, now=100,
        )
        again = kb.create_owned_failure_coordination_request(
            conn, root_task_id=task_id, organization=organization, now=999,
        )

        assert again == request
        assert request.kind == "owned_operational_failure"
        assert request.origin_session_id == "internal:owned_operational_failure"
        assert request.origin_message_id == task_id
        assert request.responsible_agent == "builder"
        assert request.max_leaf_launches == 2
        assert request.max_concurrent_leaf == 1
        assert request.max_model_calls == 20
        assert request.final_model_call_reserve == 3
        assert request.max_transient_retries == 1
        assert request.checkpoint_at == 1300
        assert kb.get_task(conn, task_id).request_root_id == request.id
        assert conn.execute(
            "SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0

        for ordinal in range(1, 18):
            assert kb.charge_coordination_model_call(
                conn, request.id, purpose="work", task_id=task_id, now=101,
            ) == ordinal
        with pytest.raises(kb.CoordinationBudgetExceeded, match="reserve preserved"):
            kb.charge_coordination_model_call(
                conn, request.id, purpose="work", task_id=task_id, now=101,
            )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', assignee = 'director' "
                "WHERE id = ?",
                (task_id,),
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_required",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "required_successes": 2,
                },
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_verified",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "success_event_ids": ["success-1", "success-2"],
                    "success_orders": [11, 12],
                    "required_successes": 2,
                },
            )
        claimed, _ = kb.claim_task_for_dispatch(
            conn,
            task_id,
            review=True,
            organization=organization,
            now=86_500,
        )
        assert claimed is not None
        for ordinal in range(18, 21):
            assert kb.charge_coordination_model_call(
                conn,
                request.id,
                purpose="terminal_review",
                task_id=task_id,
                now=86_500,
            ) == ordinal
        with pytest.raises(kb.CoordinationBudgetExceeded, match="final_return requires"):
            kb.charge_coordination_model_call(
                conn, request.id, purpose="final_return", task_id=task_id,
            )


def test_owned_failure_review_waits_for_recovery_then_launches_after_checkpoint(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        task_id = _owned_failure_task(conn)
        request = kb.create_owned_failure_coordination_request(
            conn, root_task_id=task_id, organization=organization, now=100,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', assignee = 'director' "
                "WHERE id = ?",
                (task_id,),
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_required",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "required_successes": 2,
                },
            )
        with pytest.raises(kb.CoordinationLaunchDeferred, match="awaits"):
            kb.reserve_coordination_launch(
                conn, task_id, organization=organization, now=86_500,
            )

        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_verified",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "success_event_ids": ["success-1", "success-2"],
                    "success_orders": [11, 12],
                    "required_successes": 2,
                },
            )
        reservation = kb.reserve_coordination_launch(
            conn, task_id, organization=organization, now=86_500,
        )
        assert reservation.request_root_id == request.id
        assert reservation.role == "manager"
        assert reservation.leaf_launch_ordinal is None


def test_verified_owned_failure_three_call_review_reaches_acceptance(
    kanban_home, organization, monkeypatch,
):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent import turn_context
    from hermes_cli.workforce_handoffs import acknowledge_handoff, create_handoff

    now = int(time.time())
    iso = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
    with kb.connect_closing() as conn:
        created = create_handoff(
            conn,
            source_agent="director",
            target_agent="builder",
            expected_outcome="Repair the failing scheduled workflow",
            acceptance_test="Two distinct later executions succeed",
            evidence_references=["workflow:test"],
            acknowledgment_deadline=iso(now + 60),
            checkpoint_at=iso(now + 240),
            organization=organization,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "builder",
                "director": "director",
                "workflow_id": "scheduled-repair",
                "event_id": "failure-1",
            },
            requires_source_acceptance=True,
        )
        task_id = created["task_id"]
        request = kb.create_owned_failure_coordination_request(
            conn, root_task_id=task_id, organization=organization, now=now,
        )
        acknowledge_handoff(
            conn, task_id, actor="builder", organization=organization, now=now + 1,
        )
        owner = kb.claim_task(conn, task_id, claimer="builder:test")
        assert owner is not None
        for ordinal in range(1, 18):
            assert kb.charge_coordination_model_call(
                conn, request.id, purpose="work", task_id=task_id, now=now + 2,
            ) == ordinal
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_required",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "required_successes": 2,
                },
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_verified",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "success_event_ids": ["success-1", "success-2"],
                    "success_orders": [11, 12],
                    "required_successes": 2,
                },
            )
        assert kb.request_review(
            conn,
            task_id,
            summary="Repair complete; recovery evidence attached",
            expected_run_id=owner.current_run_id,
        )
        reviewer, _ = kb.claim_task_for_dispatch(
            conn, task_id, review=True, organization=organization, now=now + 300,
        )
        assert reviewer is not None
        assert reviewer.assignee == "director"

        monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
        title_agent = SimpleNamespace(
            platform="cli",
            session_id="review-session",
            _session_db=object(),
            _session_db_created=True,
        )
        with patch("agent.title_generator.maybe_auto_title") as title_call:
            turn_context._maybe_title_session_at_turn_start(
                title_agent,
                [{"role": "user", "content": "Review the repaired task"}],
            )
        title_call.assert_not_called()

        # The implementation used every general-purpose call, leaving exactly
        # the fixed three-call terminal-review reserve. With no auxiliary title
        # charge, the reviewer can inspect the task and evidence before using
        # its third call to return the source-acceptance transition.
        for offset, ordinal in enumerate(range(18, 21), start=1):
            assert kb.charge_coordination_model_call(
                conn,
                request.id,
                purpose="terminal_review",
                task_id=task_id,
                now=now + 300 + offset,
            ) == ordinal
        assert kb.complete_task(
            conn,
            task_id,
            summary="Source accepted the verified repair",
            expected_run_id=reviewer.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.get_coordination_request(conn, request.id).status == "completed"

        # Acceptance landed before the reserve was exhausted; a later process
        # cannot spend another call or reopen the completed request.
        with pytest.raises(kb.CoordinationBudgetExceeded, match="aggregate"):
            kb.charge_coordination_model_call(
                conn,
                request.id,
                purpose="terminal_review",
                task_id=task_id,
                now=now + 304,
            )
        kb._record_worker_exit(424242, 256)
        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.get_coordination_request(conn, request.id).status == "completed"


def test_verified_owned_failure_completion_keeps_bound_tail_and_never_retries(
    kanban_home, organization,
):
    from datetime import datetime, timezone

    from hermes_cli.workforce_handoffs import acknowledge_handoff, create_handoff

    now = int(time.time())
    iso = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
    with kb.connect_closing() as conn:
        created = create_handoff(
            conn,
            source_agent="director",
            target_agent="builder",
            expected_outcome="Repair the failing scheduled workflow",
            acceptance_test="Two distinct later executions succeed",
            evidence_references=["workflow:test"],
            acknowledgment_deadline=iso(now + 60),
            checkpoint_at=iso(now + 240),
            organization=organization,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "builder",
                "director": "director",
                "workflow_id": "scheduled-repair",
                "event_id": "failure-1",
            },
            requires_source_acceptance=True,
        )
        task_id = created["task_id"]
        request = kb.create_owned_failure_coordination_request(
            conn, root_task_id=task_id, organization=organization, now=now,
        )
        acknowledge_handoff(
            conn, task_id, actor="builder", organization=organization, now=now + 1,
        )
        owner = kb.claim_task(conn, task_id, claimer="builder:test")
        assert owner is not None
        for ordinal in range(1, 18):
            assert kb.charge_coordination_model_call(
                conn, request.id, purpose="work", task_id=task_id, now=now + 2,
            ) == ordinal
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_required",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "required_successes": 2,
                },
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_verified",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "success_event_ids": ["success-1", "success-2"],
                    "success_orders": [11, 12],
                    "required_successes": 2,
                },
            )
        assert kb.request_review(
            conn,
            task_id,
            summary="Repair complete; recovery evidence attached",
            expected_run_id=owner.current_run_id,
        )
        reviewer, _ = kb.claim_task_for_dispatch(
            conn, task_id, review=True, organization=organization, now=now + 300,
        )
        assert reviewer is not None
        assert reviewer.assignee == "director"
        assert kb.charge_coordination_model_call(
            conn,
            request.id,
            purpose="terminal_review",
            task_id=task_id,
            now=now + 301,
        ) == 18
        assert kb.complete_task(
            conn,
            task_id,
            summary="Source accepted the verified repair",
            expected_run_id=reviewer.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.get_coordination_request(conn, request.id).status == "completed"

        # The same still-running reviewer turn may render its post-tool answer
        # from the reserved tail, but no later process can reopen the work.
        assert kb.charge_coordination_model_call(
            conn,
            request.id,
            purpose="terminal_review",
            task_id=task_id,
            now=now + 302,
        ) == 19
        assert kb.charge_coordination_model_call(
            conn,
            request.id,
            purpose="terminal_review",
            task_id=task_id,
            now=now + 303,
        ) == 20
        with pytest.raises(kb.CoordinationBudgetExceeded, match="aggregate"):
            kb.charge_coordination_model_call(
                conn,
                request.id,
                purpose="terminal_review",
                task_id=task_id,
                now=now + 304,
            )
        kb._record_worker_exit(424242, 256)
        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.get_coordination_request(conn, request.id).status == "completed"


def test_trusted_origin_factory_allows_zero_transient_retries(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        _, request = _accept_request(
            conn, organization, max_transient_retries=0,
        )
        assert request.max_transient_retries == 0
