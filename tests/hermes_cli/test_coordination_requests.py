from __future__ import annotations

from pathlib import Path

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
        )
        second = kb.create_coordination_request(
            conn,
            root_task_id=root_id,
            origin_session_id="session-1",
            origin_message_id="message-1",
            organization=organization,
            now=999,
        )

        assert first == second
        assert first.id == kb.coordination_request_id("session-1", "message-1")
        assert first.checkpoint_at == 1300
        assert kb.get_task(conn, root_id).request_root_id == first.id
        accepted = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'coordination_request_accepted'",
            (root_id,),
        ).fetchone()[0]
        assert accepted == 1


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

        with pytest.raises(kb.CoordinationBudgetExceeded, match="concurrency"):
            kb.reserve_coordination_launch(
                conn, second, organization=organization, now=101,
            )


def test_model_calls_reserve_two_final_attempts_and_roots_are_independent(
    kanban_home, organization,
):
    with kb.connect_closing() as conn:
        _, first = _accept_request(conn, organization)
        _, second = _accept_request(
            conn,
            organization,
            session_id="aurora-session-2",
            message_id="message-2",
        )

        for ordinal in range(1, 39):
            assert kb.charge_coordination_model_call(
                conn, first.id, purpose="work", now=101,
            ) == ordinal
        with pytest.raises(kb.CoordinationBudgetExceeded, match="reserve preserved"):
            kb.charge_coordination_model_call(
                conn, first.id, purpose="work", now=101,
            )
        assert kb.charge_coordination_model_call(
            conn, first.id, purpose="final_return", now=1301,
        ) == 39
        assert kb.charge_coordination_model_call(
            conn, first.id, purpose="final_return", now=1301,
        ) == 40
        with pytest.raises(kb.CoordinationBudgetExceeded, match="aggregate"):
            kb.charge_coordination_model_call(
                conn, first.id, purpose="final_return", now=1301,
            )

        assert kb.charge_coordination_model_call(conn, second.id, now=101) == 1
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
