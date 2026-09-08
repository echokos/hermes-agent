"""Exercise canonical task ownership through real runtime profile boundaries."""

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt


@pytest.fixture
def runtime_identity_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    for profile in ("main", "foo", "leaf", "foreignleaf"):
        (home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "main"))
    monkeypatch.setenv("HERMES_PROFILE", "main")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "workspaces"))
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_COORDINATION_REQUEST_ROOT",
        "HERMES_COORDINATION_TASK_ID", "HERMES_COORDINATION_PURPOSE",
    ):
        monkeypatch.delenv(key, raising=False)
    agents = []
    for name, profile, manager, reports in (
        ("elliott", None, None, ["root", "main"]),
        ("root", "main", "elliott", ["leaf"]),
        ("main", "foo", "elliott", ["foreignleaf"]),
        ("leaf", "leaf", "root", []),
        ("foreignleaf", "foreignleaf", "main", []),
    ):
        agents.append({
            "agent": name,
            "display_name": name.title(),
            "status": "active" if profile else "artifact",
            "operational": bool(profile),
            "department": "Operations" if profile else None,
            "function": "Manager" if reports else "Implementation",
            "manager": manager,
            "direct_reports": reports,
            "mission": "Exercise runtime identity boundaries",
            "owned_outcomes": [],
            "authority": [],
            "prohibited_actions": [],
            "escalation_target": manager,
            "cross_team_request_path": None,
            "buzz_rooms": [],
            "profile_path": str(home / "profiles" / profile) if profile else None,
        })
    organization = tmp_path / "organization.yaml"
    organization.write_text(
        yaml.safe_dump({"schema_version": 1, "agents": agents}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(organization))
    from gateway.session_context import reset_session_vars

    reset_session_vars()
    kb.init_db()
    yield home
    reset_session_vars()


def test_runtime_creator_card_passes_canonical_owner_completion(
    runtime_identity_home, monkeypatch,
):
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Root work", assignee="root")
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    result = json.loads(kt._handle_create({
        "title": "Independent follow-up", "assignee": "leaf",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child.created_by == "root"
        assert kb.parent_ids(conn, child.id) == []
        assert kb.complete_task(
            conn, parent, created_cards=[child.id], fire_lifecycle_hook=False,
        )
        assert kb.get_task(conn, parent).status == "done"


@pytest.mark.parametrize(
    ("scope", "expected_owners"),
    [("owned_outcomes", {"root"}), ("portfolio_outcomes", {"root", "leaf"})],
)
def test_runtime_outcome_scope_filters_canonical_owners(
    runtime_identity_home, scope, expected_owners,
):
    outcomes = {}
    with kb.connect_closing() as conn:
        for owner in ("root", "main", "leaf"):
            task = kb.create_task(
                conn, title=f"{owner} outcome", assignee=owner, triage=True,
            )
            assert kb.decompose_triage_task(
                conn, task, root_assignee="orchestrator",
                children=[{"title": "Execution", "assignee": "leaf", "parents": []}],
                author="test", auto_promote=False,
            )
            outcomes[owner] = task
    result = json.loads(kt._handle_list({"workforce_scope": scope}))
    assert "error" not in result
    assert {task["id"] for task in result["tasks"]} == {
        outcomes[owner] for owner in expected_owners
    }
    assert set(result["scope_profiles"]) == expected_owners


def test_secondary_api_creation_and_subscription_use_actual_runtime(
    runtime_identity_home, monkeypatch,
):
    from agent.coordination_budget import scoped_coordination_budget
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import clear_session_vars, get_session_env

    monkeypatch.setenv("HERMES_HOME", str(runtime_identity_home / "profiles" / "foo"))
    monkeypatch.setenv("HERMES_PROFILE", "foo")
    with APIServerAdapter._profile_scope("main"):
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id="api-root-origin", session_id="api-root-origin",
        )
        try:
            assert get_session_env("HERMES_SESSION_PROFILE", "") == ""
            assert kt._task_creator() == "root"
            with scoped_coordination_budget() as scope:
                # The host turn supplies its stable addressed-message identity.
                scope.origin_message_id = "api-root-message"
                result = json.loads(kt._handle_create({
                    "title": "API accepted root", "assignee": "root",
                    "report_to_origin": True, "coordination": {},
                }))
        finally:
            clear_session_vars(tokens)
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert task.created_by == "root"
        request = kb.get_coordination_request(conn, result["request_root_id"])
        assert request.responsible_agent == "root"
        route = conn.execute(
            "SELECT notifier_profile FROM kanban_notify_subs WHERE task_id = ?",
            (task.id,),
        ).fetchone()
        assert route["notifier_profile"] == "main"


def test_default_api_runtime_cannot_accept_named_process_manager_root(
    runtime_identity_home, monkeypatch,
):
    from agent.coordination_budget import scoped_coordination_budget
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import clear_session_vars

    monkeypatch.setenv("HERMES_PROFILE", "main")
    with APIServerAdapter._profile_scope("default"):
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id="api-default-origin", session_id="api-default-origin",
        )
        try:
            with scoped_coordination_budget() as scope:
                scope.origin_message_id = "api-default-message"
                result = json.loads(kt._handle_create({
                    "title": "Invalid API acceptance", "assignee": "root",
                    "report_to_origin": True, "coordination": {},
                }))
        finally:
            clear_session_vars(tokens)
    assert "error" in result
    with kb.connect_closing() as conn:
        for table in ("tasks", "coordination_requests", "kanban_notify_subs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("assignee", ["root", "main"])
def test_named_runtime_authorizes_its_canonical_manager_only(
    runtime_identity_home, monkeypatch, assignee,
):
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_PROFILE", "foo")
    tokens = set_session_vars(
        platform="telegram", chat_id="root-origin", chat_type="dm",
        session_id="root-origin", message_id="root-origin-message",
        profile="foo",
    )
    try:
        result = json.loads(kt._handle_create({
            "title": "Named runtime acceptance", "assignee": assignee,
            "report_to_origin": True, "coordination": {},
        }))
    finally:
        clear_session_vars(tokens)
    with kb.connect_closing() as conn:
        if assignee == "main":
            assert "error" in result
            for table in ("tasks", "coordination_requests", "kanban_notify_subs"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        else:
            assert result["ok"] is True
            task = kb.get_task(conn, result["task_id"])
            assert task.created_by == task.assignee == "root"
            request = kb.get_coordination_request(conn, result["request_root_id"])
            assert request.responsible_agent == "root"
            route = conn.execute(
                "SELECT notifier_profile FROM kanban_notify_subs WHERE task_id = ?",
                (task.id,),
            ).fetchone()
            assert route["notifier_profile"] == "main"


def test_generic_creation_without_organization_keeps_runtime_author(
    runtime_identity_home, monkeypatch,
):
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG", str(runtime_identity_home / "missing-organization.yaml"),
    )
    result = json.loads(kt._handle_create({
        "title": "Generic board work", "assignee": "legacy-worker",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, result["task_id"]).created_by == "main"


def test_legacy_runtime_owner_can_complete_with_canonical_created_card(
    runtime_identity_home, monkeypatch,
):
    organization = runtime_identity_home.parent / "organization.yaml"
    data = yaml.safe_load(organization.read_text(encoding="utf-8"))
    data["agents"] = [
        agent for agent in data["agents"]
        if agent["agent"] not in {"main", "foreignleaf"}
    ]
    next(agent for agent in data["agents"] if agent["agent"] == "elliott")[
        "direct_reports"
    ] = ["root"]
    organization.write_text(yaml.safe_dump(data), encoding="utf-8")
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Legacy Root work", assignee="main")
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    result = json.loads(kt._handle_create({
        "title": "Canonical follow-up", "assignee": "leaf",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child.created_by == "root"
        assert kb.complete_task(
            conn, parent, created_cards=[child.id], fire_lifecycle_hook=False,
        )
        assert kb.get_task(conn, parent).assignee == "main"
        assert kb.get_task(conn, parent).status == "done"


def test_colliding_canonical_owner_cannot_claim_runtime_owners_created_card(
    runtime_identity_home, monkeypatch,
):
    with kb.connect_closing() as conn:
        root_parent = kb.create_task(conn, title="Root work", assignee="root")
        other_parent = kb.create_task(conn, title="Canonical Main work", assignee="main")
    monkeypatch.setenv("HERMES_KANBAN_TASK", root_parent)
    result = json.loads(kt._handle_create({
        "title": "Root created follow-up", "assignee": "leaf",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child.created_by == "root"
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, other_parent, created_cards=[child.id],
                fire_lifecycle_hook=False,
            )
        assert kb.get_task(conn, other_parent).status != "done"


@pytest.mark.parametrize("registry_state", ["absent", "malformed", "ambiguous", "encoding"])
def test_untrusted_registry_does_not_grant_created_card_alias_ownership(
    runtime_identity_home, registry_state,
):
    from hermes_cli.workforce_org import WorkforceOrganizationError, load_organization

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Main assigned work", assignee="main")
    result = json.loads(kt._handle_create({
        "title": "Root created follow-up", "assignee": "leaf",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, result["task_id"]).created_by == "root"

    organization = runtime_identity_home.parent / "organization.yaml"
    if registry_state == "absent":
        organization.unlink()
    elif registry_state == "malformed":
        organization.write_text("agents: [\n", encoding="utf-8")
    elif registry_state == "encoding":
        organization.write_bytes(b"\xff")
    else:
        data = yaml.safe_load(organization.read_text(encoding="utf-8"))
        for agent in data["agents"]:
            if agent["agent"] == "elliott":
                agent["direct_reports"] = ["root", "secondary"]
            elif agent["agent"] == "main":
                agent["agent"] = "secondary"
                agent["profile_path"] = str(runtime_identity_home / "profiles" / "main")
            elif agent["agent"] == "foreignleaf":
                agent["manager"] = "secondary"
        organization.write_text(yaml.safe_dump(data), encoding="utf-8")
        org = load_organization()
        with pytest.raises(WorkforceOrganizationError, match="ambiguous"):
            org.resolve_profile("main")

    with kb.connect_closing() as conn:
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent, created_cards=[result["task_id"]],
                fire_lifecycle_hook=False,
            )
        assert kb.get_task(conn, parent).status != "done"


def test_invalid_registry_encoding_preserves_generic_creator_and_exact_proof(
    runtime_identity_home, monkeypatch,
):
    organization = runtime_identity_home.parent / "organization.yaml"
    organization.write_bytes(b"\xff")
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Generic main work", assignee="main")
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    result = json.loads(kt._handle_create({
        "title": "Generic follow-up", "assignee": "legacy-worker",
    }))
    assert result["ok"] is True
    with kb.connect_closing() as conn:
        child = kb.get_task(conn, result["task_id"])
        assert child.created_by == "main"
        assert kb.complete_task(
            conn, parent, created_cards=[child.id], fire_lifecycle_hook=False,
        )
        assert kb.get_task(conn, parent).status == "done"
