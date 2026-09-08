"""Portfolio reads expose decision intake without turning signals into work."""

import json

import pytest

from hermes_cli import kanban_db as kb
from plugins.workforce_control.store import record_signal
from tests.hermes_cli.test_coordination_requests import kanban_home, organization
from tools import kanban_tools as kt


@pytest.fixture
def portfolio(organization, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("hermes_cli.workforce_org.active_workforce_agent", lambda: organization.get("aurora"))
    return organization


def signal(conn, name="Investigate unsupported account failure"):
    return record_signal(
        conn, source_agent="builder", expected_outcome=name, goal_ref="unknown",
        observation="Existing task remains blocked; no user decision is established.",
        evidence_references=["task:existing"],
    )["task_id"]


def listed(**args):
    return json.loads(kt._handle_list({"workforce_scope": "portfolio_outcomes", "limit": 12, **args}))


def test_actual_signal_tool_to_portfolio_consumer_is_read_only_and_non_executing(portfolio, monkeypatch):
    from tools import workforce_signal_tool

    monkeypatch.setattr(workforce_signal_tool, "active_workforce_agent", lambda: portfolio.get("builder"))
    receipt = json.loads(workforce_signal_tool._handle({
        "expected_outcome": "Investigate unsupported account failure",
        "observation": "Existing task is blocked without a supported reserved decision.",
        "estimated_effort": "one bounded triage", "department_recommendation": "Review existing evidence",
        "evidence_references": ["task:existing"],
    }))
    task_id = receipt["signal_id"]
    with kb.connect_closing() as conn:
        ordinary = kb.create_task(conn, title="Unrelated dependency-free todo", assignee="builder")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (ordinary,))
        before = list(conn.iterdump())
        result = listed()
        assert result["count"] == 1 and not result["truncated"]
        row = result["tasks"][0]
        assert row["id"] == task_id and row["status"] == "blocked"
        assert row["workforce_record_kind"] == "unresolved_signal"
        assert row["decision_owner"] == "aurora" and row["triage_only"] is True
        assert row["launch_authorized"] is False and row["current_run_id"] is None
        assert result["promoted"] == 0
        assert json.loads(kt._handle_list({"workforce_scope": "owned_outcomes"}))["tasks"] == []
        assert list(conn.iterdump()) == before
        assert kb.get_task(conn, ordinary).status == "todo"
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM coordination_requests").fetchone()[0] == 0


def test_signals_and_outcomes_share_one_sorted_candidate_limit(portfolio):
    with kb.connect_closing() as conn:
        high = signal(conn, "Higher-priority qualified signal")
        low = signal(conn, "Lower-priority qualified signal")
        root = kb.create_task(conn, title="Owned outcome", assignee="builder", priority=5, triage=True)
        assert kb.decompose_triage_task(
            conn, root, root_assignee="aurora", author="test", auto_promote=False,
            children=[{"title": "Execution", "assignee": "qa", "parents": []}],
        )
        ordinary = kb.create_task(conn, title="Assigned but not owned", assignee="aurora", priority=10)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET priority = 9 WHERE id = ?", (high,))
            conn.execute("UPDATE tasks SET priority = 1 WHERE id = ?", (low,))
        result = listed(limit=2)
        assert [row["id"] for row in result["tasks"]] == [high, root]
        assert result["count"] == 2 and result["truncated"] and result["next_limit"] == 4
        assert "triage_only" not in result["tasks"][1]
        assert ordinary not in {row["id"] for row in result["tasks"]}


def test_bounded_portfolio_keeps_new_and_reobserved_old_signals_visible_among_many_roots(portfolio):
    with kb.connect_closing() as conn:
        old = signal(conn, "Old signal with a fresh canonical observation")
        recent = signal(conn, "New qualified signal")
        roots = []
        for index in range(14):
            root = kb.create_task(conn, title=f"Owned outcome {index}", assignee="builder", priority=10, triage=True)
            assert kb.decompose_triage_task(
                conn, root, root_assignee="aurora", author="test", auto_promote=False,
                children=[{"title": f"Execution {index}", "assignee": "qa", "parents": []}],
            )
            roots.append(root)
        for index in range(14):
            stale = signal(conn, f"Unchanged older signal {index}")
            with kb.write_txn(conn):
                conn.execute("UPDATE wc_items SET updated_at = 100 WHERE task_id = ?", (stale,))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET created_at = 1 WHERE id = ?", (old,))
            conn.execute("UPDATE wc_items SET updated_at = 200 WHERE task_id = ?", (old,))
            conn.execute("UPDATE wc_items SET updated_at = 300 WHERE task_id = ?", (recent,))
        before = list(conn.iterdump())
        result = listed(limit=12)
        ids = [row["id"] for row in result["tasks"]]
        assert len(ids) == 12 and result["truncated"]
        assert ids[0] == recent and ids[2] == old
        assert sum(task_id in roots for task_id in ids) == 6
        assert sum(row.get("triage_only", False) for row in result["tasks"]) == 6
        assert all(row["launch_authorized"] is False for row in result["tasks"] if row.get("triage_only"))
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize("damage", [
    "resolved", "archived", "running", "malformed_body", "wrong_owner", "wrong_identity",
    "wrong_source", "authorized", "unclassified",
])
def test_unqualified_signals_fail_closed(portfolio, damage):
    with kb.connect_closing() as conn:
        task_id = signal(conn)
        task = kb.get_task(conn, task_id)
        body = json.loads(task.body)
        with kb.write_txn(conn):
            if damage == "resolved":
                conn.execute("UPDATE wc_items SET current_state = 'complete' WHERE task_id = ?", (task_id,))
            elif damage in {"archived", "running"}:
                conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (damage, task_id))
            elif damage == "malformed_body":
                conn.execute("UPDATE tasks SET body = 'not json' WHERE id = ?", (task_id,))
            elif damage == "unclassified":
                conn.execute("DELETE FROM wc_items WHERE task_id = ?", (task_id,))
            else:
                key, value = {"wrong_owner": ("decision_owner", "qa"),
                              "wrong_identity": ("stable_key", "forged"),
                              "wrong_source": ("source_agent", "qa"),
                              "authorized": ("launch_authorized", True)}[damage]
                body[key] = value
                conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (json.dumps(body), task_id))
        assert listed(include_archived=True)["tasks"] == []


def test_signal_scope_is_exact_actor_and_honors_status_and_tenant(portfolio, monkeypatch):
    with kb.connect_closing() as conn:
        task_id = signal(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET tenant = 'finance' WHERE id = ?", (task_id,))
        assert listed(status="ready")["tasks"] == []
        assert listed(tenant="other")["tasks"] == []
        assert listed(status="blocked", tenant="finance")["tasks"][0]["id"] == task_id
        monkeypatch.setattr("hermes_cli.workforce_org.active_workforce_agent", lambda: portfolio.get("director"))
        assert listed()["tasks"] == []


def test_portfolio_listing_does_not_initialize_optional_signal_schema(portfolio):
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'wc_items'").fetchone() is None
        assert listed()["tasks"] == []
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'wc_items'").fetchone() is None
