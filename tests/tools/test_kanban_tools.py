"""Tests for the Kanban tool surface (tools/kanban_tools.py).

Verifies:
  - Tools are gated on HERMES_KANBAN_TASK: a normal chat session sees
    zero kanban tools in its schema; a worker session sees the kanban set.
  - Each handler's happy path.
  - Error paths (missing required args, bad metadata type, etc).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_kanban_tools_hidden_without_env_var(monkeypatch, tmp_path):
    """Normal `hermes chat` sessions (no HERMES_KANBAN_TASK) must have
    zero kanban_* tools in their schema."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    assert kanban == set(), (
        f"kanban tools leaked into normal chat schema: {kanban}"
    )


def test_coordination_schema_requires_root_before_worker_cards():
    from tools.kanban_tools import KANBAN_CREATE_SCHEMA

    description = KANBAN_CREATE_SCHEMA["parameters"]["properties"][
        "report_to_origin"
    ]["description"]
    assert "Create this root before worker cards" in description
    assert "Build worker cards first" not in description


# ---------------------------------------------------------------------------
# Handler happy paths
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a worker: HERMES_HOME isolated, HERMES_KANBAN_TASK set
    after we've created the task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def test_show_defaults_to_env_task_id(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_show({})
    d = json.loads(out)
    assert "task" in d
    assert d["task"]["id"] == worker_env
    assert d["task"]["status"] == "running"
    assert "worker_context" in d
    assert "runs" in d
    assert "coordination_budget" not in d


def test_list_filters_tasks(monkeypatch, worker_env):
    """kanban_list gives orchestrators filtered board discovery."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="alpha", assignee="factory", priority=5)
        b = kb.create_task(conn, title="beta", assignee="reviewer")
        c = kb.create_task(conn, title="gamma", assignee="factory", tenant="other")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({"assignee": "factory", "status": "ready", "limit": 10})
    d = json.loads(out)
    ids = [t["id"] for t in d["tasks"]]
    assert ids == [a, c]
    assert d["count"] == 2
    assert d["tasks"][0]["title"] == "alpha"
    assert d["tasks"][0]["parent_count"] == 0
    assert b not in ids

    tenant_out = kt._handle_list({
        "assignee": "factory",
        "status": "ready",
        "tenant": "other",
    })
    tenant_ids = [t["id"] for t in json.loads(tenant_out)["tasks"]]
    assert tenant_ids == [c]


def test_list_workforce_outcomes_use_explicit_root_ownership(monkeypatch, worker_env):
    """Assignments and cross-department execution are not outcome ownership."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from types import SimpleNamespace
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    agents = {
        "director": SimpleNamespace(
            agent="director", profile_path="/profiles/director", direct_reports=("worker",)
        ),
        "worker": SimpleNamespace(
            agent="worker", profile_path="/profiles/worker", direct_reports=()
        ),
    }
    org = SimpleNamespace(get=lambda agent_id: agents[agent_id])
    monkeypatch.setattr(
        "hermes_cli.workforce_org.active_workforce_agent", lambda: agents["director"]
    )
    monkeypatch.setattr("hermes_cli.workforce_org.load_organization", lambda: org)

    conn = kb.connect()
    try:
        ordinary = kb.create_task(conn, title="ordinary", assignee="director", priority=8)
        cross_team = kb.create_task(conn, title="cross-team", assignee="worker", priority=9)
        own = kb.create_task(
            conn, title="director outcome", assignee="director", priority=2, triage=True,
        )
        report = kb.create_task(
            conn, title="report outcome", assignee="worker", priority=3, triage=True,
        )
        assert kb.decompose_triage_task(
            conn,
            own,
            root_assignee="orchestrator",
            children=[{"title": "own execution", "assignee": "worker", "parents": []}],
            author="test",
            auto_promote=False,
        )
        assert kb.decompose_triage_task(
            conn,
            report,
            root_assignee="orchestrator",
            children=[{"title": "report execution", "assignee": "other", "parents": []}],
            author="test",
            auto_promote=False,
        )
        outside = kb.create_task(conn, title="outside", assignee="other", priority=9)
    finally:
        conn.close()

    result = json.loads(kt._handle_list({
        "workforce_scope": "portfolio_outcomes", "status": "todo", "limit": 12,
    }))
    ids = [task["id"] for task in result["tasks"]]
    assert ids == [report, own]
    assert outside not in ids
    assert ordinary not in ids
    assert cross_team not in ids
    assert result["scope_profiles"] == ["director", "worker"]

    error = json.loads(kt._handle_list({
        "workforce_scope": "portfolio_outcomes", "assignee": "other",
    }))
    assert "mutually exclusive" in error["error"]

    self_result = json.loads(kt._handle_list({
        "workforce_scope": "owned_outcomes", "status": "todo", "limit": 12,
    }))
    assert [task["id"] for task in self_result["tasks"]] == [own]
    assert self_result["scope_profiles"] == ["director"]

    list_properties = kt.KANBAN_LIST_SCHEMA["parameters"]["properties"]
    show_properties = kt.KANBAN_SHOW_SCHEMA["parameters"]["properties"]
    assert list_properties["workforce_scope"]["enum"] == [
        "owned_outcomes", "portfolio_outcomes",
    ]
    assert "workforce_scope" not in show_properties


def test_owned_outcomes_fail_closed_for_legacy_decomposition(monkeypatch, worker_env):
    """A legacy root without owner evidence is never guessed from children."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from types import SimpleNamespace
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    actor = SimpleNamespace(
        agent="director", profile_path="/profiles/director", direct_reports=()
    )
    monkeypatch.setattr(
        "hermes_cli.workforce_org.active_workforce_agent", lambda: actor
    )
    monkeypatch.setattr(
        "hermes_cli.workforce_org.load_organization",
        lambda: SimpleNamespace(get=lambda _agent_id: actor),
    )
    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="legacy root", assignee="orchestrator")
        kb._append_event(conn, root, "decomposed", {"child_ids": ["legacy-child"]})
        conn.commit()
    finally:
        conn.close()

    result = json.loads(kt._handle_list({"workforce_scope": "owned_outcomes"}))
    assert result["tasks"] == []


def test_archive_stale_is_orchestrator_only_and_evidence_grounded(
    monkeypatch, worker_env
):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        stale = kb.create_task(conn, title="cancelled candidate", assignee="emily")
        kb.add_comment(conn, stale, "elliott", "STOP per Elliott. Do not continue.")
    finally:
        conn.close()

    refused = json.loads(kt._handle_archive_stale({
        "task_id": stale,
        "reason": "explicit stop",
        "evidence_quote": "STOP per Elliott. Do not continue.",
    }))
    assert "orchestrator-only" in refused["error"]

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    result = json.loads(kt._handle_archive_stale({
        "task_id": stale,
        "reason": "explicit stop",
        "evidence_quote": "STOP per Elliott. Do not continue.",
    }))
    assert result["ok"] is True
    assert result["status"] == "archived"
    with kb.connect() as conn:
        assert kb.get_task(conn, stale).status == "archived"


def test_complete_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({
        "summary": "got the thing done",
        "metadata": {"files": 2},
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"] == worker_env
    # Verify via kernel
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.summary == "got the thing done"
        assert run.metadata == {"files": 2}
    finally:
        conn.close()


def test_complete_retry_with_empty_created_cards_succeeds(worker_env):
    """After a phantom rejection, retrying kanban_complete with
    created_cards=[] (the documented escape hatch) must complete the
    task. Regression for #22923."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Hit the gate first.
    rejected = json.loads(kt._handle_complete({
        "summary": "oops",
        "created_cards": ["t_phantomdeadbeef"],
    }))
    assert rejected.get("error")

    # Retry with the escape hatch.
    ok = json.loads(kt._handle_complete({
        "summary": "retry without claims",
        "created_cards": [],
    }))
    assert ok.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_complete_goal_mode_rejected_by_judge(monkeypatch, tmp_path):
    """Goal-mode tasks must pass the auxiliary judge before completion.
    Regression for #38367: workers bypassing the judge via early kanban_complete."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Set up isolated HERMES_HOME
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-test", assignee="test-worker",
            body="Must achieve X with verified evidence.", goal_mode=True
        )
        kb.claim_task(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)

    # Mock the judge to reject the completion. The gate only runs when a
    # judge is reachable, so force the availability probe True as well.
    def mock_judge_goal(goal, last_response, *, timeout=30.0, subgoals=None):
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        return "continue", "missing verification evidence", False, None, False

    monkeypatch.setattr("tools.kanban_tools.judge_goal", mock_judge_goal)
    monkeypatch.setattr("tools.kanban_tools._goal_judge_available", lambda: True)

    # Attempt to complete should be rejected
    out = kt._handle_complete({"summary": "I did some stuff but not X"})
    d = json.loads(out)
    assert "error" in d
    assert "Goal completion rejected by judge" in d["error"]
    assert "missing verification evidence" in d["error"]
    assert f"parents=[{goal_task_id}]" in d["error"]

    # Verify the task is NOT completed in the DB
    conn2 = kb.connect()
    try:
        task = kb.get_task(conn2, goal_task_id)
        assert task.status == "running"  # Should still be running, not done
    finally:
        conn2.close()


def test_block_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_block({"reason": "need clarification"})
    d = json.loads(out)
    assert d["ok"] is True
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "blocked"
    finally:
        conn.close()


def _make_goal_mode_worker_env(monkeypatch, tmp_path):
    """Set up an isolated HERMES_HOME with one claimed goal_mode task,
    matching the pattern used by the kanban_complete judge gate tests."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-block-test", assignee="test-worker",
            body="Must achieve X.", goal_mode=True,
        )
        kb.claim_task(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)
    return goal_task_id


def test_block_goal_mode_rejects_missing_kind(monkeypatch, tmp_path):
    """A goal_mode worker calling kanban_block with no kind must not be able
    to use it as an unguarded escape from the goal loop (Issue #38696,
    sibling of the kanban_complete judge gate / Issue #38367)."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    out = kt._handle_block({"reason": "giving up"})
    d = json.loads(out)
    assert "error" in d
    assert "goal_mode" in d["error"]

    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_block_goal_mode_rejects_disallowed_kind(monkeypatch, tmp_path):
    """`capability` / `transient` are valid kinds in general but must not
    let a goal_mode worker exit the loop without going through the judge."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    for kind in ("capability", "transient"):
        out = kt._handle_block({"reason": "blocked", "kind": kind})
        d = json.loads(out)
        assert "error" in d, f"kind={kind} should be rejected for goal_mode"

    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_heartbeat_extends_claim_expires(worker_env):
    """The kanban_heartbeat tool MUST extend claim_expires, not just
    update last_heartbeat_at — otherwise long-running workers loop the
    heartbeat tool diligently and still get reclaimed by
    release_stale_claims at DEFAULT_CLAIM_TTL_SECONDS.

    Regression test for the bug where _handle_heartbeat called
    heartbeat_worker but never heartbeat_claim, so claim_expires sat
    static while last_heartbeat_at advanced.
    """
    import time as _time
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Rewind claim_expires into the past so any forward movement is
    # unambiguous (avoids time.sleep flakiness).
    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (1, worker_env),
        )
        conn.commit()
        before = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()
    assert before == 1

    out = kt._handle_heartbeat({"note": "still alive"})
    assert json.loads(out).get("ok") is True

    conn = kb.connect()
    try:
        after = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()

    now = int(_time.time())
    # claim_expires should be roughly now + DEFAULT_CLAIM_TTL_SECONDS.
    # We assert a generous floor (now + half the default TTL) to keep the
    # test stable against future TTL changes.
    assert after > before, (
        f"claim_expires did not advance ({before} -> {after}); workers "
        f"would be reclaimed at TTL despite heartbeating"
    )
    assert after >= now + (kb.DEFAULT_CLAIM_TTL_SECONDS // 2), (
        f"claim_expires={after} is suspiciously close to now={now}; "
        f"expected at least now + {kb.DEFAULT_CLAIM_TTL_SECONDS // 2}"
    )


def test_comment_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env,
        "body": "hello thread",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["comment_id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        assert len(comments) == 1
        # Author defaults to HERMES_PROFILE env we set in the fixture
        assert comments[0].author == "test-worker"
        assert comments[0].body == "hello thread"
    finally:
        conn.close()


def test_comment_ignores_caller_supplied_author(worker_env):
    """``args["author"]`` is no longer honored — the author is always
    derived from ``HERMES_PROFILE`` so a worker can't forge a comment
    under an authoritative-looking name like ``hermes-system`` and
    poison the next worker's prompt context. Cross-task commenting
    itself remains unrestricted (see #19713); only the author override
    is removed.
    """
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env, "body": "hi", "author": "hermes-system",
    })
    assert json.loads(out)["ok"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        # Author comes from HERMES_PROFILE in the fixture, not the
        # caller-supplied "hermes-system" override.
        assert comments[0].author == "test-worker"
    finally:
        conn.close()


def test_create_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "child task",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"]
    assert d["status"] == "todo"  # parent isn't done yet
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        child = kb.get_task(conn, d["task_id"])
        assert child.title == "child task"
        assert child.assignee == "peer"
    finally:
        conn.close()


def test_create_binds_task_local_gateway_session_for_wake(
    monkeypatch, worker_env,
):
    """A concurrent gateway turn must not stamp the process-global session."""
    from gateway.session_context import reset_session_vars, set_session_vars
    from tools import kanban_tools as kt

    monkeypatch.setattr(
        kt,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": True}},
    )
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-process-session")
    set_session_vars(
        platform="telegram",
        chat_id="chat-42",
        chat_type="dm",
        user_id="user-42",
        session_id="current-gateway-session",
        profile="test-worker",
    )
    try:
        result = json.loads(
            kt._handle_create({"title": "wake me", "assignee": "peer"})
        )
    finally:
        reset_session_vars()

    assert result["ok"] is True
    assert result["subscribed"] is True
    assert result["session_id"] == "current-gateway-session"
    assert result["wake_attached"] is True
    assert result["delivery_mode"] == "session_wake"
    assert result["delivery_warning"] is None

    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task.session_id == "current-gateway-session"


def test_link_happy_path(worker_env):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="A", assignee="x")
        b = kb.create_task(conn, title="B", assignee="x")
    finally:
        conn.close()
    from tools import kanban_tools as kt
    out = kt._handle_link({"parent_id": a, "child_id": b})
    d = json.loads(out)
    assert d["ok"] is True


def test_unblock_happy_path(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="blocked", assignee="worker")
        kb.block_task(conn, tid, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": tid})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "ready"

    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_unblock_with_pending_parents_returns_todo(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (child,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": child})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "todo"

    conn = kb.connect()
    try:
        assert kb.get_task(conn, child).status == "todo"
    finally:
        conn.close()


def test_worker_lifecycle_through_tools(worker_env):
    """Drive the full claim -> heartbeat -> comment -> complete lifecycle
    exclusively through the tools, then verify the DB state matches what
    the dispatcher/notifier expect."""
    from tools import kanban_tools as kt

    # 1. show — worker orientation
    show = json.loads(kt._handle_show({}))
    assert show["task"]["id"] == worker_env

    # 2. heartbeat during long op
    assert json.loads(kt._handle_heartbeat({"note": "warming up"}))["ok"]

    # 3. comment for a future peer
    assert json.loads(kt._handle_comment({
        "task_id": worker_env,
        "body": "note: using stdlib sqlite3 bindings",
    }))["ok"]

    # 4. spawn a child task for follow-up
    child_out = json.loads(kt._handle_create({
        "title": "write integration test",
        "assignee": "qa",
        "parents": [worker_env],
    }))
    assert child_out["ok"]

    # 5. complete with structured handoff
    comp = json.loads(kt._handle_complete({
        "summary": "implemented + spawned QA follow-up",
        "metadata": {"child_task": child_out["task_id"]},
    }))
    assert comp["ok"]

    # Verify final state
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        parent = kb.get_task(conn, worker_env)
        assert parent.status == "done"
        assert parent.current_run_id is None
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.metadata == {"child_task": child_out["task_id"]}
        # Child is todo (parent just finished, but recompute_ready may
        # have promoted it — complete_task runs recompute internally).
        child = kb.get_task(conn, child_out["task_id"])
        assert child.status == "ready", (
            f"child should be ready after parent done, got {child.status}"
        )
        # Comment is visible
        assert len(kb.list_comments(conn, worker_env)) == 1
        # Heartbeat event recorded
        hb = [e for e in kb.list_events(conn, worker_env) if e.kind == "heartbeat"]
        assert len(hb) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# System-prompt guidance injection
# ---------------------------------------------------------------------------


def test_kanban_guidance_prompt_size_bounded():
    """KANBAN_GUIDANCE is injected into every kanban-capable process's system
    prompt and resolved once at agent init, so its size is a per-worker token
    tax paid on every spawn. Bound it as an invariant, not a change-detector:
    the ceiling (8000 chars, roughly 2000 tokens) leaves headroom above the
    current ~6.2k chars for tight additions, while catching accidental bloat
    (pasted docs, duplicated sections) before it ships to every worker.
    """
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert len(KANBAN_GUIDANCE) < 8000, (
        f"KANBAN_GUIDANCE is {len(KANBAN_GUIDANCE)} chars; it is injected into "
        "every kanban worker's system prompt — trim it or consciously re-bound "
        "this invariant with justification."
    )


def test_kanban_guidance_orchestrator_decision_ownership():
    """The orchestrator section must carry the split-brain prevention
    contract: decisions are made by the orchestrator before fan-out and
    stamped into every dependent card body."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert KANBAN_GUIDANCE.count("Decision ownership.") == 1
    assert "Never let two subtree cards decide the same question" in KANBAN_GUIDANCE
    assert "workers cannot see sibling context" in KANBAN_GUIDANCE


# ---------------------------------------------------------------------------
# Worker task-ownership enforcement (regression tests for #19534)
# ---------------------------------------------------------------------------
#
# A worker process has HERMES_KANBAN_TASK set to its own task id. The
# destructive tools (kanban_complete, kanban_block, kanban_heartbeat,
# kanban_unblock) must refuse to operate
# on any OTHER task id, even if the caller supplies an explicit `task_id`
# argument. Workers legitimately call kanban_show / kanban_list /
# kanban_comment / kanban_create / kanban_link on other tasks, so those
# are unrestricted.
#
# Orchestrator profiles (no HERMES_KANBAN_TASK in env) are intentionally
# exempt — their job is routing, and they sometimes close out child
# tasks on behalf of the child.


def test_worker_complete_rejects_foreign_task_id(worker_env):
    """A worker cannot complete a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": other, "summary": "HIJACK"})
    d = json.loads(out)
    assert d.get("ok") is not True
    assert "refusing to mutate" in d.get("error", "")

    # Sibling task must be untouched.
    conn = kb.connect()
    try:
        assert kb.get_task(conn, other).status == "ready"
    finally:
        conn.close()


def test_worker_can_comment_on_foreign_task(worker_env):
    """Cross-task commenting must remain unrestricted (#19713 policy).

    The author-forgery hardening removed args['author'] but deliberately
    did NOT add an ownership gate to kanban_comment — comments are the
    documented handoff channel between tasks. This test pins that policy
    so a future change accidentally adding ``_enforce_worker_task_ownership``
    to ``_handle_comment`` would fail CI immediately.
    """
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="sibling")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": other,
        "body": "handoff: see prior findings before starting",
    })
    d = json.loads(out)
    assert d.get("ok") is True, f"cross-task comment must succeed: {d}"

    # The comment lands on the foreign task, attributed to the worker's
    # HERMES_PROFILE — never to a caller-controlled string.
    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, other)
        assert len(comments) == 1
        assert comments[0].author == "test-worker"
        assert comments[0].body.startswith("handoff:")
    finally:
        conn.close()


def test_worker_unblock_rejects_foreign_task_id(worker_env):
    """A worker cannot unblock any task — kanban_unblock is orchestrator-only.

    The check fires before the per-task ownership check, so the error
    surface is the orchestrator-only refusal rather than the
    cross-task-ownership refusal. Either is fine — the property we're
    pinning is "worker cannot mutate foreign task via kanban_unblock".
    """
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="blocked sibling", assignee="peer")
        kb.block_task(conn, other, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": other})
    d = json.loads(out)
    err = d.get("error", "")
    assert "orchestrator-only" in err or "refusing to mutate" in err, (
        f"expected worker-rejection error, got {err}"
    )

    conn = kb.connect()
    try:
        assert kb.get_task(conn, other).status == "blocked"
    finally:
        conn.close()


def test_orchestrator_complete_any_task_allowed(monkeypatch, tmp_path):
    """Orchestrator profiles (no HERMES_KANBAN_TASK) can still complete
    any task via explicit task_id. The check only applies to workers."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="child to close out")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": tid, "summary": "orchestrator close"})
    d = json.loads(out)
    assert d.get("ok") is True and d.get("task_id") == tid


# ---------------------------------------------------------------------------
# Optional ``board`` parameter — per-call DB override
# ---------------------------------------------------------------------------
#
# The dispatcher pins the active board via HERMES_KANBAN_BOARD env var,
# but a Telegram-side orchestrator handling multiple boards needs to be
# able to route a single tool call to a specific board's DB without
# restarting Hermes. These tests pin that ``board=<slug>`` argument
# routes each handler to that board's sqlite file, and that omitting
# ``board`` preserves the legacy env-driven resolution.


@pytest.fixture
def multi_board_env(monkeypatch, tmp_path):
    """Isolated Hermes home with two distinct kanban boards seeded.

    Returns ``("default", "alt")`` slugs. The default board has one
    pre-existing task ``seed_default``; ``alt`` has ``seed_alt``. No
    HERMES_KANBAN_TASK is pinned (orchestrator context) — workers test
    the env-task case via the existing ``worker_env`` fixture.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Make sure neither HERMES_KANBAN_DB nor HERMES_KANBAN_BOARD pin a
    # board — the test is specifically about the per-call override.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    # Default board — implicit
    conn = kb.connect()
    try:
        seed_default = kb.create_task(
            conn, title="seed-default", assignee="worker-d"
        )
    finally:
        conn.close()
    # Alt board — explicit slug routes the connection to a separate DB
    conn = kb.connect(board="alt")
    try:
        seed_alt = kb.create_task(
            conn, title="seed-alt", assignee="worker-a"
        )
    finally:
        conn.close()
    return {
        "default_seed": seed_default,
        "alt_seed": seed_alt,
        "default_db": kb.kanban_db_path(),
        "alt_db": kb.kanban_db_path(board="alt"),
    }


def test_board_param_none_falls_back_to_env(worker_env):
    """When ``board`` is omitted or None, behaviour is unchanged from
    before this feature — calls land on whatever the env resolves to.
    Regression guard against accidentally rewiring default resolution."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_show({})  # no board, no task_id
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    out = kt._handle_show({"task_id": worker_env, "board": None})
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    # Sanity: the env-resolved path is the legacy default DB, NOT an
    # 'alt' board path. Confirms the override path was not silently
    # forced.
    assert kb.kanban_db_path() == kb.kanban_db_path(board="default")


# ---------------------------------------------------------------------------
# kanban_create auto-subscribe behaviour
#
# Model-created tasks do not subscribe the originating session by default.
# Operators can opt in for a profile through auto_subscribe_on_create.
# - Gateway sessions: HERMES_SESSION_PLATFORM + HERMES_SESSION_CHAT_ID set.
# - TUI sessions: HERMES_SESSION_KEY (or HERMES_SESSION_ID) set, with
#   the platform/chat_id ContextVars intentionally empty.
# - CLI / cron / test sessions: no delivery channel -> no subscription.
# - Explicit gateway /kanban create subscriptions use a separate path.
# ---------------------------------------------------------------------------

def _list_subs_for_task(task_id):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        return list(kb.list_notify_subs(conn, task_id))
    finally:
        conn.close()


def _sub_index(subs):
    """Normalise a list of notify-subs (dicts or objects) into dicts
    keyed by platform+chat_id, so assertions work regardless of the
    return shape."""
    out = []
    for s in subs:
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append({
                "platform": getattr(s, "platform", None),
                "chat_id": getattr(s, "chat_id", None),
                "thread_id": getattr(s, "thread_id", None),
                "user_id": getattr(s, "user_id", None),
                "delivery_metadata": getattr(s, "delivery_metadata", None),
                "notifier_profile": getattr(s, "notifier_profile", None),
            })
    return out


def test_create_does_not_subscribe_gateway_session_by_default(monkeypatch, worker_env):
    """Internal agent work must not leak raw events into the human chat."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-7")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-9")
    monkeypatch.setenv("HERMES_SESSION_USER_ID_ALT", "alt-user-9")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "forum")

    out = kt._handle_create({
        "title": "auto-sub gateway",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(new_tid) == []


def test_create_explicit_user_commitment_wakes_origin_without_global_opt_in(
    monkeypatch, worker_env,
):
    """One explicit commitment may close the loop without noisy fleet auto-subscribe."""
    from gateway.session_context import reset_session_vars, set_session_vars
    from tools import kanban_tools as kt

    monkeypatch.setattr(
        kt,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": False}},
    )
    set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        chat_type="dm",
        user_id="elliott",
        session_id="aurora-origin",
        profile="aurora",
    )
    try:
        result = json.loads(kt._handle_create({
            "title": "final verified user commitment",
            "assignee": "aurora",
            "report_to_origin": True,
        }))
    finally:
        reset_session_vars()

    assert result["ok"] is True
    assert result["subscribed"] is True
    assert result["report_to_origin"] is True
    subs = _sub_index(_list_subs_for_task(result["task_id"]))
    assert len(subs) == 1
    assert subs[0]["platform"] == "buzz"
    assert subs[0]["chat_id"] == "elliott-dm"
    assert subs[0]["delivery_mode"] == "wake"


def test_create_accepts_coordination_root_and_inherits_within_origin_turn(
    monkeypatch, worker_env,
):
    """The bound origin message, not model-supplied parent ids, scopes a turn."""
    from gateway.session_context import reset_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    monkeypatch.setattr(
        kt,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": True}},
    )
    set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        chat_type="dm",
        user_id="elliott",
        session_id="origin-session",
        message_id="origin-message-1",
        profile="aurora",
    )
    try:
        root = json.loads(kt._handle_create({
            "title": "coordinate accepted request",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {
                "max_leaf_launches": 3,
                "max_concurrent_leaf": 1,
                "max_model_calls": 30,
                "checkpoint_seconds": 900,
            },
        }))
        child = json.loads(kt._handle_create({
            "title": "same-turn child without model parent",
            "assignee": "sage",
        }))

        # A new inbound message in the same durable session is unrelated.
        set_session_vars(
            platform="buzz",
            chat_id="elliott-dm",
            chat_type="dm",
            user_id="elliott",
            session_id="origin-session",
            message_id="origin-message-2",
            profile="aurora",
        )
        unrelated = json.loads(kt._handle_create({
            "title": "unrelated later turn",
            "assignee": "sage",
        }))
    finally:
        reset_session_vars()

    assert root["ok"] is True
    assert root["request_root_id"].startswith("cr_")
    assert root["coordination"] == {
        "max_leaf_launches": 3,
        "max_concurrent_leaf": 1,
        "max_model_calls": 30,
        "final_model_call_reserve": 2,
        "checkpoint_at": root["coordination"]["checkpoint_at"],
    }
    with kb.connect() as conn:
        root_task = kb.get_task(conn, root["task_id"])
        child_task = kb.get_task(conn, child["task_id"])
        unrelated_task = kb.get_task(conn, unrelated["task_id"])
    assert root_task.request_root_id == root["request_root_id"]
    assert child_task.request_root_id == root["request_root_id"]
    assert unrelated_task.request_root_id is None
    assert len(_list_subs_for_task(root["task_id"])) == 1
    assert _list_subs_for_task(child["task_id"]) == []
    assert len(_list_subs_for_task(unrelated["task_id"])) == 1


def test_create_coordination_root_rolls_back_without_final_route(
    monkeypatch, worker_env,
):
    from gateway.session_context import reset_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    set_session_vars(
        session_id="origin-without-route",
        message_id="message-without-route",
        profile="aurora",
    )
    try:
        result = json.loads(kt._handle_create({
            "title": "must be atomic",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {},
        }))
    finally:
        reset_session_vars()

    assert "persistent origin route" in result["error"]
    with kb.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'must be atomic'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM coordination_requests"
        ).fetchone()[0] == 0


def test_create_coordination_tool_cannot_raise_pilot_limits(
    monkeypatch, worker_env,
):
    from gateway.session_context import reset_session_vars, set_session_vars
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        session_id="bounded-origin",
        message_id="bounded-message",
        profile="aurora",
    )
    try:
        result = json.loads(kt._handle_create({
            "title": "inflate budget",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {"max_model_calls": 41},
        }))
    finally:
        reset_session_vars()
    assert "between 1 and 40" in result["error"]


def test_concurrent_origin_turns_inherit_only_their_own_coordination_root(
    monkeypatch, worker_env,
):
    from threading import Barrier

    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    barrier = Barrier(2)

    def create_pair(index: int) -> tuple[dict, dict]:
        tokens = set_session_vars(
            platform="buzz",
            chat_id=f"origin-{index}",
            chat_type="dm",
            user_id="elliott",
            session_id=f"session-{index}",
            message_id=f"message-{index}",
            profile="aurora",
        )
        try:
            root = json.loads(kt._handle_create({
                "title": f"root-{index}",
                "assignee": "aurora",
                "report_to_origin": True,
                "coordination": {},
            }))
            barrier.wait(timeout=10)
            child = json.loads(kt._handle_create({
                "title": f"child-{index}",
                "assignee": "sage",
            }))
            return root, child
        finally:
            clear_session_vars(tokens)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(create_pair, (1, 2)))

    assert pairs[0][0]["request_root_id"] != pairs[1][0]["request_root_id"]
    with kb.connect() as conn:
        for root, child in pairs:
            child_task = kb.get_task(conn, child["task_id"])
            assert child_task.request_root_id == root["request_root_id"]


def test_same_origin_coordination_retry_returns_existing_root(
    monkeypatch, worker_env,
):
    from gateway.session_context import reset_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        session_id="retry-session",
        message_id="retry-message",
        profile="aurora",
    )
    try:
        first = json.loads(kt._handle_create({
            "title": "original root",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {},
        }))
        retry = json.loads(kt._handle_create({
            "title": "model retried after uncertain reply",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {},
        }))
    finally:
        reset_session_vars()

    assert retry["ok"] is True
    assert retry["task_id"] == first["task_id"]
    assert retry["request_root_id"] == first["request_root_id"]
    with kb.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM coordination_requests"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE request_root_id = ?",
            (first["request_root_id"],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?",
            (first["task_id"],),
        ).fetchone()[0] == 1


def test_same_origin_coordination_retry_settles_accepting_scope_calls(
    monkeypatch, worker_env,
):
    from contextlib import contextmanager

    from agent import coordination_budget
    from gateway.session_context import reset_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    set_session_vars(
        platform="buzz",
        chat_id="elliott-dm",
        session_id="settlement-race-session",
        message_id="settlement-race-message",
        profile="aurora",
    )
    try:
        first = json.loads(kt._handle_create({
            "title": "root accepted by another scope",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {},
        }))

        @contextmanager
        def raced_acceptance_binding():
            yield coordination_budget.CoordinationAcceptanceBinding(
                model_calls=2,
                scope_id="scope-that-missed-early-lookup",
            )

        monkeypatch.setattr(
            coordination_budget,
            "coordination_acceptance_binding",
            raced_acceptance_binding,
        )
        retry = json.loads(kt._handle_create({
            "title": "idempotent retry after concurrent acceptance",
            "assignee": "aurora",
            "report_to_origin": True,
            "coordination": {},
        }))
    finally:
        reset_session_vars()

    assert retry["ok"] is True
    assert retry["task_id"] == first["task_id"]
    with kb.connect() as conn:
        request = kb.get_coordination_request(conn, first["request_root_id"])
        debit = conn.execute(
            "SELECT model_calls FROM coordination_acceptance_debits "
            "WHERE request_root_id = ? AND acceptance_scope_id = ?",
            (first["request_root_id"], "scope-that-missed-early-lookup"),
        ).fetchone()
    assert request.model_calls_used == 2
    assert debit["model_calls"] == 2


def test_concurrent_child_committed_before_root_is_adopted_by_origin_message(
    monkeypatch, worker_env,
):
    import time
    from threading import Barrier

    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    barrier = Barrier(2)

    def invoke(payload: dict, delay: float) -> dict:
        tokens = set_session_vars(
            platform="buzz",
            chat_id="shared-origin",
            chat_type="dm",
            user_id="elliott",
            session_id="shared-session",
            message_id="shared-message",
            profile="aurora",
        )
        try:
            barrier.wait(timeout=10)
            time.sleep(delay)
            return json.loads(kt._handle_create(payload))
        finally:
            clear_session_vars(tokens)

    child_payload = {"title": "parallel child", "assignee": "sage"}
    root_payload = {
        "title": "parallel root",
        "assignee": "aurora",
        "report_to_origin": True,
        "coordination": {},
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        child_future = pool.submit(invoke, child_payload, 0.0)
        root_future = pool.submit(invoke, root_payload, 0.1)
        child = child_future.result(timeout=20)
        root = root_future.result(timeout=20)

    with kb.connect() as conn:
        adopted = kb.get_task(conn, child["task_id"])
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (child["task_id"],),
        ).fetchall()
    assert adopted.request_root_id == root["request_root_id"]
    assert "coordination_request_adopted" in {row["kind"] for row in events}


def test_coordinated_worker_cannot_escape_budget_on_another_board(
    monkeypatch, worker_env,
):
    from hermes_cli import kanban_db as kb
    from hermes_cli.workforce_org import load_organization
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    organization = load_organization()
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="coordinated source",
            assignee="aurora",
            session_id="source-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=source,
            platform="buzz",
            chat_id="elliott-dm",
            delivery_mode="wake",
        )
        kb.create_coordination_request(
            conn,
            root_task_id=source,
            origin_session_id="source-session",
            origin_message_id="source-message",
            organization=organization,
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", source)

    rejected = json.loads(kt._handle_create({
        "title": "cross-board escape",
        "assignee": "sage",
        "board": "other",
    }))
    assert "cannot create on another board" in rejected["error"]

    with kb.connect_closing() as conn:
        ordinary = kb.create_task(conn, title="ordinary source", assignee="sage")
    monkeypatch.setenv("HERMES_KANBAN_TASK", ordinary)
    allowed = json.loads(kt._handle_create({
        "title": "ordinary cross-board child",
        "assignee": "sage",
        "board": "other",
    }))
    assert allowed["ok"] is True
    with kb.connect_closing(board="other") as conn:
        assert kb.get_task(conn, allowed["task_id"]).request_root_id is None


def test_create_subscribes_gateway_session_when_opted_in(
    monkeypatch, worker_env, tmp_path,
):
    """The legacy model-tool behavior remains available by explicit config."""
    home = tmp_path / "opt-in-home" / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: true\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-7")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-9")
    monkeypatch.setenv("HERMES_SESSION_USER_ID_ALT", "alt-user-9")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "forum")

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "auto-sub gateway opted in",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    s = subs[0]
    assert s["platform"] == "telegram"
    assert s["chat_id"] == "chat-42"
    assert s["thread_id"] == "thread-7"
    assert s["user_id"] == "user-9"
    assert s["user_id_alt"] == "alt-user-9"
    assert s["chat_type"] == "forum"
    assert s["delivery_mode"] == "notify+wake"


def test_create_subscribes_tui_session_via_session_key(monkeypatch, worker_env):
    """TUI / desktop sessions don't have a platform/chat_id (single
    local channel), but the parent process exports HERMES_SESSION_KEY.
    We should still auto-subscribe, with platform='tui' and
    chat_id=<key>."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "tui-session-abc")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(
        kt,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": True}},
    )

    out = kt._handle_create({
        "title": "auto-sub tui",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    assert subs[0]["platform"] == "tui"
    assert subs[0]["chat_id"] == "tui-session-abc"
    assert subs[0]["chat_type"] == "dm"
    assert subs[0]["delivery_mode"] == "notify"


def test_create_does_not_subscribe_in_cli_session(monkeypatch, worker_env):
    """CLI / cron / test sessions have no persistent delivery channel.
    _maybe_auto_subscribe returns False and no row is written."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "no sub cli",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_create_respects_auto_subscribe_on_create_false(monkeypatch, worker_env, tmp_path):
    """The config gate kanban.auto_subscribe_on_create=false must
    suppress auto-subscription even when the session has a delivery
    channel. This is the knob that addresses the upstream design
    concern from PR #19718 (reverted in #19721) — users who want
    explicit kanban_notify-subscribe calls per task get that."""
    # worker_env already created <tmp>/.hermes; use a fresh sibling
    # home to avoid mkdir() colliding with the worker's directory.
    home = tmp_path / "gate-home" / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")

    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "no sub gated",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_create_fails_closed_when_subscription_config_cannot_load(
    monkeypatch, worker_env,
):
    """A config read failure cannot create an unsolicited delivery edge."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "buzz")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "elliott-dm")

    def _config_failure():
        raise RuntimeError("unreadable config")

    monkeypatch.setattr(kt, "load_config", _config_failure)

    out = kt._handle_create({
        "title": "no sub on config failure",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["subscribed"] is False, d
    assert _list_subs_for_task(d["task_id"]) == []


def test_maybe_auto_subscribe_swallows_add_notify_sub_failure(monkeypatch, worker_env):
    """If add_notify_sub itself raises (e.g. DB locked, schema drift),
    _maybe_auto_subscribe must NOT bubble that up and fail the parent
    kanban_create. The function returns False and the parent create
    still succeeds with subscribed=False."""
    from tools import kanban_tools as kt
    monkeypatch.setattr(
        kt,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": True}},
    )
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")

    from hermes_cli import kanban_db as kb

    def _boom(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(kb, "add_notify_sub", _boom)

    out = kt._handle_create({
        "title": "auto-sub tolerates add_notify_sub failure",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["subscribed"] is False, d


# ---------------------------------------------------------------------------
# Attachments — kanban_attach / kanban_attach_url / kanban_attachments
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_private_urls(monkeypatch):
    """Opt the SSRF guard into private/loopback targets for local fixtures.

    Mirrors a user setting HERMES_ALLOW_PRIVATE_URLS on a private network.
    Resets the url_safety process-lifetime cache on both sides so the
    override neither leaks in nor out of the test.
    """
    from tools import url_safety

    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def test_attach_url_rejects_non_http_scheme(worker_env):
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": "file:///etc/passwd"})
    d = json.loads(out)
    assert "error" in d
    assert "scheme" in d["error"]


# ---------------------------------------------------------------------------
# kanban_attach_url — SSRF guard (tools/url_safety.is_safe_url per hop)
# ---------------------------------------------------------------------------


@pytest.fixture
def default_url_guard(monkeypatch):
    """Force the SSRF guard to its secure default for this test.

    Clears HERMES_ALLOW_PRIVATE_URLS and resets url_safety's process-lifetime
    cache on both sides so a prior test's opt-in can't leak in.
    """
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def _assert_attach_url_blocked(worker_env, url):
    """Call kanban_attach_url with ``url`` and assert the SSRF guard fired
    (clean tool error, no attachment row, no network fetch needed)."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": url})
    d = json.loads(out)
    assert "error" in d, out
    assert "SSRF" in d["error"] or "blocked" in d["error"].lower(), out
    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, worker_env) == []
    finally:
        conn.close()


def test_attach_url_blocks_loopback(worker_env, default_url_guard):
    """http://127.0.0.1/ is rejected before any connection is made."""
    _assert_attach_url_blocked(worker_env, "http://127.0.0.1/")


def _fake_public_dns(monkeypatch, mapping):
    """Patch url_safety's getaddrinfo so hostnames in ``mapping`` resolve to
    the given (public) IPs and literal IPs resolve to themselves — no real
    DNS or network traffic."""
    import ipaddress
    import socket as _socket

    real_af, real_sock = _socket.AF_INET, _socket.SOCK_STREAM

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = mapping.get(host)
        if ip is None:
            # Literal IPs pass through; unknown hostnames fail like NXDOMAIN.
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise _socket.gaierror(f"fake DNS: unknown host {host!r}")
            ip = host
        return [(real_af, real_sock, 6, "", (ip, 0))]

    from tools import url_safety
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)


class _FakeStreamResponse:
    def __init__(self, *, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    @property
    def is_redirect(self):
        return 300 <= self.status_code < 400 and "location" in {
            k.lower() for k in self.headers
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_attach_url_happy_path_public_host(worker_env, default_url_guard, monkeypatch):
    """A public URL passes the guard and the bytes are stored (mocked fetch)."""
    from pathlib import Path

    import httpx

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    _fake_public_dns(monkeypatch, {"files.example.com": "93.184.216.34"})

    payload = b"public fetch body"

    def fake_stream(method, url, **kwargs):
        assert url == "http://files.example.com/docs/spec.pdf"
        return _FakeStreamResponse(
            status_code=200,
            headers={"content-type": "application/pdf; charset=binary"},
            body=payload,
        )

    monkeypatch.setattr(httpx, "stream", fake_stream)

    out = kt._handle_attach_url({"url": "http://files.example.com/docs/spec.pdf"})
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["size"] == len(payload)

    conn = kb.connect()
    try:
        atts = kb.list_attachments(conn, worker_env)
        assert [a.filename for a in atts] == ["spec.pdf"]
        assert atts[0].content_type == "application/pdf"
        assert Path(atts[0].stored_path).read_bytes() == payload
    finally:
        conn.close()
