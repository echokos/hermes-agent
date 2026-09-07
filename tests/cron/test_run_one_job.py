"""Characterization + unit tests for the `run_one_job` shared helper (Phase 4A).

`tick`'s per-job body (`_process_job`) is the execute → save → deliver → mark
sequence that fires ONE due job. Phase 4A extracts it into a module-level
`run_one_job(job, *, adapters=None, loop=None, verbose=False)` so the external
Chronos provider's `fire_due` can reuse the IDENTICAL body — no duplicated
correctness.

The first test characterizes the sequence as driven through `tick()` (proving
the extraction didn't change `tick`'s behavior); the rest unit-test the
extracted helper directly.
"""
import json

import cron.scheduler as s


def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, silent_marker_in=None):
    """Patch the job pipeline primitives and record the call order."""
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **_kw):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(
        s,
        "claim_job_for_fire",
        lambda job_id, **_kw: {"id": job_id, "name": "t"},
    )
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "advance_next_runs", lambda ids: 1)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_workflow_status_marker_is_stripped_and_stored(monkeypatch):
    delivered = []
    marked = []

    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **kwargs: (
            True,
            "out",
            "The source CSV has no data row.\n\n[WORKFLOW_STATUS:blocked]",
            None,
        ),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **kwargs: delivered.append(content),
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, workflow_status=None: marked.append(
            (jid, ok, workflow_status)
        ),
    )

    assert s.run_one_job(
        {"id": "workflow-job", "name": "prepare", "track_workflow_status": True}
    )

    assert delivered == ["The source CSV has no data row."]
    assert marked == [("workflow-job", True, "blocked")]


def test_failed_workflow_marker_settles_all_ledgers_as_failed(monkeypatch):
    from hermes_cli import workflow_registry as reg

    with reg.connect_closing() as conn:
        reg.create_definition(
            conn,
            id="wf-cron-marker-failure",
            slug="cron-marker-failure",
            name="Cron Marker Failure",
            owner_profile="default",
            status="active",
            runtime_kind="hermes",
        )
        reg.replace_steps(
            conn,
            "wf-cron-marker-failure",
            [{"step_key": "collect", "position": 0, "name": "Collect"}],
        )

    delivered = []
    marked = []
    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **kwargs: (
            True,
            "out",
            "Sanitized failure body.\n[WORKFLOW_STATUS:failed]",
            None,
        ),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **kwargs: delivered.append(content),
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, workflow_status=None: marked.append(
            (jid, ok, err, workflow_status)
        ),
    )

    assert s.run_one_job(
        {
            "id": "workflow-marker-failure-job",
            "name": "marker failure job",
            "workflow_id": "wf-cron-marker-failure",
            "workflow_step_key": "collect",
            "track_workflow_status": True,
        }
    )

    assert len(delivered) == 1
    assert "[WORKFLOW_STATUS:failed]" not in delivered[0]
    assert marked == [
        (
            "workflow-marker-failure-job",
            False,
            "Workflow reported failed outcome.",
            "failed",
        )
    ]
    with reg.connect_closing() as conn:
        run = reg.list_runs(conn, "wf-cron-marker-failure")[0]
        step = conn.execute("SELECT * FROM workflow_step_runs").fetchone()
    assert run.status == "failed"
    assert run.error == "Workflow reported failed outcome."
    assert step["status"] == "failed"
    assert step["summary"] == "Sanitized failure body."


def test_workflow_failure_marker_embedded_in_prose_is_not_control_data():
    response = "The literal [WORKFLOW_STATUS:failed] appears in documentation."

    status, body = s._extract_workflow_status(response)

    assert status == "unknown"
    assert body == response


def test_run_one_job_records_workflow_registry_run(monkeypatch):
    from hermes_cli import workflow_registry as reg

    with reg.connect_closing() as conn:
        reg.create_definition(
            conn,
            id="wf-cron",
            slug="cron-workflow",
            name="Cron Workflow",
            owner_profile="default",
            status="active",
            runtime_kind="hermes",
        )
        reg.replace_steps(
            conn,
            "wf-cron",
            [{"step_key": "collect", "position": 0, "name": "Collect"}],
        )

    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **kwargs: (
            True,
            "out",
            "Collected context.\n[WORKFLOW_STATUS:completed]",
            None,
        ),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(s, "_deliver_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *args, **kwargs: None)

    assert s.run_one_job(
        {
            "id": "workflow-job",
            "name": "workflow job",
            "workflow_id": "wf-cron",
            "workflow_step_key": "collect",
            "track_workflow_status": True,
        }
    )

    with reg.connect_closing() as conn:
        runs = reg.list_runs(conn, "wf-cron")
        step_runs = conn.execute("SELECT * FROM workflow_step_runs").fetchall()

    assert len(runs) == 1
    assert runs[0].status == "succeeded"
    assert runs[0].trigger_kind == "cron"
    assert runs[0].trigger_ref == "workflow-job"
    assert len(step_runs) == 1
    assert step_runs[0]["step_key"] == "collect"
    assert step_runs[0]["status"] == "succeeded"
    assert step_runs[0]["summary"] == "Collected context."


def test_blocked_cron_workflow_terminalizes_non_resumable_parent(monkeypatch):
    from hermes_cli import workflow_registry as reg

    with reg.connect_closing() as conn:
        reg.create_definition(
            conn, id="wf-cron-blocked", slug="cron-workflow-blocked",
            name="Cron Workflow Blocked", owner_profile="default",
            status="active", runtime_kind="hermes",
        )
        reg.replace_steps(
            conn, "wf-cron-blocked",
            [{"step_key": "collect", "position": 0, "name": "Collect"}],
        )
    monkeypatch.setattr(
        s, "run_job",
        lambda job, **kwargs: (
            True, "out", "Need a retained decision.\n[WORKFLOW_STATUS:blocked]", None
        ),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(s, "_deliver_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *args, **kwargs: None)

    assert s.run_one_job({
        "id": "workflow-blocked-job", "name": "blocked job",
        "workflow_id": "wf-cron-blocked", "workflow_step_key": "collect",
        "track_workflow_status": True,
    })

    with reg.connect_closing() as conn:
        run = reg.list_runs(conn, "wf-cron-blocked")[0]
        step = conn.execute("SELECT * FROM workflow_step_runs").fetchone()
    assert run.status == "cancelled"
    assert run.ended_at is not None
    assert "future fire" in run.error
    assert step["status"] == "waiting_for_approval"


def test_failed_workflow_run_does_not_create_kanban_task(monkeypatch):
    from hermes_cli import workflow_registry as reg
    from hermes_constants import get_default_hermes_root

    with reg.connect_closing() as conn:
        reg.create_definition(
            conn,
            id="wf-cron-failure",
            slug="cron-workflow-failure",
            name="Cron Workflow Failure",
            owner_profile="default",
            status="active",
            runtime_kind="hermes",
        )
        reg.replace_steps(
            conn,
            "wf-cron-failure",
            [{"step_key": "collect", "position": 0, "name": "Collect"}],
        )

    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **kwargs: (False, "", "", "injected failure"),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(s, "_deliver_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *args, **kwargs: None)

    assert s.run_one_job(
        {
            "id": "workflow-failure-job",
            "name": "workflow failure job",
            "workflow_id": "wf-cron-failure",
            "workflow_step_key": "collect",
        }
    )

    with reg.connect_closing() as conn:
        runs = reg.list_runs(conn, "wf-cron-failure")
        step_runs = conn.execute("SELECT * FROM workflow_step_runs").fetchall()

    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].error == "injected failure"
    assert runs[0].kanban_task_id is None
    assert len(step_runs) == 1
    assert step_runs[0]["status"] == "failed"
    assert not (get_default_hermes_root() / "kanban.db").exists()


def test_tracked_workflow_without_marker_is_unknown(monkeypatch):
    marked = []
    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **kwargs: (True, "out", "Prepared the report.", None),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/out")
    monkeypatch.setattr(s, "_deliver_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, workflow_status=None: marked.append(
            (ok, workflow_status)
        ),
    )

    s.run_one_job(
        {"id": "workflow-unknown", "name": "prepare", "track_workflow_status": True}
    )

    assert marked == [(True, "unknown")]


def test_run_one_job_silent_skips_delivery(monkeypatch):
    """A [SILENT] final response saves output + marks the run but does NOT
    deliver."""
    calls = _patch_pipeline(monkeypatch, silent_marker_in="[SILENT]")

    s.run_one_job({"id": "j3", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "run_job" in kinds and "save" in kinds and "mark" in kinds
    assert "deliver" not in kinds


def test_run_one_job_empty_response_is_soft_failure(monkeypatch):
    """An empty final response marks the run as NOT ok (issue #8585)."""
    calls = _patch_pipeline(monkeypatch, final="   ")

    s.run_one_job({"id": "j4", "name": "t"})

    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j4", False)


def test_run_one_job_failed_job_delivers_error(monkeypatch):
    """A failed job still delivers (the error notice) and marks not-ok."""
    calls = _patch_pipeline(monkeypatch, success=False, final="", error="boom")

    s.run_one_job({"id": "j5", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "deliver" in kinds  # failures always deliver
    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j5", False)


def test_required_workforce_signal_absence_cannot_leave_a_cron_run_green(monkeypatch):
    from cron.executions import list_executions
    from hermes_cli import workflow_registry as reg

    with reg.connect_closing() as conn:
        reg.create_definition(
            conn,
            id="wf-required-signal",
            slug="required-signal",
            name="Required Signal",
            owner_profile="chloe",
            status="active",
            runtime_kind="hermes",
        )
        reg.replace_steps(
            conn,
            "wf-required-signal",
            [{"step_key": "observe", "position": 0, "name": "Observe"}],
        )
    calls = _patch_pipeline(monkeypatch, success=True, final="apparently complete")
    marked = []
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, **kwargs: marked.append((jid, ok, err, kwargs)),
    )
    ok = s.run_one_job({
        "id": "chloe-control", "name": "Chloe control plane",
        "workflow_id": "wf-required-signal",
        "workflow_step_key": "observe",
        "track_workflow_status": True,
        "required_workforce_signal": True,
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    })
    # ``run_one_job`` returns whether the processing path completed; the
    # authoritative execution/run record must still be failed.
    assert ok is True
    assert [call[0] for call in calls] == ["run_job", "save"]
    assert len(marked) == 1
    assert marked[0][0:2] == ("chloe-control", False)
    assert "not submitted" in marked[0][2]
    assert marked[0][3]["workflow_status"] == "failed"
    executions = list_executions(job_id="chloe-control")
    assert len(executions) == 1
    assert executions[0]["status"] == "failed"
    assert "not submitted" in executions[0]["error"]
    with reg.connect_closing() as conn:
        workflow_run = reg.list_runs(conn, "wf-required-signal")[0]
        workflow_step = conn.execute(
            "SELECT * FROM workflow_step_runs WHERE workflow_run_id = ?",
            (workflow_run.id,),
        ).fetchone()
    assert workflow_run.status == "failed"
    assert workflow_step["status"] == "failed"
    assert not any(call[0] == "deliver" for call in calls)


def test_optional_observer_signal_contract_allows_verified_quiet_run(monkeypatch):
    calls = _patch_pipeline(monkeypatch, success=True, final="[SILENT]")

    assert s.run_one_job({
        "id": "chloe-quiet", "name": "Chloe factual reconciliation",
        "observe_workforce_signal_attempts": True,
    })

    assert ("mark", "chloe-quiet", True) in calls
    assert "deliver" not in [call[0] for call in calls]


def test_failed_optional_signal_attempt_cannot_be_hidden_by_no_action_prose(
    monkeypatch, tmp_path,
):
    calls = []

    def failed_signal_run(job, *, defer_agent_teardown=None, **_kwargs):
        from tools.workforce_signal_runtime import mark_failure

        mark_failure("signal persistence offline")
        return True, "output", "[NO_ACTION]", None

    monkeypatch.setattr(s, "run_job", failed_signal_run)
    monkeypatch.setattr(s, "save_job_output", lambda *args, **kwargs: "/tmp/out")
    monkeypatch.setattr(
        s, "_deliver_result", lambda *args, **kwargs: calls.append("deliver")
    )
    marked = []
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, **kwargs: marked.append((jid, ok, err)),
    )
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    assert s.run_one_job({
        "id": "chloe-attempt", "name": "Chloe factual reconciliation",
        "workflow_id": "workforce-proactive-operating-cycles",
        "observe_workforce_signal_attempts": True,
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    })

    assert marked[0][0:2] == ("chloe-attempt", False)
    assert "persistence offline" in marked[0][2]
    assert calls == []


def test_owned_failure_is_persisted_and_not_delivered(monkeypatch, tmp_path):
    calls = _patch_pipeline(
        monkeypatch, success=False, final="", error="upstream timeout"
    )
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    assert s.run_one_job({
        "id": "owned-job", "name": "owned job", "workflow_id": "owned-workflow",
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    })

    assert "deliver" not in [call[0] for call in calls]
    intake = (tmp_path / "cron" / "operational-failures.jsonl").read_text().splitlines()
    assert len(intake) == 1
    assert json.loads(intake[0])["execution_id"]


def test_owned_exception_is_persisted_and_not_delivered(monkeypatch, tmp_path):
    calls = []

    def raise_from_job(job, *, defer_agent_teardown=None, **_kwargs):
        raise RuntimeError("raised failure")

    monkeypatch.setattr(s, "run_job", raise_from_job)
    monkeypatch.setattr(
        s, "_deliver_result", lambda *args, **kwargs: calls.append("deliver")
    )
    monkeypatch.setattr(s, "mark_job_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    assert s.run_one_job({
        "id": "owned-exception", "name": "owned exception",
        "workflow_id": "owned-exception-workflow",
        "failure_ownership": {
            "technical_owner": "root", "director": "aurora",
        },
    }) is False

    assert calls == []
    intake = (tmp_path / "cron" / "operational-failures.jsonl").read_text().splitlines()
    assert len(intake) == 1
    assert "raised failure" in json.loads(intake[0])["sanitized_error"]


def test_run_one_job_operator_only_script_failure_skips_delivery(monkeypatch):
    """Pre-run collector failures remain visible to operators, not chat."""
    calls = _patch_pipeline(
        monkeypatch,
        success=False,
        final="",
        error=s.OPERATOR_ONLY_SCRIPT_FAILURE + "collector exited 2",
    )

    s.run_one_job({"id": "j5-script", "name": "collector"})

    kinds = [c[0] for c in calls]
    assert "deliver" not in kinds
    assert ("mark", "j5-script", False) in calls


def test_run_one_job_exception_marks_failure(monkeypatch):
    """If run_job raises, the helper marks the run failed and returns False
    rather than propagating."""
    def boom(job, *, defer_agent_teardown=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(s, "run_job", boom)
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: marks.append((jid, ok)),
    )

    ok = s.run_one_job({"id": "j6", "name": "t"})

    assert ok is False
    assert marks == [("j6", False)]


def test_run_one_job_installs_secret_scope_under_multiplex(monkeypatch, tmp_path):
    """Regression: under profile isolation (multiplex active), run_one_job must
    execute run_job inside a profile secret scope so credential reads
    (resolve_runtime_provider -> get_secret) don't fail-close with
    UnscopedSecretError, and must tear the scope down afterward.

    Behavior contract: a scope is present during run_job and absent after,
    regardless of the concrete secret values.
    """
    from agent import secret_scope as ss

    # Point cron's home resolution at a profile whose .env carries a secret.
    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    scope_during_run = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    ss.set_multiplex_active(True)
    try:
        ok = s.run_one_job({"id": "j7", "name": "t"})
    finally:
        ss.set_multiplex_active(False)

    assert ok is True
    # Scope was installed during run_job and the profile secret resolved.
    assert scope_during_run["scope"] is not None
    assert scope_during_run["base_url"] == "https://openrouter.ai/api/v1"
    # And it was torn down after run_one_job returned (no leak).
    assert ss.current_secret_scope() is None

def test_run_one_job_delivers_before_agent_teardown(monkeypatch):
    """Regression for #58720: the cron agent's async-resource teardown
    (agent.close + cleanup_stale_async_clients) MUST run AFTER delivery, not
    before. run_job defers teardown by appending the live agent to the holder
    list; run_one_job tears it down only after _deliver_result has run. If the
    order flips, delivery races a torn-down async client and dies with
    'cannot schedule new futures after interpreter shutdown'.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(
        job,
        *,
        defer_agent_teardown=None,
        _admitted_run=None,
    ):
        order.append("run_job")
        # Mimic run_job's deferral contract: hand the live agent back so the
        # caller tears it down after delivery instead of in run_job's finally.
        assert defer_agent_teardown is not None, "run_one_job must defer teardown"
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def fake_deliver(job, content, adapters=None, loop=None):
        order.append("deliver")
        return None

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    # cleanup_stale_async_clients is imported lazily inside _teardown_cron_agent;
    # stub it so the teardown records its own marker without touching real caches.
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j8", "name": "t"})

    assert ok is True
    # Delivery must strictly precede agent teardown + stale-client reap.
    assert order == ["run_job", "deliver", "agent.close", "cleanup_stale"], order


def test_run_one_job_tears_down_deferred_agent_when_delivery_raises(monkeypatch):
    """Even if _deliver_result raises, the deferred agent is still torn down
    (no fd/client leak — #10200). Teardown lives in a finally around delivery.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(
        job,
        *,
        defer_agent_teardown=None,
        _admitted_run=None,
    ):
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def boom_deliver(job, content, adapters=None, loop=None):
        order.append("deliver-raise")
        raise RuntimeError("send blew up")

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", boom_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j9", "name": "t"})

    assert ok is True  # delivery error is recorded, not propagated
    assert order == ["deliver-raise", "agent.close", "cleanup_stale"], order


def test_run_one_job_tears_down_deferred_agent_when_save_raises(monkeypatch):
    """#58720 W1: if save_job_output (or the [SILENT]/empty computation) raises
    AFTER run_job hands the agent back but BEFORE delivery, the deferred agent
    must still be torn down. The outer `except` would otherwise swallow the
    error and leak the agent (#10200). Teardown lives in a finally spanning
    save→deliver.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(
        job,
        *,
        defer_agent_teardown=None,
        _admitted_run=None,
    ):
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def boom_save(jid, out):
        order.append("save-raise")
        raise RuntimeError("disk full")

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", boom_save)
    monkeypatch.setattr(s, "_deliver_result",
                        lambda *a, **k: order.append("deliver"))
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j10", "name": "t"})

    # Save raised, so the outer handler sends a failure alert and returns
    # False, while still tearing down the deferred agent without a leak.
    assert ok is False
    assert "deliver" in order
    assert order == ["save-raise", "agent.close", "cleanup_stale", "deliver"], order
