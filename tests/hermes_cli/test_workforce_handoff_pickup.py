"""CLI-process boundaries for owned workforce handoff pickup."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli._subprocess_compat import IS_WINDOWS
from hermes_cli.workforce_handoff_pickup import _pickup_command, _pickup_env


def _claimed_pickup(monkeypatch, tmp_path):
    from hermes_cli import kanban_db
    from hermes_cli.workforce_handoffs import (
        claim_owned_failure_handoff_pickup,
        create_handoff,
    )
    from hermes_cli.workforce_org import load_organization

    root = tmp_path / ".hermes"
    (root / "profiles" / "alina").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    organization = load_organization()
    now = int(time.time())
    iso = lambda offset: datetime.fromtimestamp(now + offset, timezone.utc).isoformat()
    db_path = root / "kanban.db"
    with kanban_db.connect_closing(db_path) as conn:
        created = create_handoff(
            conn,
            source_agent="aurora",
            target_agent="alina",
            expected_outcome="Repair the owned operational failure",
            acceptance_test="A later probe succeeds",
            evidence_references=["execution:failure-1"],
            acknowledgment_deadline=iso(120),
            checkpoint_at=iso(3600),
            organization=organization,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "alina",
                "director": "aurora",
                "workflow_id": "owned-failure-test",
                "event_id": "failure-1",
            },
            requires_source_acceptance=True,
        )
        pickup = claim_owned_failure_handoff_pickup(
            conn, target_agent="alina", organization=organization, now=now + 1
        )
    assert pickup is not None
    return db_path, created, pickup, organization, now


def test_pickup_command_is_one_turn_tool_sourced_workforce_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup._resolve_hermes_argv",
        lambda: ["hermes"],
    )

    command = _pickup_command(
        target_agent="alina", request_root_id="cr_pickup_123", task_id="t_pickup_123"
    )

    assert command[:5] == ["hermes", "-p", "alina", "--cli", "chat"]
    assert "-Q" in command
    assert command[command.index("--max-turns") + 1] == "2"
    assert command[command.index("-t") + 1] == "workforce"
    assert command[command.index("-c") + 1] == "workforce-handoff:cr_pickup_123:alina"


def test_pickup_env_scrubs_worker_identity_and_sets_exact_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "unrelated-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "stale-lock")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "telegram")
    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup.resolve_profile_env", lambda _target: "/profiles/alina"
    )

    env = _pickup_env(
        database_path=tmp_path / "kanban.db",
        task_id="t_pickup_123",
        request_root_id="cr_pickup_123",
        target_agent="alina",
        source_agent="aurora",
    )

    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env
    assert env["HERMES_SESSION_SOURCE"] == "tool"
    assert env["HERMES_COORDINATION_REQUEST_ROOT"] == "cr_pickup_123"
    assert env["HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET"] == "alina"


def test_root_execution_profile_validation_rejects_wrong_or_nonoperational_profile(
    monkeypatch,
):
    from hermes_cli.workforce_handoff_pickup import _canonical_execution_profile

    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    assert _canonical_execution_profile(None, target_agent="root") == "main"
    assert _canonical_execution_profile("main", target_agent="root") == "main"
    with pytest.raises(ValueError, match="does not match"):
        _canonical_execution_profile("aurora", target_agent="root")
    with pytest.raises(ValueError):
        _canonical_execution_profile("amy", target_agent="root")


@pytest.mark.skipif(IS_WINDOWS, reason="pickup log descriptor hardening is POSIX-only")
def test_pickup_log_is_exclusive_owner_only_and_never_reopens(monkeypatch, tmp_path):
    from hermes_cli.workforce_handoff_pickup import _open_pickup_log

    database_path = tmp_path / "kanban.db"
    database_path.touch()
    log_path, log_file = _open_pickup_log(database_path, "t_pickup_123")
    with log_file:
        log_file.write(b"bounded diagnostic\n")

    assert log_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _open_pickup_log(database_path, "t_pickup_123")


def test_pickup_log_fails_closed_without_posix_descriptor_permissions(monkeypatch, tmp_path):
    from hermes_cli.workforce_handoff_pickup import _open_pickup_log

    database_path = tmp_path / "kanban.db"
    database_path.touch()
    monkeypatch.setattr("hermes_cli.workforce_handoff_pickup.IS_WINDOWS", True)

    with pytest.raises(OSError, match="requires POSIX descriptor permissions"):
        _open_pickup_log(database_path, "t_pickup_123")


def test_fresh_acknowledgment_survives_immediate_task_state_advance(monkeypatch, tmp_path):
    from hermes_cli import kanban_db
    from hermes_cli.workforce_handoff_pickup import _fresh_acknowledgment
    from hermes_cli.workforce_handoffs import acknowledge_handoff, record_checkpoint

    db_path, created, pickup, organization, now = _claimed_pickup(monkeypatch, tmp_path)
    with kanban_db.connect_closing(db_path) as conn:
        acknowledge_handoff(
            conn, created["task_id"], actor="alina", organization=organization, now=now + 2
        )
        record_checkpoint(
            conn,
            created["task_id"],
            actor="alina",
            evidence_references=["execution:repair-started"],
            organization=organization,
            now=now + 3,
        )

    assert _fresh_acknowledgment(
        database_path=db_path,
        task_id=created["task_id"],
        request_root_id=pickup["request_root_id"],
        target_agent="alina",
        source_agent="aurora",
    ) is True


@pytest.mark.skipif(IS_WINDOWS, reason="pickup process hardening is POSIX-only")
def test_pickup_reports_committed_ack_when_child_finalization_exits_nonzero(
    monkeypatch, tmp_path
):
    from hermes_cli import kanban_db
    from hermes_cli.workforce_handoff_pickup import run_workforce_handoff_pickup
    from hermes_cli.workforce_handoffs import acknowledge_handoff

    db_path, created, pickup, organization, now = _claimed_pickup(monkeypatch, tmp_path)

    class FinalizationFailure:
        returncode = 23

        def wait(self, timeout=None):
            assert timeout == 120
            with kanban_db.connect_closing(db_path) as conn:
                acknowledge_handoff(
                    conn,
                    created["task_id"],
                    actor="alina",
                    organization=organization,
                    now=now + 2,
                )
            return self.returncode

    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup.subprocess.Popen",
        lambda *_args, **_kwargs: FinalizationFailure(),
    )
    result = asyncio.run(run_workforce_handoff_pickup(
        task_id=created["task_id"],
        request_root_id=pickup["request_root_id"],
        target_agent="alina",
        source_agent="aurora",
        database_path=db_path,
    ))

    assert result.acknowledged is True
    assert result.returncode == 23
    assert result.timed_out is False
    assert result.reason == "pickup exited with 23 after durable acknowledgment"


@pytest.mark.skipif(IS_WINDOWS, reason="pickup process hardening is POSIX-only")
def test_pickup_cancellation_kills_and_reaps_the_dedicated_process(
    monkeypatch, tmp_path
):
    from hermes_cli.workforce_handoff_pickup import run_workforce_handoff_pickup

    db_path, created, pickup, _organization, _now = _claimed_pickup(monkeypatch, tmp_path)
    entered_wait = threading.Event()
    release_wait = threading.Event()
    killed: list[object] = []

    class BlockingProcess:
        returncode = None

        def wait(self, timeout=None):
            if timeout == 120:
                entered_wait.set()
                release_wait.wait()
                return 0
            assert timeout == 1
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup.subprocess.Popen",
        lambda *_args, **_kwargs: BlockingProcess(),
    )
    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup.kill_process_tree",
        lambda proc: killed.append(proc),
    )

    async def cancel_pickup() -> None:
        task = asyncio.create_task(run_workforce_handoff_pickup(
            task_id=created["task_id"],
            request_root_id=pickup["request_root_id"],
            target_agent="alina",
            source_agent="aurora",
            database_path=db_path,
        ))
        assert await asyncio.to_thread(entered_wait.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_wait.set()

    asyncio.run(cancel_pickup())
    assert len(killed) == 1


@pytest.mark.skipif(IS_WINDOWS, reason="pickup process hardening is POSIX-only")
@pytest.mark.parametrize(
    ("target_agent", "execution_profile"),
    [("alina", "alina"), ("root", "main")],
)
def test_pickup_runs_actual_cli_and_real_registry_with_loopback_provider(
    monkeypatch, tmp_path, target_agent, execution_profile,
):
    """Pickup must run the concrete profile with canonical tool authority."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from hermes_cli import kanban_db
    from hermes_cli.workforce_handoff_pickup import run_workforce_handoff_pickup
    from hermes_cli.workforce_handoffs import (
        claim_owned_failure_handoff_pickup,
        create_handoff,
    )
    from hermes_cli.workforce_org import load_organization

    seen_requests: list[tuple[str, dict]] = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 -- BaseHTTPRequestHandler contract
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            seen_requests.append((self.path, request))
            task_id = self.server.task_id
            chat_attempt = sum(
                path.endswith("/chat/completions") for path, _ in seen_requests
            )
            if chat_attempt == 2:
                final_chunk = {
                    "id": "pickup-summary",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "pickup-model",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Acknowledged."},
                        "finish_reason": "stop",
                    }],
                }
                body = f"data: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            response = {
                "id": "pickup-call",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "pickup-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_pickup",
                            "type": "function",
                            "function": {
                                "name": "workforce_handoff",
                                "arguments": json.dumps({
                                    "action": "acknowledge", "task_id": task_id,
                                }),
                            },
                        }],
                    },
                }],
            }
            if request.get("stream") is True:
                call_chunk = {
                    "id": "pickup-call",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "pickup-model",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                **response["choices"][0]["message"]["tool_calls"][0],
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                finish_chunk = {
                    "id": "pickup-call",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "pickup-model",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }],
                }
                body = (
                    f"data: {json.dumps(call_chunk)}\n\n"
                    f"data: {json.dumps(finish_chunk)}\n\n"
                    "data: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = tmp_path / ".hermes"
        profile = root / "profiles" / execution_profile
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv(
            "HERMES_WORKFORCE_ORG",
            str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
        )
        profile.joinpath("config.yaml").write_text(
            "model:\n"
            "  provider: custom:pickup-fake\n"
            "  default: pickup-model\n"
            "providers:\n"
            "  pickup-fake:\n"
            f"    base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "    api_key: pickup-test-key\n"
            "    default_model: pickup-model\n",
            encoding="utf-8",
        )
        organization = load_organization()
        now = int(time.time())
        iso = lambda offset: datetime.fromtimestamp(now + offset, timezone.utc).isoformat()
        db_path = root / "kanban.db"
        with kanban_db.connect_closing(db_path) as conn:
            created = create_handoff(
                conn,
                source_agent="aurora",
                target_agent=target_agent,
                expected_outcome="Repair the owned operational failure",
                acceptance_test="A later probe succeeds",
                evidence_references=["execution:failure-1"],
                acknowledgment_deadline=iso(120),
                checkpoint_at=iso(3600),
                organization=organization,
                context={
                    "kind": "owned_operational_failure",
                    "technical_owner": target_agent,
                    "director": "aurora",
                    "workflow_id": "owned-failure-test",
                    "event_id": "failure-1",
                },
                requires_source_acceptance=True,
            )
            pickup = claim_owned_failure_handoff_pickup(
                conn,
                target_agent=target_agent,
                organization=organization,
                now=now + 1,
            )
        assert pickup is not None
        server.task_id = created["task_id"]
        monkeypatch.setattr(
            "hermes_cli.workforce_handoff_pickup._resolve_hermes_argv",
            lambda: [sys.executable, "-m", "hermes_cli.main"],
        )

        result = asyncio.run(run_workforce_handoff_pickup(
            task_id=created["task_id"],
            request_root_id=pickup["request_root_id"],
            target_agent=target_agent,
            source_agent="aurora",
            database_path=db_path,
        ))

        with kanban_db.connect_closing(db_path) as conn:
            debug_task = kanban_db.get_task(conn, created["task_id"])
            debug_events = [event.kind for event in kanban_db.list_events(conn, created["task_id"])]
        assert result.acknowledged is True, (
            result.log_path.read_text(encoding="utf-8"), debug_task.body, debug_events
        )
        assert result.returncode == 0
        assert result.log_path.stat().st_mode & 0o777 == 0o600
        chat_requests = [
            request for path, request in seen_requests
            if path.endswith("/chat/completions")
        ]
        assert len(chat_requests) == 2
        assert any(
            tool["function"]["name"] == "workforce_handoff"
            for tool in chat_requests[0].get("tools", [])
        )
        with kanban_db.connect_closing(db_path) as conn:
            task = kanban_db.get_task(conn, created["task_id"])
            assert json.loads(task.body)["state"] == "accepted"
            acknowledged = [
                event for event in kanban_db.list_events(conn, created["task_id"])
                if event.kind == "workforce_handoff_acknowledged"
            ]
            assert len(acknowledged) == 1
            assert acknowledged[0].payload["actor"] == target_agent

        launched: dict[str, object] = {}

        class Worker:
            pid = 4242

        def launch_worker(command, **kwargs):
            launched["command"] = list(command)
            launched["env"] = dict(kwargs["env"])
            return Worker()

        monkeypatch.setattr(subprocess, "Popen", launch_worker)
        monkeypatch.setattr(kanban_db, "_memory_pressure_level", lambda: "normal")
        with kanban_db.connect_closing(db_path) as conn:
            dispatch = kanban_db.dispatch_once(conn, max_spawn=1)
            task = kanban_db.get_task(conn, created["task_id"])

        assert dispatch.spawned[0][:2] == (created["task_id"], target_agent)
        assert launched["command"][1:3] == ["-p", execution_profile]
        assert launched["env"]["HERMES_PROFILE"] == execution_profile
        assert launched["env"]["HERMES_HOME"] == str(profile)
        assert organization.validate_execution_profile(
            launched["env"]["HERMES_PROFILE"]
        ).agent == target_agent
        assert task.status == "running"
        assert task.assignee == target_agent
    finally:
        server.shutdown()
        server.server_close()
