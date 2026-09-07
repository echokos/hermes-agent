import json
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import kanban_db
from tools import workforce_signal_tool as signal
from tools.workforce_signal_runtime import activate, reset


SOURCE = Path(__file__).parents[2] / "workforce" / "organization.yaml"


def _payload():
    return {
        "expected_outcome": "Reduce failed releases",
        "approved_goal": "Reliable product delivery",
        "observation": "Three approved releases failed the same validation",
        "evidence_references": ["run:1", "run:2", "run:3"],
        "estimated_effort": "30 minutes to scope",
        "dependencies": ["release logs"],
        "risks": ["unknown shared cause"],
        "needed_capabilities": ["product", "agent systems"],
        "department_recommendation": "Investigate the common validator",
    }


def test_signal_is_fixed_nonexecuting_record_for_aurora(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "emily").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "emily"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda **_kwargs: db_path)
    result = json.loads(signal._handle(_payload()))
    assert result["success"] is True
    assert result["assignee"] == "aurora"
    assert result["status"] == "blocked"
    assert result["launch_authorized"] is False
    with kanban_db.connect_closing(db_path) as conn:
        task = kanban_db.get_task(conn, result["signal_id"])
        packet = json.loads(task.body)
    assert task.status == "blocked"
    assert packet["decision_owner"] == "aurora"
    assert packet["source_agent"] == "emily"
    assert packet["launch_authorized"] is False


def test_signal_deduplicates_exact_packet(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "main").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "main"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda **_kwargs: db_path)
    first = json.loads(signal._handle(_payload()))
    second = json.loads(signal._handle(_payload()))
    assert first["signal_id"] == second["signal_id"]


def test_friend_profile_cannot_submit(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "amy").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "amy"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    result = json.loads(signal._handle(_payload()))
    assert result.get("success") is not True
    assert "not eligible" in result["error"]


def test_callers_cannot_choose_launch_or_assignee():
    properties = signal.WORKFORCE_SIGNAL_SCHEMA["parameters"]["properties"]
    assert "assignee" not in properties
    assert "priority" not in properties
    assert "status" not in properties
    assert "launch" not in properties


def test_chloe_can_only_make_mechanical_record_under_aurora_assignment(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "chloe").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "chloe"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda **_kwargs: db_path)
    missing_assignment = _payload()
    missing_assignment.pop("department_recommendation")
    denied = json.loads(signal._handle(missing_assignment))
    assert "aurora_assignment_id" in denied["error"]
    with_recommendation = {**_payload(), "aurora_assignment_id": "task-aurora-1"}
    denied = json.loads(signal._handle(with_recommendation))
    assert "may not provide a recommendation" in denied["error"]
    allowed = {
        **missing_assignment,
        "aurora_assignment_id": "task-aurora-1",
    }
    result = json.loads(signal._handle(allowed))
    assert result["success"] is True
    with kanban_db.connect_closing(db_path) as conn:
        packet = json.loads(kanban_db.get_task(conn, result["signal_id"]).body)
    assert packet["source_agent"] == "chloe"
    assert packet["aurora_assignment_id"] == "task-aurora-1"
    assert packet["launch_authorized"] is False


def test_chloe_offline_write_failure_is_observed_and_a_later_retry_can_recover(tmp_path, monkeypatch):
    """Each Cron attempt gets fresh host state; a prior outage cannot go green."""
    profiles = tmp_path / "profiles"
    (profiles / "chloe").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "chloe"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    payload = {
        **_payload(),
        "department_recommendation": "",
        "aurora_assignment_id": "t_aurora-1",
    }
    token, failed_attempt = activate(True)
    try:
        monkeypatch.setattr(signal, "record_signal", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
        failure = json.loads(signal._handle(payload))
    finally:
        reset(token)
    assert failure.get("success") is not True
    assert failed_attempt.failure == "offline"
    assert failed_attempt.completed is False

    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda **_kwargs: db_path)
    monkeypatch.undo()
    monkeypatch.setenv("HERMES_HOME", str(profiles / "chloe"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    token, recovered_attempt = activate(True)
    try:
        recovery = json.loads(signal._handle(payload))
    finally:
        reset(token)
    assert recovery["success"] is True
    assert recovered_attempt.failure is None
    assert recovered_attempt.completed is True


def test_mel_cannot_route_a_signal(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "mel").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "mel"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    result = json.loads(signal._handle(_payload()))
    assert result.get("success") is not True
    assert "may not route" in result["error"]


def test_signal_uses_task_profile_override_before_process_environment(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    (profiles / "amy").mkdir(parents=True)
    (profiles / "emily").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "amy"))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(SOURCE))
    db_path = tmp_path / "kanban.db"
    monkeypatch.setattr(kanban_db, "kanban_db_path", lambda **_kwargs: db_path)

    token = set_hermes_home_override(profiles / "emily")
    try:
        result = json.loads(signal._handle(_payload()))
    finally:
        reset_hermes_home_override(token)

    assert result["success"] is True
    assert result["source_agent"] == "emily"
