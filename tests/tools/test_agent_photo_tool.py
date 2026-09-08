"""Regression coverage for the personal-profile agent-photo capability."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.workforce_org import WorkforceOrganizationError, load_organization
from model_tools import get_tool_definitions


REPO_ROOT = Path(__file__).parents[2]
ORG_PATH = REPO_ROOT / "workforce" / "organization.yaml"


@pytest.fixture
def trusted_wrapper(monkeypatch, tmp_path):
    """Provide the pinned, owner-only wrapper without relying on the host install."""
    from tools import agent_photo_tool

    tmp_path.chmod(0o700)
    wrapper = tmp_path / "operator-bin" / "hermes-agent-photo"
    wrapper.parent.mkdir()
    wrapper.parent.chmod(0o700)
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", wrapper)
    return wrapper


@pytest.fixture
def personal_profile(monkeypatch, tmp_path, trusted_wrapper):
    """Configure one personal-only profile and a trusted shared skill root."""
    def configure(name: str) -> Path:
        tmp_path.chmod(0o700)
        profile = tmp_path / "profiles" / name
        profile.mkdir(parents=True)
        profile.parent.chmod(0o775)
        profile.chmod(0o700)
        organization = yaml.safe_load(ORG_PATH.read_text(encoding="utf-8"))
        for agent in organization["agents"]:
            if agent["agent"] == name:
                agent["profile_path"] = str(profile)
        organization_path = tmp_path / "organization" / "organization.yaml"
        organization_path.parent.mkdir(parents=True, exist_ok=True)
        organization_path.write_text(yaml.safe_dump(organization), encoding="utf-8")
        organization_path.parent.chmod(0o755)
        organization_path.chmod(0o644)
        shared = tmp_path / "shared-skills" / "agent-photo"
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "SKILL.md").write_text(
            "# Agent Photo\n\nUse only the fixed wrapper.\n", encoding="utf-8"
        )
        shared.parent.chmod(0o755)
        shared.chmod(0o755)
        (shared / "SKILL.md").chmod(0o644)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.delenv("HERMES_WORKFORCE_ORG", raising=False)
        monkeypatch.delenv("HERMES_SHARED_SKILLS_DIR", raising=False)
        return profile

    return configure


@pytest.mark.windows_only
def test_agent_photo_imports_but_is_unavailable_without_posix_descriptor_security():
    from tools import agent_photo_tool

    assert agent_photo_tool._secure_descriptor_capability_available() is False
    assert agent_photo_tool.WRAPPER_PATH is None
    assert agent_photo_tool.check_personal_agent_photo_requirements() is False
    assert json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"})) == {
        "error": "agent-photo is unavailable on this platform"
    }


def test_personal_profile_directory_matches_runner_mode_contract(tmp_path):
    from tools import agent_photo_tool

    root = tmp_path / "hermes"
    profiles = root / "profiles"
    profile = profiles / "amy"
    profile.mkdir(parents=True)
    root.chmod(0o700)
    profiles.chmod(0o775)
    profile.chmod(0o700)

    descriptor = agent_photo_tool._open_personal_profile_directory(root, "amy")
    try:
        assert os.fstat(descriptor).st_mode & 0o777 == 0o700
    finally:
        os.close(descriptor)


def test_generation_timeout_covers_the_fixed_provider_download_and_runner_budgets():
    from tools import agent_photo_tool

    assert agent_photo_tool._wrapper_timeout("generate") == (
        agent_photo_tool._GENERATION_PROVIDER_TIMEOUT_SECONDS
        + agent_photo_tool._GENERATION_DOWNLOAD_TIMEOUT_SECONDS
        + agent_photo_tool._GENERATION_RUNNER_SETUP_TIMEOUT_SECONDS
    )
    assert agent_photo_tool._wrapper_timeout("generate") > agent_photo_tool._wrapper_timeout(
        "preview"
    )


@pytest.mark.parametrize(
    ("root_mode", "profiles_mode", "profile_mode"),
    [
        (0o775, 0o775, 0o700),
        (0o700, 0o755, 0o700),
        (0o700, 0o775, 0o750),
        (0o700, 0o775, 0o775),
    ],
)
def test_personal_profile_directory_refuses_unsafe_or_incompatible_modes(
    tmp_path, root_mode, profiles_mode, profile_mode
):
    from tools import agent_photo_tool

    root = tmp_path / "hermes"
    profiles = root / "profiles"
    profile = profiles / "amy"
    profile.mkdir(parents=True)
    root.chmod(root_mode)
    profiles.chmod(profiles_mode)
    profile.chmod(profile_mode)

    with pytest.raises(ValueError, match="agent-photo profile path is unsafe"):
        agent_photo_tool._open_personal_profile_directory(root, "amy")


def test_personal_profile_directory_refuses_symlinked_profile(tmp_path):
    from tools import agent_photo_tool

    root = tmp_path / "hermes"
    profiles = root / "profiles"
    target = root / "target"
    profile = profiles / "amy"
    target.mkdir(parents=True)
    profiles.mkdir(parents=True)
    root.chmod(0o700)
    profiles.chmod(0o775)
    target.chmod(0o700)
    profile.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="agent-photo profile path is unsafe"):
        agent_photo_tool._open_personal_profile_directory(root, "amy")


@pytest.mark.parametrize("profile_name", ["", ".", "..", "amy/other"])
def test_personal_profile_directory_refuses_non_profile_names(tmp_path, profile_name):
    from tools import agent_photo_tool

    with pytest.raises(ValueError, match="agent-photo profile path is unsafe"):
        agent_photo_tool._open_personal_profile_directory(tmp_path, profile_name)


@pytest.mark.parametrize("profile_name", ["amy", "kourtnie"])
def test_personal_profiles_can_discover_skill_and_use_no_spend_actions(
    monkeypatch, personal_profile, profile_name
):
    from tools import agent_photo_tool

    profile = personal_profile(profile_name)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="safe output\n", stderr="")

    monkeypatch.setattr(agent_photo_tool.subprocess, "run", fake_run)

    instructions = json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"}))
    preview = json.loads(
        agent_photo_tool.agent_photo_tool(
            {"action": "preview", "prompt": "portrait in warm window light"}
        )
    )
    status = json.loads(agent_photo_tool.agent_photo_tool({"action": "characters_status"}))

    assert instructions["skill"] == "agent-photo"
    assert "fixed wrapper" in instructions["instructions"]
    assert preview == {"success": True, "action": "preview", "output": "safe output"}
    assert status == {"success": True, "action": "characters_status", "output": "safe output"}
    assert [call[0][1:] for call in calls] == [
        ["--preview-prompt", "portrait in warm window light"],
        ["--characters-status"],
    ]
    for command, kwargs in calls:
        assert command[0] == f"/proc/self/fd/{kwargs['pass_fds'][0]}"
        profile_fd = kwargs["pass_fds"][1]
        assert kwargs == {
            "stdin": agent_photo_tool.subprocess.DEVNULL,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": agent_photo_tool._wrapper_timeout("preview"),
            "env": agent_photo_tool._wrapper_environment(profile, profile_fd=profile_fd),
            "pass_fds": kwargs["pass_fds"],
        }


@pytest.mark.parametrize("profile_name", ["amy", "kourtnie"])
def test_no_spend_actions_allow_the_cooperative_profiles_ancestor(
    monkeypatch, personal_profile, profile_name
):
    """Match the live layout: profiles is cooperative, each profile is private."""
    from tools import agent_photo_tool

    profile = personal_profile(profile_name)
    profile.parent.chmod(0o775)
    profile.chmod(0o700)
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="safe output\n", stderr=""),
    )

    instructions = json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"}))
    preview = json.loads(
        agent_photo_tool.agent_photo_tool(
            {"action": "preview", "prompt": "portrait in warm window light"}
        )
    )
    status = json.loads(agent_photo_tool.agent_photo_tool({"action": "characters_status"}))

    assert instructions["skill"] == "agent-photo"
    assert "fixed wrapper" in instructions["instructions"]
    assert preview == {"success": True, "action": "preview", "output": "safe output"}
    assert status == {"success": True, "action": "characters_status", "output": "safe output"}


def test_rejects_world_writable_or_symlinked_profiles_ancestor(tmp_path):
    from tools import agent_photo_tool

    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    profiles = root / "profiles"
    profiles.mkdir(mode=0o777)
    profiles.chmod(0o777)
    (profiles / "amy").mkdir(mode=0o700)

    with pytest.raises(ValueError, match="profile path is unsafe"):
        agent_photo_tool._open_personal_profile_directory(root, "amy")

    profiles.chmod(0o700)
    linked_root = tmp_path / "linked-root"
    linked_root.mkdir(mode=0o700)
    (linked_root / "profiles").symlink_to(profiles, target_is_directory=True)

    with pytest.raises(ValueError, match="profile path is unsafe"):
        agent_photo_tool._open_personal_profile_directory(linked_root, "amy")


def test_rejected_private_profile_validation_does_not_leak_descriptors(tmp_path):
    """Unsafe profile roots must be closed before the next request can retry."""
    from tools import agent_photo_tool

    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    profiles = root / "profiles"
    profiles.mkdir(mode=0o775)
    profiles.chmod(0o775)
    profile = profiles / "amy"
    profile.mkdir(mode=0o775)
    profile.chmod(0o775)

    baseline_fds = len(os.listdir("/proc/self/fd"))
    for _ in range(32):
        with pytest.raises(ValueError, match="profile path is unsafe"):
            agent_photo_tool._open_personal_profile_directory(root, "amy")

    assert len(os.listdir("/proc/self/fd")) == baseline_fds


def test_wrapper_receives_the_canonical_profile_path_required_by_the_fixed_runner(personal_profile):
    from tools import agent_photo_tool

    profile = personal_profile("amy")

    assert agent_photo_tool._wrapper_environment(profile, profile_fd=42)["HERMES_HOME"] == str(profile)


def test_generation_requires_executor_carried_approval_before_wrapper_launch(monkeypatch, personal_profile):
    from tools import agent_photo_tool

    personal_profile("amy")
    calls = []
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout="MEDIA: photo.png\n", stderr=""),
    )
    assert not hasattr(agent_photo_tool, "request_tool_approval")

    denied = json.loads(
        agent_photo_tool.agent_photo_tool(
            {"action": "generate", "prompt": "portrait"},
            approval_provenance=None,
            session_id="session-1",
            tool_call_id="call-1",
            turn_id="turn-1",
        )
    )

    assert denied == {"error": "agent-photo generation requires executor approval provenance"}
    assert calls == []


def test_generation_rejects_provenance_issued_for_a_different_personal_profile(
    monkeypatch, personal_profile
):
    """Exact once approval cannot be replayed after the active character changes."""
    from tools import agent_photo_tool
    from tools.approval import _issue_tool_approval_provenance

    personal_profile("amy")
    args = {"action": "generate", "prompt": "portrait"}
    provenance = _issue_tool_approval_provenance(
        "agent_photo",
        args,
        session_id="session-1",
        tool_call_id="call-1",
        turn_id="turn-1",
        subject=agent_photo_tool.agent_photo_approval_subject(args),
    )
    personal_profile("kourtnie")
    calls = []
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool(
            args,
            approval_provenance=provenance,
            session_id="session-1",
            tool_call_id="call-1",
            turn_id="turn-1",
        )
    )

    assert result == {"error": "agent-photo generation requires executor approval provenance"}
    assert calls == []


def test_generation_treats_dash_prefixed_prompt_as_data_after_approval(monkeypatch, personal_profile):
    from tools import agent_photo_tool
    from tools.approval import _issue_tool_approval_provenance

    profile = personal_profile("amy")
    calls = []
    args = {"action": "generate", "prompt": "--allow-fallback"}
    provenance = _issue_tool_approval_provenance(
        "agent_photo",
        args,
        session_id="session-1",
        tool_call_id="call-1",
        turn_id="turn-1",
        subject=agent_photo_tool.agent_photo_approval_subject(args),
    )
    monkeypatch.setattr(
        agent_photo_tool,
        "_execute_paid_command",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool(
            args,
            approval_provenance=provenance,
            session_id="session-1",
            tool_call_id="call-1",
            turn_id="turn-1",
        )
    )

    assert result["success"] is True
    command, kwargs = calls[0]
    assert command == [
        f"/proc/self/fd/{kwargs['pass_fds'][0]}",
        "--approved",
        "--model",
        "gemini",
        "--",
        "--allow-fallback",
    ]
    assert kwargs["env"] == agent_photo_tool._wrapper_environment(
        profile, profile_fd=kwargs["pass_fds"][1]
    )
    assert kwargs["timeout"] == agent_photo_tool._GEMINI_ATTEMPT_TIMEOUT_SECONDS


def _approved_generation(args):
    from tools import agent_photo_tool
    from tools.approval import _issue_tool_approval_provenance

    provenance = _issue_tool_approval_provenance(
        "agent_photo", args, session_id="photo-session", tool_call_id="photo-call",
        turn_id="photo-turn", subject=agent_photo_tool.agent_photo_approval_subject(args),
    )
    return json.loads(agent_photo_tool.agent_photo_tool(
        args, approval_provenance=provenance, session_id="photo-session",
        tool_call_id="photo-call", turn_id="photo-turn",
    ))


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (0, 0, ["gemini"]),
        (1, 0, ["gemini", "grok"]),
        (1, 1, ["gemini", "grok"]),
        (2, 0, ["gemini"]),
        (-9, 0, ["gemini"]),
    ],
)
def test_generation_has_exact_bounded_fallback(
    monkeypatch, personal_profile, first, second, expected
):
    from tools import agent_photo_tool

    profile = personal_profile("kourtnie")
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        code = first if len(calls) == 1 else second
        return SimpleNamespace(returncode=code, stdout="result", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    args = {"action": "generate", "prompt": "portrait"}
    subject = agent_photo_tool.agent_photo_approval_subject(args)
    result = _approved_generation(args)

    assert result["providers_attempted"] == expected
    assert [command[3] for command, _ in calls] == expected
    assert subject["provider_sequence"] == ["gemini", "grok"]
    assert subject["attempts_per_provider"] == 1
    assert subject["max_generation_seconds"] < 420
    for command, kwargs in calls:
        assert command[1:] == ["--approved", "--model", command[3], "--", "portrait"]
        assert kwargs["env"] == agent_photo_tool._wrapper_environment(profile)
        assert 0 < kwargs["timeout"] <= agent_photo_tool._GENERATION_TIMEOUT_SECONDS
    assert bool(result.get("success")) == (first == 0 or (first == 1 and second == 0))


@pytest.mark.parametrize("model", ["grok", "seedream"])
def test_explicit_other_provider_never_falls_back(monkeypatch, personal_profile, model):
    from tools import agent_photo_tool

    personal_profile("amy")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    result = _approved_generation({"action": "generate", "prompt": "portrait", "model": model})
    assert result["providers_attempted"] == [model]
    assert len(calls) == 1


@pytest.mark.parametrize("reason,expected", [
    ("timeout", ["gemini", "grok"]),
    ("cancelled", ["gemini"]),
    ("cleanup_unverified", ["gemini"]),
])
def test_only_reaped_timeout_can_fall_back(monkeypatch, personal_profile, reason, expected):
    from tools import agent_photo_tool

    personal_profile("amy")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise agent_photo_tool._GenerationStopped(reason)
        return SimpleNamespace(returncode=0, stdout="photo", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    result = _approved_generation({"action": "generate", "prompt": "portrait"})
    assert result["providers_attempted"] == expected
    assert [command[3] for command in calls] == expected


def test_revoked_request_stops_before_fallback(monkeypatch, personal_profile):
    from tools import agent_photo_tool

    personal_profile("amy")
    calls = []
    monkeypatch.setattr(agent_photo_tool, "_generation_cancelled", lambda: bool(calls))

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    result = _approved_generation({"action": "generate", "prompt": "portrait"})
    assert result["providers_attempted"] == ["gemini"]
    assert "cancelled" in result["error"]


def test_human_approved_run_without_request_origin_stops_on_run_end(monkeypatch, personal_profile):
    from agent.agent_photo_request import (
        finish_agent_photo_request_run, get_current_agent_photo_request_authorization,
        start_agent_photo_request_run,
    )
    from tools import agent_photo_tool

    personal_profile("amy")
    agent = SimpleNamespace()
    run, token = start_agent_photo_request_run(agent)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        run.finish()
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    try:
        assert get_current_agent_photo_request_authorization() is None
        result = _approved_generation({"action": "generate", "prompt": "portrait"})
        assert result["providers_attempted"] == ["gemini"]
        assert "cancelled" in result["error"]
    finally:
        finish_agent_photo_request_run(agent, run, token)


def test_gemini_only_request_preserves_single_attempt(monkeypatch, personal_profile):
    from tools import agent_photo_tool

    personal_profile("amy")
    calls = []
    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", lambda command, **kwargs:
        calls.append(command) or SimpleNamespace(returncode=1, stdout="failed", stderr=""))
    args = {"action": "generate", "prompt": "portrait", "fallback_to_grok": False}
    result = _approved_generation(args)
    assert result["providers_attempted"] == ["gemini"]
    assert len(calls) == 1
    assert agent_photo_tool.agent_photo_approval_subject(args)["provider_sequence"] == ["gemini"]


def test_fallback_choice_cannot_change_after_approval(monkeypatch, personal_profile):
    from tools import agent_photo_tool
    from tools.approval import _issue_tool_approval_provenance

    personal_profile("amy")
    args = {"action": "generate", "prompt": "portrait", "fallback_to_grok": False}
    provenance = _issue_tool_approval_provenance(
        "agent_photo", args, session_id="photo-session", tool_call_id="photo-call", turn_id="photo-turn",
        subject=agent_photo_tool.agent_photo_approval_subject(args),
    )
    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", lambda *a, **k:
        pytest.fail("changed fallback policy must not launch"))
    result = json.loads(agent_photo_tool.agent_photo_tool(
        {**args, "fallback_to_grok": True}, approval_provenance=provenance,
        session_id="photo-session", tool_call_id="photo-call", turn_id="photo-turn",
    ))
    assert "requires executor approval provenance" in result["error"]


@pytest.mark.parametrize("options", [
    {"fallback_to_grok": "true"},
    {"fallback_to_grok": 1},
    {"fallback_to_grok": None},
    {"model": "grok", "fallback_to_grok": True},
    {"model": "seedream", "fallback_to_grok": True},
])
def test_invalid_fallback_never_launches(monkeypatch, personal_profile, options):
    from tools import agent_photo_tool

    personal_profile("amy")
    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", lambda *a, **k:
        pytest.fail("invalid fallback policy must not launch"))
    result = json.loads(agent_photo_tool.agent_photo_tool(
        {"action": "generate", "prompt": "portrait", **options},
    ))
    assert "error" in result


def test_exhausted_generation_deadline_stops_fallback(monkeypatch, personal_profile):
    from tools import agent_photo_tool

    personal_profile("amy")
    now = [100.0]
    calls = []
    monkeypatch.setattr(agent_photo_tool.time, "monotonic", lambda: now[0])

    def execute(command, **kwargs):
        calls.append(command)
        now[0] += agent_photo_tool._GENERATION_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr(agent_photo_tool, "_execute_paid_command", execute)
    result = _approved_generation({"action": "generate", "prompt": "portrait"})
    assert result["providers_attempted"] == ["gemini"]
    assert "deadline reached" in result["error"]


def test_paid_timeout_reaps_wrapper_and_child(monkeypatch, tmp_path):
    import sys
    from tools import agent_photo_tool

    monkeypatch.setattr(agent_photo_tool, "_generation_cancelled", lambda: False)
    marker = tmp_path / "child-pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8'); time.sleep(30)"
    )
    with pytest.raises(agent_photo_tool._GenerationStopped, match="timeout"):
        agent_photo_tool._execute_paid_command(
            [sys.executable, "-c", script, str(marker)], env={}, pass_fds=(), timeout=1,
        )
    child_pid = int(marker.read_text(encoding="utf-8"))
    state_path = Path(f"/proc/{child_pid}/stat")
    assert not state_path.exists() or state_path.read_text(encoding="utf-8").split()[2] == "Z"


def test_unverified_process_group_cleanup_is_not_a_reaped_timeout(monkeypatch):
    import sys
    from tools import agent_photo_tool

    monkeypatch.setattr(agent_photo_tool, "_generation_cancelled", lambda: False)
    monkeypatch.setattr(agent_photo_tool, "_process_group_running", lambda group: True)
    monkeypatch.setattr(agent_photo_tool, "_GENERATION_CLEANUP_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(agent_photo_tool._GenerationStopped, match="cleanup_unverified"):
        agent_photo_tool._execute_paid_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env={}, pass_fds=(), timeout=0.1,
        )


@pytest.mark.parametrize(
    "args",
    [
        {"action": "preview", "prompt": "portrait", "command": "id"},
        {"action": "generate", "prompt": "portrait", "source": "/home/elliott/.hermes/profiles/sloane/assets/seed.png"},
        {"action": "generate", "prompt": "portrait", "approved": True},
        {"action": "generate", "prompt": "portrait", "current_request": "please make one"},
        {"action": "shell", "prompt": "id"},
        {"action": "preview", "prompt": "../other-profile"},
    ],
)
def test_rejects_arbitrary_commands_paths_and_cross_profile_assets(personal_profile, args):
    from tools import agent_photo_tool

    personal_profile("amy")
    result = json.loads(agent_photo_tool.agent_photo_tool(args))

    assert "error" in result


def test_rejects_absolute_prompt_paths_malformed_fields_and_untrusted_skill_root(
    monkeypatch, personal_profile, tmp_path
):
    from tools import agent_photo_tool

    personal_profile("amy")
    hostile = tmp_path / "profiles" / "sloane" / "skills" / "agent-photo"
    hostile.mkdir(parents=True)
    (hostile / "SKILL.md").write_text("# Sloane private instructions", encoding="utf-8")
    monkeypatch.setenv("HERMES_SHARED_SKILLS_DIR", str(hostile.parent))

    absolute_path = json.loads(
        agent_photo_tool.agent_photo_tool(
            {"action": "preview", "prompt": "/home/elliott/.hermes/profiles/sloane/assets/seed.png"}
        )
    )
    malformed_action = json.loads(agent_photo_tool.agent_photo_tool({"action": []}))
    malformed_model = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "generate", "prompt": "portrait", "model": []})
    )
    instructions = json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"}))

    assert absolute_path["error"] == "prompt must not contain a path"
    assert "action must be one of" in malformed_action["error"]
    assert malformed_model["error"] == "model must be a string"
    assert "fixed wrapper" in instructions["instructions"]


def test_ignores_environment_organization_override_and_rejects_symlinked_wrapper(
    monkeypatch, personal_profile, tmp_path
):
    from tools import agent_photo_tool

    personal_profile("sloane")
    hostile_org = yaml.safe_load(ORG_PATH.read_text(encoding="utf-8"))
    for agent in hostile_org["agents"]:
        if agent["agent"] == "sloane":
            agent["status"] = "friend"
            agent["operational"] = False
    hostile_org_path = tmp_path / "hostile-organization.yaml"
    hostile_org_path.write_text(yaml.safe_dump(hostile_org), encoding="utf-8")
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(hostile_org_path))

    rejected_profile = json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"}))

    personal_profile("amy")
    target = tmp_path / "target-wrapper"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o700)
    wrapper_link = tmp_path / "hermes-agent-photo"
    wrapper_link.symlink_to(target)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", wrapper_link)
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a symlinked wrapper must not run"),
    )
    rejected_wrapper = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "preview", "prompt": "portrait"})
    )

    assert "authorized personal profiles" in rejected_profile["error"]
    assert rejected_wrapper["error"] == "agent-photo wrapper must be a regular file"


def test_rejects_symlinked_wrapper_ancestor_before_spawn(monkeypatch, personal_profile, tmp_path):
    from tools import agent_photo_tool

    personal_profile("amy")
    target = tmp_path / "wrapper-target"
    target.mkdir()
    wrapper = target / "hermes-agent-photo"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o700)
    linked_parent = tmp_path / "linked-bin"
    linked_parent.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", linked_parent / wrapper.name)
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a wrapper below a symlinked ancestor must not run"),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "preview", "prompt": "portrait"})
    )

    assert result == {"error": "agent-photo wrapper path is unsafe"}


def test_rejects_symlinked_local_wrapper_ancestor_before_spawn(
    monkeypatch, personal_profile, tmp_path
):
    from tools import agent_photo_tool

    personal_profile("amy")
    target = tmp_path / "wrapper-target"
    (target / "bin").mkdir(parents=True)
    wrapper = target / "bin" / "hermes-agent-photo"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o700)
    local = tmp_path / ".local"
    local.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", local / "bin" / wrapper.name)
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a wrapper below a symlinked .local ancestor must not run"),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "preview", "prompt": "portrait"})
    )

    assert result == {"error": "agent-photo wrapper path is unsafe"}


def test_rejects_group_writable_organization_before_reading_personal_identity(
    personal_profile,
):
    from tools import agent_photo_tool

    profile = personal_profile("amy")
    organization_dir = profile.parent.parent / "organization"
    organization_dir.chmod(0o775)

    result = json.loads(agent_photo_tool.agent_photo_tool({"action": "instructions"}))

    assert result == {"error": "agent-photo organization path is unsafe"}


def test_preview_executes_the_checked_wrapper_descriptor(monkeypatch, personal_profile, tmp_path):
    """A pathname swap after validation cannot change the launched wrapper."""
    from tools import agent_photo_tool

    personal_profile("amy")
    wrapper = tmp_path / "hermes-agent-photo"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", wrapper)
    calls = []
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout="safe output", stderr=""),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "preview", "prompt": "portrait"})
    )

    assert result["success"] is True
    command, kwargs = calls[0]
    wrapper_fd = kwargs["pass_fds"][0]
    assert command == [f"/proc/self/fd/{wrapper_fd}", "--preview-prompt", "portrait"]
    assert kwargs["pass_fds"] == (wrapper_fd, kwargs["pass_fds"][1])
    with pytest.raises(OSError):
        os.fstat(wrapper_fd)


def test_preview_validates_a_held_profile_descriptor_but_passes_the_canonical_path(
    monkeypatch, personal_profile, tmp_path
):
    from tools import agent_photo_tool

    profile = personal_profile("amy")
    wrapper = tmp_path / "hermes-agent-photo"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(agent_photo_tool, "WRAPPER_PATH", wrapper)
    calls = []
    monkeypatch.setattr(
        agent_photo_tool.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or SimpleNamespace(returncode=0, stdout="safe output", stderr=""),
    )

    result = json.loads(
        agent_photo_tool.agent_photo_tool({"action": "preview", "prompt": "portrait"})
    )

    assert result["success"] is True
    _command, kwargs = calls[0]
    profile_fd = kwargs["pass_fds"][1]
    assert kwargs["env"]["HERMES_HOME"] == str(profile)
    assert kwargs["pass_fds"] == (kwargs["pass_fds"][0], profile_fd)
    with pytest.raises(OSError):
        os.fstat(profile_fd)


def test_toolset_is_limited_to_personal_profiles_and_does_not_widen_cli(
    monkeypatch, personal_profile
):
    from tools import agent_photo_tool

    personal_profile("amy")
    personal_names = {
        item["function"]["name"]
        for item in get_tool_definitions(
            enabled_toolsets=["agent_photo"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    }
    cli_names = {
        item["function"]["name"]
        for item in get_tool_definitions(
            enabled_toolsets=["hermes-cli"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    }

    assert personal_names == {"agent_photo"}
    assert "terminal" not in personal_names
    assert "skill_view" not in personal_names
    assert "kanban_complete" not in personal_names
    assert "agent_photo" not in cli_names

    personal_profile("sloane")
    assert agent_photo_tool.check_personal_agent_photo_requirements() is False
    with pytest.raises(WorkforceOrganizationError):
        load_organization(ORG_PATH).validate_execution_profile("amy")


def test_schema_prompt_impact_stays_small(personal_profile):
    from tools import agent_photo_tool

    personal_profile("amy")
    schema_bytes = len(
        json.dumps(agent_photo_tool.AGENT_PHOTO_SCHEMA, separators=(",", ":")).encode("utf-8")
    )

    assert schema_bytes <= 2_000
