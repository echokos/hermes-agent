"""Narrow, personal-profile access to the identity-locked agent-photo wrapper.

This tool deliberately owns one fixed shared procedure and one fixed executable.
It is not a shell, skill browser, or generic file interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root, get_hermes_home
from tools.approval import consume_tool_approval_provenance
from hermes_cli.workforce_org import WorkforceOrganizationError, load_organization
from tools.registry import registry, tool_error, tool_result


TOOLSET = "agent_photo"
_MAX_PROMPT_CHARS = 4_000
_MAX_OUTPUT_CHARS = 12_000
_NO_SPEND_TIMEOUT_SECONDS = 60
_GENERATION_PROVIDER_TIMEOUT_SECONDS = 240
_GENERATION_DOWNLOAD_TIMEOUT_SECONDS = 60
_GENERATION_RUNNER_SETUP_TIMEOUT_SECONDS = 60
_GENERATION_TIMEOUT_SECONDS = (
    _GENERATION_PROVIDER_TIMEOUT_SECONDS
    + _GENERATION_DOWNLOAD_TIMEOUT_SECONDS
    + _GENERATION_RUNNER_SETUP_TIMEOUT_SECONDS
)
_GENERATION_CLEANUP_TIMEOUT_SECONDS = 5
_GENERATION_POLL_SECONDS = 0.25
_GEMINI_ATTEMPT_TIMEOUT_SECONDS = 180

_ACTION_ALLOWED_KEYS = {
    "instructions": {"action"},
    "preview": {"action", "prompt"},
    "characters_status": {"action"},
    "generate": {"action", "prompt", "model", "fallback_to_grok"},
}

AGENT_PHOTO_SCHEMA = {
    "name": "agent_photo",
    "description": (
        "Use the active personal profile's identity-locked agent-photo procedure. "
        "It can load only the shared agent-photo instructions, preview a prompt, "
        "check the bound character status, or make one user-requested generation. "
        "It cannot run shell commands, accept file paths, or access another profile. "
        "Generation requires a direct current-message request or fresh human approval, "
        "and always passes --approved to the wrapper. By default Gemini failure falls back to "
        "Grok once within the same request; no further provider attempts are made."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["instructions", "preview", "characters_status", "generate"],
                "description": "The fixed agent-photo action to perform.",
            },
            "prompt": {
                "type": "string",
                "description": "Requested photo scene. Required for preview and generate; never a path or command.",
            },

            "model": {
                "type": "string",
                "enum": ["gemini", "grok", "seedream"],
                "description": "Generation provider. Defaults to gemini with one Grok fallback. Explicit grok or seedream makes one attempt without fallback.",
            },
            "fallback_to_grok": {
                "type": "boolean",
                "description": "Gemini only: defaults to true. Set false when the user asks for Gemini without fallback. Never enables any other fallback provider.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _secure_descriptor_capability_available() -> bool:
    """Return whether this host can enforce the wrapper's POSIX trust model."""
    if os.name != "posix":
        return False
    try:
        import pwd
    except ImportError:
        return False
    return all(hasattr(os, attribute) for attribute in ("getuid", "O_DIRECTORY", "O_NOFOLLOW"))


def _require_secure_descriptor_capability() -> None:
    if not _secure_descriptor_capability_available():
        raise ValueError("agent-photo is unavailable on this platform")


def _current_uid() -> int:
    _require_secure_descriptor_capability()
    get_uid = getattr(os, "getuid", None)
    if not callable(get_uid):
        raise ValueError("agent-photo is unavailable on this platform")
    return get_uid()


def _default_wrapper_path() -> Path | None:
    if not _secure_descriptor_capability_available():
        return None
    import pwd

    return Path(pwd.getpwuid(_current_uid()).pw_dir) / ".local" / "bin" / "hermes-agent-photo"


WRAPPER_PATH = _default_wrapper_path()


def _wrapper_timeout(action: str) -> int:
    """Keep paid generation alive through its fixed provider and runner budgets."""
    if action == "generate":
        return _GENERATION_TIMEOUT_SECONDS
    return _NO_SPEND_TIMEOUT_SECONDS


def _active_personal_profile() -> Path:
    """Return the active profile only when it is an authorized personal profile."""
    _require_secure_descriptor_capability()
    root = get_default_hermes_root().resolve(strict=True)
    home = get_hermes_home().expanduser().resolve(strict=True)
    if home.parent != root / "profiles" or not home.is_dir():
        raise ValueError("agent-photo requires an active named personal profile")
    try:
        organization_text = _read_fixed_file(
            root,
            ("organization", "organization.yaml"),
            resource="organization",
        )
        agent = load_organization(
            root / "organization" / "organization.yaml",
            source_text=organization_text,
        ).from_profile_path(home)
    except WorkforceOrganizationError as exc:
        raise ValueError("agent-photo is unavailable for an unknown profile") from exc
    if agent.operational or agent.status != "friend":
        raise ValueError("agent-photo is available only to authorized personal profiles")
    if not agent.profile_path or Path(agent.profile_path).resolve(strict=True) != home:
        raise ValueError("agent-photo profile path does not match the canonical organization")
    return home


def check_personal_agent_photo_requirements() -> bool:
    """Expose this schema only to authorized personal profiles."""
    try:
        _active_personal_profile()
    except (OSError, ValueError):
        return False
    return True


def _validate_fixed_directory(descriptor: int, resource: str) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != _current_uid()
        or metadata.st_mode & 0o022
    ):
        raise ValueError(f"agent-photo {resource} path is unsafe")


def _validate_profile_root_directory(descriptor: int) -> None:
    """Require the scoped runner's exact shared profile-root policy."""
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != _current_uid()
        or stat.S_IMODE(metadata.st_mode) != 0o775
    ):
        raise ValueError("agent-photo profile path is unsafe")


def _validate_personal_profile_directory(descriptor: int) -> None:
    """Keep each authorized personal profile private to its owner."""
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != _current_uid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("agent-photo profile path is unsafe")


def _open_fixed_directory(root: Path, parts: tuple[str, ...], *, resource: str) -> int:
    """Open a fixed directory beneath a held no-follow directory-FD chain."""
    _require_secure_descriptor_capability()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        current_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise ValueError(f"agent-photo {resource} path is unsafe") from exc
    try:
        _validate_fixed_directory(current_fd, resource)
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
            _validate_fixed_directory(current_fd, resource)
        if parts:
            next_fd = os.open(parts[-1], directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
            _validate_fixed_directory(current_fd, resource)
    except OSError as exc:
        os.close(current_fd)
        raise ValueError(f"agent-photo {resource} path is unsafe") from exc
    except Exception:
        os.close(current_fd)
        raise
    return current_fd


def _open_personal_profile_directory(root: Path, profile_name: str) -> int:
    """Open a private personal profile below the one cooperative ancestor."""
    if not profile_name or Path(profile_name).name != profile_name:
        raise ValueError("agent-photo profile path is unsafe")
    root_fd = _open_fixed_directory(root, (), resource="profile")
    profiles_fd: int | None = None
    profile_fd: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        profiles_fd = os.open("profiles", directory_flags, dir_fd=root_fd)
        _validate_profile_root_directory(profiles_fd)
        profile_fd = os.open(profile_name, directory_flags, dir_fd=profiles_fd)
        _validate_personal_profile_directory(profile_fd)
        return profile_fd
    except OSError as exc:
        if profile_fd is not None:
            os.close(profile_fd)
        raise ValueError("agent-photo profile path is unsafe") from exc
    except BaseException:
        if profile_fd is not None:
            os.close(profile_fd)
        raise
    finally:
        os.close(root_fd)
        if profiles_fd is not None:
            os.close(profiles_fd)


def _open_fixed_file(root: Path, parts: tuple[str, ...], *, resource: str) -> int:
    """Open a fixed file beneath a held no-follow directory-FD chain."""
    if not parts:
        raise ValueError(f"agent-photo {resource} path is unsafe")
    current_fd = _open_fixed_directory(root, parts[:-1], resource=resource)
    try:
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
    except OSError as exc:
        raise ValueError(f"agent-photo {resource} path is unsafe") from exc
    finally:
        os.close(current_fd)
    metadata = os.fstat(file_fd)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != _current_uid()
        or metadata.st_mode & 0o022
    ):
        os.close(file_fd)
        raise ValueError(f"agent-photo {resource} path is unsafe")
    return file_fd


def _read_fixed_file(root: Path, parts: tuple[str, ...], *, resource: str) -> str:
    descriptor = _open_fixed_file(root, parts, resource=resource)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8-sig") as stream:
            return stream.read()
    except OSError as exc:
        raise ValueError(f"agent-photo {resource} path is unsafe") from exc


def _shared_skill_instructions(profile: Path) -> str:
    shared = _read_fixed_file(
        profile.parent.parent,
        ("shared-skills", "agent-photo", "SKILL.md"),
        resource="shared procedure",
    )
    return (
        "# Native agent_photo execution\n\n"
        "Use this native tool, not terminal commands. A direct current user photo "
        "request authorizes generation without another approval prompt. The native "
        "generate action defaults to one Gemini attempt followed, on failure, by "
        "one Grok attempt within the same request. It manages that fallback itself; "
        "do not call generate again to retry it. Set fallback_to_grok=false for "
        "an explicit Gemini-only request. Explicit Grok or Seedream selections "
        "make one attempt and never fall back. This native contract supersedes "
        "the standalone CLI fallback instructions below; it never passes the "
        "broad --allow-fallback flag. No terminal access or credential changes "
        "are needed to invoke the native tool. Deliver each returned MEDIA line "
        "once. After timeout, report the uncertain provider outcome rather than "
        "claiming no remote image could have been generated.\n\n"
        + shared
    )


def _wrapper_environment(profile: Path, *, profile_fd: int | None = None) -> dict[str, str]:
    """Pass only the profile identity required by the fixed wrapper."""
    # The fixed wrapper validates this canonical path against its authorized
    # profile root; its public contract does not accept /proc/self/fd paths.
    return {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "HERMES_HOME": str(profile),
    }


def _clean_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    if (
        "\x00" in text
        or ".." in text
        or "/" in text
        or "\\" in text
        or text.startswith("~")
        or text.lower().startswith("file:")
    ):
        raise ValueError(f"{field} must not contain a path")
    return text


def agent_photo_approval_subject(args: dict[str, Any]) -> dict[str, Any]:
    """Bind one paid approval to this active character and fixed wrapper argv."""
    profile = _active_personal_profile()
    prompt = _clean_text(args.get("prompt"), "prompt", _MAX_PROMPT_CHARS)
    model = args.get("model", "gemini")
    if not isinstance(model, str) or model not in {"gemini", "grok", "seedream"}:
        raise ValueError("model must be one of: gemini, grok, seedream")
    providers = _generation_providers(args)
    return {
        "profile_name": profile.name,
        "profile_path": str(profile),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model": model,
        "output_options": ["--approved", "--model", model, "--"],
        "provider_sequence": providers,
        "attempts_per_provider": 1,
        "gemini_attempt_seconds": _GEMINI_ATTEMPT_TIMEOUT_SECONDS,
        "max_generation_seconds": _GENERATION_TIMEOUT_SECONDS
        + len(providers) * _GENERATION_CLEANUP_TIMEOUT_SECONDS,
    }


def _generation_providers(args: dict[str, Any]) -> list[str]:
    model = args.get("model", "gemini")
    fallback = args.get("fallback_to_grok", model == "gemini")
    if type(fallback) is not bool:
        raise ValueError("fallback_to_grok must be a boolean")
    if fallback and model != "gemini":
        raise ValueError("fallback_to_grok is available only for Gemini")
    return ["gemini", "grok"] if fallback else [model]


def _generation_cancelled() -> bool:
    from agent.agent_photo_request import (
        get_current_agent_photo_request_authorization,
        get_current_agent_photo_request_run,
    )
    from tools.interrupt import is_interrupted

    authorization = get_current_agent_photo_request_authorization()
    run = get_current_agent_photo_request_run()
    return (
        is_interrupted()
        or (run is not None and not run.is_active())
        or (authorization is not None and not authorization.is_active())
    )


class _GenerationStopped(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _process_group_running(group_id: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if os.getpgid(int(entry.name)) != group_id:
                continue
            state = (entry / "stat").read_text(encoding="utf-8", errors="replace").rpartition(")")[2].split()[0]
            if state != "Z":
                return True
        except ProcessLookupError:
            continue
        except FileNotFoundError:
            continue
    return False


def _execute_paid_command(
    command: list[str], *, env: dict[str, str], pass_fds: tuple[int, ...], timeout: float
) -> subprocess.CompletedProcess:
    """Reap the wrapper and its generator before allowing a fallback attempt."""
    _require_secure_descriptor_capability()
    if _generation_cancelled():
        raise _GenerationStopped("cancelled")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        pass_fds=pass_fds,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            if _generation_cancelled():
                raise _GenerationStopped("cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _GenerationStopped("timeout")
            try:
                stdout, stderr = process.communicate(
                    timeout=min(_GENERATION_POLL_SECONDS, remaining)
                )
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        cleanup_deadline = time.monotonic() + _GENERATION_CLEANUP_TIMEOUT_SECONDS
        try:
            os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok - POSIX capability checked before launch
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=_GENERATION_CLEANUP_TIMEOUT_SECONDS)
            while _process_group_running(process.pid):
                if time.monotonic() >= cleanup_deadline:
                    raise _GenerationStopped("cleanup_unverified")
                time.sleep(min(0.05, max(0, cleanup_deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            raise _GenerationStopped("cleanup_unverified") from None
        raise


def _trusted_wrapper_fd() -> int:
    """Open the fixed wrapper once so a later path swap cannot redirect it."""
    _require_secure_descriptor_capability()
    if WRAPPER_PATH is None:
        raise ValueError("agent-photo is unavailable on this platform")
    try:
        if (
            WRAPPER_PATH.parent.name == "bin"
            and WRAPPER_PATH.parent.parent.name == ".local"
        ):
            return _open_fixed_file(
                WRAPPER_PATH.parent.parent.parent,
                (".local", "bin", WRAPPER_PATH.name),
                resource="wrapper",
            )
        return _open_fixed_file(
            WRAPPER_PATH.parent,
            (WRAPPER_PATH.name,),
            resource="wrapper",
        )
    except ValueError as exc:
        if WRAPPER_PATH.is_symlink():
            raise ValueError("agent-photo wrapper must be a regular file") from exc
        raise


def _run_wrapper(
    profile: Path, command: list[str], action: str, *, timeout: float | None = None
) -> str:
    wrapper_fd = _trusted_wrapper_fd()
    try:
        profile_fd = _open_personal_profile_directory(
            get_default_hermes_root(),
            profile.name,
        )
    except Exception:
        os.close(wrapper_fd)
        raise
    try:
        argv = [f"/proc/self/fd/{wrapper_fd}", *command]
        kwargs = {
            "timeout": _wrapper_timeout(action) if timeout is None else timeout,
            "env": _wrapper_environment(profile, profile_fd=profile_fd),
            "pass_fds": (wrapper_fd, profile_fd),
        }
        if action == "generate":
            completed = _execute_paid_command(argv, **kwargs)
        else:
            completed = subprocess.run(
                argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", **kwargs
            )
    except _GenerationStopped as exc:
        return tool_error(
            f"agent-photo {action} stopped: {exc.reason}",
            failure_kind=exc.reason,
        )
    except FileNotFoundError:
        return tool_error("the fixed operator-installed hermes-agent-photo wrapper is not installed")
    except subprocess.TimeoutExpired:
        return tool_error(f"agent-photo {action} timed out")
    except OSError as exc:
        return tool_error(f"agent-photo {action} could not start: {exc}")
    finally:
        os.close(wrapper_fd)
        os.close(profile_fd)

    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n[output truncated]"
    if completed.returncode:
        return tool_error(
            f"agent-photo {action} failed",
            output=output,
            failure_kind="attempt_failed" if action == "generate" and completed.returncode == 1 else "runner_refused",
        )
    return tool_result({"success": True, "action": action, "output": output})


def agent_photo_tool(
    args: dict[str, Any],
    *,
    approval_provenance: Any = None,
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    **_: Any,
) -> str:
    """Handle one fixed personal-profile agent-photo action."""
    if not isinstance(args, dict):
        return tool_error("agent-photo arguments must be an object")
    action = args.get("action")
    if not isinstance(action, str) or action not in _ACTION_ALLOWED_KEYS:
        return tool_error("action must be one of: instructions, preview, characters_status, generate")
    unexpected = set(args) - _ACTION_ALLOWED_KEYS[action]
    if unexpected:
        return tool_error("agent-photo does not accept commands, paths, or extra options")
    try:
        profile = _active_personal_profile()
        if action == "instructions":
            return tool_result(
                {
                    "success": True,
                    "skill": "agent-photo",
                    "instructions": _shared_skill_instructions(profile),
                }
            )
        if action == "characters_status":
            return _run_wrapper(profile, ["--characters-status"], action)

        prompt = _clean_text(args.get("prompt"), "prompt", _MAX_PROMPT_CHARS)
        if action == "preview":
            if prompt.startswith("-"):
                return tool_error("preview prompt must not start with an option")
            return _run_wrapper(profile, ["--preview-prompt", prompt], action)

        model = args.get("model", "gemini")
        if not isinstance(model, str):
            return tool_error("model must be a string")
        if model not in {"gemini", "grok", "seedream"}:
            return tool_error("model must be one of: gemini, grok, seedream")
        if not consume_tool_approval_provenance(
            approval_provenance,
            "agent_photo",
            args,
            session_id=session_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            subject=agent_photo_approval_subject(args),
        ):
            return tool_error("agent-photo generation requires executor approval provenance")
        providers_attempted: list[str] = []
        deadline = time.monotonic() + _GENERATION_TIMEOUT_SECONDS
        for provider in _generation_providers(args):
            if _generation_cancelled():
                return tool_error("agent-photo generation cancelled", providers_attempted=providers_attempted)
            if _active_personal_profile() != profile:
                return tool_error("agent-photo profile changed before generation")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return tool_error("agent-photo generation deadline reached", providers_attempted=providers_attempted)
            attempt_timeout = (
                min(remaining, _GEMINI_ATTEMPT_TIMEOUT_SECONDS)
                if provider == "gemini"
                else remaining
            )
            providers_attempted.append(provider)
            result = json.loads(
                _run_wrapper(
                    profile,
                    ["--approved", "--model", provider, "--", prompt],
                    action,
                    timeout=attempt_timeout,
                )
            )
            result["providers_attempted"] = list(providers_attempted)
            if result.get("success") or result.get("failure_kind") not in {"attempt_failed", "timeout"}:
                break
        return tool_result(result)
    except (OSError, ValueError) as exc:
        return tool_error(str(exc))


registry.register(
    name="agent_photo",
    toolset=TOOLSET,
    schema=AGENT_PHOTO_SCHEMA,
    handler=agent_photo_tool,
    check_fn=check_personal_agent_photo_requirements,
    emoji="📷",
    max_result_size_chars=_MAX_OUTPUT_CHARS,
)
