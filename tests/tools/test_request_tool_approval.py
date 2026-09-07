"""Tests for tools.approval.request_tool_approval - the plugin pre_tool_call
``{"action": "approve"}`` escalation into the human-approval gate.

These verify that a plugin-driven approval reuses the SAME machinery as a
Tier-2 dangerous-command match: session/permanent allowlist, the CLI prompt,
the gateway submit_pending path, cron_mode, and fail-closed timeouts.
"""

import contextvars
import multiprocessing

import pytest

import tools.approval as approval
from tools.approval import request_tool_approval


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Give each test a clean session key and empty allowlists."""
    monkeypatch.setattr(
        approval, "get_current_session_key",
        lambda default="default": "test-session",
    )
    # Empty session + permanent approval stores so nothing pre-approves.
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    # Not a yolo session (the shared gate checks this first).
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
    # No thread-registered CLI callback by default.
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback", lambda: None, raising=False
    )
    yield


class TestToolApprovalProvenance:
    @staticmethod
    def _approved_call(monkeypatch, *, tool_call_id="call-1"):
        from hermes_cli import plugins

        args = {"draft_id": "honk-2026-06", "sha256": "a" * 64}
        resolver = getattr(plugins, "resolve_pre_tool_call", None)
        assert callable(resolver), (
            "approval resolution must return executor-carried provenance"
        )
        monkeypatch.setattr(
            plugins,
            "invoke_hook",
            lambda hook_name, **kwargs: [
                {
                    "action": "approve",
                    "message": "real-money action",
                    "allow_session": False,
                    "allow_permanent": False,
                    "allow_yolo": False,
                    "allow_cron": False,
                    "allow_single_query": False,
                }
            ],
        )
        monkeypatch.setattr(
            approval,
            "request_tool_approval",
            lambda *args, **kwargs: {"approved": True, "message": None},
        )
        resolution = resolver(
            "finance_execute_qbo_invoice",
            args,
            task_id="task-1",
            session_id="session-1",
            tool_call_id=tool_call_id,
            turn_id="turn-1",
        )
        assert resolution.block_message is None
        assert resolution.approval_provenance is not None
        return args, resolution.approval_provenance

    @staticmethod
    def _consume(provenance, args, *, tool="finance_execute_qbo_invoice", call="call-1"):
        consumer = getattr(approval, "consume_tool_approval_provenance", None)
        assert callable(consumer), "core provenance consumer is required"
        return consumer(
            provenance,
            tool,
            args,
            session_id="session-1",
            tool_call_id=call,
            turn_id="turn-1",
        )

    def test_exact_call_consumes_once_and_rejects_replay(self, monkeypatch):
        args, provenance = self._approved_call(monkeypatch)

        assert self._consume(provenance, dict(args))
        assert not self._consume(provenance, dict(args))

    def test_agent_photo_generate_uses_core_exact_once_policy(self, monkeypatch):
        from hermes_cli import plugins
        from tools import agent_photo_tool

        calls = []
        monkeypatch.setattr(plugins, "invoke_hook", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            agent_photo_tool,
            "agent_photo_approval_subject",
            lambda _args: {"profile_name": "amy", "prompt_sha256": "a" * 64},
        )
        monkeypatch.setattr(
            approval,
            "request_tool_approval",
            lambda *args, **kwargs: calls.append((args, kwargs))
            or {"approved": True, "message": None},
        )

        result = plugins.resolve_pre_tool_call(
            "agent_photo",
            {"action": "generate", "prompt": "portrait", "model": "gemini"},
            session_id="session-1",
            tool_call_id="call-1",
            turn_id="turn-1",
        )

        assert result.block_message is None
        assert result.approval_provenance is not None
        assert calls == [
            (
                ("agent_photo", "Approve this paid agent-photo generation for the active personal profile?"),
                {
                    "rule_key": "agent_photo_generation",
                    "allow_session": False,
                    "allow_permanent": False,
                    "allow_yolo": False,
                    "allow_cron": False,
                    "allow_single_query": False,
                },
            )
        ]

    @pytest.mark.parametrize(
        ("tool", "args_patch", "call"),
        [
            ("terminal", {}, "call-1"),
            ("finance_execute_qbo_invoice", {"sha256": "b" * 64}, "call-1"),
            ("finance_execute_qbo_invoice", {}, "call-2"),
        ],
    )
    def test_wrong_tool_args_or_call_id_do_not_consume(
        self, monkeypatch, tool, args_patch, call
    ):
        args, provenance = self._approved_call(monkeypatch)
        changed = dict(args)
        changed.update(args_patch)

        assert not self._consume(provenance, changed, tool=tool, call=call)
        assert self._consume(provenance, args)

    def test_copied_contexts_share_one_consumption_state(self, monkeypatch):
        args, provenance = self._approved_call(monkeypatch)
        slot = contextvars.ContextVar("copied_provenance")
        slot.set(provenance)
        first = contextvars.copy_context()
        second = contextvars.copy_context()

        assert first.run(lambda: self._consume(slot.get(), args))
        assert not second.run(lambda: self._consume(slot.get(), args))

    def test_expired_provenance_fails_without_consuming(self, monkeypatch):
        now = {"value": 100.0}
        monkeypatch.setattr(approval.time, "monotonic", lambda: now["value"])
        args, provenance = self._approved_call(monkeypatch)
        now["value"] += 61.0

        assert not self._consume(provenance, args)

    def test_provenance_is_bound_to_the_issuing_process(self, monkeypatch):
        args, provenance = self._approved_call(monkeypatch)
        issuing_pid = approval.os.getpid()
        monkeypatch.setattr(approval.os, "getpid", lambda: issuing_pid + 1)

        assert not self._consume(provenance, args)

    def test_forked_process_cannot_consume_provenance(self, monkeypatch):
        args, provenance = self._approved_call(monkeypatch)
        process_context = multiprocessing.get_context("fork")
        receive_result, send_result = process_context.Pipe(duplex=False)

        def consume_in_child():
            send_result.send(self._consume(provenance, args))
            send_result.close()

        process = process_context.Process(target=consume_in_child)
        process.start()
        send_result.close()
        child_result = receive_result.recv()
        process.join(timeout=5)

        assert process.exitcode == 0
        assert child_result is False
        assert self._consume(provenance, args)

    def test_structurally_similar_object_cannot_forge_provenance(self):
        fake = type(
            "FakeProvenance",
            (),
            {
                "tool_name": "finance_execute_qbo_invoice",
                "args": {"draft_id": "honk-2026-06", "sha256": "a" * 64},
                "tool_call_id": "call-1",
                "turn_id": "turn-1",
                "session_id": "session-1",
            },
        )()

        assert not self._consume(fake, fake.args)

    def test_reflective_state_copy_cannot_forge_provenance(self, monkeypatch):
        args, provenance = self._approved_call(monkeypatch)
        forged = object.__new__(type(provenance))
        for slot in type(provenance).__slots__:
            if slot == "__weakref__":
                continue
            setattr(forged, slot, getattr(provenance, slot))

        assert not self._consume(forged, args)
        assert self._consume(provenance, args)


class TestRequestToolApproval:
    def test_session_cached_approval_short_circuits(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        # Should NOT prompt at all.
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("should not prompt when already approved"),
        )
        res = request_tool_approval("write_file", "sensitive path", rule_key="ssh")
        assert res == {"approved": True, "message": None}

    def test_cli_approve_once(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "once")
        res = request_tool_approval("write_file", "writing ~/.ssh/authorized_keys")
        assert res["approved"] is True

    def test_cli_deny_blocks(self, monkeypatch):
        from hermes_cli import lifecycle

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        events = []
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda hook_name, **kwargs: events.append((hook_name, kwargs)) or [],
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-1",
            tool_call_id="call-1",
        )
        try:
            res = request_tool_approval("terminal", "curl PUT to external API")
        finally:
            approval.reset_current_observability_context(tokens)
        assert res["approved"] is False
        assert "denied" in res["message"].lower()
        assert res["pattern_key"].startswith("plugin_rule:")
        assert [name for name, _ in events] == [
            "pre_approval_request",
            "post_approval_response",
        ]
        assert all(event["turn_id"] == "turn-1" for _, event in events)
        assert all(event["tool_call_id"] == "call-1" for _, event in events)
        assert events[-1][1]["choice"] == "deny"

    def test_cli_session_persists_session_only(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "session")
        calls = {"session": [], "permanent": []}
        monkeypatch.setattr(approval, "approve_session",
                            lambda sk, pk: calls["session"].append(pk))
        monkeypatch.setattr(approval, "approve_permanent",
                            lambda pk: calls["permanent"].append(pk))
        monkeypatch.setattr(approval, "save_permanent_allowlist", lambda x: None)
        res = request_tool_approval("write_file", "reason", rule_key="ssh-writes")
        assert res["approved"] is True
        assert calls["session"] == ["plugin_rule:ssh-writes"]
        assert calls["permanent"] == []  # session != always


    def test_cron_deny_mode_blocks(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "cron" in res["message"].lower()

    def test_cron_approve_mode_allows(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "approve")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is True

    def test_explicitly_disallowed_single_query_auto_approval_blocks(self, monkeypatch):
        """A fresh-consent action cannot inherit global -q auto-approval."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: True)
        monkeypatch.setattr(approval, "_get_single_query_approval_mode", lambda: "approve")

        res = request_tool_approval(
            "agent_photo",
            "fresh human approval required",
            allow_session=False,
            allow_permanent=False,
            allow_yolo=False,
            allow_cron=False,
            allow_single_query=False,
        )

        assert res["approved"] is False
        assert "single-query" in res["message"].lower()


    def test_distinct_reasons_get_distinct_keys(self, monkeypatch):
        """Two different reasons on the SAME tool must not share an [a]lways
        allowlist entry (Finding 3: tool_name alone was too coarse)."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        k1 = request_tool_approval("write_file", "write to ~/.ssh")["pattern_key"]
        k2 = request_tool_approval("write_file", "send an email")["pattern_key"]
        assert k1 != k2

    def test_explicit_rule_key_overrides_derivation(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval("terminal", "any", rule_key="my-rule")
        assert res["pattern_key"] == "plugin_rule:my-rule"

    def test_no_human_non_cron_fails_closed(self, monkeypatch):
        """Non-interactive, non-gateway, NON-cron context blocks (fail-closed)
        — a plugin-flagged action never runs ungated without a human."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "no interactive user or gateway" in res["message"].lower()

    def test_yolo_session_bypasses_gate(self, monkeypatch):
        """A --yolo session skips the plugin approval gate (parity with the
        dangerous-command path, via the shared _run_approval_gate)."""
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("yolo must not prompt"),
        )
        res = request_tool_approval("terminal", "curl PUT", rule_key="ext")
        assert res == {"approved": True, "message": None}

    def test_once_only_ignores_cached_and_yolo_bypasses(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        prompted = {}

        def prompt(*args, **kwargs):
            prompted.update(kwargs)
            return "once"

        monkeypatch.setattr(approval, "prompt_dangerous_approval", prompt)
        res = request_tool_approval(
            "terminal",
            "real-money action",
            allow_session=False,
            allow_permanent=False,
            allow_yolo=False,
            allow_cron=False,
        )

        assert res == {"approved": True, "message": None}
        assert prompted["allow_session"] is False
        assert prompted["allow_permanent"] is False

    @pytest.mark.parametrize("choice", ["session", "always"])
    def test_once_only_rejects_disallowed_persistent_choice(
        self, monkeypatch, choice
    ):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: choice
        )

        res = request_tool_approval(
            "terminal",
            "real-money action",
            allow_session=False,
            allow_permanent=False,
            allow_yolo=False,
            allow_cron=False,
        )

        assert res["approved"] is False
        assert "not permitted" in res["message"].lower()

    def test_once_only_blocks_cron_even_when_cron_mode_approves(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(
            approval, "env_var_enabled", lambda name: name == "HERMES_CRON_SESSION"
        )
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "approve")

        res = request_tool_approval(
            "terminal",
            "real-money action",
            allow_session=False,
            allow_permanent=False,
            allow_yolo=False,
            allow_cron=False,
        )

        assert res["approved"] is False
        assert "cron" in res["message"].lower()

    def test_once_only_gateway_request_exposes_only_once_and_deny(
        self, monkeypatch
    ):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        submitted = {}
        monkeypatch.setattr(
            approval, "submit_pending", lambda sk, data: submitted.update(data)
        )

        res = request_tool_approval(
            "terminal",
            "real-money action",
            allow_session=False,
            allow_permanent=False,
            allow_yolo=False,
            allow_cron=False,
        )

        assert res["status"] == "approval_required"
        assert submitted["allow_session"] is False
        assert submitted["allow_permanent"] is False
