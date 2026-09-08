"""Tail budgets must charge transport-visible content, not display-only text."""

from copy import deepcopy

import pytest

from agent.context_compressor import (
    ContextCompressor, _MAX_TAIL_MESSAGE_FLOOR, _estimate_msg_budget_tokens,
)
from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import substitute_api_content


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize("sidecar", [None, "", 123, ["invalid sidecar"], "memory " * 5000])
@pytest.mark.parametrize("content", [
    "brief display",
    [{"type": "text", "text": "typed display"}],
    [{"type": "text", "text": "image caption"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "a" * 80}}],
])
@pytest.mark.parametrize("charge_stale_thinking", [False, True])
def test_budget_matches_transport_substitution_without_mutation(
    role, sidecar, content, charge_stale_thinking
):
    message = {
        "role": role,
        "content": content,
        "api_content": sidecar,
        "reasoning_content": "reasoning " * 100,
        "codex_reasoning_items": [{"encrypted_content": "replay" * 100}],
        "tool_calls": [{"id": "call-fixture", "type": "function", "function": {
            "name": "fixture", "arguments": "{}",
        }}],
    }
    original = deepcopy(message)
    wire_message = deepcopy(message)
    substitute_api_content(wire_message)
    assert _estimate_msg_budget_tokens(message, charge_stale_thinking) == (
        _estimate_msg_budget_tokens(wire_message, charge_stale_thinking)
    )
    assert message == original


def test_memory_augmented_short_user_turns_do_not_overfill_protected_tail():
    compressor = ContextCompressor(
        model="test-model", provider="test", quiet_mode=True,
        config_context_length=200_000,
    )
    compressor.tail_token_budget = 10_000
    messages = [{"role": "system", "content": "fixture system"}]
    for turn in range(50):
        display = f"Short user turn {turn}."
        messages.extend([
            {"role": "user", "content": display,
             "api_content": display + "\n\n<memory>" + "context " * 4000 + "</memory>"},
            {"role": "assistant", "content": f"Brief reply {turn}."},
        ])
    original = deepcopy(messages)
    head = compressor._protect_head_size(messages)
    cut = compressor._find_tail_cut_by_tokens(messages, head)
    protected_tokens = estimate_messages_tokens_rough(messages[cut:])
    assert cut > len(messages) // 2
    # The existing message floor can exceed the token budget, but no longer
    # keeps the hundreds of thousands of tokens hidden in older sidecars.
    floor_tokens = estimate_messages_tokens_rough(messages[-_MAX_TAIL_MESSAGE_FLOOR:])
    assert protected_tokens <= max(floor_tokens, compressor.tail_token_budget * 1.5)
    assert estimate_messages_tokens_rough(messages[head:cut]) > protected_tokens
    assert messages[-2] in messages[cut:]
    assert messages == original
    wire_messages = deepcopy(messages)
    for message in wire_messages:
        substitute_api_content(message)
    assert cut == compressor._find_tail_cut_by_tokens(wire_messages, head)
