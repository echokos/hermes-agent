"""Persistence and reclaim gates for per-turn micro-compaction."""

from copy import deepcopy
import hashlib
import json
import logging
import sqlite3

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _DB_PERSISTED_MARKER,
)
from agent.model_metadata import estimate_messages_tokens_rough
from hermes_state import SessionDB


def _compressor(summary: str = "ROLLING SUMMARY") -> ContextCompressor:
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    compressor._micro_compact_enabled = True
    compressor._micro_summarize_one = lambda _text: summary
    return compressor


def _conversation(exchanges: int = 10) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": "system prompt",
            "custom_metadata": {"nested": ["system", {"keep": True}]},
        }
    ]
    for index in range(exchanges):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"question {index}",
                    "api_content": f"api question {index}",
                    "display_kind": "text",
                    "display_metadata": {"index": index, "nested": [1, 2]},
                    "custom_metadata": {"owner": "user", "index": index},
                },
                {
                    "role": "assistant",
                    "content": f"answer {index} " + "z" * 400,
                    "reasoning": f"reasoning {index}",
                    "reasoning_details": [{"type": "text", "text": str(index)}],
                    "finish_reason": "stop",
                    "custom_metadata": {"owner": "assistant", "index": index},
                },
            ]
        )
    return messages


@pytest.fixture
def bound_session_factory(tmp_path):
    databases: list[SessionDB] = []
    serial = 0

    def create(summary: str = "ROLLING SUMMARY"):
        nonlocal serial
        serial += 1
        db_path = tmp_path / f"state-{serial}.db"
        session_id = f"micro-reclaim-{serial}"
        database = SessionDB(db_path=db_path)
        databases.append(database)
        database.create_session(session_id, source="test")
        messages = _conversation()
        database.append_messages_batch(session_id, messages)
        # Match a live agent after its append-only flush. These internal stamps
        # are intentionally absent from SQLite but must survive every rejection.
        for message in messages:
            message[_DB_PERSISTED_MARKER] = True
        compressor = _compressor(summary)
        compressor.bind_session_state(database, session_id)
        return compressor, database, db_path, session_id, messages

    yield create

    for database in databases:
        database.close()


def _raw_db_rows(db_path) -> list[tuple]:
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM messages ORDER BY id"
        ).fetchall()


def _db_counts(db_path, session_id) -> tuple[int, int]:
    with sqlite3.connect(db_path) as connection:
        active, archived = connection.execute(
            "SELECT SUM(active = 1), SUM(active = 0 AND compacted = 1) "
            "FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(active or 0), int(archived or 0)


def _telemetry_payloads(caplog) -> list[dict]:
    prefix = "micro compaction telemetry: "
    return [
        json.loads(record.getMessage().split(prefix, 1)[1])
        for record in caplog.records
        if prefix in record.getMessage()
    ]


def _assert_input_unchanged(messages, before, identities) -> None:
    assert messages == before
    assert [id(message) for message in messages] == identities


def _context_tokens(messages) -> int:
    """Estimate without private SessionDB bookkeeping fields."""
    normalized = [
        {
            key: value
            for key, value in message.items()
            if key not in {_DB_PERSISTED_MARKER, "_row_id"}
        }
        if isinstance(message, dict)
        else message
        for message in messages
    ]
    return estimate_messages_tokens_rough(normalized)


def _install_post_write_failure(database: SessionDB, monkeypatch) -> None:
    """Run the real archive callback, then roll it back and raise."""

    def fail_after_write(operation, patience_s=None):
        del patience_s
        with database._lock:
            database._conn.execute("BEGIN IMMEDIATE")
            try:
                operation(database._conn)
                raise sqlite3.OperationalError("forced failure before commit")
            except BaseException:
                database._conn.rollback()
                raise

    monkeypatch.setattr(database, "_execute_write", fail_after_write)


def test_intentional_first_marker_growth_still_commits(bound_session_factory):
    compressor, database, _, session_id, messages = bound_session_factory()
    before_tokens = estimate_messages_tokens_rough(messages)

    result = compressor._micro_compact(list(messages))

    assert estimate_messages_tokens_rough(result) > before_tokens
    assert len([m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]) == 1
    reloaded = database.get_messages_as_conversation(session_id)
    assert [(m["role"], m["content"]) for m in reloaded] == [
        (m["role"], m["content"]) for m in result
    ]
    assert any(compressor._is_context_summary_message(m) for m in reloaded)


def test_cumulative_growth_rejection_is_copy_on_write_and_skips_db(
    bound_session_factory, caplog, monkeypatch
):
    compressor, database, db_path, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    old_summary = compressor._micro_compact_rolling_summary
    old_cursor = compressor._micro_compact_cursor
    old_saved = compressor._micro_compact_tokens_saved_total
    attempt = list(messages)
    before = deepcopy(attempt)
    identities = [id(message) for message in attempt]
    db_before = _raw_db_rows(db_path)
    sync_calls = 0
    real_sync = database.archive_and_compact

    def count_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(database, "archive_and_compact", count_sync)
    compressor._micro_summarize_one = lambda _text: "EXPANDED " + "x" * 8000

    with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
        result = compressor._micro_compact(attempt)

    assert result is attempt
    _assert_input_unchanged(attempt, before, identities)
    assert _raw_db_rows(db_path) == db_before
    assert sync_calls == 0
    assert compressor._micro_compact_rolling_summary == old_summary
    assert compressor._micro_compact_cursor > old_cursor
    assert compressor._micro_compact_tokens_saved_total == old_saved
    payload = _telemetry_payloads(caplog)[-1]
    assert payload["outcome"] == "no_reclaim"
    assert payload["tokens_delta"] == 0
    assert payload["candidate_tokens_delta"] > 0


def test_rejected_exchange_consumes_cadence_and_next_attempt_moves_on(
    bound_session_factory
):
    compressor, _, _, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    compressor._micro_compact_every_n_turns = 3
    compressor._micro_compact_turns_since_pass = 2
    summarized_inputs: list[str] = []

    def expanding(text):
        summarized_inputs.append(text)
        return "EXPANDED " + "x" * 8000

    compressor._micro_summarize_one = expanding
    unchanged = compressor._micro_compact(list(messages))
    assert len(summarized_inputs) == 1
    assert "answer 1" in summarized_inputs[0]

    assert compressor._micro_compact(list(unchanged)) == unchanged
    assert compressor._micro_compact(list(unchanged)) == unchanged
    assert len(summarized_inputs) == 1

    compressor._micro_compact(list(unchanged))
    assert len(summarized_inputs) == 2
    assert "answer 2" in summarized_inputs[1]
    assert "answer 1" not in summarized_inputs[1]


def test_rejected_tail_boundary_is_not_rescanned_and_resumes_after_growth(
    bound_session_factory, monkeypatch
):
    compressor, _, _, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    summarized_inputs: list[str] = []

    def expanding(text):
        summarized_inputs.append(text)
        return "EXPANDED " + "x" * 8000

    compressor._micro_summarize_one = expanding

    # Fix the tail boundary at the end of the third remaining exchange. This
    # makes the exhausted cursor exactly equal to tail_start, the edge that a
    # strict `< tail_start` cursor-validity check used to discard.
    head_end = compressor._align_boundary_forward(
        messages, compressor._protect_head_size(messages)
    )
    cursor = compressor._micro_compact_cursor
    exchange_ends: list[int] = []
    for _ in range(4):
        exchange = compressor._find_one_exchange(messages, cursor, len(messages) - 1)
        assert exchange is not None
        _, cursor = exchange
        exchange_ends.append(cursor)

    tail_limit = exchange_ends[2]
    monkeypatch.setattr(
        compressor,
        "_find_tail_cut_by_tokens",
        lambda _messages, _head_start: tail_limit,
    )
    assert head_end < tail_limit

    # Reject every exchange currently outside the protected tail.
    for _ in range(3):
        assert compressor._micro_compact(messages) is messages

    attempted_before_exhaustion = list(summarized_inputs)
    assert len(attempted_before_exhaustion) == 3
    assert len(attempted_before_exhaustion) == len(set(attempted_before_exhaustion))

    # Repeated same-transcript passes must preserve the exhausted cursor rather
    # than re-scan from the persisted marker and buy an old rejection again.
    for _ in range(3):
        assert compressor._micro_compact(messages) is messages
    assert summarized_inputs == attempted_before_exhaustion

    # Moving the protected tail after transcript growth makes the same cursor
    # eligible naturally and processing resumes at the next exchange.
    tail_limit = exchange_ends[3]
    compressor._micro_compact(messages)
    assert len(summarized_inputs) == len(attempted_before_exhaustion) + 1
    assert summarized_inputs[-1] not in attempted_before_exhaustion


def test_exact_context_boundary_rejects_before_db_bookkeeping_can_mask_it(
    bound_session_factory, monkeypatch
):
    compressor, database, db_path, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    before_context_tokens = _context_tokens(messages)
    before_raw_tokens = estimate_messages_tokens_rough(messages)

    head_end = compressor._align_boundary_forward(
        messages, compressor._protect_head_size(messages)
    )
    tail_start = compressor._find_tail_cut_by_tokens(messages, head_end)
    cursor = compressor._resolve_compact_cursor(messages, head_end, tail_start)
    exchange = compressor._find_one_exchange(messages, cursor, tail_start)
    assert exchange is not None
    exchange_start, exchange_end = exchange

    def build_candidate(length):
        return compressor._splice_micro_compact_result(
            messages,
            exchange_start,
            exchange_end,
            supersede=True,
            summary_text="B" * length,
            previous_summary=compressor._micro_compact_rolling_summary,
        )

    low, high = 1, 12_000
    while low < high:
        midpoint = (low + high) // 2
        if _context_tokens(build_candidate(midpoint)) < before_context_tokens:
            low = midpoint + 1
        else:
            high = midpoint

    boundary_summary = "B" * low
    boundary_candidate = build_candidate(low)
    assert _context_tokens(boundary_candidate) == before_context_tokens
    assert estimate_messages_tokens_rough(boundary_candidate) < before_raw_tokens
    compressor._micro_summarize_one = lambda _text: boundary_summary
    db_before = _raw_db_rows(db_path)
    sync_calls = 0
    real_sync = database.archive_and_compact

    def count_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(database, "archive_and_compact", count_sync)
    result = compressor._micro_compact(messages)

    assert result is messages
    assert sync_calls == 0
    assert _raw_db_rows(db_path) == db_before


def test_defrag_equal_size_is_rejected(bound_session_factory, monkeypatch, caplog):
    compressor, database, db_path, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    old_summary = compressor._micro_compact_rolling_summary
    before = deepcopy(messages)
    identities = [id(message) for message in messages]
    db_before = _raw_db_rows(db_path)
    compressor._micro_compact_defrag_threshold_tokens = 1
    compressor._micro_summarize_one = lambda _text: old_summary
    sync_calls = 0
    real_sync = database.archive_and_compact

    def count_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(database, "archive_and_compact", count_sync)
    with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
        result = compressor._micro_compact(messages)

    assert result is messages
    _assert_input_unchanged(messages, before, identities)
    assert sync_calls == 0
    assert _raw_db_rows(db_path) == db_before
    payload = _telemetry_payloads(caplog)[-1]
    assert payload["outcome"] == "defrag_no_reclaim"
    assert payload["tokens_delta"] == 0
    assert payload["candidate_tokens_delta"] == 0


def test_positive_defrag_commits_reloads_and_flushes_without_duplicates(
    bound_session_factory, monkeypatch
):
    from run_agent import AIAgent

    compressor, database, db_path, session_id, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    active_before, archived_before = _db_counts(db_path, session_id)
    old_summary = compressor._micro_compact_rolling_summary
    old_marker_content = compressor._render_micro_marker_content(old_summary)
    compressor._micro_compact_defrag_threshold_tokens = 1
    compressor._micro_summarize_one = lambda _text: "SHORT"

    result = compressor._micro_compact(messages)

    assert result is not messages
    assert _context_tokens(result) < _context_tokens(messages)
    assert all(message.get(_DB_PERSISTED_MARKER) for message in result)
    result_markers = [
        message for message in result
        if compressor._is_context_summary_message(message)
    ]
    assert len(result_markers) == 1
    assert "SHORT" in result_markers[0]["content"]

    reloaded = database.get_messages_as_conversation(session_id)
    assert [(message["role"], message["content"]) for message in reloaded] == [
        (message["role"], message["content"]) for message in result
    ]
    assert len(
        [
            message for message in reloaded
            if compressor._is_context_summary_message(message)
        ]
    ) == 1
    assert _db_counts(db_path, session_id) == (
        len(result),
        archived_before + active_before,
    )
    with sqlite3.connect(db_path) as connection:
        archived_marker_count = connection.execute(
            "SELECT COUNT(*) FROM messages "
            "WHERE session_id = ? AND active = 0 AND compacted = 1 "
            "AND content = ?",
            (session_id, old_marker_content),
        ).fetchone()[0]
    assert archived_marker_count == 1

    # Exercise the real append-only flush path that follows finalize_turn.
    # Every committed candidate row is already stamped, so it must issue no
    # second append and leave the DB cardinality unchanged.
    append_calls = 0
    real_append = database.append_messages_batch

    def count_append(*args, **kwargs):
        nonlocal append_calls
        append_calls += 1
        return real_append(*args, **kwargs)

    monkeypatch.setattr(database, "append_messages_batch", count_append)
    agent = AIAgent.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = database
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = len(result)
    agent._db_flush_scan_prefix = None
    assert agent._flush_messages_to_session_db_unlocked(result) is True
    assert append_calls == 0
    assert _db_counts(db_path, session_id) == (
        len(result),
        archived_before + active_before,
    )


def test_defrag_growth_is_rejected_once_per_summary_digest(
    bound_session_factory, caplog
):
    compressor, _, _, _, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    old_summary = compressor._micro_compact_rolling_summary
    attempt = list(messages)
    before = deepcopy(attempt)
    identities = [id(message) for message in attempt]
    compressor._micro_compact_defrag_threshold_tokens = 1
    compressor._micro_summarize_one = lambda _text: "EXPANDED " + "x" * 8000

    with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
        result = compressor._micro_compact(attempt)

    assert result is attempt
    _assert_input_unchanged(attempt, before, identities)
    expected_digest = hashlib.sha256(old_summary.encode()).hexdigest()
    assert compressor._micro_compact_defrag_blocked_digest == expected_digest
    assert _telemetry_payloads(caplog)[-1]["outcome"] == "defrag_no_reclaim"

    summarized_inputs: list[str] = []

    def compact_absorption(text):
        summarized_inputs.append(text)
        return "UPDATED SUMMARY"

    compressor._micro_summarize_one = compact_absorption
    absorbed = compressor._micro_compact(list(attempt))
    assert len(summarized_inputs) == 1
    assert "[ASSISTANT]" in summarized_inputs[0]
    assert summarized_inputs[0] != old_summary
    assert absorbed != attempt
    assert compressor._micro_compact_defrag_blocked_digest == ""
    assert compressor._needs_defrag() is True


@pytest.mark.parametrize("pass_kind", ["first", "cumulative", "defrag"])
def test_bound_db_failure_rolls_back_without_adopting_candidate(
    pass_kind, bound_session_factory, caplog, monkeypatch
):
    initial_summary = (
        "OLD SUMMARY " + "x" * 4000 if pass_kind == "defrag" else "ROLLING SUMMARY"
    )
    compressor, database, db_path, _, messages = bound_session_factory(
        initial_summary
    )
    if pass_kind != "first":
        messages = compressor._micro_compact(list(messages))
    if pass_kind == "defrag":
        compressor._micro_compact_defrag_threshold_tokens = 1
        compressor._micro_summarize_one = lambda _text: "SHORT"
    elif pass_kind == "cumulative":
        compressor._micro_summarize_one = lambda _text: "UPDATED"

    attempt = list(messages)
    before = deepcopy(attempt)
    identities = [id(message) for message in attempt]
    db_before = _raw_db_rows(db_path)
    summary_before = compressor._micro_compact_rolling_summary
    cursor_before = compressor._micro_compact_cursor
    saved_before = compressor._micro_compact_tokens_saved_total
    _install_post_write_failure(database, monkeypatch)

    with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
        result = compressor._micro_compact(attempt)

    assert result is attempt
    _assert_input_unchanged(attempt, before, identities)
    assert _raw_db_rows(db_path) == db_before
    assert compressor._micro_compact_rolling_summary == summary_before
    assert compressor._micro_compact_tokens_saved_total == saved_before
    payload = _telemetry_payloads(caplog)[-1]
    expected_outcome = (
        "defrag_db_sync_failed" if pass_kind == "defrag" else "db_sync_failed"
    )
    assert payload["outcome"] == expected_outcome
    assert payload["tokens_delta"] == 0
    assert payload["candidate_tokens_delta"] is not None
    if pass_kind == "defrag":
        assert compressor._micro_compact_cursor == cursor_before
        assert compressor._micro_compact_defrag_blocked_digest
    else:
        assert compressor._micro_compact_cursor > cursor_before


def test_restart_may_reconsider_rejected_exchange_without_losing_content(
    bound_session_factory
):
    compressor, database, _, session_id, messages = bound_session_factory()
    messages = compressor._micro_compact(list(messages))
    compressor._micro_summarize_one = lambda _text: "EXPANDED " + "x" * 8000
    rejected = compressor._micro_compact(list(messages))
    assert any("answer 1" in str(m.get("content")) for m in rejected)

    # The skip cursor is deliberately process-local. A fresh compressor may
    # reconsider this exchange once, but rehydrates the old marker first and
    # must neither drop the old summary nor leak duplicate markers.
    reloaded = database.get_messages_as_conversation(session_id)
    restarted = _compressor("REHYDRATED PLUS ANSWER 1")
    restarted.bind_session_state(database, session_id)
    summarized_inputs: list[str] = []

    def summarize(text):
        summarized_inputs.append(text)
        return "REHYDRATED PLUS ANSWER 1"

    restarted._micro_summarize_one = summarize
    result = restarted._micro_compact(reloaded)

    assert len(summarized_inputs) == 1
    assert "answer 1" in summarized_inputs[0]
    assert restarted._micro_compact_rolling_summary == "REHYDRATED PLUS ANSWER 1"
    assert len([m for m in result if restarted._is_context_summary_message(m)]) == 1
    assert any("question 1" in str(m.get("content")) for m in result)


def test_micro_retry_state_does_not_leak_across_session_boundaries(
    bound_session_factory
):
    compressor, database, _, _, _ = bound_session_factory()
    compressor._micro_compact_cursor = 7
    compressor._micro_compact_rolling_summary = "OLD SESSION"
    compressor._micro_compact_consecutive_failures = 2
    compressor._micro_compact_last_failure_cursor = 6
    compressor._micro_compact_defrag_blocked_digest = "digest"
    compressor._micro_compact_passes = 4
    compressor._micro_compact_tokens_saved_total = 123
    compressor._micro_compact_turns_since_pass = 2
    compressor._flush_scan_cursor_invalidated = True

    compressor.on_session_end("old-session", [])

    assert compressor._micro_compact_cursor == 0
    assert compressor._micro_compact_rolling_summary == ""
    assert compressor._micro_compact_consecutive_failures == 0
    assert compressor._micro_compact_last_failure_cursor == -1
    assert compressor._micro_compact_defrag_blocked_digest == ""
    assert compressor._micro_compact_passes == 0
    assert compressor._micro_compact_tokens_saved_total == 0
    assert compressor._micro_compact_turns_since_pass == 0
    assert compressor._flush_scan_cursor_invalidated is False

    database.create_session("next-session", source="test")
    compressor._micro_compact_cursor = 9
    compressor._micro_compact_defrag_blocked_digest = "other"
    compressor.bind_session_state(database, "next-session")
    assert compressor._micro_compact_cursor == 0
    assert compressor._micro_compact_defrag_blocked_digest == ""
