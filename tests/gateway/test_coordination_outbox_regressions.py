"""Process-level final-return arbitration and durable receipt retention."""

from contextlib import closing
import multiprocessing
from pathlib import Path
import sqlite3
import time

import pytest

from gateway import delivery_ledger as ledger


def _arguments():
    return dict(
        request_root_id="cr_regression", task_id="t_regression", event_id=7,
        responsible_agent="aurora", board_path="unused-by-arbitration-test",
        session_key="origin-session", platform="telegram", chat_id="origin-chat",
        thread_id=None,
    )


def _claim_in_process(path, ready, start, results, claim_token):
    ledger._db_path = lambda: Path(path)
    ledger._validate_coordination_final_return_authority = lambda **kwargs: None
    ready.put(True)
    if not start.wait(15):
        results.put(("error", "start timeout"))
        return
    try:
        result = (
            ledger.claim_coordination_final_return_delivery(
                **_arguments(), content="verified final", claim_token=claim_token,
            )
            if claim_token
            else ledger.admit_coordination_final_return_turn(**_arguments())
        )
        results.put(("ok", result.send_claimed, result.state))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    monkeypatch.setattr(ledger, "_db_path", lambda: path)
    monkeypatch.setattr(ledger, "_validate_coordination_final_return_authority", lambda **kwargs: None)
    with ledger._transaction():
        pass
    return path


def _compete(database, claim_token=None):
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    processes = [
        context.Process(target=_claim_in_process, args=(str(database), ready, start, results, claim_token))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=30)
        start.set()
        outcomes = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
        assert all(outcome[2] == "sending" for outcome in outcomes)
        with closing(sqlite3.connect(database)) as conn:
            assert conn.execute("SELECT count(*) FROM delivery_obligations").fetchone()[0] == 1
        return outcomes
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
        for queue in (ready, results):
            queue.close()
            queue.join_thread()


def test_separate_processes_admit_only_one_final_turn(database):
    assert sum(outcome[1] for outcome in _compete(database)) == 1


def test_replayed_owner_ticket_starts_only_one_platform_send(database):
    admission = ledger.admit_coordination_final_return_turn(**_arguments())
    assert sum(outcome[1] for outcome in _compete(database, admission.claim_token)) == 1


def test_only_admitted_ticket_can_send_and_uncertain_never_readmits(database):
    admission = ledger.admit_coordination_final_return_turn(**_arguments())
    assert admission.send_claimed
    unrelated = ledger.claim_coordination_final_return_delivery(
        **_arguments(), content="verified final", claim_token="f" * 32,
    )
    assert not unrelated.send_claimed
    own = ledger.claim_coordination_final_return_delivery(
        **_arguments(), content="verified final", claim_token=admission.claim_token,
    )
    assert own.send_claimed
    ledger.mark_coordination_final_return_uncertain("cr_regression", 7, error="transport_timeout")
    assert not ledger.admit_coordination_final_return_turn(**_arguments()).send_claimed
    assert not ledger.claim_coordination_final_return_delivery(
        **_arguments(), content="verified final", claim_token=admission.claim_token,
    ).send_claimed


def test_nonterminal_root_can_start_final_turn_but_cannot_send(database, monkeypatch):
    terminal = False
    checks = []

    def validate(*, require_terminal=True, **kwargs):
        checks.append(require_terminal)
        if require_terminal and not terminal:
            raise ValueError("coordination final return root task is not terminal")

    monkeypatch.setattr(ledger, "_validate_coordination_final_return_authority", validate)
    admission = ledger.admit_coordination_final_return_turn(**_arguments())
    assert admission.send_claimed and checks == [False]
    with pytest.raises(ValueError, match="not terminal"):
        ledger.claim_coordination_final_return_delivery(
            **_arguments(), content="premature", claim_token=admission.claim_token,
        )
    terminal = True
    assert ledger.claim_coordination_final_return_delivery(
        **_arguments(), content="verified final", claim_token=admission.claim_token,
    ).send_claimed
    assert checks == [False, True, True]


@pytest.mark.parametrize("state", ["pending", "sending", "uncertain", "acknowledged"])
def test_generic_cleanup_retains_coordination_rows_under_age_and_capacity_pressure(
    database, monkeypatch, state,
):
    admission = ledger.admit_coordination_final_return_turn(**_arguments())
    if state == "pending":
        ledger.mark_coordination_final_return_pending("cr_regression", 7, error="pre_send_rejection")
    elif state == "uncertain":
        ledger.mark_coordination_final_return_uncertain("cr_regression", 7, error="transport_timeout")
    elif state == "acknowledged":
        ledger.mark_coordination_final_return_acknowledged(
            "cr_regression", 7, returned_message_id="platform-message:telegram:origin-chat:123",
        )
    old = time.time() - 366 * 86400
    with ledger._transaction() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET created_at = ?, updated_at = ? WHERE obligation_id = ?",
            (old, old, admission.obligation_id),
        )
        conn.execute(
            "INSERT INTO delivery_obligations "
            "(obligation_id,session_key,platform,chat_id,content,state,attempts,created_at,updated_at) "
            "VALUES ('ordinary','session','telegram','chat','old ordinary result','delivered',0,?,?)",
            (old, old),
        )
    monkeypatch.setattr(ledger, "_MAX_ROWS", 0)
    ledger._prune()
    record = ledger.get_coordination_final_return_delivery("cr_regression", 7)
    assert record is not None and record.state == state
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute(
            "SELECT count(*) FROM delivery_obligations WHERE obligation_id = 'ordinary'",
        ).fetchone()[0] == 0


def test_acknowledged_receipt_survives_new_admission_and_conflicting_receipt(database):
    ledger.admit_coordination_final_return_turn(**_arguments())
    receipt = "platform-message:telegram:origin-chat:123"
    ledger.mark_coordination_final_return_acknowledged(
        "cr_regression", 7, returned_message_id=receipt,
    )
    existing = ledger.admit_coordination_final_return_turn(**_arguments())
    assert not existing.send_claimed
    assert existing.state == "acknowledged"
    assert existing.returned_message_id == receipt
    with pytest.raises(RuntimeError, match="receipt conflict"):
        ledger.mark_coordination_final_return_acknowledged(
            "cr_regression", 7, returned_message_id="platform-message:telegram:origin-chat:456",
        )
    assert ledger.get_coordination_final_return_delivery("cr_regression", 7).returned_message_id == receipt
