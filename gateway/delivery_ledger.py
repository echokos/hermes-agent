"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup, ``sweep_recoverable()`` claims rows whose owning process is
dead and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; the restart is a natural
  retry boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped, stale rows expire, and both
transition to ``abandoned`` (kept briefly for inspection, then pruned).

Everything here is best-effort by design: ledger failures must never block
or delay an actual send. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)

# Coordination final returns are deliberately separate from ordinary gateway
# redelivery.  A crash while a platform send is in flight is unknowable; never
# let the generic startup sweeper silently replay one of these rows.
COORDINATION_FINAL_RETURN_PURPOSE = "final_return"
_COORDINATION_FINAL_RETURN_STATES = frozenset({
    "pending", "sending", "acknowledged", "uncertain",
})


@dataclass(frozen=True)
class CoordinationFinalReturnDelivery:
    """One durable final-return outbox record.

    ``pending`` means no send is known to have started. ``sending`` and
    ``uncertain`` both prohibit automatic replay; a caller must reconcile a
    concrete receipt before it can acknowledge the return.
    """

    obligation_id: str
    state: str
    request_root_id: str
    event_id: int
    returned_message_id: str = ""
    last_error: str = ""
    send_claimed: bool = False
    claim_token: str = ""


def coordination_final_return_obligation_id(request_root_id: str, event_id: int) -> str:
    """Return the stable root/purpose/version identity for one final return."""
    root = str(request_root_id or "").strip()
    if not root or any(char in root for char in "\x00\r\n:"):
        raise ValueError("invalid coordination request root id")
    if isinstance(event_id, bool):
        raise ValueError("invalid coordination final-return event id")
    try:
        parsed_event_id = int(event_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid coordination final-return event id") from exc
    if parsed_event_id <= 0:
        raise ValueError("invalid coordination final-return event id")
    return f"coordination:{root}:final_return:{parsed_event_id}"


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            delivery_purpose TEXT,
            coordination_root_id TEXT,
            coordination_event_id INTEGER,
            returned_message_id TEXT,
            claim_token TEXT
        )"""
    )
    # Existing state.db files predate the coordination columns. Keep this
    # additive and local to the ledger rather than introducing a second
    # durable outbox in the Kanban board.
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(delivery_obligations)").fetchall()
    }
    for name, sql_type in (
        ("delivery_purpose", "TEXT"),
        ("coordination_root_id", "TEXT"),
        ("coordination_event_id", "INTEGER"),
        ("returned_message_id", "TEXT"),
        ("claim_token", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE delivery_obligations ADD COLUMN {name} {sql_type}")


@contextmanager
def _transaction(*, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector. On a long-running
    gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this
    bug was #69567 / PR #69594). ``record_obligation`` runs on every outbound
    final response, so this ledger is the highest-frequency leaker.
    """
    conn = _connect()
    try:
        if immediate:
            # `_DB_LOCK` coordinates threads in this process only. The
            # outbox also protects concurrent gateway processes, so take the
            # SQLite writer reservation before examining a coordination row.
            conn.execute("BEGIN IMMEDIATE")
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists. Route through the
        # cross-platform probe: ``os.kill(pid, 0)`` on Windows is NOT a
        # no-op (bpo-14484 — CPython maps sig=0 to
        # ``GenerateConsoleCtrlEvent(0, pid)``), so a raw probe here could
        # Ctrl+C the gateway's own console group whenever psutil failed to
        # read the start time of a live pid. ``_pid_exists`` keeps the
        # EPERM-means-alive semantics (exists but owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                # Never fall back to a raw sig-0 probe on Windows.
                return False
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
            return True
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while
    distinct threads/topics on the same chat can never collide (the
    session_key carries platform, chat and thread; ``message_ref`` is the
    triggering inbound message id, distinguishing turns in one session)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started),
        )
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def _coordination_row(row: Any) -> CoordinationFinalReturnDelivery:
    if not hasattr(row, "keys"):
        row = {
            "obligation_id": row[0],
            "state": row[1],
            "coordination_root_id": row[2],
            "coordination_event_id": row[3],
            "returned_message_id": row[4],
            "last_error": row[5],
            "claim_token": row[6] if len(row) > 6 else "",
        }
    return CoordinationFinalReturnDelivery(
        obligation_id=str(row["obligation_id"]),
        state=str(row["state"]),
        request_root_id=str(row["coordination_root_id"] or ""),
        event_id=int(row["coordination_event_id"]),
        returned_message_id=str(row["returned_message_id"] or ""),
        last_error=str(row["last_error"] or ""),
        claim_token=str(row["claim_token"] or ""),
    )


def get_coordination_final_return_delivery(
    request_root_id: str,
    event_id: int,
) -> Optional[CoordinationFinalReturnDelivery]:
    """Load one coordination return record without changing its state."""
    obligation_id = coordination_final_return_obligation_id(request_root_id, event_id)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ? "
            "AND delivery_purpose = ?",
            (obligation_id, COORDINATION_FINAL_RETURN_PURPOSE),
        ).fetchone()
    return _coordination_row(row) if row is not None else None


def _coordination_claim_token(value: str = "") -> str:
    token = str(value or "").strip()
    if not token:
        return secrets.token_hex(16)
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("invalid coordination final-return claim token")
    return token


def _validate_coordination_final_return_authority(
    *,
    request_root_id: str,
    task_id: str,
    event_id: int,
    responsible_agent: str,
    board_path: str,
    require_terminal: bool = True,
) -> None:
    """Require the durable board still authorizes this terminal return."""
    from hermes_cli import kanban_db

    conn = kanban_db.connect(Path(board_path))
    try:
        kanban_db.validate_coordination_final_return_authority(
            conn,
            request_root_id=request_root_id,
            task_id=task_id,
            event_id=event_id,
            responsible_agent=responsible_agent,
            require_terminal=require_terminal,
        )
    finally:
        conn.close()


def claim_coordination_final_return_delivery(
    *,
    request_root_id: str,
    task_id: str,
    event_id: int,
    responsible_agent: str,
    board_path: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    claim_token: str = "",
) -> CoordinationFinalReturnDelivery:
    """Durably claim the sole final-return send immediately before I/O.

    A prior ``sending`` or ``uncertain`` state is intentionally returned
    without another platform call. This makes restart behavior explicit: a
    human/adapter-specific reconciliation must establish a receipt before the
    system treats the final return as delivered.
    """
    obligation_id = coordination_final_return_obligation_id(request_root_id, event_id)
    root = str(request_root_id).strip()
    body = str(content or "")
    if not body:
        raise ValueError("coordination final-return content is required")
    # Validate immediately before transitioning from safe pending to a
    # potentially-visible send. A final response that ceased to be terminal
    # while its agent turn ran is withheld instead of leaking stale output.
    _validate_coordination_final_return_authority(
        request_root_id=root,
        task_id=str(task_id).strip(),
        event_id=event_id,
        responsible_agent=str(responsible_agent).strip(),
        board_path=str(board_path),
        require_terminal=True,
    )

    now = time.time()
    pid, started = _owner_stamp()
    token = _coordination_claim_token(claim_token)
    did_claim = False
    with _DB_LOCK, _transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ?",
            (obligation_id,),
        ).fetchone()
        if row is not None:
            record = _coordination_row(row)
            if record.state == "acknowledged" or record.state == "uncertain":
                return record
            if record.state == "sending":
                if record.claim_token == token:
                    # A ticket admits exactly one visible send. The model
                    # worker can be re-entered with its own header/event, so
                    # fence the send itself separately from turn admission.
                    cursor = conn.execute(
                        "UPDATE delivery_obligations SET content = ?, updated_at = ?, attempts = attempts + 1 "
                        "WHERE obligation_id = ? AND state = 'sending' AND claim_token = ? "
                        "AND attempts = 0",
                        (body, now, obligation_id, token),
                    )
                    return CoordinationFinalReturnDelivery(
                        **{**record.__dict__, "send_claimed": bool(cursor.rowcount)}
                    )
                return record
            if (
                record.state != "pending"
                or record.request_root_id != root
                or record.event_id != int(event_id)
            ):
                raise RuntimeError("coordination final-return delivery identity conflict")
            cursor = conn.execute(
                "UPDATE delivery_obligations SET state = 'sending', content = ?, "
                "session_key = ?, platform = ?, chat_id = ?, thread_id = ?, "
                "updated_at = ?, owner_pid = ?, owner_started_at = ?, last_error = NULL, claim_token = ?, attempts = 1 "
                "WHERE obligation_id = ? AND state = 'pending'",
                (body, session_key, platform, str(chat_id),
                 str(thread_id) if thread_id else None, now, pid, started,
                 token, obligation_id),
            )
            did_claim = bool(cursor.rowcount)
        else:
            # Store a pending row first, then claim it in the same critical
            # section. A hard crash can therefore leave only a safe pending
            # record, never an untracked send attempt.
            conn.execute(
                "INSERT INTO delivery_obligations "
                "(obligation_id, session_key, platform, chat_id, thread_id, "
                "content, state, attempts, created_at, updated_at, owner_pid, "
                "owner_started_at, delivery_purpose, coordination_root_id, "
                "coordination_event_id, claim_token) "
                "VALUES (?, ?, ?, ?, ?, ?, 'sending', 1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (obligation_id, session_key, platform, str(chat_id),
                 str(thread_id) if thread_id else None, body, now, now, pid,
                 started, COORDINATION_FINAL_RETURN_PURPOSE, root, int(event_id), token),
            )
            did_claim = True
        claimed = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ?",
            (obligation_id,),
        ).fetchone()
    if claimed is None:  # pragma: no cover - transaction invariant
        raise RuntimeError("coordination final-return delivery claim disappeared")
    record = _coordination_row(claimed)
    return CoordinationFinalReturnDelivery(
        **{**record.__dict__, "send_claimed": did_claim}
    )


def admit_coordination_final_return_turn(
    *,
    request_root_id: str,
    task_id: str,
    event_id: int,
    responsible_agent: str,
    board_path: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
) -> CoordinationFinalReturnDelivery:
    """Reserve one final-return turn before a provider call can begin.

    ``sending`` here means its opaque delivery ticket is live, not that a
    platform call has happened. This prevents a timed-out watcher from
    starting another model turn while the first worker might still finish.
    """
    obligation_id = coordination_final_return_obligation_id(request_root_id, event_id)
    root = str(request_root_id).strip()
    _validate_coordination_final_return_authority(
        request_root_id=root,
        task_id=str(task_id).strip(),
        event_id=event_id,
        responsible_agent=str(responsible_agent).strip(),
        board_path=str(board_path),
        require_terminal=False,
    )
    now = time.time()
    pid, started = _owner_stamp()
    token = _coordination_claim_token()
    with _DB_LOCK, _transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ?",
            (obligation_id,),
        ).fetchone()
        if row is not None:
            record = _coordination_row(row)
            if record.state in {"acknowledged", "sending", "uncertain"}:
                return record
            if (
                record.state != "pending"
                or record.request_root_id != root
                or record.event_id != int(event_id)
            ):
                raise RuntimeError("coordination final-return delivery identity conflict")
            cursor = conn.execute(
                "UPDATE delivery_obligations SET state = 'sending', updated_at = ?, "
                "owner_pid = ?, owner_started_at = ?, last_error = NULL, claim_token = ?, attempts = 0 "
                "WHERE obligation_id = ? AND state = 'pending'",
                (now, pid, started, token, obligation_id),
            )
            if not cursor.rowcount:  # pragma: no cover - BEGIN IMMEDIATE fences this
                raise RuntimeError("coordination final-return turn admission lost")
        else:
            conn.execute(
                "INSERT INTO delivery_obligations "
                "(obligation_id, session_key, platform, chat_id, thread_id, content, "
                "state, attempts, created_at, updated_at, owner_pid, owner_started_at, "
                "delivery_purpose, coordination_root_id, coordination_event_id, claim_token) "
                "VALUES (?, ?, ?, ?, ?, '', 'sending', 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                (obligation_id, session_key, platform, str(chat_id),
                 str(thread_id) if thread_id else None, now, now, pid, started,
                 COORDINATION_FINAL_RETURN_PURPOSE, root, int(event_id), token),
            )
    return CoordinationFinalReturnDelivery(
        obligation_id=obligation_id,
        state="sending",
        request_root_id=root,
        event_id=int(event_id),
        send_claimed=True,
        claim_token=token,
    )


def mark_coordination_final_return_acknowledged(
    request_root_id: str,
    event_id: int,
    *,
    returned_message_id: str,
) -> CoordinationFinalReturnDelivery:
    """Persist the only acceptable terminal outcome: a concrete receipt."""
    receipt = str(returned_message_id or "").strip()
    if not (
        receipt.startswith("session-message:")
        or receipt.startswith("platform-message:")
    ) or any(
        char in receipt for char in "\x00\r\n"
    ):
        raise ValueError("coordination final-return receipt is invalid")
    obligation_id = coordination_final_return_obligation_id(request_root_id, event_id)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ? AND delivery_purpose = ?",
            (obligation_id, COORDINATION_FINAL_RETURN_PURPOSE),
        ).fetchone()
        if row is None:
            raise ValueError("coordination final-return delivery was never claimed")
        record = _coordination_row(row)
        if record.state == "acknowledged":
            if record.returned_message_id != receipt:
                raise RuntimeError("coordination final-return receipt conflict")
            return record
        if record.state not in {"sending", "uncertain"}:
            raise RuntimeError("coordination final-return delivery is not reconcilable")
        conn.execute(
            "UPDATE delivery_obligations SET state = 'acknowledged', "
            "returned_message_id = ?, updated_at = ?, last_error = NULL "
            "WHERE obligation_id = ? AND state IN ('sending', 'uncertain')",
            (receipt, time.time(), obligation_id),
        )
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ?",
            (obligation_id,),
        ).fetchone()
    return _coordination_row(row)


def mark_coordination_final_return_pending(
    request_root_id: str,
    event_id: int,
    *,
    error: str = "",
    claim_token: str = "",
    only_unstarted: bool = False,
) -> CoordinationFinalReturnDelivery:
    """Record a definite no-send outcome that a watcher may safely retry."""
    return _set_coordination_final_return_state(
        request_root_id,
        event_id,
        state="pending",
        error=error,
        claim_token=claim_token,
        only_unstarted=only_unstarted,
    )


def mark_coordination_final_return_uncertain(
    request_root_id: str,
    event_id: int,
    *,
    error: str = "",
) -> CoordinationFinalReturnDelivery:
    """Record an ambiguous send; it must be reconciled, never replayed."""
    return _set_coordination_final_return_state(
        request_root_id, event_id, state="uncertain", error=error,
    )


def _set_coordination_final_return_state(
    request_root_id: str,
    event_id: int,
    *,
    state: str,
    error: str,
    claim_token: str = "",
    only_unstarted: bool = False,
) -> CoordinationFinalReturnDelivery:
    if state not in {"pending", "uncertain"}:  # pragma: no cover - private guard
        raise ValueError("invalid coordination final-return state")
    obligation_id = coordination_final_return_obligation_id(request_root_id, event_id)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ? AND delivery_purpose = ?",
            (obligation_id, COORDINATION_FINAL_RETURN_PURPOSE),
        ).fetchone()
        if row is None:
            raise ValueError("coordination final-return delivery was never claimed")
        record = _coordination_row(row)
        if record.state == "acknowledged":
            return record
        if record.state != "sending":
            return record
        where = "WHERE obligation_id = ? AND state = 'sending'"
        params: list[Any] = [state, time.time(), str(error or "")[:500] or None, obligation_id]
        if claim_token:
            where += " AND claim_token = ?"
            params.append(_coordination_claim_token(claim_token))
        if only_unstarted:
            where += " AND attempts = 0"
        conn.execute(
            "UPDATE delivery_obligations SET state = ?, updated_at = ?, last_error = "
            f"? {where}",
            params,
        )
        row = conn.execute(
            "SELECT obligation_id, state, coordination_root_id, "
            "coordination_event_id, returned_message_id, last_error, claim_token "
            "FROM delivery_obligations WHERE obligation_id = ?",
            (obligation_id,),
        ).fetchone()
    return _coordination_row(row)


def reconcile_coordination_final_return(
    request_root_id: str,
    event_id: int,
    *,
    returned_message_id: str = "",
) -> Optional[CoordinationFinalReturnDelivery]:
    """Return durable state, or explicitly settle a known concrete receipt.

    There is intentionally no adapter polling or automatic resend here. A
    caller that cannot prove a receipt receives the existing ``uncertain``
    record and must retain its cursor/claim for human or adapter-specific
    reconciliation.
    """
    record = get_coordination_final_return_delivery(request_root_id, event_id)
    if record is None or not returned_message_id:
        return record
    return mark_coordination_final_return_acknowledged(
        request_root_id, event_id, returned_message_id=returned_message_id,
    )


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')
                 AND (delivery_purpose IS NULL OR delivery_purpose = '')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state,
             attempts, created_at, owner_pid, owner_started_at) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (now, oid),
                )
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ):
                # No adapter for this platform this boot — the caller cannot
                # send, so claiming would spend an attempt on a no-op.
                continue
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    # pending = send never started, redeliver plainly;
                    # attempting/failed = ambiguous or rejected, carry marker.
                    "needs_marker": state != "pending",
                    "attempts": attempts + 1,
                })
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'acknowledged', 'abandoned')
                     AND (delivery_purpose IS NULL OR delivery_purpose = '')
                     AND updated_at < ?""",
                (cutoff,),
            )
            total = conn.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         WHERE delivery_purpose IS NULL OR delivery_purpose = ''
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
