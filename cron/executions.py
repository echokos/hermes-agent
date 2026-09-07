"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so
# dashboard operations that temporarily enter another profile cannot leak that
# profile's execution records into the import-time home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown")
_IMPORT_EXECUTIONS_FILE = EXECUTIONS_FILE
_locks_guard = threading.Lock()
_locks: Dict[Path, threading.RLock] = {}
_PROCESS_ID = uuid.uuid4().hex


class _PrivateStatePermissionError(OSError):
    """A private cron state path could not be secured and verified."""


def _secure_execution_traversal_supported() -> bool:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    return bool(
        nofollow
        and directory_flag
        and os.open in supports_dir_fd
        and os.mkdir in supports_dir_fd
    )


def _reject_windows_execution_state_symlinks(path: Path) -> str:
    """Best-effort Windows symlink guard without POSIX-only open flags."""
    absolute = os.path.abspath(os.fspath(path))
    candidates = []
    current = absolute
    while True:
        candidates.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for candidate in reversed(candidates):
        if os.path.islink(candidate):
            raise _PrivateStatePermissionError(
                f"cron private state path contains a symlink: {path}"
            )
    return absolute


def _chmod_windows_execution_state_path(
    path: Path,
    mode: int,
    *,
    directory: bool,
    create_directories: bool = False,
    missing_ok: bool = False,
) -> None:
    """Preserve private-state initialization on Windows without dir_fd."""
    try:
        absolute = _reject_windows_execution_state_symlinks(path)
        if directory and create_directories:
            os.makedirs(absolute, mode=0o700, exist_ok=True)
        absolute = _reject_windows_execution_state_symlinks(path)
        if not os.path.lexists(absolute):
            if missing_ok:
                return
            raise FileNotFoundError(absolute)
        path_stat = os.stat(absolute)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(path_stat.st_mode):
            raise _PrivateStatePermissionError(
                f"cron private state path has unexpected type: {path}"
            )
        os.chmod(absolute, mode)
    except _PrivateStatePermissionError:
        raise
    except OSError as exc:
        raise _PrivateStatePermissionError(
            f"cannot secure cron private state path {path}: {exc}"
        ) from exc


def _open_execution_state_path(
    path: Path,
    *,
    directory: bool,
    create_directories: bool = False,
    missing_ok: bool = False,
) -> int | None:
    """Open *path* component-by-component without following symlinks."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not _secure_execution_traversal_supported():
        raise _PrivateStatePermissionError(
            "secure cron state traversal is unsupported on this platform"
        )

    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        raise _PrivateStatePermissionError(
            f"invalid cron private state path: {path}"
        )

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | directory_flag | nofollow | close_on_exec
    file_flags = os.O_RDONLY | nofollow | close_on_exec
    current_fd: int | None = None
    try:
        current_fd = os.open(absolute.anchor, directory_flags)
        components = parts[1:]
        for index, component in enumerate(components):
            is_target = index == len(components) - 1
            flags = directory_flags if not is_target or directory else file_flags
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if directory and create_directories:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        # A concurrent safe creator won the race. Reopen below
                        # with O_NOFOLLOW|O_DIRECTORY; files and symlinks fail.
                        pass
                    next_fd = os.open(component, flags, dir_fd=current_fd)
                elif is_target and missing_ok:
                    os.close(current_fd)
                    current_fd = None
                    return None
                else:
                    raise
            os.close(current_fd)
            current_fd = next_fd
        result_fd = current_fd
        current_fd = None
        return result_fd
    except OSError as exc:
        raise _PrivateStatePermissionError(
            f"cannot safely open cron private state path {path}: {exc}"
        ) from exc
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _chmod_execution_state_path(
    path: Path,
    mode: int,
    *,
    directory: bool,
    create_directories: bool = False,
    missing_ok: bool = False,
) -> None:
    if os.name == "nt":
        _chmod_windows_execution_state_path(
            path,
            mode,
            directory=directory,
            create_directories=create_directories,
            missing_ok=missing_ok,
        )
        return

    fd = _open_execution_state_path(
        path,
        directory=directory,
        create_directories=create_directories,
        missing_ok=missing_ok,
    )
    if fd is None:
        return
    try:
        path_stat = os.fstat(fd)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(path_stat.st_mode):
            raise _PrivateStatePermissionError(
                f"cron private state path has unexpected type: {path}"
            )
        os.fchmod(fd, mode)
        verified_mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if verified_mode != mode:
            raise _PrivateStatePermissionError(
                f"cron private state path {path} has mode "
                f"{verified_mode:#o}, expected {mode:#o}"
            )
    except _PrivateStatePermissionError:
        raise
    except OSError as exc:
        raise _PrivateStatePermissionError(
            f"cannot secure cron private state path {path}: {exc}"
        ) from exc
    finally:
        os.close(fd)


def _normalize_execution_store_permissions(
    executions_file: Path,
    *,
    require_database: bool = False,
) -> None:
    """Secure cron-owned state paths or raise without following symlinks."""
    absolute_db = Path(os.path.abspath(os.fspath(executions_file)))
    cron_dir = absolute_db.parent
    _chmod_execution_state_path(
        cron_dir,
        0o700,
        directory=True,
        create_directories=True,
    )
    _chmod_execution_state_path(
        absolute_db,
        0o600,
        directory=False,
        missing_ok=not require_database,
    )
    for name in (
        absolute_db.name + "-wal",
        absolute_db.name + "-shm",
        ".tick.lock",
        ".jobs.lock",
    ):
        _chmod_execution_state_path(
            cron_dir / name,
            0o600,
            directory=False,
            missing_ok=True,
        )


def _open_private_cron_lock(lock_path: Path) -> int:
    """Create or open a cron lock without following any path symlink."""
    absolute_lock = Path(os.path.abspath(os.fspath(lock_path)))
    if absolute_lock.name not in {".jobs.lock", ".tick.lock"}:
        raise _PrivateStatePermissionError(
            f"unexpected cron private lock path: {lock_path}"
        )

    # Keep the complete cron-store normalization contract in force before
    # creating a previously missing lock file.
    _normalize_execution_store_permissions(
        absolute_lock.with_name("executions.db")
    )
    if os.name == "nt":
        fd = None
        try:
            lock_name = _reject_windows_execution_state_symlinks(absolute_lock)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
            fd = os.open(lock_name, flags, 0o600)
            path_stat = os.lstat(lock_name)
            fd_stat = os.fstat(fd)
            if (
                stat.S_ISLNK(path_stat.st_mode)
                or not stat.S_ISREG(fd_stat.st_mode)
                or (path_stat.st_dev, path_stat.st_ino)
                != (fd_stat.st_dev, fd_stat.st_ino)
            ):
                raise _PrivateStatePermissionError(
                    f"cron private lock changed during open: {lock_path}"
                )
            os.chmod(lock_name, 0o600)
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    parent_fd = _open_execution_state_path(
        absolute_lock.parent,
        directory=True,
    )
    assert parent_fd is not None
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(absolute_lock.name, flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise _PrivateStatePermissionError(
            f"cannot safely open cron private lock {lock_path}: {exc}"
        ) from exc
    finally:
        os.close(parent_fd)

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _PrivateStatePermissionError(
                f"cron private lock has unexpected type: {lock_path}"
            )
        os.fchmod(fd, 0o600)
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
            raise _PrivateStatePermissionError(
                f"cron private lock {lock_path} is not mode 0o600"
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _current_executions_file() -> Path:
    """Resolve the ledger once from the active immutable profile context."""
    if EXECUTIONS_FILE is not None:
        return Path(os.path.abspath(os.fspath(EXECUTIONS_FILE.expanduser())))
    return Path(
        os.path.abspath(
            os.fspath(
                get_hermes_home().expanduser() / "cron" / "executions.db"
            )
        )
    )


def _lock_for(path: Path) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(path, threading.RLock())


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = _current_executions_file() if path is None else path
    _normalize_execution_store_permissions(path)
    return sqlite3.connect(path, timeout=5)


def _initialize_schema(
    conn: sqlite3.Connection,
    executions_file: Path,
) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             identity TEXT PRIMARY KEY,
             execution_id TEXT NOT NULL,
             job_id TEXT NOT NULL,
             platform TEXT NOT NULL,
             target_fingerprint TEXT NOT NULL,
             state TEXT NOT NULL CHECK(state IN
               ('pending','confirmed_sent','confirmed_absent','unknown')),
             fallback_used INTEGER NOT NULL DEFAULT 0 CHECK(fallback_used IN (0, 1)),
             created_at TEXT NOT NULL,
             updated_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_deliveries_execution "
        "ON deliveries(execution_id, created_at DESC)"
    )
    _normalize_execution_store_permissions(
        executions_file,
        require_database=True,
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, always close.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back
    the transaction; it does not close the connection. Relying on that alone
    leaks a connection (and its WAL/SHM file descriptors) on every call,
    since closing then depends on the garbage collector. Schema init runs
    inside the ``try`` too, so a PRAGMA/DDL failure after a successful
    ``connect()`` still closes the connection instead of leaking it.
    """
    executions_file = _current_executions_file()
    with _lock_for(executions_file):
        conn = _connect(executions_file)
        try:
            _initialize_schema(conn, executions_file)
            try:
                os.chmod(executions_file, 0o600)
            except OSError:
                pass
            with conn:
                yield conn
        finally:
            conn.close()


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )
    # Delivery identities are only useful while their owning execution remains
    # in the durable execution history. Keeping the same retention boundary
    # prevents an unbounded second ledger while preserving restart/retry safety
    # for every retained execution.
    conn.execute(
        "DELETE FROM deliveries WHERE execution_id NOT IN (SELECT id FROM executions)"
    )


_DELIVERY_TERMINAL_STATES = {
    "confirmed_sent",
    "confirmed_absent",
    "unknown",
}


def claim_delivery(
    identity: str,
    *,
    execution_id: str,
    job_id: str,
    platform: str,
    target_fingerprint: str,
    fallback: bool = False,
) -> Dict[str, Any]:
    """Atomically reserve one logical delivery or reject a replay.

    A reservation is written *before* the transport is touched.  A second
    process, a retry after restart, or a fallback can therefore only proceed
    when the preceding send was durably proven absent.  The one permitted
    fallback consumes ``fallback_used`` atomically, so an absent primary cannot
    fan out into multiple native retries.
    """
    identity = str(identity)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        if not fallback:
            cur = conn.execute(
                """INSERT OR IGNORE INTO deliveries
                   (identity, execution_id, job_id, platform, target_fingerprint,
                    state, fallback_used, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                (
                    identity,
                    str(execution_id),
                    str(job_id),
                    str(platform),
                    str(target_fingerprint),
                    now,
                    now,
                ),
            )
            if cur.rowcount == 1:
                return {"claimed": True, "state": "pending"}
        else:
            cur = conn.execute(
                """UPDATE deliveries
                   SET state='pending', fallback_used=1, updated_at=?
                   WHERE identity=? AND state='confirmed_absent' AND fallback_used=0""",
                (now, identity),
            )
            if cur.rowcount == 1:
                return {"claimed": True, "state": "pending"}

        row = conn.execute(
            "SELECT state FROM deliveries WHERE identity=?", (identity,)
        ).fetchone()
    # A missing row is only possible if a caller races a retention prune.  Do
    # not infer safety from that exceptional shape; fail closed as unknown.
    return {"claimed": False, "state": str(row["state"]) if row else "unknown"}


def resolve_delivery(identity: str, state: str) -> Optional[Dict[str, Any]]:
    """Commit one terminal outcome without rewriting an earlier conclusion."""
    if state not in _DELIVERY_TERMINAL_STATES:
        raise ValueError(f"unsupported delivery state: {state!r}")
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        conn.execute(
            """UPDATE deliveries SET state=?, updated_at=?
               WHERE identity=? AND state='pending'""",
            (state, now, str(identity)),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE identity=?", (str(identity),)
        ).fetchone()
    return _record(row)


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    record = _record(row)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET status='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')""",
            (status, now, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    return record


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: List[Dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            cur = conn.execute(
                """UPDATE executions SET status='unknown', finished_at=?, error=?
                   WHERE id=? AND status IN ('claimed','running')""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 row["id"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _record(conn.execute(
                    "SELECT * FROM executions WHERE id=?", (row["id"],)
                ).fetchone())
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
