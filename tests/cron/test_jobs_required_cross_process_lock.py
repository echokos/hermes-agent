"""Focused coverage for callers that require the jobs cross-process lock."""

import os

import pytest

from cron import jobs


pytestmark = pytest.mark.skipif(
    jobs.fcntl is None,
    reason="strict jobs-lock semantics require POSIX fcntl",
)


def _force_flock_timeout(monkeypatch):
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.0)

    def _always_blocked(_fd, _operation):
        raise BlockingIOError("held by another process")

    monkeypatch.setattr(jobs.fcntl, "flock", _always_blocked)


def test_default_timeout_preserves_degraded_scheduler_behavior(monkeypatch):
    jobs.ensure_dirs()
    _force_flock_timeout(monkeypatch)

    entered = False
    with jobs._jobs_lock():
        entered = True
        assert jobs._jobs_lock_state.cross_process_acquired is False

    assert entered is True


def test_required_lock_timeout_fails_before_critical_section(monkeypatch):
    jobs.ensure_dirs()
    _force_flock_timeout(monkeypatch)

    entered = False
    with pytest.raises(jobs.CrossProcessJobsLockUnavailable, match="timed out"):
        with jobs._jobs_lock(require_cross_process=True):
            entered = True

    assert entered is False
    assert jobs._jobs_lock_state.depth == 0
    assert jobs._jobs_lock_state.cross_process_acquired is False


def test_required_lock_open_failure_fails_before_critical_section(monkeypatch):
    import cron.executions as executions

    jobs.ensure_dirs()

    def _open_failure(_path):
        raise OSError("synthetic open failure")

    monkeypatch.setattr(executions, "_open_private_cron_lock", _open_failure)

    entered = False
    with pytest.raises(jobs.CrossProcessJobsLockUnavailable, match="unavailable"):
        with jobs._jobs_lock(require_cross_process=True):
            entered = True

    assert entered is False


def test_required_lock_without_backend_closes_opened_handle(monkeypatch):
    import cron.executions as executions

    jobs.ensure_dirs()
    real_open = executions._open_private_cron_lock
    opened_fds = []

    def _tracked_open(path):
        fd = real_open(path)
        opened_fds.append(fd)
        return fd

    monkeypatch.setattr(executions, "_open_private_cron_lock", _tracked_open)
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", None)

    with pytest.raises(jobs.CrossProcessJobsLockUnavailable, match="no supported"):
        with jobs._jobs_lock(require_cross_process=True):
            pytest.fail("strict section must not run without a lock backend")

    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])
    assert jobs._jobs_lock_state.depth == 0
    assert jobs._jobs_lock_state.cross_process_acquired is False


def test_required_nested_lock_rejects_degraded_outer_lock(monkeypatch):
    jobs.ensure_dirs()
    _force_flock_timeout(monkeypatch)

    with jobs._jobs_lock():
        assert jobs._jobs_lock_state.depth == 1
        assert jobs._jobs_lock_state.cross_process_acquired is False
        with pytest.raises(jobs.CrossProcessJobsLockUnavailable, match="outer"):
            with jobs._jobs_lock(require_cross_process=True):
                pytest.fail("strict nested section must not run")
        assert jobs._jobs_lock_state.depth == 1

    assert jobs._jobs_lock_state.depth == 0


def test_required_nested_lock_reuses_acquired_outer_lock(monkeypatch):
    jobs.ensure_dirs()
    real_flock = jobs.fcntl.flock
    operations = []

    def _tracked_flock(fd, operation):
        operations.append(operation)
        return real_flock(fd, operation)

    monkeypatch.setattr(jobs.fcntl, "flock", _tracked_flock)

    with jobs._jobs_lock():
        assert jobs._jobs_lock_state.cross_process_acquired is True
        with jobs._jobs_lock(require_cross_process=True):
            assert jobs._jobs_lock_state.depth == 2
            assert jobs._jobs_lock_state.cross_process_acquired is True

    acquire = jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB
    assert operations.count(acquire) == 1
    assert operations.count(jobs.fcntl.LOCK_UN) == 1


def test_required_lock_unwind_clears_thread_local_state():
    jobs.ensure_dirs()

    with pytest.raises(RuntimeError, match="synthetic body failure"):
        with jobs._jobs_lock(require_cross_process=True):
            assert jobs._jobs_lock_state.cross_process_acquired is True
            raise RuntimeError("synthetic body failure")

    assert jobs._jobs_lock_state.depth == 0
    assert jobs._jobs_lock_state.load_stamp is None
    assert jobs._jobs_lock_state.cross_process_acquired is False

    with jobs._jobs_lock(require_cross_process=True):
        assert jobs._jobs_lock_state.cross_process_acquired is True
