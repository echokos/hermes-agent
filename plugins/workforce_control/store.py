"""Additive outcome control over the existing Hermes Kanban kernel.

Kanban remains the only task system. These tables add classifications,
evidence, typed relations, draft plans, correction provenance, and observer
cursors without redefining Kanban statuses or duplicating task execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Iterable, Mapping

from hermes_cli import kanban_db
from hermes_cli.workforce_org import WorkforceOrganization, load_organization


write_txn = kanban_db.write_txn


SCHEMA_VERSION = 2
ITEM_KINDS = {"signal", "outcome", "execution", "verification"}
RELATION_KINDS = {"duplicate", "supersedes", "discovered_from", "verifies", "remediates"}
PLAN_STATES = {"draft", "materialized", "rejected", "deferred", "stopped"}
CORRECTION_CLASSES = {
    "one_off_fact", "tentative_not_execution", "current_state_verification",
    "delivery_route_exception", "authority_boundary", "quality_standard",
    "workflow_defect", "organization_rule",
}
OBSERVATION_CLASSES = {
    "already_complete", "duplicate", "superseded", "failed_verification",
    "external_blocker", "broken_record",
}
MAX_PLAN_NODES = 8


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wc_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_runtime (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL DEFAULT 'paused' CHECK (mode IN ('paused','shadow','apply')),
    kill_switch INTEGER NOT NULL DEFAULT 1,
    pause_reason TEXT,
    max_actions_per_tick INTEGER NOT NULL DEFAULT 25,
    max_materialized_nodes INTEGER NOT NULL DEFAULT 8,
    daily_model_cost_ceiling_usd REAL NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_items (
    task_id TEXT PRIMARY KEY,
    item_kind TEXT NOT NULL CHECK (item_kind IN ('signal','outcome','execution','verification')),
    stable_key TEXT NOT NULL UNIQUE,
    goal_ref TEXT NOT NULL DEFAULT 'unknown',
    goal_evidence_at INTEGER,
    desired_outcome TEXT NOT NULL,
    acceptance_test TEXT,
    action_class TEXT,
    target_ref TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL DEFAULT '[]',
    verification_state TEXT NOT NULL DEFAULT 'pending' CHECK (verification_state IN ('pending','passed','failed','not_required')),
    current_state TEXT NOT NULL DEFAULT 'open',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_relations (
    source_task_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('duplicate','supersedes','discovered_from','verifies','remediates')),
    target_task_id TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    confidence TEXT NOT NULL DEFAULT 'high' CHECK (confidence IN ('low','medium','high')),
    created_by TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (source_task_id, relation, target_task_id)
);
CREATE TABLE IF NOT EXISTS wc_plans (
    plan_id TEXT PRIMARY KEY,
    stable_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    goal_ref TEXT NOT NULL,
    goal_evidence_at INTEGER,
    desired_outcome TEXT NOT NULL,
    acceptance_test TEXT NOT NULL,
    priority_rationale TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    capacity_assessment TEXT NOT NULL,
    deadline_dependencies TEXT NOT NULL,
    displaced_work TEXT NOT NULL,
    unresolved_decisions_json TEXT NOT NULL,
    defer_or_stop TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('draft','materialized','rejected','deferred','stopped')),
    materialized_root_task_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_corrections (
    correction_id TEXT PRIMARY KEY,
    classification TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('one_off','profile','department','workforce','system')),
    description TEXT NOT NULL,
    rule_target TEXT,
    provenance_ref TEXT NOT NULL,
    privacy_class TEXT NOT NULL CHECK (privacy_class IN ('organizational','personal_private','relationship_private')),
    status TEXT NOT NULL CHECK (status IN ('pending','implemented','superseded')),
    regression_ref TEXT,
    supersedes_id TEXT,
    recorded_by TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_reconcile_actions (
    action_id TEXT PRIMARY KEY,
    stable_key TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    classification TEXT NOT NULL,
    target_task_id TEXT,
    proposed_action TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    evidence_at INTEGER NOT NULL,
    confidence TEXT NOT NULL CHECK (confidence IN ('low','medium','high')),
    proposed_by TEXT NOT NULL,
    required_authority TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('shadow','proposed','applied','quarantined','dismissed')),
    created_at INTEGER NOT NULL,
    applied_at INTEGER
);
CREATE TABLE IF NOT EXISTS wc_cursors (
    observer TEXT PRIMARY KEY,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_goal_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    source_guid TEXT NOT NULL,
    source_title TEXT NOT NULL,
    source_updated_at INTEGER NOT NULL,
    captured_at INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    goals_json TEXT NOT NULL,
    recorded_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS wc_vision_reviews (
    review_id TEXT PRIMARY KEY,
    stable_key TEXT NOT NULL UNIQUE,
    source_ref TEXT NOT NULL,
    goal_ref TEXT NOT NULL,
    brief TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    requested_by TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','completed','cancelled')),
    response_json TEXT,
    responded_by TEXT,
    requested_at INTEGER NOT NULL,
    responded_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_wc_items_kind_state ON wc_items(item_kind, current_state);
CREATE INDEX IF NOT EXISTS idx_wc_actions_state ON wc_reconcile_actions(state, created_at);
CREATE INDEX IF NOT EXISTS idx_wc_corrections_status ON wc_corrections(status, scope);
CREATE INDEX IF NOT EXISTS idx_wc_goals_captured ON wc_goal_snapshots(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_wc_vision_status ON wc_vision_reviews(status, requested_at);
"""


def _now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    try:
        parsed = json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return default
    return parsed


def _id(prefix: str, stable_key: str) -> str:
    return f"{prefix}_{hashlib.sha256(stable_key.encode()).hexdigest()[:16]}"


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def stable_identity(
    *, item_kind: str, desired_outcome: str, action_class: str = "", target_ref: str = ""
) -> str:
    if item_kind not in ITEM_KINDS:
        raise ValueError(f"invalid item_kind: {item_kind}")
    parts = [item_kind, _normalized(desired_outcome), _normalized(action_class), _normalized(target_ref)]
    if not parts[1]:
        raise ValueError("desired_outcome is required")
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Apply only plugin-owned, forward-compatible tables."""
    conn.executescript(SCHEMA_SQL)
    with write_txn(conn):
        row = conn.execute("SELECT version FROM wc_schema WHERE singleton = 1").fetchone()
        if row and int(row["version"]) > SCHEMA_VERSION:
            raise RuntimeError("workforce-control database schema is newer than this plugin")
        conn.execute(
            "INSERT INTO wc_schema(singleton,version,updated_at) VALUES(1,?,?) "
            "ON CONFLICT(singleton) DO UPDATE SET version=excluded.version,updated_at=excluded.updated_at",
            (SCHEMA_VERSION, _now()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO wc_runtime(singleton,mode,kill_switch,pause_reason,updated_at) "
            "VALUES(1,'paused',1,'requires explicit Elliott cutover authorization',?)",
            (_now(),),
        )
        # Schema v1 signals were created blocked, but their typed blocker
        # fields were left empty. Repair only plugin-owned signal rows so the
        # ordinary Kanban blocker view is accurate without touching unrelated
        # historical tasks.
        conn.execute(
            "UPDATE tasks SET block_kind='needs_input',block_recurrences="
            "CASE WHEN block_recurrences < 1 THEN 1 ELSE block_recurrences END "
            "WHERE status='blocked' AND (block_kind IS NULL OR block_kind='') "
            "AND id IN (SELECT task_id FROM wc_items WHERE item_kind='signal')"
        )


def schema_present(conn: sqlite3.Connection) -> bool:
    required = {"wc_schema", "wc_runtime", "wc_items", "wc_reconcile_actions", "wc_cursors"}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'wc_%'"
    ).fetchall()
    return required <= {str(row["name"]) for row in rows}


def runtime_state(conn: sqlite3.Connection) -> dict[str, Any]:
    ensure_schema(conn)
    return dict(conn.execute("SELECT * FROM wc_runtime WHERE singleton = 1").fetchone())


def set_runtime_mode(
    conn: sqlite3.Connection, *, mode: str, kill_switch: bool, reason: str
) -> dict[str, Any]:
    """Administrative primitive; intentionally not exposed as an agent tool."""
    if mode not in {"paused", "shadow", "apply"}:
        raise ValueError("mode must be paused, shadow, or apply")
    ensure_schema(conn)
    with write_txn(conn):
        conn.execute(
            "UPDATE wc_runtime SET mode=?,kill_switch=?,pause_reason=?,updated_at=? WHERE singleton=1",
            (mode, 1 if kill_switch else 0, reason.strip() or None, _now()),
        )
    return runtime_state(conn)


def _merge_list(existing: str | None, incoming: Iterable[Any]) -> str:
    merged = list(_loads(existing, []))
    for value in incoming:
        if value not in merged:
            merged.append(value)
    return _json(merged)


def record_signal(
    conn: sqlite3.Connection,
    *, source_agent: str,
    expected_outcome: str,
    goal_ref: str,
    observation: str,
    evidence_references: list[str],
    action_class: str = "opportunity",
    target_ref: str = "",
    packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_schema(conn)
    stable_key = stable_identity(
        item_kind="signal", desired_outcome=expected_outcome,
        action_class=action_class, target_ref=target_ref,
    )
    existing = conn.execute(
        "SELECT i.*,t.status,t.assignee FROM wc_items i JOIN tasks t ON t.id=i.task_id WHERE i.stable_key=?",
        (stable_key,),
    ).fetchone()
    provenance = {"source_agent": source_agent, "observation": observation, "recorded_at": _now()}
    if existing:
        with write_txn(conn):
            conn.execute(
                "UPDATE wc_items SET evidence_json=?,provenance_json=?,updated_at=? WHERE task_id=?",
                (
                    _merge_list(existing["evidence_json"], evidence_references),
                    _merge_list(existing["provenance_json"], [provenance]),
                    _now(), existing["task_id"],
                ),
            )
            kanban_db._append_event(
                conn, existing["task_id"], "workforce_signal_reobserved",
                {"source_agent": source_agent, "stable_key": stable_key},
            )
        return {
            "task_id": existing["task_id"], "status": existing["status"],
            "assignee": existing["assignee"], "stable_key": stable_key,
            "created": False,
        }

    body = dict(packet or {})
    body.update({
        "kind": "workforce_signal", "decision_owner": "aurora",
        "launch_authorized": False, "source_agent": source_agent,
        "expected_outcome": expected_outcome, "approved_goal": goal_ref,
        "observation": observation, "evidence_references": evidence_references,
        "stable_key": stable_key, "action_class": action_class,
        "target_ref": target_ref or None,
    })
    now = _now()
    with write_txn(conn):
        task_id = kanban_db.create_task(
            conn, title=f"Signal: {expected_outcome[:120]}", body=json.dumps(body, indent=2, sort_keys=True),
            assignee="aurora", created_by=source_agent, workspace_kind="scratch",
            triage=False, initial_status="blocked",
            idempotency_key=f"workforce-signal:{stable_key}",
        )
        kanban_db._append_event(
            conn,
            task_id,
            "blocked",
            {
                "reason": "non-executing workforce signal awaiting Aurora decision",
                "kind": "needs_input",
            },
        )
        conn.execute(
            "UPDATE tasks SET block_kind='needs_input',block_recurrences=1 WHERE id=?",
            (task_id,),
        )
        conn.execute(
            "INSERT INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,action_class,target_ref,evidence_json,provenance_json,verification_state,current_state,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,'not_required','open',?,?)",
            (task_id, "signal", stable_key, goal_ref or "unknown", expected_outcome,
             action_class, target_ref or None, _json(evidence_references), _json([provenance]), now, now),
        )
    return {"task_id": task_id, "status": "blocked", "assignee": "aurora", "stable_key": stable_key, "created": True}


def publish_goal_snapshot(
    conn: sqlite3.Connection,
    *,
    actor: str,
    source_guid: str,
    source_title: str,
    source_updated_at: str | int,
    goals: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish a workforce-safe projection of Aurora's verified goals note.

    The Evernote note remains canonical.  This projection deliberately carries
    only the bounded fields operational agents need for alignment; private note
    text is never copied into the Kanban database.
    """
    if actor != "aurora":
        raise PermissionError("only Aurora may publish the workforce goals projection")
    guid = source_guid.strip()
    title = source_title.strip()
    if not guid or not title:
        raise ValueError("source_guid and source_title are required")
    source_ts = _parse_time(source_updated_at)
    if source_ts is None:
        raise ValueError("source_updated_at is required")
    if not goals or len(goals) > 24:
        raise ValueError("goals must contain between 1 and 24 entries")

    allowed = {"goal_id", "title", "desired_outcome", "priority", "status", "departments"}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(goals):
        if not isinstance(raw, Mapping):
            raise ValueError(f"goal {index} must be an object")
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"goal {index} contains unsupported fields: {sorted(extra)}")
        goal_id = str(raw.get("goal_id") or "").strip()
        goal_title = str(raw.get("title") or "").strip()
        desired = str(raw.get("desired_outcome") or "").strip()
        if not goal_id or not goal_title or not desired:
            raise ValueError(f"goal {index} requires goal_id, title, and desired_outcome")
        if goal_id in seen:
            raise ValueError(f"duplicate goal_id: {goal_id}")
        seen.add(goal_id)
        departments = sorted({str(v).strip() for v in raw.get("departments") or [] if str(v).strip()})
        normalized.append({
            "goal_id": goal_id,
            "title": goal_title,
            "desired_outcome": desired,
            "priority": str(raw.get("priority") or "").strip() or "unspecified",
            "status": str(raw.get("status") or "active").strip() or "active",
            "departments": departments,
        })

    payload = _json(normalized)
    content_hash = hashlib.sha256(payload.encode()).hexdigest()
    snapshot_id = _id("goals", f"{guid}:{source_ts}:{content_hash}")
    captured_at = _now()
    ensure_schema(conn)
    with write_txn(conn):
        latest = conn.execute(
            "SELECT source_updated_at,content_hash FROM wc_goal_snapshots "
            "ORDER BY source_updated_at DESC,captured_at DESC LIMIT 1"
        ).fetchone()
        if latest and source_ts < int(latest["source_updated_at"]):
            raise ValueError("goal source is older than the current verified snapshot")
        if (
            latest
            and source_ts == int(latest["source_updated_at"])
            and content_hash != str(latest["content_hash"])
        ):
            raise ValueError(
                "goal source timestamp matches the current snapshot but content differs"
            )
        conn.execute(
            "INSERT INTO wc_goal_snapshots(snapshot_id,source_guid,source_title,source_updated_at,captured_at,content_hash,goals_json,recorded_by) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(snapshot_id) DO UPDATE SET "
            "captured_at=excluded.captured_at,recorded_by=excluded.recorded_by",
            (snapshot_id, guid, title, source_ts, captured_at, content_hash, payload, actor),
        )
    return {
        "snapshot_id": snapshot_id,
        "source_guid": guid,
        "source_updated_at": source_ts,
        "captured_at": captured_at,
        "content_hash": content_hash,
        "goal_count": len(normalized),
    }


def current_goal_snapshot(conn: sqlite3.Connection, *, max_age_hours: int = 36) -> dict[str, Any] | None:
    ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM wc_goal_snapshots "
        "ORDER BY source_updated_at DESC,captured_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["goals"] = _loads(result.pop("goals_json"), [])
    result["age_seconds"] = max(0, _now() - int(result["captured_at"]))
    result["stale"] = result["age_seconds"] > max(1, int(max_age_hours)) * 3600
    return result


def request_vision_review(
    conn: sqlite3.Connection,
    *,
    actor: str,
    source_ref: str,
    goal_ref: str,
    brief: str,
    evidence_references: list[str],
) -> dict[str, Any]:
    if actor != "aurora":
        raise PermissionError("only Aurora may request a Vision review")
    source = source_ref.strip()
    goal = goal_ref.strip()
    request_brief = brief.strip()
    if not source or not goal or not request_brief:
        raise ValueError("source_ref, goal_ref, and brief are required")
    stable_key = hashlib.sha256(f"{source}\0{goal}\0{_normalized(request_brief)}".encode()).hexdigest()
    ensure_schema(conn)
    existing = conn.execute(
        "SELECT review_id,status FROM wc_vision_reviews WHERE stable_key=?", (stable_key,)
    ).fetchone()
    if existing:
        return {"review_id": existing["review_id"], "status": existing["status"], "created": False}
    review_id = _id("vision", stable_key)
    with write_txn(conn):
        conn.execute(
            "INSERT INTO wc_vision_reviews(review_id,stable_key,source_ref,goal_ref,brief,evidence_json,requested_by,status,requested_at) "
            "VALUES(?,?,?,?,?,?,?,'pending',?)",
            (review_id, stable_key, source, goal, request_brief, _json(evidence_references), actor, _now()),
        )
    return {"review_id": review_id, "status": "pending", "created": True}


def list_vision_reviews(conn: sqlite3.Connection, *, status: str = "pending", limit: int = 10) -> list[dict[str, Any]]:
    ensure_schema(conn)
    if status not in {"pending", "completed", "cancelled"}:
        raise ValueError("invalid Vision review status")
    rows = conn.execute(
        "SELECT * FROM wc_vision_reviews WHERE status=? ORDER BY requested_at LIMIT ?",
        (status, max(1, min(int(limit), 20))),
    ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["evidence_references"] = _loads(item.pop("evidence_json"), [])
        item["response"] = _loads(item.pop("response_json"), None)
        results.append(item)
    return results


def complete_vision_review(
    conn: sqlite3.Connection,
    *,
    actor: str,
    review_id: str,
    response: Mapping[str, Any],
) -> dict[str, Any]:
    if actor != "mel":
        raise PermissionError("only Mel may complete the formal Vision review")
    required = {"reframe", "ten_x_option", "assumptions", "value_case", "risks", "smallest_test"}
    if set(response) != required:
        raise ValueError(f"Vision response must contain exactly {sorted(required)}")
    for key in required:
        value = response[key]
        if isinstance(value, list):
            if not value or not all(str(v).strip() for v in value):
                raise ValueError(f"{key} must not be empty")
        elif not str(value or "").strip():
            raise ValueError(f"{key} must not be empty")
    ensure_schema(conn)
    now = _now()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE wc_vision_reviews SET status='completed',response_json=?,responded_by=?,responded_at=? "
            "WHERE review_id=? AND status='pending'",
            (_json(dict(response)), actor, now, review_id.strip()),
        )
        if cur.rowcount != 1:
            raise ValueError("Vision review is missing or no longer pending")
    return {"review_id": review_id.strip(), "status": "completed", "responded_at": now}


def _parse_time(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return int(parsed.timestamp())


def _validate_graph(nodes: list[Mapping[str, Any]], org: WorkforceOrganization) -> None:
    if not nodes:
        raise ValueError("plan must contain at least one execution node")
    if len(nodes) > MAX_PLAN_NODES:
        raise ValueError(f"plan exceeds the {MAX_PLAN_NODES}-node safety limit")
    keys = [str(node.get("key") or "").strip() for node in nodes]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("every plan node needs a unique key")
    known = set(keys)
    graph: dict[str, list[str]] = {}
    for node in nodes:
        key = str(node["key"])
        assignee = str(node.get("assignee") or "")
        org.validate_execution_profile(assignee)
        responsibility = str(node.get("responsibility") or "").strip()
        if responsibility:
            owner = org.technical_ownership.get(responsibility)
            if owner is None:
                raise ValueError(f"node {key}: unknown responsibility {responsibility!r}")
            resolved = org.resolve_profile(assignee)
            if owner == "department_director":
                if resolved.function != "Director":
                    raise ValueError(f"node {key}: {responsibility} must be assigned to a department director")
            elif resolved.agent != owner:
                raise ValueError(f"node {key}: {responsibility} is owned by {owner}")
        authority = str(node.get("authority_class") or "routine")
        if authority not in {"routine", "reserved"}:
            raise ValueError(f"node {key}: authority_class must be routine or reserved")
        parents = [str(value) for value in node.get("parents") or []]
        if any(parent not in known for parent in parents):
            raise ValueError(f"node {key}: unknown parent")
        graph[key] = parents
        for required in ("title", "acceptance_test", "action_class"):
            if not str(node.get(required) or "").strip():
                raise ValueError(f"node {key}: {required} is required")
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("plan graph contains a cycle")
        if key in visited:
            return
        visiting.add(key)
        for parent in graph[key]:
            visit(parent)
        visiting.remove(key)
        visited.add(key)
    for key in keys:
        visit(key)


def record_plan(conn: sqlite3.Connection, *, actor: str, payload: Mapping[str, Any], organization: WorkforceOrganization | None = None) -> dict[str, Any]:
    org = organization or load_organization()
    if org.resolve_profile(actor).agent != "aurora":
        raise PermissionError("only Aurora may draft a workforce execution plan")
    ensure_schema(conn)
    required = (
        "title", "goal_ref", "desired_outcome", "acceptance_test", "priority_rationale",
        "checkpoint", "capacity_assessment", "deadline_dependencies", "displaced_work", "defer_or_stop",
    )
    values = {name: str(payload.get(name) or "").strip() for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("missing plan fields: " + ", ".join(missing))
    nodes = list(payload.get("nodes") or [])
    if not all(isinstance(node, Mapping) for node in nodes):
        raise ValueError("nodes must be objects")
    _validate_graph(nodes, org)
    decisions = list(payload.get("unresolved_decisions") or [])
    evidence = list(payload.get("evidence_references") or [])
    stable_key = stable_identity(
        item_kind="outcome", desired_outcome=values["desired_outcome"],
        action_class="plan", target_ref=values["goal_ref"],
    )
    plan_id = _id("plan", stable_key)
    now = _now()
    with write_txn(conn):
        existing = conn.execute("SELECT state FROM wc_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if existing and existing["state"] == "materialized":
            raise ValueError("materialized plans are immutable; reconcile the existing outcome")
        conn.execute(
            "INSERT INTO wc_plans(plan_id,stable_key,title,goal_ref,goal_evidence_at,desired_outcome,acceptance_test,priority_rationale,checkpoint,capacity_assessment,deadline_dependencies,displaced_work,unresolved_decisions_json,defer_or_stop,graph_json,evidence_json,requested_by,state,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'draft',?,?) "
            "ON CONFLICT(plan_id) DO UPDATE SET title=excluded.title,goal_ref=excluded.goal_ref,goal_evidence_at=excluded.goal_evidence_at,desired_outcome=excluded.desired_outcome,acceptance_test=excluded.acceptance_test,priority_rationale=excluded.priority_rationale,checkpoint=excluded.checkpoint,capacity_assessment=excluded.capacity_assessment,deadline_dependencies=excluded.deadline_dependencies,displaced_work=excluded.displaced_work,unresolved_decisions_json=excluded.unresolved_decisions_json,defer_or_stop=excluded.defer_or_stop,graph_json=excluded.graph_json,evidence_json=excluded.evidence_json,updated_at=excluded.updated_at",
            (plan_id, stable_key, values["title"], values["goal_ref"], _parse_time(payload.get("goal_evidence_at")),
             values["desired_outcome"], values["acceptance_test"], values["priority_rationale"], values["checkpoint"],
             values["capacity_assessment"], values["deadline_dependencies"], values["displaced_work"], _json(decisions),
             values["defer_or_stop"], _json(nodes), _json(evidence), actor, now, now),
        )
    return {"plan_id": plan_id, "state": "draft", "execution_cards_created": 0, "unresolved_decisions": decisions}


def _require_fresh_evidence(evidence: list[str], evidence_at: str | int | None) -> int:
    if not evidence:
        raise ValueError("current_state_evidence is required")
    timestamp = _parse_time(evidence_at)
    if timestamp is None:
        raise ValueError("current_state_evidence_at is required")
    now = _now()
    if timestamp > now + 300 or now - timestamp > 86400:
        raise ValueError("current-state evidence must be no more than 24 hours old")
    return timestamp


def _validated_materialization_coordination(
    conn: sqlite3.Connection,
    *,
    actor: str,
    context: tuple[str, str, str] | None,
    organization: WorkforceOrganization,
) -> dict[str, Any] | None:
    if context is None:
        return None
    try:
        request_root_id, source_task_id, purpose = context
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid current coordination execution context") from exc
    request_root_id = str(request_root_id or "").strip()
    source_task_id = str(source_task_id or "").strip()
    if not request_root_id or not source_task_id or purpose != "work":
        raise ValueError("materialization requires a current coordination work context")
    request = kanban_db.get_coordination_request(conn, request_root_id)
    source = kanban_db.get_task(conn, source_task_id)
    actor_id = organization.resolve_profile(actor).agent
    if request is None or request.status != "active":
        raise ValueError("materialization requires an active accepted coordination request")
    if request.responsible_agent != actor_id:
        raise PermissionError("current coordination request is owned by another manager")
    if source is None or source.request_root_id != request.id:
        raise ValueError("current coordination task is not bound to the accepted request")
    return {
        "request_root_id": request.id,
        "source_task_id": source.id,
        "origin_session_id": request.origin_session_id,
        "origin_message_id": request.origin_message_id,
    }


def _resolved_materialization_context(
    conn: sqlite3.Connection,
    *,
    context: tuple[str, str, str] | None,
    origin: tuple[str, str] | None,
) -> tuple[tuple[str, str, str] | None, tuple[str, str] | None]:
    normalized_origin = None
    if origin is not None:
        try:
            origin_session_id, origin_message_id = origin
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid current coordination origin") from exc
        origin_session_id = str(origin_session_id or "").strip()
        origin_message_id = str(origin_message_id or "").strip()
        if bool(origin_session_id) != bool(origin_message_id):
            raise ValueError("current coordination origin is incomplete")
        if origin_session_id:
            normalized_origin = (origin_session_id, origin_message_id)
    if context is None and normalized_origin is not None:
        request = kanban_db.get_coordination_request(
            conn,
            kanban_db.coordination_request_id(*normalized_origin),
        )
        if request is not None:
            context = (request.id, request.root_task_id, "work")
    return context, normalized_origin


def _materialized_plan_result(
    conn: sqlite3.Connection,
    *,
    actor: str,
    plan: sqlite3.Row,
    coordination_context: tuple[str, str, str] | None,
    coordination_origin: tuple[str, str] | None,
    organization: WorkforceOrganization,
) -> dict[str, Any]:
    coordination_context, coordination_origin = _resolved_materialization_context(
        conn,
        context=coordination_context,
        origin=coordination_origin,
    )
    coordination = _validated_materialization_coordination(
        conn,
        actor=actor,
        context=coordination_context,
        organization=organization,
    )
    root_task_id = str(plan["materialized_root_task_id"] or "")
    if coordination is not None:
        root = kanban_db.get_task(conn, root_task_id)
        if root is None or root.request_root_id != coordination["request_root_id"]:
            raise ValueError(
                "materialized plan is not bound to the current coordination request"
            )
    elif coordination_origin is not None:
        root = kanban_db.get_task(conn, root_task_id)
        created = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='created' "
            "ORDER BY id DESC LIMIT 1",
            (root_task_id,),
        ).fetchone()
        created_payload = _loads(created["payload"] if created else None, {})
        if (
            root is None
            or root.request_root_id is not None
            or root.session_id != coordination_origin[0]
            or not isinstance(created_payload, dict)
            or created_payload.get("coordination_origin_message_id")
            != coordination_origin[1]
        ):
            raise ValueError(
                "materialized plan is not pending adoption by the current coordination origin"
            )
    result = {
        "plan_id": str(plan["plan_id"]),
        "root_task_id": root_task_id,
        "created": False,
    }
    if coordination is not None:
        result["request_root_id"] = coordination["request_root_id"]
    return result


def materialize_plan(
    conn: sqlite3.Connection, *, actor: str, plan_id: str,
    current_state_evidence: list[str], current_state_evidence_at: str | int,
    confirmed_execution_ready: bool, organization: WorkforceOrganization | None = None,
    coordination_context: tuple[str, str, str] | None = None,
    coordination_origin: tuple[str, str] | None = None,
    coordination_acceptance_pending: bool = False,
) -> dict[str, Any]:
    org = organization or load_organization()
    if org.resolve_profile(actor).agent != "aurora":
        raise PermissionError("only Aurora may materialize workforce execution")
    state = runtime_state(conn)
    if state["kill_switch"] or state["mode"] != "apply":
        raise RuntimeError("workforce materialization is paused; explicit cutover activation is required")
    evidence_at = _require_fresh_evidence(current_state_evidence, current_state_evidence_at)
    if not confirmed_execution_ready:
        raise ValueError("tentative or exploratory intake cannot be materialized")
    plan = conn.execute("SELECT * FROM wc_plans WHERE plan_id=?", (plan_id,)).fetchone()
    if plan is None:
        raise ValueError("unknown workforce plan")
    if plan["state"] == "materialized":
        return _materialized_plan_result(
            conn,
            actor=actor,
            plan=plan,
            coordination_context=coordination_context,
            coordination_origin=coordination_origin,
            organization=org,
        )
    if plan["state"] != "draft":
        raise ValueError(f"plan cannot be materialized from {plan['state']}")

    now = _now()
    with write_txn(conn):
        # Another connection may have materialized this draft while this call
        # waited for the write lock. Re-read before any idempotent task create
        # can return rows owned by that other request.
        plan = conn.execute(
            "SELECT * FROM wc_plans WHERE plan_id=?", (plan_id,)
        ).fetchone()
        if plan is None:
            raise ValueError("unknown workforce plan")
        if plan["state"] == "materialized":
            return _materialized_plan_result(
                conn,
                actor=actor,
                plan=plan,
                coordination_context=coordination_context,
                coordination_origin=coordination_origin,
                organization=org,
            )
        if plan["state"] != "draft":
            raise ValueError(f"plan cannot be materialized from {plan['state']}")
        if plan["goal_ref"].strip().casefold() == "unknown":
            raise ValueError("plans with an unknown goal remain in discovery")
        unresolved = list(_loads(plan["unresolved_decisions_json"], []))
        if unresolved:
            raise ValueError("unresolved decisions must be resolved before materialization")
        nodes = list(_loads(plan["graph_json"], []))
        _validate_graph(nodes, org)
        if any(
            str(node.get("authority_class") or "routine") == "reserved"
            for node in nodes
        ):
            raise PermissionError(
                "reserved-authority nodes require Elliott and cannot be materialized by Aurora"
            )
        max_nodes = min(MAX_PLAN_NODES, int(state["max_materialized_nodes"]))
        if len(nodes) > max_nodes:
            raise ValueError("plan exceeds the configured materialization limit")
        coordination_context, coordination_origin = _resolved_materialization_context(
            conn,
            context=coordination_context,
            origin=coordination_origin,
        )
        coordination = _validated_materialization_coordination(
            conn, actor=actor, context=coordination_context, organization=org,
        )
        acceptance_pending = coordination is None and coordination_acceptance_pending
        if acceptance_pending and coordination_origin is None:
            raise ValueError("pending coordination acceptance requires a current origin")
        origin_session_id, origin_message_id = coordination_origin or ("", "")
        if coordination is not None:
            if (origin_session_id, origin_message_id) not in {
                ("", ""),
                (coordination["origin_session_id"], coordination["origin_message_id"]),
            }:
                raise ValueError("current coordination origin does not match the accepted request")
            origin_session_id = coordination["origin_session_id"]
            origin_message_id = coordination["origin_message_id"]
        task_context = {
            "session_id": origin_session_id or None,
            "coordination_source_task_id": coordination["source_task_id"] if coordination else None,
            "coordination_origin_message_id": origin_message_id or None,
        }
        by_key: dict[str, str] = {}
        remaining = {str(node["key"]): node for node in nodes}
        while remaining:
            progressed = False
            for key, node in list(remaining.items()):
                parents = [str(value) for value in node.get("parents") or []]
                if any(parent not in by_key for parent in parents):
                    continue
                task_id = kanban_db.create_task(
                    conn, title=str(node["title"]),
                    body=json.dumps({
                        "kind": "workforce_execution", "plan_id": plan_id,
                        "desired_outcome": plan["desired_outcome"],
                        "acceptance_test": node["acceptance_test"],
                        "current_state_evidence": current_state_evidence,
                        "current_state_evidence_at": evidence_at,
                        **(
                            {"coordination_acceptance_pending": True}
                            if acceptance_pending else {}
                        ),
                    }, indent=2, sort_keys=True),
                    assignee=str(node["assignee"]), created_by="aurora",
                    tenant=str(node.get("tenant") or "company"),
                    project_id=node.get("project_id"), workspace_kind=str(node.get("workspace_kind") or "scratch"),
                    parents=[by_key[parent] for parent in parents],
                    idempotency_key=f"workforce-plan:{plan_id}:{key}", goal_mode=bool(node.get("goal_mode", True)),
                    **task_context,
                )
                stable_key = stable_identity(
                    item_kind="execution", desired_outcome=str(node["title"]),
                    action_class=str(node["action_class"]),
                    target_ref=str(node.get("target_ref") or f"{plan_id}:{key}"),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO wc_items(task_id,item_kind,stable_key,goal_ref,goal_evidence_at,desired_outcome,acceptance_test,action_class,target_ref,evidence_json,provenance_json,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,'pending','open',?,?)",
                    (task_id, "execution", stable_key, plan["goal_ref"], plan["goal_evidence_at"], str(node["title"]),
                     str(node["acceptance_test"]), str(node["action_class"]), str(node.get("target_ref") or "") or None,
                     _json(current_state_evidence), _json([{"plan_id": plan_id, "node": key}]), now, now),
                )
                by_key[key] = task_id
                del remaining[key]
                progressed = True
            if not progressed:
                raise RuntimeError("plan graph could not be topologically materialized")
        root_id = kanban_db.create_task(
            conn, title=f"Outcome: {plan['title']}",
            body=json.dumps({
                "kind": "workforce_outcome", "plan_id": plan_id,
                "desired_outcome": plan["desired_outcome"],
                "acceptance_test": plan["acceptance_test"],
                **(
                    {"coordination_acceptance_pending": True}
                    if acceptance_pending else {}
                ),
            }, indent=2, sort_keys=True),
            assignee="aurora", created_by="aurora", tenant="company", parents=list(by_key.values()),
            idempotency_key=f"workforce-outcome:{plan['stable_key']}", workspace_kind="scratch", goal_mode=False,
            **task_context,
        )
        outcome_key = stable_identity(item_kind="outcome", desired_outcome=plan["desired_outcome"], action_class="outcome", target_ref=plan["goal_ref"])
        conn.execute(
            "INSERT OR IGNORE INTO wc_items(task_id,item_kind,stable_key,goal_ref,goal_evidence_at,desired_outcome,acceptance_test,evidence_json,provenance_json,verification_state,current_state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,'pending','open',?,?)",
            (root_id, "outcome", outcome_key, plan["goal_ref"], plan["goal_evidence_at"], plan["desired_outcome"], plan["acceptance_test"],
             _json(current_state_evidence), _json([{"plan_id": plan_id}]), now, now),
        )
        for child_id in by_key.values():
            conn.execute(
                "INSERT OR IGNORE INTO wc_relations(source_task_id,relation,target_task_id,evidence_json,confidence,created_by,created_at) VALUES(?,'discovered_from',?,?,'high','aurora',?)",
                (child_id, root_id, _json(current_state_evidence), now),
            )
        conn.execute("UPDATE wc_plans SET state='materialized',materialized_root_task_id=?,updated_at=? WHERE plan_id=?", (root_id, now, plan_id))
    result = {
        "plan_id": plan_id,
        "root_task_id": root_id,
        "execution_tasks": by_key,
        "created": True,
    }
    if coordination is not None:
        result["request_root_id"] = coordination["request_root_id"]
    return result


def propose_reconciliation(
    conn: sqlite3.Connection, *, actor: str, observations: list[Mapping[str, Any]],
    mode: str = "shadow", organization: WorkforceOrganization | None = None,
) -> list[dict[str, Any]]:
    org = organization or load_organization()
    actor_id = org.validate_execution_profile(actor).agent
    if mode not in {"shadow", "proposed"}:
        raise ValueError("mode must be shadow or proposed")
    ensure_schema(conn)
    results: list[dict[str, Any]] = []
    for observation in observations:
        classification = str(observation.get("classification") or "")
        if classification not in OBSERVATION_CLASSES:
            raise ValueError(f"invalid reconciliation classification: {classification}")
        task_id = str(observation.get("task_id") or "")
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise ValueError(f"unknown task {task_id}")
        evidence = list(observation.get("evidence_references") or [])
        evidence_at = _require_fresh_evidence(evidence, observation.get("evidence_at"))
        confidence = str(observation.get("confidence") or "low")
        if confidence not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")
        target_id = str(observation.get("target_task_id") or "") or None
        action_map = {
            "already_complete": "close_after_acceptance_check",
            "duplicate": "link_duplicate_and_archive",
            "superseded": "link_supersedes_and_archive",
            "failed_verification": "reopen_outcome_and_create_remediation",
            "external_blocker": "retain_block_and_set_decision_condition",
            "broken_record": "quarantine_for_repair",
        }
        if classification in {"duplicate", "superseded", "failed_verification"} and not target_id:
            raise ValueError(f"{classification} requires target_task_id")
        if target_id and kanban_db.get_task(conn, target_id) is None:
            raise ValueError(f"unknown target task {target_id}")
        proposed_action = action_map[classification]
        required_authority = "aurora"
        stable = hashlib.sha256(_json({"task": task_id, "class": classification, "target": target_id, "evidence": evidence}).encode()).hexdigest()
        action_id = _id("recon", stable)
        state_name = "shadow" if mode == "shadow" else "proposed"
        rationale = str(observation.get("rationale") or "").strip()
        if not rationale:
            raise ValueError("reconciliation rationale is required")
        with write_txn(conn):
            conn.execute(
                "INSERT INTO wc_reconcile_actions(action_id,stable_key,task_id,classification,target_task_id,proposed_action,rationale,evidence_json,evidence_at,confidence,proposed_by,required_authority,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(stable_key) DO UPDATE SET rationale=excluded.rationale,evidence_json=excluded.evidence_json,evidence_at=excluded.evidence_at,confidence=excluded.confidence,proposed_by=excluded.proposed_by,state=CASE WHEN wc_reconcile_actions.state='applied' THEN 'applied' ELSE excluded.state END",
                (action_id, stable, task_id, classification, target_id, proposed_action, rationale, _json(evidence), evidence_at,
                 confidence, actor_id, required_authority, state_name, _now()),
            )
        results.append({"action_id": action_id, "task_id": task_id, "classification": classification, "state": state_name, "proposed_action": proposed_action})
    return results


def apply_reconciliation(
    conn: sqlite3.Connection, *, actor: str, action_ids: list[str],
    organization: WorkforceOrganization | None = None,
) -> list[dict[str, Any]]:
    org = organization or load_organization()
    if org.resolve_profile(actor).agent != "aurora":
        raise PermissionError("only Aurora may apply reconciler transitions")
    runtime = runtime_state(conn)
    if runtime["kill_switch"] or runtime["mode"] != "apply":
        raise RuntimeError("reconciler apply mode is paused")
    if len(action_ids) > int(runtime["max_actions_per_tick"]):
        raise ValueError("action batch exceeds the configured concurrency ceiling")
    results: list[dict[str, Any]] = []
    for action_id in action_ids:
        action = conn.execute("SELECT * FROM wc_reconcile_actions WHERE action_id=?", (action_id,)).fetchone()
        if action is None:
            raise ValueError(f"unknown reconciliation action {action_id}")
        if action["state"] == "applied":
            results.append({"action_id": action_id, "state": "applied", "created": False})
            continue
        if action["state"] not in {"proposed", "shadow"}:
            raise ValueError(f"action cannot be applied from {action['state']}")
        if action["confidence"] != "high" or _now() - int(action["evidence_at"]) > 86400:
            with write_txn(conn):
                conn.execute("UPDATE wc_reconcile_actions SET state='quarantined' WHERE action_id=?", (action_id,))
            results.append({"action_id": action_id, "state": "quarantined", "reason": "insufficient confidence or stale evidence"})
            continue
        task_id = action["task_id"]
        classification = action["classification"]
        target_id = action["target_task_id"]
        evidence = list(_loads(action["evidence_json"], []))
        with write_txn(conn):
            if classification == "already_complete":
                item = conn.execute("SELECT * FROM wc_items WHERE task_id=?", (task_id,)).fetchone()
                if item and item["item_kind"] == "outcome" and item["verification_state"] != "passed":
                    conn.execute("UPDATE wc_reconcile_actions SET state='quarantined' WHERE action_id=?", (action_id,))
                    results.append({"action_id": action_id, "state": "quarantined", "reason": "outcome lacks passing verification"})
                    continue
                conn.execute("UPDATE tasks SET status='done',completed_at=? WHERE id=?", (_now(), task_id))
                conn.execute("UPDATE wc_items SET current_state='complete',updated_at=? WHERE task_id=?", (_now(), task_id))
                kanban_db._append_event(conn, task_id, "workforce_reconciled_complete", {"actor": actor, "evidence": evidence})
            elif classification in {"duplicate", "superseded"}:
                relation = "duplicate" if classification == "duplicate" else "supersedes"
                conn.execute(
                    "INSERT OR IGNORE INTO wc_relations(source_task_id,relation,target_task_id,evidence_json,confidence,created_by,created_at) VALUES(?,?,?,?,?,'aurora',?)",
                    (task_id, relation, target_id, _json(evidence), "high", _now()),
                )
                conn.execute("UPDATE tasks SET status='archived',claim_lock=NULL,claim_expires=NULL,worker_pid=NULL WHERE id=?", (task_id,))
                conn.execute("UPDATE wc_items SET current_state=?,updated_at=? WHERE task_id=?", (classification, _now(), task_id))
                kanban_db._append_event(conn, task_id, "workforce_reconciled_archived", {"classification": classification, "target_task_id": target_id})
            elif classification == "failed_verification":
                conn.execute("UPDATE tasks SET status='triage',completed_at=NULL WHERE id=?", (target_id,))
                conn.execute("UPDATE wc_items SET verification_state='failed',current_state='open',updated_at=? WHERE task_id=?", (_now(), target_id))
                remediation_id = kanban_db.create_task(
                    conn, title=f"Remediate failed verification for {target_id}",
                    body=json.dumps({"kind": "workforce_remediation", "verification_task_id": task_id, "outcome_task_id": target_id, "evidence": evidence}, indent=2, sort_keys=True),
                    assignee="aurora", created_by="aurora", triage=True,
                    idempotency_key=f"workforce-remediation:{task_id}:{target_id}",
                )
                remediation_key = stable_identity(item_kind="execution", desired_outcome=f"remediate {target_id}", action_class="remediation", target_ref=target_id)
                conn.execute(
                    "INSERT OR IGNORE INTO wc_items(task_id,item_kind,stable_key,goal_ref,desired_outcome,acceptance_test,action_class,target_ref,evidence_json,provenance_json,verification_state,current_state,created_at,updated_at) VALUES(?,'execution',?,'unknown',?,?,?,?,?,?,'pending','open',?,?)",
                    (remediation_id, remediation_key, f"Remediate {target_id}", "Outcome passes its acceptance test", "remediation", target_id,
                     _json(evidence), _json([{"verification_task_id": task_id}]), _now(), _now()),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO wc_relations(source_task_id,relation,target_task_id,evidence_json,confidence,created_by,created_at) VALUES(?,'remediates',?,?,?,'aurora',?)",
                    (remediation_id, target_id, _json(evidence), "high", _now()),
                )
                kanban_db._append_event(conn, target_id, "workforce_verification_failed", {"verification_task_id": task_id, "remediation_task_id": remediation_id})
            elif classification == "external_blocker":
                conn.execute("UPDATE tasks SET status='blocked',block_kind='needs_input' WHERE id=?", (task_id,))
                kanban_db._append_event(conn, task_id, "workforce_external_blocker_confirmed", {"evidence": evidence, "no_auto_unblock": True})
            else:
                conn.execute("UPDATE wc_reconcile_actions SET state='quarantined' WHERE action_id=?", (action_id,))
                results.append({"action_id": action_id, "state": "quarantined", "reason": "broken records require repair review"})
                continue
            conn.execute("UPDATE wc_reconcile_actions SET state='applied',applied_at=? WHERE action_id=?", (_now(), action_id))
        results.append({"action_id": action_id, "state": "applied", "created": True})
    return results


def record_correction(
    conn: sqlite3.Connection, *, actor: str, classification: str, scope: str,
    description: str, provenance_ref: str, privacy_class: str,
    rule_target: str | None = None, regression_ref: str | None = None,
    supersedes_id: str | None = None, organization: WorkforceOrganization | None = None,
) -> dict[str, Any]:
    org = organization or load_organization()
    actor_id = org.validate_execution_profile(actor).agent
    if classification not in CORRECTION_CLASSES:
        raise ValueError("invalid correction classification")
    if scope not in {"one_off", "profile", "department", "workforce", "system"}:
        raise ValueError("invalid correction scope")
    if privacy_class == "relationship_private" and scope in {"department", "workforce", "system"}:
        raise PermissionError("private relationship context may never propagate as an organizational rule")
    if scope in {"workforce", "system"} and actor_id not in {"aurora", "alina", "root"}:
        raise PermissionError("workforce and system corrections require Aurora, Alina, or Root")
    if not description.strip() or not provenance_ref.strip():
        raise ValueError("description and provenance_ref are required")
    ensure_schema(conn)
    stable = hashlib.sha256(_json({"class": classification, "scope": scope, "description": _normalized(description), "target": rule_target}).encode()).hexdigest()
    correction_id = _id("corr", stable)
    status = "implemented" if rule_target and regression_ref else "pending"
    now = _now()
    with write_txn(conn):
        if supersedes_id:
            prior = conn.execute("SELECT correction_id FROM wc_corrections WHERE correction_id=?", (supersedes_id,)).fetchone()
            if prior is None:
                raise ValueError("superseded correction does not exist")
            conn.execute("UPDATE wc_corrections SET status='superseded',updated_at=? WHERE correction_id=?", (now, supersedes_id))
        conn.execute(
            "INSERT INTO wc_corrections(correction_id,classification,scope,description,rule_target,provenance_ref,privacy_class,status,regression_ref,supersedes_id,recorded_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(correction_id) DO UPDATE SET rule_target=excluded.rule_target,provenance_ref=excluded.provenance_ref,privacy_class=excluded.privacy_class,status=excluded.status,regression_ref=excluded.regression_ref,supersedes_id=excluded.supersedes_id,updated_at=excluded.updated_at",
            (correction_id, classification, scope, description.strip(), rule_target, provenance_ref.strip(), privacy_class, status, regression_ref, supersedes_id, actor_id, now, now),
        )
    return {"correction_id": correction_id, "status": status, "scope": scope, "organizational_propagation": privacy_class != "relationship_private"}


def dashboard_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    if not schema_present(conn):
        return {
            "available": False,
            "runtime": None,
            "items": [],
            "plans": [],
            "exceptions": [],
            "corrections": [],
        }
    def grouped(query: str) -> list[dict[str, Any]]:
        return [dict(row) for row in conn.execute(query).fetchall()]
    runtime = conn.execute(
        "SELECT * FROM wc_runtime WHERE singleton = 1"
    ).fetchone()
    table_names = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'wc_%'"
        ).fetchall()
    }
    latest_goals = (
        conn.execute(
            "SELECT snapshot_id,source_title,source_updated_at,captured_at,content_hash "
            "FROM wc_goal_snapshots ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        if "wc_goal_snapshots" in table_names
        else None
    )
    return {
        "available": True,
        # Dashboard reads must not run migrations or update schema timestamps.
        "runtime": dict(runtime) if runtime is not None else None,
        "items": grouped("SELECT item_kind,current_state,COUNT(*) count FROM wc_items GROUP BY item_kind,current_state ORDER BY item_kind,current_state"),
        "plans": grouped("SELECT state,COUNT(*) count FROM wc_plans GROUP BY state ORDER BY state"),
        "exceptions": grouped("SELECT state,classification,COUNT(*) count FROM wc_reconcile_actions WHERE state!='applied' GROUP BY state,classification ORDER BY state,classification"),
        "corrections": grouped("SELECT status,scope,COUNT(*) count FROM wc_corrections GROUP BY status,scope ORDER BY status,scope"),
        "goals": dict(latest_goals) if latest_goals is not None else None,
        "vision": (
            grouped(
                "SELECT status,COUNT(*) count FROM wc_vision_reviews "
                "GROUP BY status ORDER BY status"
            )
            if "wc_vision_reviews" in table_names
            else []
        ),
    }


def observe_dispatch_tick(
    conn: sqlite3.Connection, *, observer: str = "dispatch_tick",
    organization: WorkforceOrganization | None = None,
) -> dict[str, Any]:
    """Consume explicit Workforce Control events without guessing or auto-applying.

    The observer is inert while paused or killed, never calls a model, and only
    proposes transitions for plugin-classified items with machine-checkable
    evidence. Aurora still owns application of every proposed transition.
    """
    if not schema_present(conn):
        return {"available": False, "processed": 0, "proposed": 0}
    runtime = dict(conn.execute("SELECT * FROM wc_runtime WHERE singleton=1").fetchone())
    if runtime["kill_switch"] or runtime["mode"] == "paused":
        return {"available": True, "paused": True, "processed": 0, "proposed": 0}
    row = conn.execute(
        "SELECT last_event_id FROM wc_cursors WHERE observer=?", (observer,)
    ).fetchone()
    cursor = int(row["last_event_id"]) if row else 0
    limit = max(1, int(runtime["max_actions_per_tick"]))
    events = conn.execute(
        "SELECT e.id,e.task_id,e.kind,e.created_at,i.item_kind,i.verification_state,"
        "t.status,t.block_kind,t.block_recurrences FROM task_events e "
        "JOIN wc_items i ON i.task_id=e.task_id JOIN tasks t ON t.id=e.task_id "
        "WHERE e.id>? ORDER BY e.id LIMIT ?",
        (cursor, limit),
    ).fetchall()
    proposals: list[dict[str, Any]] = []
    for event in events:
        observation: dict[str, Any] | None = None
        evidence = [f"kanban:event/{event['id']}"]
        if (
            event["item_kind"] == "outcome"
            and event["kind"] in {"completed", "status"}
            and event["status"] == "done"
            and event["verification_state"] != "passed"
        ):
            observation = {
                "task_id": event["task_id"],
                "classification": "already_complete",
                "confidence": "high",
                "rationale": "Outcome was marked done without passing verification",
                "evidence_references": evidence,
                "evidence_at": int(event["created_at"]),
            }
        elif (
            event["status"] == "blocked"
            and event["block_kind"] in {"needs_input", "capability"}
            and int(event["block_recurrences"] or 0) >= 3
        ):
            observation = {
                "task_id": event["task_id"],
                "classification": "external_blocker",
                "confidence": "high",
                "rationale": "Typed external blocker repeated at least three times",
                "evidence_references": evidence,
                "evidence_at": int(event["created_at"]),
            }
        if observation:
            proposals.extend(
                propose_reconciliation(
                    conn,
                    actor="aurora",
                    observations=[observation],
                    mode="shadow" if runtime["mode"] == "shadow" else "proposed",
                    organization=organization,
                )
            )
    last_id = int(events[-1]["id"]) if events else cursor
    with write_txn(conn):
        conn.execute(
            "INSERT INTO wc_cursors(observer,last_event_id,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(observer) DO UPDATE SET last_event_id=excluded.last_event_id,updated_at=excluded.updated_at",
            (observer, last_id, _now()),
        )
    return {
        "available": True,
        "paused": False,
        "processed": len(events),
        "proposed": len(proposals),
        "cursor": last_id,
    }
