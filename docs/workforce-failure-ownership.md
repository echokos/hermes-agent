# Workforce Failure Ownership

This control plane is opt-in. Deploying the source does not change any Cron job,
host schedule, delivery route, or live profile. Activate only the bounded pilot
sources below after the monitor is running against the shared Kanban database.

## Profile Cron activation

Set `failure_ownership.enabled_at` to the timezone-aware UTC instant immediately
before saving each job. The monitor ignores older execution-ledger rows. If the
field is absent on first adoption, it bootstraps from only the latest terminal
execution: a latest success is a healthy baseline and a latest failure is opened.
Subsequent ledger reads advance a per-job terminal cursor and never rescan from
the activation baseline.

Add this object to Grace job `bfb34552cae2` (X bookmarks) and job
`49384cf381de` (GitHub stars), using a fresh timestamp for each atomic update:

```json
{
  "failure_ownership": {
    "technical_owner": "root",
    "director": "aurora",
    "severity": "warning",
    "enabled_at": "2026-09-07T19:00:00+00:00",
    "ack_timeout_seconds": 900,
    "repair_timeout_seconds": 3600,
    "recovery_successes_required": 2
  }
}
```

The timestamp above is an example, not a value to copy. A failed opted-in run is
persisted and its direct failure delivery is suppressed. Successful useful
outputs retain the job's existing delivery. Two distinct later execution IDs
provide recovery evidence but do not replace Root's acknowledgment or Aurora's
source-owned review acceptance. Grace remains the collector source and business
owner; Root owns repair of the shared integration.

For Chloe job `44f44143a9bd`, use Alina as the technical owner of local scheduler/tool
persistence and Aurora as director. Chloe is the factual-record actor because
the tool derives its actor from the executing profile; it cannot be overridden
in tool input.

```json
{
  "failure_ownership": {
    "technical_owner": "alina",
    "director": "aurora",
    "severity": "warning",
    "enabled_at": "2026-09-07T19:00:00+00:00",
    "ack_timeout_seconds": 900,
    "repair_timeout_seconds": 3600,
    "recovery_successes_required": 2
  },
  "required_workforce_signal": false,
  "observe_workforce_signal_attempts": true
}
```

Chloe's reconciliation contract explicitly permits a host-bounded quiet success
when there is no material fact to record, so `required_workforce_signal` must
remain false. Attempt observation is still enabled: once Chloe invokes
`workforce_signal`, invalid input, exhausted call budget, or a failed write makes
the Cron execution and tracked workflow fail even if the model returns
`[NO_ACTION]` or success prose. Invalid preflight spends a call attempt but not
the one-write allowance, so a valid retry can still commit. Repeated invalid
attempts stop at the existing six-call job budget.

Before activating Chloe, its prompt/runbook must supply a durable, explicit
Aurora assignment ID as `aurora_assignment_id` whenever a material signal is
written. Do not increase its tool-call or write quota and do not require a
synthetic signal on quiet runs.

## Host activation

Host jobs call `scripts/workforce_failure_intake.py` once per terminal outcome
with a stable, unique `execution_id`. The JSON object requires `workflow_id`,
`source_id`, `technical_owner`, `director`, `execution_id`, and `error`.
Recovery uses a new execution ID plus `"outcome": "recovered"`. Gate each host
adapter off by default; suppress its legacy direct alert only after the append
returns success and the deployed monitor path has passed a harmless canary.

## Rollback

Remove `failure_ownership` and the Chloe `observe_workforce_signal_attempts`
flag from the pilot jobs, or disable the host adapter gate. This immediately
restores legacy failure-delivery behavior for later runs. Do not delete monitor
state, intake records, execution rows, or active Kanban incidents during
rollback; they are audit evidence and existing handoffs keep their original
owner and director until accepted or explicitly closed.
