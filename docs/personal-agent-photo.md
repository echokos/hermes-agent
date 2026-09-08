# Personal Agent Photos

The native `agent_photo` tool is limited to canonical personal profiles whose
organization status is `friend` and which are not operational workforce agents.
It invokes the fixed administrator-installed `hermes-agent-photo` wrapper,
never a shell command supplied by the model. The wrapper owns credential
injection, Characters bindings, identity assets and output paths.

## Current Requests

A direct current user request can authorize one photo without a second human
approval. The native ingress captures authored text before upload, history and
other prompt enrichment. A dedicated classifier uses the configured approval
auxiliary route. It does not treat tool arguments, quoted text, prior messages,
background work or internal delegation as user authorization.

The resulting capability is profile/session/turn-bound, one-use and bound to
the exact tool arguments and provider chain. Control messages, run completion,
errors and cancellation revoke it. Duplicate or overlapping calls cannot
reuse it. Classification failure blocks without generation or a redundant
human prompt. Unsupported origins and requests without direct authorization
retain the existing fresh human-approval path.

## Provider Attempts

Gemini is the default. One failed Gemini attempt can fall back to one Grok
attempt within the same authorized call. Set `fallback_to_grok=false` when the
user requests Gemini only. Explicit Grok or Seedream selections make one
attempt without fallback; neither can enable `fallback_to_grok=true`.

The native tool never forwards the standalone script's broad
`--allow-fallback` flag, which includes other providers. Instead, each attempt
uses the fixed wrapper with exact single-provider arguments. Successful output
stops the chain, and no provider is retried. A generic attempt failure does
not establish whether a provider call occurred. Explicit wrapper refusal,
start errors, cancellation and unverified cleanup stop without fallback.

The generation chain retains a 360-second shared budget, with a 180-second
Gemini attempt cap and up to five seconds of cleanup per stopped attempt.
The remaining budget limits Grok. On timeout or cancellation, the native tool
kills its isolated subprocess group, reaps the wrapper and verifies no live
group members remain before a timeout can permit fallback. It also checks the
captured run lifetime before each attempt, including human-approved calls
without a direct-request grant. Local cleanup cannot prove that a remote
provider did not finish a request; do not describe timeout as proof that no
image was generated.

## Deployment Boundary

These changes do not require replacing the live shared photo skill or its
Characters extensions. Do not deploy an older repository snapshot over an
independently extended shared snapshot. Validate the existing trusted wrapper
and provider credential injection separately, without generating paid media.
Deploy Hermes source through the maintained fork's normal protected release
and stock update path, then verify real user-requested generation and delivery.
