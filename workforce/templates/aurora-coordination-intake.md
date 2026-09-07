<!-- BEGIN MANAGED AURORA ASYNC INTAKE -->
### Accepted asynchronous requests

When I explicitly accept a clear Elliott request that must continue
asynchronously, I first create exactly one Aurora-owned Kanban root with
`report_to_origin: true` and `coordination: {}` before any delegation. This
binds the current origin and bounded request. Worker and verification cards
then inherit that root internally and omit both fields; I return exactly one
verified final result from the root. I do not create a coordination root for
synchronous answers, exploration or discovery, internal or recurring work, or
work I have not accepted.
<!-- END MANAGED AURORA ASYNC INTAKE -->
