# Enterkey Maintained Fork

The Enterkey deployment uses the stock Hermes update command and dashboard
update action. The production checkout has these remotes:

```text
origin    https://github.com/enterkey-io/hermes-agent.git
nous-upstream  https://github.com/NousResearch/hermes-agent.git
```

`main` tracks `origin/main`. That branch is the deployable release line and
contains both reviewed Nous changes and the Enterkey features required by the
running agent fleet.

Keep the persistent Nous remote named `nous-upstream`. Stock Hermes reserves a
remote named exactly `upstream` for its direct-fork update probe; using that
name would make `hermes update --check` advertise Nous commits before the
Enterkey integration workflow has tested and promoted them.

## Deploying an approved release

Use the normal Hermes interface:

```bash
hermes update --check
hermes update --yes
```

The dashboard's **Update now** action invokes the same updater. Do not add an
Enterkey-specific update script or alter the dashboard action. The stock
updater backs up state, fast-forwards the checkout from `origin/main`, refreshes
dependencies and bundled assets, and restarts managed Hermes services.

Production must remain on a clean `main` checkout. Develop fixes on named
branches and merge them into the Enterkey fork before deployment; never depend
on an uncommitted production edit surviving an update.

## Incorporating Nous releases

The stock updater does not merge `nous-upstream/main` during a routine Enterkey
deployment. That merge is a code-integration operation and must be reviewed and
tested before it reaches the deployable branch.

Enterkey's `Enterkey upstream integration` workflow performs that integration
every six hours. It merges current Nous `main` into
`automation/upstream-sync`, maintains one pull request, and enables promotion
only after the repository's aggregate `All required checks pass` gate
succeeds. Branch protection prevents an untested integration from reaching
`main`.

Merge conflicts, CI failures, and CI-sensitive changes remain administrator
review events. They do not change `origin/main`, so the dashboard updater stays
on the last tested release. The user-facing update procedure remains one action:
press **Update now** to deploy the latest promoted Enterkey release.

This separation makes the routine update path safe while keeping upstream code
integration observable and reversible.

## Configured fallback policy

The fleet's top-level fallback chain is eligible only for classified service
failures and rate limits: timeouts, overloads, server errors, and provider or
upstream throttling. Authentication, billing, model/configuration errors,
context limits, invalid responses, and provider or content-policy refusals stay
on their native recovery or terminal path. Auxiliary tasks inherit this same
restriction when consulting the top-level chain; explicit task-specific
auxiliary routing remains independent.

## Verification

After an update, confirm:

```bash
git status --short --branch
git rev-parse HEAD origin/main
systemctl --user --failed --no-pager
systemctl --user list-units --type=service 'hermes*' --no-pager
```

`HEAD` must equal `origin/main`, the tree must be clean, and all enabled Hermes
gateway and dashboard units must be active.
