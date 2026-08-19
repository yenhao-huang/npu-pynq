# Branch Model

```text
main      stable, always synthesizable, protected
dev       integration; feature branches merge here
feat/*    individual work
```

Releases are **tags**, not a branch: `v0.1.0-mac-npu`. Hardware versions are
discrete artifacts tied to one bitstream, which is what a tag denotes and a
long-lived `deploy` branch does not.

## Why not a `deploy` branch

A `deploy` branch is a moving pointer. A bitstream is immutable and bound to one
source revision, one tool version, and one board. Asking "which commit is on the
board?" must have an exact answer; a branch tip changes under you. Tag the
commit that produced the bitstream and attach the bitstream to that tag's
Release.

Use a `deploy` branch only when a hosting service deploys from a branch by
contract and cannot read tags.

## Protection

On `main`: require the `ci` check to pass, require a pull request, disallow
force-push. Leave `dev` unprotected so integration stays cheap.
