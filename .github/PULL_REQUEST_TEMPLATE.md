## Summary

Describe the focused change and why it is needed.

## Validation

List the commands and relevant model runs used to validate the change, and
the commit they ran on.

## Review

The independent review's findings ledger: each finding, its severity and
verdict, the failing-first test and the fix commit. See `PR_Requirement.md`
at the repository root.

## Memory and performance

For model, importer, streaming, or Metal changes, describe the bounded-memory
behavior and include before/after measurements when performance changes.
Write `Not applicable` when this section does not apply.

## Remaining limitations

List any behavior that was intentionally left unchanged or still needs
validation.

- [ ] The change does not load a complete checkpoint, shard, or large model
      tensor into Swift heap memory.
- [ ] Logs and artifacts contain no credentials, private paths, or model
      weights.
- [ ] Every gate for this diff's scope passed on the final commit, and the
      review has no open medium-or-higher finding.
