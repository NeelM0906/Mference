# Family-PR acceptance gate

The testing profile every model-family PR (internal or external) passes
before merge. First executed against PR #15 (Maple Preview, 2026-08-11);
Phase A's `bringup-check.sh` will automate steps 1–6, at which point this
document becomes its specification.

## Gate steps

1. **Full suite ×3, consecutive, green.** One pass is not enough: the
   Maple review caught a cross-suite race (a new `@Suite` mutating the
   shared `FakeHFURLProtocol` statics outside the serialized remote suite)
   that failed a *different* test on each run. Any test that touches
   shared fixture state must join the fixture's serialized suite — in the
   remote-install stack, that means `extension RemotePayloadCopyTests`,
   never a new suite type.
2. **Release build** (`swift build -c release`) clean.
3. **Static checks:** `git diff --check` and
   `LC_ALL=en_US.UTF-8 ruby Scripts/check_markdown_links.rb`
   (the UTF-8 locale is required; the script misreports under US-ASCII).
4. **Pinned install + smoke:** the family's `SupportedModelSource` entry
   installs from its pinned revision with strict verification; raw and
   chat CLI complete a short generation with a normal stats footer.
5. **Community-protocol page:** the three frozen `real-generation-v1`
   cases, one discarded warmup, fresh-process measured runs, every footer
   `stop=endOfTurn`, medians recorded in the family's docs page with host
   details. Raw probes (any prompt, any settings) do not substitute.
6. **Phases snapshot:** one `MFERENCE_PHASES=1` decode attribution run,
   recorded on the family page — this is the baseline later optimization
   A/Bs will be judged against.
7. **Approximate or optional features** (e.g. FlashHead-style heads) are
   opt-in flags, clearly labeled, with no default-path claims.
8. **Provenance:** licenses and `THIRD_PARTY_NOTICES.md` updated for any
   imported reference code; no credentials, private paths, or weights in
   logs or fixtures.
9. **Merge-compatibility note:** the reviewer records the file overlap
   against active development branches and the agreed land order.

## Maintenance contract

A merged family is supported while its gate stays green in CI. A family
whose gate breaks and has no maintainer response gets flagged in its docs
page rather than silently degrading.
