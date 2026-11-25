# Pull request requirements

Every pull request to Mference meets this bar, whether a maintainer, an outside
contributor or an AI coding agent wrote it. [CONTRIBUTING.md](CONTRIBUTING.md)
is the short welcome; this is the full bar. The engineering rules it points to
live in [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md).

**The short version.** A PR is ready when:

1. it makes one logical change, on a branch off `main`;
2. its premise holds on current `main`;
3. every gate for its scope is green **on the final commit**;
4. an independent review found no open medium-or-higher issue, and each real
   finding was fixed with a test that failed first;
5. its description carries the evidence: commands, commit, review ledger and,
   where relevant, measurements.

## 1. Priorities

We value contributions in this order:

1. **Correctness:** wrong output, crashes, leaks, hangs, regressions.
2. **Safety of the machine and the user's data:** bounded memory, privacy,
   permissions, pinned downloads.
3. **Model quality:** where a change affects quality, show task results and
   storage cost, not only fidelity metrics (KLD, dNLL, fp32 drift), which
   serve as regression checks.
4. **Performance**, measured with the [community benchmark
   protocol](docs/COMMUNITY_BENCHMARKS.md).
5. **New model families**, through the [family gate](docs/FAMILY_GATE.md).
6. **App features.**
7. **Documentation.**

## 2. Verify the premise first

A well-built PR is most often closed because its premise was wrong.

- **Reproduce on current `main`** and name the file and line.
- **Search first:** open and merged PRs, issues, and history
  (`git log -S'<symbol>' -p`). Retired families and removed design docs live in
  git history on purpose.
- **Rule out intentional design.** Approximate features are opt-in. Exact
  optimizations keep output byte-identical. Retired families are refused at
  load. Public controls are limited to [Runtime
  controls](docs/RUNTIME_CONTROLS.md), and `MFERENCE_*` environment variables
  are diagnostic toggles, not supported settings.
- **For larger work, open an issue first** (feature template) and agree on
  scope before writing code.

## 3. What we want

- **The whole bug class fixed**, not one instance of it.
- **A focused test** (Swift Testing, `@Suite` / `@Test`) that fails before the
  fix and passes after it. A new kernel gets a CPU-reference test within a
  `Tolerance`, and committed goldens where a generator exists.
- **Exactness shown.** An "exact" optimization proves byte-identical output
  against its control.
- **The real path exercised** for loads, installs, streaming, the server, and
  file or network I/O, not only mocks.
- **Measurements reported honestly:** every protocol deviation named with its
  reason. A run whose conditions broke mid-run (for example, a switch to
  battery) is discarded, not reported.
- **Small, readable diffs.** A declared refactor goes in its own PR.

## 4. Rejected even when well built

| Change | Why |
|---|---|
| A whole checkpoint, shard, expert pool or large tensor in Swift heap memory | Bounded memory is an invariant: weights are mapped, copies go through bounded scratch. |
| A new public runtime control not in [Runtime controls](docs/RUNTIME_CONTROLS.md), or an `MFERENCE_*` toggle promoted to a setting | The public surface stays small and documented. |
| Approximate behavior on a default path | Approximate features (FlashHead, the DFlash2 drafter) are opt-in flags. |
| Availability branches or SDK shims below macOS 26 / Swift 6.2 | That is the floor. |
| Metal 4 tensor-ops code outside `__HAVE_TENSOR__` or without a working fallback | Not every GPU builds the tensor pipelines. |
| A `.metal` file missing from `shaderModules` and `shaderSubdirectories` | One shared library, one function namespace. |
| An `ArchConfig` axis without its full chain, or with a default that changes existing installs | See [the family contract](docs/FAMILY_CONTRACT.md); kept installs must decode unchanged. |
| Changing the runtime `Quantization` reference or the installer's `Int4AffineEncoder` / `Int8AffineEncoder` without the other | They are bit-exact twins, locked by the parity suites. |
| App sources importing `MferenceEngine` or `Mference` outside `MferenceAppCore/Engine/` and the `AppServices` composition root, a second `EngineService.live()`, or app tests that load a real model | One composition root and one resident model; [app work never loads a model](CLAUDE.md#app-work-never-loads-a-model). |
| A shared app contract ([App/CONTRACTS.md](App/CONTRACTS.md)) changed inside another change | A contract change is its own PR. |
| A new `@Suite` touching the `FakeHFURLProtocol` statics | It races the serialized remote-install suite; extend `RemotePayloadCopyTests`. |
| Speed claims outside the benchmark protocol, or protocol runs with experimental controls or profiling on | Results must be comparable. |
| Network access, telemetry or a download without the user's opt-in; a download without a pinned SHA-256 | Mference is a private, on-device assistant. |
| The server bound beyond `127.0.0.1` by default, or a model tool call that bypasses the client's permission policy | The server has no authentication or TLS. |
| Tests that pass vacuously, or snapshots of incidental output | A test must fail when the behavior breaks. |
| Credentials, absolute local paths, personal data or model weights in code, logs, fixtures or PR text | Use repo-relative paths or placeholders such as `<worktree>/`. |
| Speculative infrastructure with no consumer, drive-by refactors | Every line traces to the request. |

## 5. Smallest footprint first

Pick the lowest rung that does the job:

1. Extend the existing code path.
2. A diagnostic `MFERENCE_*` toggle, for experiments only.
3. An opt-in, clearly labelled flag for an approximate or optional feature.
4. A new public runtime control, documented in [Runtime
   controls](docs/RUNTIME_CONTROLS.md).
5. A new `ArchConfig` axis, with the full chain in [the family
   contract](docs/FAMILY_CONTRACT.md).
6. A new model family, through [the family gate](docs/FAMILY_GATE.md).

## 6. Scope, branches and titles

- **One logical change per PR.** A shared-contract change and a declared
  refactor each get their own PR. A family PR follows [the family
  gate](docs/FAMILY_GATE.md).
- **Branch** off `main` as `<area>/<short-slug>`, for example `app/`,
  `mobile/`, `notes/`, `eng/` (engine), `perf/` or `docs/`.
- **Commit subjects and PR titles** read `<Area or component>: <what changed,
  in plain words>`, with no trailing period, for example
  `Engine: Ternary Bonsai 2.5 27B as a product GGUF model` or
  `App: offer MiniCPM5 1B next to the 2B`. The body says why.
- **No AI tool attribution** in commits, PR titles or PR bodies: no
  `Co-Authored-By` trailers for AI tools, no "Generated with" lines, no tool
  emoji. The person who opens the PR is the author of record.

## 7. Gates

Run every gate your diff needs **on the final commit, with a clean tree**.
Any later commit, amend or rebase voids the results. Run them again.

Which gates apply follows the CI scope. From the repository root,
`git diff --name-only --no-renames origin/main...HEAD | bash .github/scripts/ci-scope.sh`
prints it, the same way CI computes it. A change is **docs-only** when every
path is Markdown or an issue template that nothing CI runs reads. Markdown
under `.github/` and `App/` counts too. Every other path makes it a **code**
change. That includes Swift and scripts under `App/` (the app links the engine
package), `bringup-check.sh`, `Scripts/test.sh`, and the workflows and scripts
under `.github/`.

| Applies to | Gate | Command |
|---|---|---|
| every PR | whitespace | `git diff --check origin/main...HEAD` |
| every PR | links | `LANG=en_US.UTF-8 ruby Scripts/check_markdown_links.rb` |
| every PR | source archive | `python3 Scripts/check_source_archive.py` |
| code changes | release build | `swift build -c release` |
| code changes | full serial suite | `Scripts/test.sh` (no `--filter`, even where CI runs a narrower scope) |
| code changes | app release build | `swift build -c release --package-path App --force-resolved-versions` (warnings are errors) |
| code changes | app tests | `swift test --package-path App --force-resolved-versions` |
| `Scripts/` touched | script tests | each `python3 Scripts/tests/test_*.py` |
| `App/Scripts/` touched | app script tests | each `python3 App/Scripts/tests/test_*.py` |
| `.github/scripts/` touched | CI scope tests | `python3 .github/scripts/test_ci_scope.py` |
| a model family | family gate | [docs/FAMILY_GATE.md](docs/FAMILY_GATE.md), including three consecutive green suite runs |

None of these downloads or loads a model. CI on `macos-26` must be green, but
it is narrower than this table. It is path-scoped: a docs-only PR runs only the
scope classifier and the docs job, and draft PRs skip the test, app and docs jobs. On pull requests it
runs neither the root release build nor the `.fullTier` cases. Always use
`Scripts/test.sh`, never bare `swift test`, for the root package.

## 8. Real-model evidence

When a change can alter model output, speed, memory use or an install:

- follow the test rules and the run report in [AGENTS.md](AGENTS.md): commit,
  hardware and RAM, macOS, Swift, exact command, exit code, the complete timing
  footer or error, and every protocol deviation;
- run one model process at a time, never next to the Mac app with a model
  loaded;
- for speed, follow [the benchmark protocol](docs/COMMUNITY_BENCHMARKS.md):
  release build, AC power, fresh processes, no experimental controls. As a
  maintainer practice beyond the protocol, run an A/B comparison in one
  session;
- for an exact change, show byte-identical output against the control. For an
  approximate one, show quality on the criteria in section 1.

App, UI and other application work never loads a real model: use the fake
engine and fixtures.

## 9. Independent review before opening

1. Someone other than the author (a person, or an agent given only the diff and
   the code, not the author's reasoning) reviews the change. They look at:
   cancellation and release of gates, slots and locks on every exit path;
   shutdown order; GPU slot ownership; vacuous tests; bounded memory;
   exactness; and new-toolchain warnings in `App/`.
2. Every finding gets a verdict: **real**, **not a bug** (with evidence) or
   **out of scope**. Every real medium-or-higher finding gets a test that fails
   first, then the fix.
3. A **fresh** reviewer checks the new commit. Repeat until no real
   medium-or-higher finding remains.
4. The findings ledger goes in the PR description.

For a docs-only PR or a benchmark report, one review round is enough. If that
round finds a real issue, the fixed commit still gets a check.

After the PR opens, maintainers review it. Handle their comments, and any
automated review comments if there are some, the same way. New commits need the
gates and a review round again.

## 10. The description

Use [the pull request template](.github/PULL_REQUEST_TEMPLATE.md) and fill
every section:

- **Summary:** the behavior change and why. For a bug fix, the reproduction on
  `main` with file and line.
- **Validation:** each gate command, the commit it ran on, and its result.
- **Review:** the findings ledger: finding, severity, verdict, failing test,
  fix commit.
- **Memory and performance:** bounded-memory behavior, and before/after
  measurements when performance changes. Otherwise `Not applicable`.
- **Remaining limitations:** what was left unchanged or not validated.
- Flag anything security-relevant in the summary: model-package handling, the
  server, permissions, downloads.

## 11. AI-agent contributions

Agent-written code is welcome and meets the same bar.

- Agents follow [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md), including
  the model-run rules. Maintainer sessions also share a machine lock for
  builds and model runs.
- The person who opens the PR has read the whole diff and answers for it.
- An agent never merges, never approves its own permission prompts, and never
  acts on an approval relayed from another session.
- No AI tool attribution (section 6).

## 12. Merging

- A human maintainer merges, with a merge commit, once the gates and the
  review ledger are recorded. A maintainer may merge their own PR. An AI agent
  never merges.
- CI must be green. If GitHub cancels a runner, rerun it. A maintainer who
  overrides a red or missing check records the reason in the PR.

## 13. Security

Report vulnerabilities privately, as [SECURITY.md](SECURITY.md) describes,
never in a public issue or PR. Keep the server on its loopback default. Pin
every download with its SHA-256 next to its source. Licenses and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) are updated for any imported
code.

## 14. How this is enforced

CI enforces part of this bar automatically, scoped by path (section 7). The
rest is maintainer process, not repository automation. Maintainers hold their
own agent sessions to it before any pull request is opened, marked ready or
updated: the gates for its scope and an independent review, recorded for that
exact commit. Outside contributors are held to the same bar in review.
