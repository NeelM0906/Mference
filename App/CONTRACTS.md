# Mference app: shared contracts (A0)

These files are the contracts the four app teams build against in parallel.
Changing a name, a case, a stored property or a protocol requirement in them
is one PR that the app coordinator approves; the other teams then rebase.
Adding an implementation behind a contract needs no approval.

| Contract | File | Implementation owner |
| --- | --- | --- |
| Package layout and targets | `Package.swift` | U1 |
| Design tokens (color, type, spacing, radius, layout) | `Sources/MferenceDesign/Tokens.swift` | U1 |
| Engine API for the app | `Sources/MferenceAppCore/Engine/EngineServicing.swift` | U2 (`EngineService.swift`) |
| SwiftData schema, store location | `Sources/MferenceAppCore/Persistence/Schema.swift` | U2 |
| Sections in the pill bar | `Sources/MferenceAppCore/Navigation/AppSection.swift` | U1 (shell) |
| Just-in-time permissions | `Sources/MferenceAppCore/Permissions/PermissionBroker.swift` | U2 |

Rules that follow from them:

- Views and view models import `MferenceAppCore`, `MferenceDesign` and
  SwiftUI only. Only `MferenceAppCore/Engine` imports `MferenceEngine` or
  `Mference`.
- Every color, font, spacing and radius in a view comes from
  `MferenceDesign`; no raw hex or point literals in views.
- Tests drive the engine through `EngineService` over an `EngineHost` with
  `FakeEngineBackend`; no test loads a model or touches the network.
- Rich per-feature values (citations, steps, scan fields) are Codable types
  owned by the feature and stored as `Data` in the schema's `*JSON` fields, so
  a feature can evolve them without a schema migration.

## Build and test

The app builds with Xcode's toolchain (SwiftData's `@Model` macro is not in
the Command Line Tools). On the shared machine every build and test goes
through the lock (`scratch/program/lock.sh`, workstreams `app`, `a1`..`a4`):

```bash
swift build -c release --package-path App
swift test --package-path App
```
