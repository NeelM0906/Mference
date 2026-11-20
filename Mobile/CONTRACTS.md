# Mference for iPhone: shared contracts (M0)

The four mobile teams build against these in parallel. Changing a name, a
case, a stored property, a protocol requirement or the wire shape in one of
them is its own small PR titled "contract: …" that the mobile coordinator
approves; the other teams then rebase on `mobile-app`. Adding an
implementation behind a contract needs no approval. The Mac app's contracts
are in [App/CONTRACTS.md](../App/CONTRACTS.md); the ones both apps share are
listed in both places.

| Contract | File | Owner |
| --- | --- | --- |
| Design tokens (shared with the Mac) | `App/Sources/MferenceDesign/Tokens.swift` | coordinator |
| Phone type and control sizes | `App/Sources/MferenceDesign/PhoneTokens.swift` | mobile-shell with the coordinator |
| Phone sections | `Mobile/MferenceMobileKit/Sources/MferenceMobileUI/Model/MobileSection.swift` | mobile-shell |
| Engine API for both apps | `App/Sources/MferenceAppCore/Engine/EngineServicing.swift` | mobile-chat |
| SwiftData schema | `App/Sources/MferenceAppCore/Persistence/Schema.swift` | mobile-chat |
| iPhone to Mac link protocol (messages, v1) | `App/Sources/MferenceLink/LinkProtocol.swift` | mobile-link |
| Mac package targets | `App/Package.swift` | coordinator |
| Phone package targets | `Mobile/MferenceMobileKit/Package.swift` | coordinator |
| Calendar screen model | `Mobile/MferenceMobileKit/Sources/MferenceMobileUI/Calendar/CalendarScreenModel.swift` | mobile-apple |
| Scan screen model | `Mobile/MferenceMobileKit/Sources/MferenceMobileUI/Scan/ScanScreenModel.swift` | mobile-apple |
| Sample screen models (target `MferenceMobileSamples`) | `Mobile/MferenceMobileKit/Sources/MferenceMobileSamples` | mobile-shell |
| Phone project build settings | `Mobile/Config/MferenceMobile.xcconfig` | mobile-shell |

Rules that follow from them:

- `MferenceMobileCore` has no SwiftUI views; `MferenceMobileUI` is SwiftUI and
  reaches services through protocols, so previews and tests run on fakes.
- Each screen in `MferenceMobileUI` depends only on its `*ScreenModel`
  protocol (`CalendarScreenModel`, `ScanScreenModel`; `ChatScreenModel` from
  mobile-chat next). The owner team writes the live model behind it.
- `MferenceMobileSamples` holds the sample screen models (`SampleCalendarModel`,
  `SampleScanModel`, ...) and will hold the `#Preview`s that need sample
  data (there are none yet). It depends on `MferenceMobileUI`;
  `MferenceMobileUI` and `MferenceMobileCore` never import it. **The app target links it only for now**: each owner swaps
  in its live model in `MferenceMobileApp.swift`, and the link is removed when
  the last live model lands. Samples do not ship at C-final.
- Platform-neutral logic lives once, in `MferenceAppCore`, built for both
  apps (decision M5); Mac-only parts sit behind `#if os(macOS)`. No copy of
  Mac logic under `Mobile/`.
- Until engine ask E1 lands, `MferenceAppCore` links `MferenceEngine` on macOS
  only, and iOS code reaches the engine behind `#if canImport(MferenceEngine)`.
- `MferenceLink` uses Foundation, CryptoKit and Security only (no sockets:
  the transport lives in each app). `LinkProtocol.swift` is the messages, the
  contract; key agreement, the pairing code, encryption, framing and replay
  checks are the link implementation beside it. After the plain hello and
  pairing exchange, every envelope crosses authenticated and encrypted.
- Tests never load a model, touch the network or raise a permission prompt.

## Signing

`Mobile/Config/MferenceMobile.xcconfig` includes the gitignored
`Mobile/Config/Signing.local.xcconfig` when it exists. The maintainer copies
`Signing.local.xcconfig.example` to it once and sets the personal team; nothing
personal is committed. Simulator builds need no signing file.
