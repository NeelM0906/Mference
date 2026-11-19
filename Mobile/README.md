# Mference for iPhone and iPhone Duo

The phone app. Its design was locked on 2026-11-19 (tabs Chat, Calendar, Scan,
Settings; the system tab bar with ink selection; the Mac's chat bubbles; on an
open iPhone Duo, two panes split at the fold).

| Part | Path | Notes |
| --- | --- | --- |
| App target | `MferenceMobile.xcodeproj`, `MferenceMobile/` | Thin `@main` over `MferenceMobileUI`. Bundle ID `ai.mference.ios`, product name Mference. Copies the bundled Geist faces from `App/Resources/Fonts`. |
| Screens | `MferenceMobileKit/Sources/MferenceMobileUI` | SwiftUI. `MobileRootView` picks tabs (iPhone, Duo closed) or two panes (window 700 pt or wider: Duo open). |
| Tokens | `App/Sources/MferenceDesign` | Shared with the Mac app. `MFColor` unchanged; phone type and control sizes are `MFPhoneFont` / `MFPhoneMetrics` (`PhoneTokens.swift`). |

This first slice runs on **sample data and scripted replies**. It never loads
a model, and pairing with the Mac is simulated. Web search on the phone is
labelled "Web search via paired Mac": searches will go through SearXNG on the
paired Mac (not wired yet).

## Build and run

```bash
xcodebuild -project Mobile/MferenceMobile.xcodeproj -scheme MferenceMobile \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro' build
```

Tests are model-free and run on a simulator (the kit is iOS only):

```bash
cd Mobile/MferenceMobileKit && xcodebuild test -scheme MferenceMobileKit-Package \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro'
```

Launch with `-MferenceShowFirstRun YES` to see first run again.

Measured screen sizes: iPhone 18 Pro 402 × 874 pt; iPhone Duo closed (outer)
466 × 678 pt; iPhone Duo open (inner) 951 × 669 pt, fold at x = 475.5.
