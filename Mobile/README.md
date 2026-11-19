# Mference for iPhone and iPhone Duo

The phone app. Its design was locked on 2026-11-19 (tabs Chat, Calendar, Scan,
Settings; the system tab bar with ink selection; the Mac's chat bubbles; on an
open iPhone Duo, two panes split at the fold).

| Part | Path | Notes |
| --- | --- | --- |
| App target | `MferenceMobile.xcodeproj`, `MferenceMobile/` | Thin `@main` over `MferenceMobileUI`. Bundle ID `ai.mference.ios`, product name Mference. Copies the bundled Geist faces from `App/Resources/Fonts`. |
| Screens | `MferenceMobileKit/Sources/MferenceMobileUI` | SwiftUI. `MobileRootView` picks tabs (iPhone, Duo closed) or two panes (window 700 pt or wider: Duo open). |
| Phone services | `MferenceMobileKit/Sources/MferenceMobileCore` | No views: model storage and download, the Mac link, camera, Files export, memory budget. |
| Link protocol | `App/Sources/MferenceLink` | The iPhone to Mac messages, shared with the Mac app. |
| Build settings | `Config/MferenceMobile.xcconfig` | Simulator builds arm64 only; includes the gitignored `Config/Signing.local.xcconfig`. |
| Tokens | `App/Sources/MferenceDesign` | Shared with the Mac app. `MFColor` unchanged; phone type and control sizes are `MFPhoneFont` / `MFPhoneMetrics` (`PhoneTokens.swift`). |

This first slice runs on **sample data and scripted replies**. It never loads
a model, and pairing with the Mac is simulated. Web search on the phone is
labelled "Web search via paired Mac": searches will go through SearXNG on the
paired Mac (not wired yet).

The shared contracts are listed in [CONTRACTS.md](CONTRACTS.md).

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

## Run on your iPhone

The app is signed with a free Apple ID (a personal team). Once:

1. In Xcode, Settings › Accounts, add your Apple ID. Xcode lists its
   "Personal Team".
2. Copy `Config/Signing.local.xcconfig.example` to
   `Config/Signing.local.xcconfig` and set `DEVELOPMENT_TEAM` to that team's
   10-character ID. The file is gitignored; nothing personal is committed.
   Without it, a device build stops with "requires a development team" and
   nothing else.
3. Connect the iPhone with a cable, unlock it and tap Trust on "Trust This
   Computer?".
4. Turn on Developer Mode: on the iPhone, Settings › Privacy & Security ›
   Developer Mode (it appears after the phone has been connected to Xcode
   once, or after the first install attempt). The phone restarts; confirm
   Turn On after it unlocks.

Build and install, from the repository root (`scratch/` is gitignored):

```bash
xcodebuild -project Mobile/MferenceMobile.xcodeproj -scheme MferenceMobile \
  -configuration Debug -destination 'generic/platform=iOS' \
  -derivedDataPath scratch/dd-device -allowProvisioningUpdates build
```

```bash
xcrun devicectl list devices
```

```bash
xcrun devicectl device install app --device <iPhone identifier> \
  scratch/dd-device/Build/Products/Debug-iphoneos/Mference.app
```

The first launch says "Untrusted Developer". On the iPhone, Settings ›
General › VPN & Device Management, pick the Apple Development entry for your
Apple ID and tap Trust. Then the app opens.

What a free team means:

- The provisioning profile lasts **7 days**. After that the app no longer
  opens; build and install again (installing over the top keeps the app's
  data).
- At most **3 apps** signed by a free team can be installed on the phone at
  once (the limit Xcode enforces today; Apple does not document it). Remove
  one to make room. A free team can also register only about 10 new app IDs
  in 7 days.
- No TestFlight, no App Store, no iCloud, and no entitlements that need a
  paid team.

Measured screen sizes: iPhone 18 Pro 402 × 874 pt; iPhone Duo closed (outer)
466 × 678 pt; iPhone Duo open (inner) 951 × 669 pt, fold at x = 475.5.
