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
| Speech for Notes (dictation, meetings, files) | `Sources/MferenceAppCore/Speech/SpeechServicing.swift`, `SpeechTypes.swift` | N2 (`SpeechService`) |
| Speech model pins (repo, revision, size, SHA-256 per file) | `Sources/MferenceAppCore/Speech/SpeechModelPins.swift`, `Resources/Speech/SOURCES.md` | coordinator |

Rules that follow from them:

- Views and view models import `MferenceAppCore`, `MferenceDesign` and
  SwiftUI only. Only `MferenceAppCore/Engine` imports `MferenceEngine` or
  `Mference`.
- Only `MferenceAppCore/Speech` imports `FluidAudio`, uses AVFoundation
  capture (`AVAudioEngine` input, `AVCaptureDevice`) or creates Core Audio
  process taps; everything else talks to `SpeechServicing`. Tests use
  `FakeSpeechService`: no speech model, microphone, call tap or TCC prompt.
  FluidAudio is used only through APIs that load local, verified folders
  (`AsrModels.loadLocal`, `VadManager(config:vadModel:)`, …); never
  `load(from:)`, `downloadAndLoad` or `ModelHub` (`NetworkAllowlistTests`).
  (The permission broker's existing microphone probe stays in `Permissions`.)
- Every color, font, spacing and radius in a view comes from
  `MferenceDesign`; no raw hex or point literals in views. Deviation
  (coordinator ruling 2026-11-18): a screen may keep its own layout sizes
  (column widths, panel widths, icon frames) as named constants in a
  per-screen `*Metrics` enum; inside `MferenceDesign`, component sizes are
  named in `MFMetrics`.
- Tests drive the engine through `EngineService` over an `EngineHost` with
  `FakeEngineBackend`; no test loads a model or touches the network.
- Find (2026-11-19) adds `AppSection.find` after Chat, and one exception to
  "one resident model": its search model (EmbeddingGemma 2, text, ~310 MB,
  plus its image and audio encoders, ~555 MB, when Photos, video and audio is
  on) runs through `MferenceEngine.LlamaEmbedder` in `Engine/FindEmbedderLive.swift`,
  a small second llama.cpp context that is never an `EngineHost` model and is
  freed when Find is idle. Find's tests use `HashingTextEmbedder`. Find also
  adds `AppPermission.photos` (`NSPhotoLibraryUsageDescription`), asked only
  when the person turns on Settings › Find › Photos library; the library is
  read through PhotoKit (`Find/FindPhotos.swift`), never as files. Find's
  views also use AppKit for the floating panel, the ⌥Space hot key, open
  panels and Quick Look thumbnails.
- Rich per-feature values (citations, steps, scan fields) are Codable types
  owned by the feature and stored as `Data` in the schema's `*JSON` fields, so
  a feature can evolve them without a schema migration.

## Speech: call audio heard (N2, 2026-11-20)

macOS can't report whether call audio (the Core Audio process tap) was
allowed: a refused tap does not fail, it delivers exact silence. So the
permission reads `.granted` once it has been asked, and is never stored or
shown as denied from silence. Two additions to `SpeechServicing` cover the gap:

- `SpeechServicing.hasHeardCallAudio` (read-only, never prompts): whether a
  meeting's tap has ever heard other apps (`SystemAudioPermission`). Settings
  › Speech shows call audio as "System audio recording · asked" until it is
  true, then "· allowed". `FakeSpeechService.setHasHeardCallAudio(_:)` sets it.
- `MeetingEvent.noCallAudioHeard`, a hint sent at most once per meeting.
  It is sent when call audio is on, the tap has delivered only exact silence
  for 30 s of recording, and the tapped apps are playing sound
  (`kAudioProcessPropertyIsRunningOutput`: every app but Mference for All
  apps, else the chosen app). When Core Audio can't say whether they play,
  30 s of silence alone sends it. While they don't play, it is checked again
  every 5 s. The meeting keeps recording both tracks. Show it with
  `MeetingEvent.noCallAudioHeardMessage`: "No call audio yet. If the other
  side is talking, check Call audio in System Settings."
  `FakeSpeechService.MeetingScript(noCallAudioHeard: true)` sends it right
  after `.started` when the meeting records call audio. `MeetingSession`
  keeps it as `callAudioHint` (cleared once the call track carries speech,
  when call audio stops, on dismiss, and when the meeting ends; never written
  into the note), and the meeting screen and the menu bar show it.

## Build and test

The app builds with Xcode's toolchain (SwiftData's `@Model` macro is not in
the Command Line Tools). On the shared machine every build and test goes
through the lock (`scratch/program/lock.sh`, workstreams `app`, `a1`..`a4`):

```bash
swift build -c release --package-path App
swift test --package-path App
```

## Bundle signing (P4, as changed 2026-11-19)

`App/Scripts/bundle.sh` signs `Mference.app` with, in order:
`MFERENCE_SIGN_IDENTITY` when set (`-` forces ad hoc); else the
maintainer's self-signed code-signing identity whose name starts with
**"Mference Local Signing"** (any case, trailing punctuation ignored; used by
its SHA-1), when `security find-identity -p codesigning` lists it; else ad hoc, with a
notice. With the local identity, the designated requirement is
certificate-based, so TCC grants (Full Disk Access, Calendar, Reminders,
Camera) survive rebuilds. With ad hoc it is a cdhash, and every build is a new
app to TCC. The bundle log prints the identity and the designated
requirement. bundle.sh never creates or trusts a certificate: the maintainer
makes that one in Keychain Access. `bundle.sh --copy-to DIR` also publishes
the app to `DIR/Mference.app` with `DIR/COMMIT` (commit, dirty flag, time,
identity), and refuses while a process runs from it.

## Engine: image understanding (Phase 2, 2026-11-24)

`EngineServicing` gains what lets Quick and Deep see pictures:

- `ChatTurn.images: [ChatImage]` (encoded picture bytes) on user turns.
  `EngineService` decodes them for a model whose engine traits say
  `supportsVision` (a GGUF model with its vision projector paired) and drops
  them for every other model. `ConversationStore.turns(for:seesImages:)` sends
  a message's pictures in every later turn too, so prompt reuse holds; the
  text read from them stays in the turn either way.
- `AppModel.supportsImageUnderstanding` (Quick, Deep) and
  `ModelOption.seesImages` / `imageUnderstandingBytes` (the add-on is installed
  and paired).
- The add-on's pins are `DownloadSource.imageUnderstanding`; it downloads
  through its own `DownloadManager` (`AppServices.imageDownloads`) into
  `~/llm-models/gguf`, where the engine pairs it with the model. Quick's is
  saved as `Qwen3.5-4B-mmproj-F16.gguf`. The Model Library shows it as
  "Image understanding" under Quick and Deep (download after approval,
  remove), and the chat offers it once, on the first picture attached to
  Quick or Deep without it (`AppSettings.hasOfferedImageUnderstanding`).
  Until it is installed, pictures reach the model as their OCR text.

Apple's on-device model (2026-11-24): where FoundationModels reports the
`.vision` capability (macOS 27), `AppleModelInfo.seesImages` makes the Apple
row `seesImages`, and `FoundationChatModel` sends a user turn's pictures as
image attachments ahead of its text, in history turns too. Nothing to
download, so the chat never offers image understanding for it.

## AppModel: Bonsai 2.5 27B (2026-11-25)

`AppModel.bonsai27b` (`bonsai-2.5-27b-q2_0`, `EngineProductSlot.bonsai27b`):
PrismML's Ternary Bonsai 2.5 27B, llama.cpp's stock `Q2_0` with its MTP head,
in the "Also on this Mac" group. It follows Deep's rules: offered only on Macs
with more than 16 GB, downloaded after approval
(`DownloadSource.bonsai27b`), and image understanding through its own Q8_0
projector (`DownloadSource.bonsai27bImageUnderstanding`).
