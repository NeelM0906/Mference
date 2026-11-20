# Mac app design: menu bar, first run, updates, terminal status

Design sketches and a build spec for four additions to the Mac app in
[`App/`](../../App/CONTRACTS.md), plus changes to the published site in
`site/`. Nothing here is built. Every state below maps to a field the app
already has, or is marked **new**.

Open the sketches in a browser; they need no build step:

| Sketch | What it shows |
| --- | --- |
| [menu-bar.html](menu-bar.html) | The menu-bar extra: main menu, model submenu, 13 states |
| [onboarding.html](onboarding.html) | A motion pass over the existing first run (it plays), and the site changes |

## 1. Menu-bar extra

A `MenuBarExtra` scene in `MferenceMacApp`, beside the `Window("Mference")`
scene, in menu style so macOS draws the material, highlight and dark mode. It
uses the app's one `AppServices`; it never builds a second engine.

| Row | Source |
| --- | --- |
| Status: loaded, loading, unloading, no model | `EngineServiceStatus.phase`, `residentModel`, `targetModel` |
| Answering, requests waiting | `isGenerating`, `queuedRequests` |
| Load failed, Retry | `lastLoadFailure`, `lastLoadFailureModel` |
| Model submenu: Recommended, Also on this Mac, Built into macOS | `ModelGroup`, `ModelAvailability`, `ModelOption` |
| Fits, Tight, Too large, estimated memory | `MemoryFit.verdict`, `estimatedResidentBytes` |
| Downloads: progress, paused, verifying, failed | `DownloadState`, `DownloadError` |
| Offline Mode, Unload Model | `AppSettings.offlineMode`, the engine's unload |
| Speed of the last reply | **new**: keep the last reply's timing on the status |
| Minutes until idle unload | **new**: the idle timer's start time |
| Update rows | **new**: section 3 |

Items: Open Mference, New Chat, Model, Unload Model, Offline Mode, Check for
Updates, Settings, Quit Mference. Switch and Unload are disabled with their
reason during a load ("after this load"), because loads cannot be
interrupted. Quitting from the menu runs the app's existing quit sequence.

## 2. First run, motion pass

`OnboardingView` keeps its steps and rules: no account (L5), nothing asked up
front (L6), Quick installs on every Mac with room, Deep is offered only above
16 GB and downloads only after a yes (L4). Each step becomes one moving
picture with a short title; the long sentences move to tooltips:

| Step | Picture |
| --- | --- |
| Welcome | Chat, Mail, Calendar and Scan tiles settle into a Mac; "No account · no usage data sent" |
| Models | A memory bar with Quick and Deep sized to their estimated memory; Quick's download fills a grid |
| Permissions | Four permission cards light one by one with the moment each is asked |
| Mail (optional) | Letters sort from Inbox into Important and To-dos |
| First answer (**new**) | The first answer streams from Quick; "Update automatically" starts off |

## 3. Automatic updates

The app has no update path today. Proposed: Sparkle 2 with the app's own
windows.

- No network use until the user turns automatic updates on or presses Check
  for Updates. Offline Mode blocks update checks.
- Checks 15 s after launch, then every hour. The feed is
  `site/appcast.xml`, published by the existing Pages workflow; the disk image
  goes on the GitHub release. Updates are signed with an EdDSA key; an
  unsigned update is refused.
- An update relaunches the app and releases the model, so it installs only
  after 30 s with no load, generation, download or scan running.
- A manual update asks once, in a What's New window. Chats, settings and
  models are never part of an update.
- **Signing.** `bundle.sh` signs ad hoc or with a local self-signed identity.
  An ad hoc app is a new app to TCC on every build, so each update would drop
  the Calendar, Reminders, Full Disk Access and Camera grants. Shipped updates
  need one stable identity: Developer ID with notarization, or the
  maintainer's local signing identity for builds that are not distributed.

## 4. Terminal status

`mference status` prints one block; `--watch` redraws it in place. Built with
[Noora](https://github.com/tuist/Noora) (live tables and progress bars). The
app owns the one resident model, so the command never loads a model: it reads
a status file the app writes (**new**,
`~/Library/Application Support/Mference/status.json`, the same fields as the
menu bar).

```text
● loaded   Quick 4B
  memory    4.52 GB estimated · unloads after 10 min idle
  answering no · 0 waiting
  downloads Deep 27B 41% · 4.97 of 12.12 GB
  offline   off
```

## 5. Site changes

On the published page, nothing removed:

- Nav button: Download for Mac, after the first signed release.
- Watch it answer: a recorded Qwen 3.6 run at its measured speed.
- What it does: chat, mail, calendar, scan, as the README leads.
- A Download block at the end, and `site/appcast.xml`.
- Fix the share text "A 175 GB model in 2.36 GB of memory": that model was
  retired; the hero now shows Qwen 3.6 (19.6 GB to 1.45 GB).

## Open decisions

- Developer ID and notarization for released builds, or local signing only.
- Whether the menu-bar extra is on by default.
- Whether the first-answer step joins the first run.
