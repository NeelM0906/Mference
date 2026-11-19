#!/usr/bin/env bash
# Assemble App/build/Mference.app from the SwiftPM release build (spec 4.2).
#
#   App/Scripts/bundle.sh [--no-build] [--copy-to DIR]
#
# 1. Builds `MferenceMac` in release with -j 4 (skipped with --no-build),
#    directly. With MFERENCE_LOCK set to a lock.sh it builds only while that
#    lock is free and unreserved or held for a build (exit 75 otherwise),
#    through `lock.sh build` (exit 3 while the quiet flag is set).
# 2. Lays out the bundle:
#      Contents/MacOS/Mference                 the MferenceMac executable
#      Contents/Info.plist, PkgInfo            from App/Resources/Info.plist
#      Contents/Resources/*.bundle             every SwiftPM resource bundle the
#                                              build produced (Mference_Mference
#                                              with the .metal sources,
#                                              Mference_LlamaCpp with the
#                                              ggml-metal kernels, swift-
#                                              transformers', ...), test bundles
#                                              excluded
#      Contents/Resources/Fonts                App/Resources/Fonts (Geist, OFL)
#      Contents/Resources/SearXNG              App/Resources/SearXNG (see hook)
# 3. Signs with App/Resources/Mference.entitlements (no App Sandbox),
#    verifies the signature, and runs `Mference --verify-bundle`, which loads
#    both engine resource bundles the way their `Bundle.module` accessors do
#    and checks their files. The signing identity (P4, as changed 2026-11-19):
#      - MFERENCE_SIGN_IDENTITY when set ("-" forces ad hoc);
#      - else "Mference Local Signing" when `security find-identity -p
#        codesigning` lists it: a self-signed certificate the maintainer
#        creates in their login keychain, so the designated requirement is
#        certificate-based and TCC grants (Full Disk Access, Calendar, ...)
#        survive rebuilds;
#      - else ad hoc, with a notice (grants may not survive a rebuild).
#    The identity and the designated requirement are printed to the log.
#    bundle.sh never creates or trusts a certificate.
# 4. With --copy-to DIR: copies the app to DIR/Mference.app (refusing while a
#    process runs from there) and writes DIR/COMMIT with the source commit.
#
# SearXNG hook (team U3): when U3's App/Scripts/searxng/prepare.sh exists,
# bundle.sh runs `prepare.sh --offline`, which builds the relocatable runtime
# into App/Resources/SearXNG/runtime from the already-verified download cache
# only. bundle.sh never downloads: if the cache is missing it stops and says
# to run `App/Scripts/searxng/prepare.sh --fetch` once (pinned, SHA-256-checked
# downloads, decision P10). It copies App/Resources/SearXNG (runtime, licence
# texts, notices) verbatim (ditto,
# so symlinks and modes survive) to Contents/Resources/SearXNG and signs every
# Mach-O file inside it with the same identity before sealing the app, so nothing else is
# needed here. Keep large or downloaded payloads gitignored inside that
# directory and record their URL and SHA-256 next to them.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$APP_DIR/build"
APP_BUNDLE="$OUT_DIR/Mference.app"
RESOURCES_SRC="$APP_DIR/Resources"

build=1
copy_to=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) build=0; shift ;;
        --copy-to)
            [[ $# -ge 2 && -n "$2" ]] || { echo "bundle.sh: --copy-to needs a directory" >&2; exit 64; }
            copy_to="$2"; shift 2 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) echo "bundle.sh: unknown argument: $1" >&2; exit 64 ;;
    esac
done

# Signing identity (see the header). Decided before anything is built.
LOCAL_IDENTITY="Mference Local Signing"
SIGN_IDENTITY="${MFERENCE_SIGN_IDENTITY:-}"
if [[ -z "$SIGN_IDENTITY" ]]; then
    if security find-identity -p codesigning 2>/dev/null | grep -qF "\"$LOCAL_IDENTITY\""; then
        SIGN_IDENTITY="$LOCAL_IDENTITY"
    else
        SIGN_IDENTITY="-"
        echo "bundle.sh: notice: no \"$LOCAL_IDENTITY\" code-signing identity in the keychain; signing ad hoc," >&2
        echo "bundle.sh: notice: so macOS may ask again for Full Disk Access, Calendar and the rest after a rebuild." >&2
    fi
fi
echo "bundle.sh: signing identity: $([[ "$SIGN_IDENTITY" == "-" ]] && echo "ad hoc" || echo "$SIGN_IDENTITY")"

# App builds run without the machine lock (TEAM_RULES "GO PARALLEL",
# 2026-11-18). Set MFERENCE_LOCK to a lock.sh to build under its rules
# instead (free lock or a build hold, never with the quiet flag set).
# App builds share the machine only with a free, unreserved lock or a build
# hold (TEAM_RULES update 2026-11-18 ~21:00Z). A build hold's purpose starts
# with its workstream (eng, app, a1-a4) and has "build" as a word, or says
# "eng build" / "app build"; "post-build smoke" and other model runs do not
# count. Plain patterns only (BSD grep has no \b).
app_build_allowed() {
    local lock_text="$1"
    if grep -q '^reserved: ' <<<"$lock_text"; then
        return 1
    fi
    if grep -q '^machine lock: free' <<<"$lock_text"; then
        return 0
    fi
    grep -qiE '^ *purpose=((eng|app|a[1-4])( [^ ]+)* build( |$)|.*(eng|app) build( |$))' <<<"$lock_text"
}

if [[ "$build" == 1 ]]; then
    build_cmd=(swift build -c release -j 4 --package-path "$APP_DIR" --product MferenceMac)
    if [[ -n "${MFERENCE_LOCK:-}" ]]; then
        lock_status="$("$MFERENCE_LOCK" status)"
        if ! app_build_allowed "$lock_status"; then
            echo "bundle.sh: the machine is held or reserved for a run, not a build; not building now:" >&2
            echo "$lock_status" >&2
            exit 75
        fi
        "$MFERENCE_LOCK" build -- "${build_cmd[@]}"
    else
        "${build_cmd[@]}"
    fi
fi

BIN_DIR="$(swift build -c release --package-path "$APP_DIR" --disable-automatic-resolution --show-bin-path)"
EXECUTABLE="$BIN_DIR/MferenceMac"
[[ -x "$EXECUTABLE" ]] || { echo "bundle.sh: $EXECUTABLE not found; build first" >&2; exit 1; }
for required in Mference_Mference Mference_LlamaCpp; do
    [[ -d "$BIN_DIR/$required.bundle" ]] || { echo "bundle.sh: $BIN_DIR/$required.bundle not found" >&2; exit 1; }
done

# SearXNG (U3): prepare.sh builds App/Resources/SearXNG/runtime from its
# hash-verified cache only (--offline, also its default). It never downloads
# here: a missing or mismatched cache exits non-zero with the fix
# (`App/Scripts/searxng/prepare.sh --fetch`), and bundle.sh stops.
SEARXNG_PREPARE="$APP_DIR/Scripts/searxng/prepare.sh"
# The call is exactly U3's (notes/chat.md "bundle.sh and SearXNG"): offline,
# also the script's default, spelled out.
if [[ -x "$SEARXNG_PREPARE" ]]; then
    if ! "$SEARXNG_PREPARE" --offline; then
        echo "bundle.sh: SearXNG is not prepared; see the line above (run its --fetch once), then bundle again." >&2
        exit 1
    fi
else
    echo "bundle.sh: App/Scripts/searxng/prepare.sh is not in this checkout: Mference.app ships without web search." >&2
fi
if [[ ! -x "$RESOURCES_SRC/SearXNG/runtime/python/bin/python3" ]]; then
    echo "bundle.sh: no SearXNG runtime under App/Resources/SearXNG/runtime: Mference.app ships without web search." >&2
fi

STAGING="$OUT_DIR/Mference.app.staging"
rm -rf "$STAGING"
mkdir -p "$STAGING/Contents/MacOS" "$STAGING/Contents/Resources"

cp "$EXECUTABLE" "$STAGING/Contents/MacOS/Mference"
cp "$RESOURCES_SRC/Info.plist" "$STAGING/Contents/Info.plist"
printf 'APPL????' > "$STAGING/Contents/PkgInfo"

shopt -s nullglob
for resource_bundle in "$BIN_DIR"/*.bundle; do
    case "$(basename "$resource_bundle")" in
        *Tests*.bundle|*Test.bundle) continue ;;
    esac
    ditto --norsrc --noextattr "$resource_bundle" "$STAGING/Contents/Resources/$(basename "$resource_bundle")"
done
shopt -u nullglob

if [[ -d "$RESOURCES_SRC/Fonts" ]]; then
    ditto --norsrc --noextattr "$RESOURCES_SRC/Fonts" "$STAGING/Contents/Resources/Fonts"
fi

# SearXNG hook (U3), see the header.
if [[ -d "$RESOURCES_SRC/SearXNG" ]]; then
    ditto --norsrc --noextattr "$RESOURCES_SRC/SearXNG" "$STAGING/Contents/Resources/SearXNG"
    while IFS= read -r -d '' candidate; do
        if [[ "$(file -b "$candidate")" == *Mach-O* ]]; then
            codesign --force --sign "$SIGN_IDENTITY" --timestamp=none "$candidate"
        fi
    done < <(find "$STAGING/Contents/Resources/SearXNG" -type f -print0)
fi

# Extended attributes (quarantine, Finder info) break the seal.
xattr -cr "$STAGING"
codesign --force --sign "$SIGN_IDENTITY" --timestamp=none \
    --entitlements "$RESOURCES_SRC/Mference.entitlements" \
    "$STAGING"
codesign --verify --strict --verbose=1 "$STAGING"
# The designated requirement TCC keys grants on: certificate-based with the
# local identity, a cdhash when ad hoc.
# (Ad hoc prints it as a comment: "# designated => cdhash ...".)
echo "bundle.sh: designated requirement: $(codesign -dr - "$STAGING" 2>&1 | sed -n 's/^#* *designated => //p')"

rm -rf "$APP_BUNDLE"
mv "$STAGING" "$APP_BUNDLE"

"$APP_BUNDLE/Contents/MacOS/Mference" --verify-bundle

# The engine finds its resource bundles through SwiftPM's generated
# accessors: Swift `Bundle.module` (Mference) and the Objective-C
# SWIFTPM_MODULE_BUNDLE that ggml-metal uses (LlamaCpp). Check that this
# toolchain's accessors search the app's Contents/Resources first, where the
# bundles are placed, rather than relying on a build-directory fallback.
check_accessor() {
    local pattern="$1" name="$2" needle="$3" found=0 file
    while IFS= read -r -d '' file; do
        found=1
        if ! grep -qF "$needle" "$file"; then
            echo "bundle.sh: $file does not search $needle; the moved app would not find its resources" >&2
            exit 1
        fi
    done < <(find "$APP_DIR/.build" -path "*Release*" -path "$pattern" -name "$name" -print0 2>/dev/null)
    if [[ "$found" == 0 ]]; then
        echo "bundle.sh: no release resource accessor matching $pattern to check" >&2
        exit 1
    fi
}
check_accessor '*LlamaCpp*' 'resource_bundle_accessor.m' 'NSBundle.mainBundle.resourceURL'
check_accessor '*/Mference-t*' 'resource_bundle_accessor.swift' 'Bundle.main.resourceURL'

# Moved-app check: a copy outside the build tree still finds every bundle.
MOVED_DIR="$(mktemp -d)"
ditto "$APP_BUNDLE" "$MOVED_DIR/Mference.app"
"$MOVED_DIR/Mference.app/Contents/MacOS/Mference" --verify-bundle
rm -rf "${MOVED_DIR:?}"

echo "bundle.sh: $APP_BUNDLE"

if [[ -n "$copy_to" ]]; then
    mkdir -p "$copy_to"
    # Canonical spellings, so "dir/", "dir//" or a symlinked path cannot slip
    # past the running-app check: the logical path (as a launch from Finder or
    # a shell would show it) and the physical one (symlinks resolved).
    copy_to_logical="$(cd "$copy_to" && pwd)"
    copy_to_physical="$(cd "$copy_to" && pwd -P)"
    for candidate in "$copy_to_logical" "$copy_to_physical"; do
        if pgrep -f "$candidate/Mference.app/Contents/MacOS/" >/dev/null; then
            echo "bundle.sh: $candidate/Mference.app is running; not replacing it (quit it, then run bundle.sh --no-build --copy-to again)" >&2
            exit 75
        fi
    done
    copy_to="$copy_to_physical"
    target="$copy_to/Mference.app"
    rm -rf "${target:?}.incoming"
    ditto "$APP_BUNDLE" "$target.incoming"
    codesign --verify --strict "$target.incoming"
    rm -rf "${target:?}"
    mv "$target.incoming" "$target"
    {
        echo "commit=$(git -C "$APP_DIR" rev-parse HEAD)"
        # The whole repository: App links the engine from the repository root.
        echo "dirty=$([[ -z "$(git -C "$APP_DIR" status --porcelain 2>/dev/null)" ]] && echo no || echo yes)"
        echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "identity=$([[ "$SIGN_IDENTITY" == "-" ]] && echo "ad hoc" || echo "$SIGN_IDENTITY")"
    } > "$copy_to/COMMIT"
    echo "bundle.sh: copied to $target ($(sed -n 's/^commit=//p' "$copy_to/COMMIT" | cut -c1-8))"
fi
