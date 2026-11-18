#!/usr/bin/env bash
# Assemble App/build/Mference.app from the SwiftPM release build (spec 4.2).
#
#   App/Scripts/bundle.sh [--no-build]
#
# 1. Builds `MferenceMac` in release with -j 4 (skipped with --no-build). On
#    the shared Mac it follows the app build rule (TEAM_RULES update
#    2026-11-18): `lock.sh build -- swift build ...`, not wrapped in
#    `lock.sh run`, and only while `lock.sh status` shows a free lock or a
#    build hold (exit 75 otherwise). `lock.sh build` itself refuses (exit 3)
#    while the quiet flag is set. Without a lock.sh it builds directly.
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
# 3. Signs ad hoc (`codesign --sign -`, P4) with App/Resources/Mference.entitlements
#    (no App Sandbox), verifies the signature, and runs
#    `Mference --verify-bundle`, which loads both engine resource bundles the
#    way their `Bundle.module` accessors do and checks their files.
#
# SearXNG hook (team U3): first runs U3's App/Scripts/searxng/prepare.sh
# when it exists (pinned, SHA-256-checked downloads, decision P10; cached
# under App/.build/searxng-cache, so later runs reuse it), which builds the
# relocatable runtime under App/Resources/SearXNG/. This script copies that directory verbatim (ditto,
# so symlinks and modes survive) to Contents/Resources/SearXNG and signs every
# Mach-O file inside it ad hoc before sealing the app, so nothing else is
# needed here. Keep large or downloaded payloads gitignored inside that
# directory and record their URL and SHA-256 next to them.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$APP_DIR/build"
APP_BUNDLE="$OUT_DIR/Mference.app"
RESOURCES_SRC="$APP_DIR/Resources"

build=1
for arg in "$@"; do
    case "$arg" in
        --no-build) build=0 ;;
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) echo "bundle.sh: unknown argument: $arg" >&2; exit 64 ;;
    esac
done

# The shared lock lives in the main checkout's scratch/, also from worktrees.
if [[ -z "${MFERENCE_LOCK:-}" ]]; then
    common_dir="$(git -C "$APP_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [[ -n "$common_dir" && -x "$(dirname "$common_dir")/scratch/program/lock.sh" ]]; then
        MFERENCE_LOCK="$(dirname "$common_dir")/scratch/program/lock.sh"
    fi
fi

if [[ "$build" == 1 ]]; then
    build_cmd=(swift build -c release -j 4 --package-path "$APP_DIR" --product MferenceMac)
    if [[ -n "${MFERENCE_LOCK:-}" ]]; then
        lock_status="$("$MFERENCE_LOCK" status)"
        # App builds share the machine only with a free lock or a build hold
        # (TEAM_RULES update 2026-11-18 ~21:00Z).
        if ! grep -q '^machine lock: free' <<<"$lock_status" \
            && ! grep -qiE '^ *purpose=.*\bbuild\b' <<<"$lock_status"; then
            echo "bundle.sh: the machine lock is held by a run, not a build; not building now:" >&2
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

SEARXNG_PREPARE="$APP_DIR/Scripts/searxng/prepare.sh"
if [[ -x "$SEARXNG_PREPARE" ]]; then
    "$SEARXNG_PREPARE"
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
            codesign --force --sign - --timestamp=none "$candidate"
        fi
    done < <(find "$STAGING/Contents/Resources/SearXNG" -type f -print0)
fi

# Extended attributes (quarantine, Finder info) break the seal.
xattr -cr "$STAGING"
codesign --force --sign - --timestamp=none \
    --entitlements "$RESOURCES_SRC/Mference.entitlements" \
    "$STAGING"
codesign --verify --strict --verbose=1 "$STAGING"

rm -rf "$APP_BUNDLE"
mv "$STAGING" "$APP_BUNDLE"

"$APP_BUNDLE/Contents/MacOS/Mference" --verify-bundle

echo "bundle.sh: $APP_BUNDLE"
