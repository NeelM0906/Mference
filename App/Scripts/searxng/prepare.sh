#!/bin/bash
# Builds the SearXNG runtime the app bundles (App Program spec 4.5, P10):
# a relocatable CPython from python-build-standalone, SearXNG's pinned source
# (unmodified) and its pinned wheels, plus the AGPL source tarball shipped
# alongside. Every download is checked against the SHA-256 in pins.env or
# wheels.lock before anything from it is used.
#
# Usage: App/Scripts/searxng/prepare.sh [--fetch] [--force] [--out DIR] [--cache DIR]
#   (default) offline: uses only the hash-verified cache and never touches the
#            network; fails, naming the missing file, if anything is absent
#            or does not match its SHA-256. This is how bundle.sh calls it.
#   --fetch  also download what the cache lacks (pinned URLs, verified before
#            use). Run it once by hand, e.g. after changing a pin.
#   --force  rebuild the runtime even when its MANIFEST already matches the pins
#   --out    where the runtime goes (default App/Resources/SearXNG/runtime,
#            gitignored); bundle.sh copies App/Resources/SearXNG into
#            Mference.app/Contents/Resources/SearXNG and signs it
#   --cache  verified downloads, reused across runs (default App/.build/searxng-cache)
#
# Re-running is cheap: when the runtime's MANIFEST matches the pins and the
# cached files verify, nothing is rebuilt.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$HERE/../.." && pwd)"
OUT="$APP_DIR/Resources/SearXNG/runtime"
CACHE="$APP_DIR/.build/searxng-cache"
FETCH=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fetch) FETCH=1; shift ;;
        --force) FORCE=1; shift ;;
        --out) OUT="$2"; shift 2 ;;
        --cache) CACHE="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

# shellcheck source=pins.env
source "$HERE/pins.env"
PY_MINOR="${PYTHON_VERSION%.*}"                 # 3.12

# Refuse to delete anything but a SearXNG runtime directory.
case "$OUT" in
    /SearXNG|/SearXNG/runtime) echo "refusing --out $OUT" >&2; exit 64 ;;
    */SearXNG/runtime|*/SearXNG) ;;
    *) echo "--out must end in /SearXNG/runtime or /SearXNG (got: $OUT)" >&2; exit 64 ;;
esac

# Temporary directories are removed however the script ends.
TEMP_DIRS=()
cleanup() { for dir in "${TEMP_DIRS[@]+"${TEMP_DIRS[@]}"}"; do rm -rf "$dir"; done; }
trap cleanup EXIT

sha256() { shasum -a 256 "$1" | awk '{print $1}'; }

# fetch URL SHA256 NAME: verified file at $CACHE/NAME. Offline (the default)
# it only verifies the cache. With --fetch a fresh download lands in its own
# new empty directory and moves into the cache only once verified.
fetch() {
    local url="$1" expected="$2" name="$3"
    local target="$CACHE/$name"
    if [[ -f "$target" ]] && [[ "$(sha256 "$target")" == "$expected" ]]; then
        return 0
    fi
    if [[ "$FETCH" != 1 ]]; then
        if [[ -f "$target" ]]; then
            echo "error: cached $name does not match its pinned SHA-256 ($expected)." >&2
        else
            echo "error: $name is not in the SearXNG cache ($CACHE)." >&2
        fi
        echo "Offline mode never downloads. Run once: App/Scripts/searxng/prepare.sh --fetch" >&2
        exit 2
    fi
    rm -f "$target"
    local staging file
    staging="$(mktemp -d "$CACHE/.download.XXXXXX")"
    TEMP_DIRS+=("$staging")
    file="$staging/$(basename "$name")"
    curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
        --output "$file" "$url"
    local actual
    actual="$(sha256 "$file")"
    if [[ "$actual" != "$expected" ]]; then
        rm -rf "$staging"
        echo "SHA-256 mismatch for $name: expected $expected, got $actual" >&2
        exit 1
    fi
    mv "$file" "$target"
    rm -rf "$staging"
}

mkdir -p "$CACHE/wheels"

echo "==> Python $PYTHON_VERSION ($PYTHON_RELEASE)"
fetch "$PYTHON_URL" "$PYTHON_SHA256" "$PYTHON_FILE"

echo "==> SearXNG $SEARXNG_COMMIT"
fetch "$SEARXNG_URL" "$SEARXNG_SHA256" "$SEARXNG_FILE"

echo "==> wheels"
while read -r file url hash; do
    [[ -z "$file" || "$file" == \#* ]] && continue
    fetch "$url" "$hash" "wheels/$file"
done < "$HERE/wheels.lock"

# The MANIFEST names every pin; a runtime built from these pins is reused.
EXPECTED_MANIFEST="python=$PYTHON_VERSION+$PYTHON_RELEASE sha256=$PYTHON_SHA256
searxng=$SEARXNG_COMMIT sha256=$SEARXNG_SHA256
wheels=$(grep -c '\.whl' "$HERE/wheels.lock") lock_sha256=$(sha256 "$HERE/wheels.lock")
prepare_sha256=$(sha256 "$0") settings_sha256=$(sha256 "$APP_DIR/Resources/SearXNG/settings.yml")"
if [[ "$FORCE" != 1 && -f "$OUT/MANIFEST" && -x "$OUT/python/bin/python3" ]] \
    && [[ "$(cat "$OUT/MANIFEST")" == "$EXPECTED_MANIFEST" ]]; then
    echo "==> up to date: $OUT"
    exit 0
fi

echo "==> assembling $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/source" "$OUT/licenses"
tar -xzf "$CACHE/$PYTHON_FILE" -C "$OUT"          # -> $OUT/python
PY="$OUT/python/bin/python3"
SITE="$("$PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

wheels=()
while read -r file _ _; do
    [[ -z "$file" || "$file" == \#* ]] && continue
    wheels+=("$CACHE/wheels/$file")
done < "$HERE/wheels.lock"
"$PY" -I -m pip install --quiet --no-index --no-deps --no-compile --disable-pip-version-check \
    --target "$SITE" "${wheels[@]}"

# SearXNG's package, unmodified, plus the version file its own `freeze`
# command would write (the app has no git checkout to ask).
EXTRACT="$(mktemp -d)"
TEMP_DIRS+=("$EXTRACT")
tar -xzf "$CACHE/$SEARXNG_FILE" -C "$EXTRACT"
SRC="$EXTRACT/searxng-$SEARXNG_COMMIT"
cp -R "$SRC/searx" "$SITE/searx"
cat > "$SITE/searx/version_frozen.py" <<EOF
# SPDX-License-Identifier: AGPL-3.0-or-later
# Written by Mference's prepare.sh, as \`python -m searx.version freeze\` would.
VERSION_STRING = "$SEARXNG_VERSION"
VERSION_TAG = "$SEARXNG_VERSION"
DOCKER_TAG = "${SEARXNG_VERSION/+/-}"
GIT_URL = "https://github.com/searxng/searxng"
GIT_BRANCH = "master"
EOF

# The corresponding source, shipped alongside (AGPL-3.0-or-later).
cp "$CACHE/$SEARXNG_FILE" "$OUT/source/$SEARXNG_FILE"
cp "$SRC/LICENSE" "$OUT/licenses/SearXNG-AGPL-3.0.txt"
cp "$OUT/python/lib/python$PY_MINOR/LICENSE.txt" "$OUT/licenses/Python-PSF.txt"
cp "$APP_DIR/Resources/SearXNG/THIRD_PARTY_NOTICES.md" "$OUT/licenses/THIRD_PARTY_NOTICES.md"
# Every wheel's own licence files, gathered in one place.
for dist in "$SITE"/*.dist-info; do
    name="$(basename "$dist" .dist-info)"
    while IFS= read -r -d '' file; do
        mkdir -p "$OUT/licenses/wheels/$name"
        cp "$file" "$OUT/licenses/wheels/$name/"
    done < <(find "$dist" -type f \( -iname 'LICEN*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -path '*/licenses/*' \) -print0)
done
cp "$APP_DIR/Resources/SearXNG/settings.yml" "$OUT/settings.yml"
cp "$APP_DIR/Resources/SearXNG/SOURCES.md" "$OUT/SOURCES.md"

# Drop what the runtime never uses (pip, tests, IDLE, Tcl/Tk) to keep the bundle small.
STDLIB="$OUT/python/lib/python$PY_MINOR"
rm -rf "$SITE"/pip "$SITE"/pip-*.dist-info \
       "$STDLIB/test" "$STDLIB/idlelib" "$STDLIB/tkinter" "$STDLIB/turtledemo" "$STDLIB/turtle.py" \
       "$STDLIB"/lib-dynload/_tkinter.* \
       "$OUT"/python/lib/libtcl* "$OUT"/python/lib/tcl* "$OUT"/python/lib/tk* \
       "$OUT"/python/lib/itcl* "$OUT"/python/lib/thread* \
       "$OUT/python/share"
"$PY" -I -m compileall -q -j 0 -x 'lib2to3/tests' "$SITE" "$STDLIB" >/dev/null

# Smoke: the package imports with the bundled interpreter, isolated.
SEARXNG_SETTINGS_PATH="$OUT/settings.yml" SEARXNG_SECRET=prepare-check \
    "$PY" -I -B -c 'import searx.webapp' >/dev/null 2>&1 \
    || { echo "searx.webapp does not import with the bundled Python" >&2; exit 1; }

# Written last: a runtime with a MANIFEST is complete and passed the smoke.
printf '%s\n' "$EXPECTED_MANIFEST" > "$OUT/MANIFEST"

echo "==> done: $OUT ($(du -sh "$OUT" | awk '{print $1}'))"
