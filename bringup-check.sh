#!/bin/bash
# Family bring-up conformance check: one command to decide whether a model
# family port is trustworthy. It automates steps 1-6 of docs/FAMILY_GATE.md for
# a single family. Design:
# docs/superpowers/specs/2026-08-08-family-bringup-kit-design.md, Workstream 3.
#
# Stage contract. Stages run in order; a FAILED stage stops the script, a
# SKIPPED stage does not. Each stage prints one result line and the script ends
# with a single verdict line.
#
#   0  preflight          macOS 15+, Swift 6.1+, memory headroom, free disk on
#                         the results volume (and the model volume when a
#                         gturbo-dir is given), no other model or installer
#                         process, release MferenceCLI and MferenceRepack
#                         present.
#   1  toy suite          Scripts/test.sh --filter <family regex>. SKIPPED when
#                         the family has no suites registered yet.
#   2  install verify     MferenceRepack --verify-install. Needs <gturbo-dir>.
#   3  ladder smoke       three fresh-process greedy generations at 16 / 32 /
#                         auto expert-cache slots; the three stdouts must be
#                         byte-identical. Needs <gturbo-dir>.
#   4  protocol scaffold  creates the results directory and prints the
#                         run-benchmark.sh command to run next. It never runs
#                         the benchmark: that stays a deliberate invocation.
#
# Every non-dry run writes <results>/bringup-report.txt with the provenance
# AGENTS.md requires a run report to carry: commit (and dirty state), hardware
# and RAM, macOS, Swift, the exact invocation, per-stage exit codes, and the
# deviations (skipped stages). A PASS without that file is not a report.
#
# env: MIN_FREE_GB=5     minimum free disk required on the results volume
#      MIN_FREE_PCT=20   minimum system-wide free memory percentage required
#
# Stage 1's per-family filter table is this kit's current seam for the future
# manifest-driven toy generator (spec W3.1). Once one ToySynthetic emits
# fixtures for any axis selection, the case statement below collapses into a
# lookup on that generator's suite name instead of a hand-kept regex.
#
# usage: ./bringup-check.sh [--dry-run] <family> [gturbo-dir]
#        ./bringup-check.sh --help
set -uo pipefail
cd "$(dirname "$0")"

fail() { echo "ABORT: $*" >&2; exit 1; }
show() { printf '    $ %s\n' "$*"; }
ok()   { echo "✔ stage ${1}: ${2}"; stage_log="${stage_log}✔ stage ${1}: ${2} (exit 0)"$'\n'; }
skip() { echo "○ stage ${1}: SKIPPED -- ${2}"; skipped=$((skipped + 1)); stage_log="${stage_log}○ stage ${1}: SKIPPED -- ${2}"$'\n'; }
bad()  {
  echo "✘ stage ${1}: FAILED -- ${2}" >&2
  stage_log="${stage_log}✘ stage ${1}: FAILED -- ${2} (exit 1)"$'\n'
  write_report "NOT SUPPORTED (${family}, stage ${1})"
  echo "verdict: NOT SUPPORTED (${family}, stage ${1})" >&2
  exit 1
}
# passed <stage> <label> -- the OK line, tagged when nothing actually ran.
passed() { if [ "${dry_run}" -eq 1 ]; then ok "$1" "$2 (dry run)"; else ok "$1" "$2"; fi; }

# The provenance record AGENTS.md requires of any reported run: commit,
# hardware and RAM, macOS, Swift, exact command, per-stage exit codes, and
# every deviation (here: the skipped stages). Written for PASS and FAIL alike;
# a dry run writes nothing because nothing ran.
write_report() {
  [ "${dry_run}" -eq 1 ] && return 0
  mkdir -p "${out_root}"
  report="${out_root}/bringup-report.txt"
  {
    echo "bringup-check ${family} -- ${1}"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "invocation: ${invocation}"
    dirty=""
    [ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=" (working tree dirty)"
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)${dirty}"
    sw_vers 2>/dev/null
    swift --version 2>&1 | head -1
    system_profiler SPHardwareDataType 2>/dev/null |
      awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
    echo "free memory: ${free_pct:-unchecked}%  free disk: ${free_gb:-unchecked} GiB"
    [ -n "${model_dir}" ] && [ -f "${model_dir}/manifest.json" ] \
      && shasum -a 256 "${model_dir}/manifest.json"
    echo "stages (each stage's own exit code; a skip is a protocol deviation):"
    printf '%s' "${stage_log}"
    echo "deviations: ${skipped} stage(s) skipped"
  } > "${report}"
  echo "report: ${report}"
}

usage() {
  cat <<'EOF'
usage: ./bringup-check.sh [--dry-run] <family> [gturbo-dir]

Runs the family bring-up conformance stages in order and prints a verdict.
Stages: 0 preflight, 1 toy suite, 2 install verify, 3 ladder smoke,
4 protocol scaffold.

  <family>      one of: gemma4 qwen36 qwen38 deepseekv4flash inklingsmall maple
  [gturbo-dir]  an installed .gturbo directory. Without it, stages 2 and 3
                (install verify, ladder smoke) are SKIPPED.
  --dry-run     print every command each stage would execute and run nothing.
  --help, -h    this text.

See docs/FAMILY_GATE.md for the gate this automates and
docs/families/TEMPLATE.md for the page to fill in afterwards.
EOF
}

dry_run=0
family=""
model_dir=""
for arg in "$@"; do
  case "${arg}" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1 ;;
    -*) usage >&2; fail "unknown option: ${arg}" ;;
    *) if [ -z "${family}" ]; then family="${arg}"
       elif [ -z "${model_dir}" ]; then model_dir="${arg}"
       else usage >&2; fail "unexpected extra argument: ${arg}"; fi ;;
  esac
done
[ -n "${family}" ] || { usage >&2; exit 1; }

# Per-family toy/parity suite filter. Empty means "none registered yet".
case "${family}" in
  # Gemma 4 has no family-named toy or parity suite: its coverage currently
  # lives inside shared kernel suites (attention scale 1.0, router k8, tied
  # INT4 embedding, INT8 shared expert), which no family filter can isolate.
  gemma4)          suite_filter="" ;;
  qwen36)          suite_filter='Qwen(Runner|ResidentParity|SlotMapParity|RepackPlanner|ToolCallParser|Install)' ;;
  qwen38)          suite_filter='Qwen38' ;;
  deepseekv4flash) suite_filter='(DSV4|Deepseek)' ;;
  inklingsmall)    suite_filter='Inkling' ;;
  maple)           suite_filter='Maple' ;;
  *) usage >&2; fail "unknown family: ${family}" ;;
esac

skipped=0
stage_log=""
invocation="./bringup-check.sh$([ "${dry_run}" -eq 1 ] && printf ' %s' --dry-run) ${family}${model_dir:+ ${model_dir}}"
out_root="benchmark-results/${family}-bringup"
[ "${dry_run}" -eq 1 ] && echo "(dry run: no command below is executed)"
echo "bringup-check ${family}${model_dir:+ ${model_dir}}"

# ---------------------------------------------------------------------------
# Stage 0 -- preflight. AGENTS.md requires every one of these before any model
# run, and each aborts rather than warns.
# ---------------------------------------------------------------------------
pgrep_pattern='(\.build/release/|/)(MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferenceRepack)( |$)|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'

if [ "${dry_run}" -eq 1 ]; then
  show "sw_vers -productVersion   # require 15+"
  show "swift --version           # require 6.1+"
  show "memory_pressure -Q        # require \${MIN_FREE_PCT:-20}% free"
  show "df -g .                   # require \${MIN_FREE_GB:-5} GiB free for run output"
  [ -n "${model_dir}" ] && show "df -g ${model_dir}          # require 1 GiB free beside the model (verify receipt)"
  show "pgrep -fl '${pgrep_pattern}'"
  show "test -x .build/release/MferenceCLI -a -x .build/release/MferenceRepack"
  ok 0 "preflight (dry run)"
else
  os_version=$(sw_vers -productVersion) || fail "could not read macOS version"
  [ "${os_version%%.*}" -ge 15 ] 2>/dev/null \
    || bad 0 "macOS 15 or later required; found ${os_version}"

  swift_version=$(swift --version 2>&1 |
    sed -n 's/.*Apple Swift version \([0-9][0-9.]*\).*/\1/p' | head -1)
  [ -n "${swift_version}" ] || bad 0 "could not determine the Swift version"
  swift_major=${swift_version%%.*}
  swift_tail=${swift_version#*.}
  swift_minor=${swift_tail%%.*}
  [ "${swift_major}" -gt 6 ] 2>/dev/null ||
    { [ "${swift_major}" -eq 6 ] && [ "${swift_minor}" -ge 1 ]; } 2>/dev/null ||
    bad 0 "Swift 6.1 or later required; found ${swift_version}"

  free_pct=$(memory_pressure -Q 2>/dev/null |
    sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p' | head -1)
  [ -n "${free_pct}" ] || bad 0 "could not read 'memory_pressure -Q'"
  [ "${free_pct}" -ge "${MIN_FREE_PCT:-20}" ] \
    || bad 0 "memory pressure too high: ${free_pct}% free, need ${MIN_FREE_PCT:-20}%"

  # Disk, before any model executes: ladder outputs and the report land on this
  # volume, and --verify-install writes its receipt beside the model. Same
  # convention as run-benchmark.sh (MIN_FREE_GB, df -g).
  free_gb=$(df -g . | awk 'NR==2 { print $4 }')
  [ -n "${free_gb}" ] || bad 0 "could not determine free disk space"
  [ "${free_gb}" -ge "${MIN_FREE_GB:-5}" ] \
    || bad 0 "need at least ${MIN_FREE_GB:-5} GiB free for run output; found ${free_gb} GiB"
  if [ -n "${model_dir}" ] && [ -d "${model_dir}" ]; then
    model_free_gb=$(df -g "${model_dir}" | awk 'NR==2 { print $4 }')
    [ -n "${model_free_gb}" ] && [ "${model_free_gb}" -ge 1 ] \
      || bad 0 "need at least 1 GiB free on the model volume for the verify receipt; found ${model_free_gb:-unknown} GiB"
  fi

  # Match only actual executables, not shells whose command line mentions them.
  live=$(pgrep -fl "${pgrep_pattern}" \
    | grep -v -e 'bringup-check.sh' -e '/bin/zsh' -e '/bin/bash' -e 'pgrep' || true)
  [ -z "${live}" ] || bad 0 "another model or installer process is running:
${live}"

  for product in MferenceCLI MferenceRepack; do
    [ -x ".build/release/${product}" ] \
      || bad 0 "release ${product} missing; run: swift build -c release --product ${product}"
  done
  ok 0 "preflight (macOS ${os_version}, Swift ${swift_version}, ${free_pct}% mem free, ${free_gb} GiB disk free)"
fi

# ---------------------------------------------------------------------------
# Stage 1 -- toy suite. The cross-import-overlay flags match how the family
# suites are run elsewhere in the kit.
# ---------------------------------------------------------------------------
if [ -z "${suite_filter}" ]; then
  skip 1 "no toy suites registered for ${family}"
else
  show "./Scripts/test.sh -Xswiftc -Xfrontend -Xswiftc -disable-cross-import-overlays --filter '${suite_filter}'"
  if [ "${dry_run}" -eq 0 ]; then
    ./Scripts/test.sh -Xswiftc -Xfrontend -Xswiftc \
      -disable-cross-import-overlays --filter "${suite_filter}"
    [ "$?" -eq 0 ] || bad 1 "toy suite --filter ${suite_filter} did not pass"
  fi
  passed 1 "toy suite (--filter ${suite_filter})"
fi

# ---------------------------------------------------------------------------
# Stage 2 -- install verify.
# ---------------------------------------------------------------------------
if [ -z "${model_dir}" ]; then
  skip 2 "no gturbo-dir given; pass one to verify an install"
else
  show ".build/release/MferenceRepack --verify-install --input-gturbo ${model_dir}"
  if [ "${dry_run}" -eq 0 ]; then
    [ -d "${model_dir}" ] || bad 2 "model directory not found: ${model_dir}"
    .build/release/MferenceRepack --verify-install --input-gturbo "${model_dir}"
    [ "$?" -eq 0 ] || bad 2 "--verify-install rejected ${model_dir}"
  fi
  passed 2 "install verify (${model_dir})"
fi

# ---------------------------------------------------------------------------
# Stage 3 -- ladder smoke. Greedy decode must not depend on how many expert
# cache slots the ladder chose, so the three stdouts have to be byte-identical.
# Each rung is a fresh process, as the benchmark protocol requires.
# ---------------------------------------------------------------------------
rungs=(16 32 auto)
if [ -z "${model_dir}" ]; then
  skip 3 "no gturbo-dir given; ladder smoke needs an installed model"
else
  ladder_dir="${out_root}/ladder-smoke"
  show "mkdir -p ${ladder_dir}"
  [ "${dry_run}" -eq 1 ] || mkdir -p "${ladder_dir}"
  for slots in "${rungs[@]}"; do
    prefix="${ladder_dir}/slots-${slots}"
    show ".build/release/MferenceCLI --model ${model_dir}" \
      "--prompt 'The capital of France is' --max-new 24 --temperature 0" \
      "--expert-cache-slots ${slots} > ${prefix}.stdout 2> ${prefix}.stderr"
    [ "${dry_run}" -eq 1 ] && continue
    .build/release/MferenceCLI \
      --model "${model_dir}" \
      --prompt "The capital of France is" \
      --max-new 24 \
      --temperature 0 \
      --expert-cache-slots "${slots}" \
      > "${prefix}.stdout" 2> "${prefix}.stderr"
    [ "$?" -eq 0 ] || bad 3 "slots=${slots} run failed; see ${prefix}.stderr"
    footer=$(grep -h '^\[stop=' "${prefix}.stderr" 2>/dev/null || true)
    [ -n "${footer}" ] || bad 3 "slots=${slots} produced no timing footer; see ${prefix}.stderr"
    echo "    slots=${slots} ${footer}"
  done
  for slots in 32 auto; do
    show "cmp ${ladder_dir}/slots-16.stdout ${ladder_dir}/slots-${slots}.stdout"
    [ "${dry_run}" -eq 1 ] && continue
    cmp -s "${ladder_dir}/slots-16.stdout" "${ladder_dir}/slots-${slots}.stdout" \
      || bad 3 "slots=16 and slots=${slots} produced different bytes; diff them under ${ladder_dir}"
  done
  passed 3 "ladder smoke (${rungs[*]} slots, byte-identical)"
fi

# ---------------------------------------------------------------------------
# Stage 4 -- protocol scaffold. Prints the next command; never runs it. The
# protocol benchmark is long (three cases x warmup + reps) and must be a
# deliberate invocation on a quiet machine.
# ---------------------------------------------------------------------------
show "mkdir -p ${out_root}"
[ "${dry_run}" -eq 1 ] || mkdir -p "${out_root}"
passed 4 "protocol scaffold (${out_root})"
cat <<EOF

Next, on a quiet machine, run the community protocol (docs/COMMUNITY_BENCHMARKS.md):

    ./run-benchmark.sh ${family}-bringup ${model_dir:-<gturbo-dir>} 3

Then fill in the family page from the template docs/families/TEMPLATE.md, and
add the MFERENCE_PHASES=1 attribution run that docs/FAMILY_GATE.md step 6 wants.
EOF

write_report "PASS (${family}, ${skipped} stage(s) skipped)"
echo "verdict: PASS (${family}, ${skipped} stage(s) skipped)"
