#!/bin/bash
# Community benchmark protocol runner (docs/COMMUNITY_BENCHMARKS.md).
#
# Extends the published protocol with N measured repetitions per case so
# medians and run-to-run spread can be reported. Each run is a fresh process.
#
# Aborts before invoking the CLI unless every precondition in AGENTS.md holds,
# and aborts as soon as any run fails or does not reach a natural end of turn --
# the protocol requires rejecting such a run, so it must never be summarized.
#
# usage: ./run-benchmark.sh <label> <gturbo-dir> [reps] [extra MferenceCLI args...]
#
# env: BENCH_CASES=a,b     restrict to these cases
#      WARMUP_CASES=a,b    restrict which cases get a discarded warmup
#      MIN_FREE_GB=5       minimum free disk required
#      MIN_FREE_PCT=20     minimum system-wide free memory percentage required
set -uo pipefail
cd "$(dirname "$0")"

fail() { echo "ABORT: $*" >&2; exit 1; }

label="${1:?model label}"
model_dir="${2:?gturbo dir}"
reps="${3:-3}"
shift 3 2>/dev/null || shift 2
extra=("$@")

case "${reps}" in
  ''|*[!0-9]*) fail "reps must be a positive integer; got \"${reps}\"" ;;
  0) fail "reps must be at least 1" ;;
esac

# ---------------------------------------------------------------------------
# Preflight. AGENTS.md requires macOS 15+, Swift 6.1+, enough disk, acceptable
# memory pressure, a completed model, and no other model process, before any
# model run. Every one of these aborts rather than warns.
# ---------------------------------------------------------------------------

os_version=$(sw_vers -productVersion) || fail "could not read macOS version"
[ "${os_version%%.*}" -ge 15 ] 2>/dev/null \
  || fail "macOS 15 or later required; found ${os_version}"

swift_version=$(swift --version 2>&1 |
  sed -n 's/.*Apple Swift version \([0-9][0-9.]*\).*/\1/p' | head -1)
[ -n "${swift_version}" ] || fail "could not determine the Swift version"
swift_major=${swift_version%%.*}
swift_tail=${swift_version#*.}
swift_minor=${swift_tail%%.*}
[ "${swift_major}" -gt 6 ] 2>/dev/null ||
  { [ "${swift_major}" -eq 6 ] && [ "${swift_minor}" -ge 1 ]; } 2>/dev/null ||
  fail "Swift 6.1 or later required; found ${swift_version}"

free_gb=$(df -g . | awk 'NR==2 { print $4 }')
[ -n "${free_gb}" ] || fail "could not determine free disk space"
[ "${free_gb}" -ge "${MIN_FREE_GB:-5}" ] \
  || fail "need at least ${MIN_FREE_GB:-5} GiB free for run output; found ${free_gb} GiB"

free_pct=$(memory_pressure -Q 2>/dev/null |
  sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p' | head -1)
[ -n "${free_pct}" ] || fail "could not read 'memory_pressure -Q'"
[ "${free_pct}" -ge "${MIN_FREE_PCT:-20}" ] \
  || fail "memory pressure too high: ${free_pct}% free, need ${MIN_FREE_PCT:-20}%"

[ -x .build/release/MferenceCLI ] \
  || fail "release CLI missing; run: swift build -c release --product MferenceCLI"

[ -d "${model_dir}" ] || fail "model directory not found: ${model_dir}"
for required in manifest.json verified-install.json; do
  [ -f "${model_dir}/${required}" ] \
    || fail "model install incomplete: ${model_dir}/${required} is missing (run --verify-install)"
done
# .partial and .resume.json mark an interrupted install (--resume /
# --discard-partial). .install.lock is deliberately not checked: it survives a
# successful install, so it says nothing about completeness.
for leftover in "${model_dir}.partial" "${model_dir}.resume.json"; do
  if [ -e "${leftover}" ]; then
    fail "unfinished install present: ${leftover} (use --resume or --discard-partial)"
  fi
done

for case_seed in short-explanation:x medium-review:x long-synthesis:x; do
  prompt_file="docs/benchmark-prompts/real-generation-v1/${case_seed%%:*}.json"
  [ -f "${prompt_file}" ] || fail "benchmark prompt missing: ${prompt_file}"
done

# Match only actual executables, not shells whose command line mentions them.
live=$(pgrep -fl '(\.build/release/|/)(MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferenceRepack)( |$)|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' \
  | grep -v -e 'run-benchmark.sh' -e '/bin/zsh' -e '/bin/bash' -e 'pgrep' || true)
[ -z "${live}" ] || fail "another model or installer process is running:
${live}"

# ---------------------------------------------------------------------------

root="benchmark-results/${label}"
mkdir -p "${root}/system" "${root}/warmup" "${root}/measured"

{
  git status --short
  git rev-parse HEAD
  sw_vers
  swift --version
  system_profiler SPHardwareDataType |
    awk -F': ' '/Model Name|Model Identifier|Chip|Total Number of Cores|Memory/ { print $1 ": " $2 }'
  shasum -a 256 "${model_dir}/manifest.json"
  shasum -a 256 docs/benchmark-prompts/real-generation-v1/*.json
  echo "measured repetitions: ${reps}"
  echo "extra CLI args: ${extra[*]+${extra[*]}}"
} 2>&1 | tee "${root}/system/system.txt"

cases=(short-explanation:20260721 medium-review:20260722 long-synthesis:20260723)

# BENCH_CASES=short-explanation,medium-review restricts the run to those cases.
if [ -n "${BENCH_CASES:-}" ]; then
  filtered=()
  for case_seed in "${cases[@]}"; do
    case ",${BENCH_CASES}," in
      *",${case_seed%%:*},"*) filtered+=("${case_seed}") ;;
    esac
  done
  cases=(${filtered[@]+"${filtered[@]}"})
  [ "${#cases[@]}" -gt 0 ] || fail "no cases matched BENCH_CASES=${BENCH_CASES}"
fi

warmup_filter="${WARMUP_CASES:-all}"

# run_case <case_id> <seed> <output-prefix> -- returns the CLI's exit status.
run_case() {
  /usr/bin/time -l .build/release/MferenceCLI \
    --model "${model_dir}" \
    --messages-file "docs/benchmark-prompts/real-generation-v1/${1}.json" \
    --max-new 1024 \
    --max-context 4096 \
    --temperature 0.2 \
    --top-k 64 \
    --top-p 0.95 \
    --seed "${2}" \
    ${extra[@]+"${extra[@]}"} \
    > "${3}.stdout" 2> "${3}.stderr"
}

# check_run <output-prefix> <description> -- abort unless the run is valid.
check_run() {
  local prefix="$1" what="$2" status="$3" footer
  footer=$(grep -h '^\[stop=' "${prefix}.stderr" 2>/dev/null || true)
  if [ "${status}" -ne 0 ]; then
    fail "${what} exited ${status}; see ${prefix}.stderr
${footer:-(no timing footer)}"
  fi
  if [ -z "${footer}" ]; then
    fail "${what} produced no timing footer; see ${prefix}.stderr"
  fi
  case "${footer}" in
    *stop=endOfTurn*) ;;
    *) fail "${what} did not reach a natural end of turn, so the protocol rejects it:
${footer}" ;;
  esac
  echo "${footer}" >&2
}

for case_seed in "${cases[@]}"; do
  case_id="${case_seed%%:*}"; seed="${case_seed##*:}"
  if [ "${warmup_filter}" != "all" ]; then
    case ",${warmup_filter}," in
      *",${case_id},"*) ;;
      *) echo "[${label}] warmup ${case_id} SKIPPED (WARMUP_CASES)" >&2; continue ;;
    esac
  fi
  echo "[${label}] warmup ${case_id} ..." >&2
  run_case "${case_id}" "${seed}" "${root}/warmup/${case_id}"
  check_run "${root}/warmup/${case_id}" "warmup ${case_id}" "$?"
done

for rep in $(seq 1 "${reps}"); do
  for case_seed in "${cases[@]}"; do
    case_id="${case_seed%%:*}"; seed="${case_seed##*:}"
    echo "[${label}] measured rep${rep} ${case_id} ..." >&2
    run_case "${case_id}" "${seed}" "${root}/measured/${case_id}.rep${rep}"
    check_run "${root}/measured/${case_id}.rep${rep}" \
      "measured rep${rep} ${case_id}" "$?"
  done
done

# Single table generator, so the published statistic cannot drift from the doc.
./summarize-benchmarks.sh "${label}" | tee "${root}/summary.txt" || \
  fail "summary rejected the results for ${label}"

echo "[${label}] complete" >&2
