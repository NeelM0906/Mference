#!/bin/bash
# Community benchmark protocol runner (docs/COMMUNITY_BENCHMARKS.md)
# Extends the protocol with N measured repetitions per case so medians and
# run-to-run spread can be reported. Each run is a fresh process.
#
# usage: ./run-benchmark.sh <label> <gturbo-dir> [reps] [extra MferenceCLI args...]
set -uo pipefail

label="${1:?model label}"
model_dir="${2:?gturbo dir}"
reps="${3:-3}"
shift 3 2>/dev/null || shift 2
extra=("$@")

root="benchmark-results/${label}"
mkdir -p "${root}/system" "${root}/warmup" "${root}/measured"

# Match only actual executables, not shells whose command line mentions them.
live=$(pgrep -fl '(\.build/release/|/)(MferenceServer|MferenceMac|MferenceDecodeService|MferenceCLI|MferenceRepack)( |$)|MferencePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' \
  | grep -v -e 'run-benchmark.sh' -e '/bin/zsh' -e '/bin/bash' -e 'pgrep' || true)
if [ -n "${live}" ]; then
  echo "ABORT: model or installer process already running:" >&2
  echo "${live}" >&2
  exit 1
fi

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
# Used only where a case is not feasible on the current prefill path.
if [ -n "${BENCH_CASES:-}" ]; then
  filtered=()
  for case_seed in "${cases[@]}"; do
    case ",${BENCH_CASES}," in
      *",${case_seed%%:*},"*) filtered+=("${case_seed}") ;;
    esac
  done
  cases=(${filtered[@]+"${filtered[@]}"})
  [ "${#cases[@]}" -gt 0 ] || { echo "no cases matched BENCH_CASES=${BENCH_CASES}" >&2; exit 1; }
fi

# WARMUP_CASES restricts which cases get a discarded warmup (default: all).
warmup_filter="${WARMUP_CASES:-all}"

run_case() {  # run_case <case_id> <seed> <outfile-prefix>
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

# One discarded warmup per case.
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
  grep -h '^\[stop=' "${root}/warmup/${case_id}.stderr" >&2 || true
done

# Measured repetitions, each in a fresh process.
for rep in $(seq 1 "${reps}"); do
  for case_seed in "${cases[@]}"; do
    case_id="${case_seed%%:*}"; seed="${case_seed##*:}"
    echo "[${label}] measured rep${rep} ${case_id} ..." >&2
    run_case "${case_id}" "${seed}" "${root}/measured/${case_id}.rep${rep}"
    grep -h '^\[stop=' "${root}/measured/${case_id}.rep${rep}.stderr" >&2 || true
  done
done

# Summarize: median tok/s across reps, plus min-max and peak RSS.
{
  printf '%-18s %8s %8s %10s %10s %10s %10s %8s\n' \
    case prompt gen prefill_s median min max peakMiB
  for case_seed in "${cases[@]}"; do
    case_id="${case_seed%%:*}"
    stats=$(for rep in $(seq 1 "${reps}"); do
      err="${root}/measured/${case_id}.rep${rep}.stderr"
      [ -f "${err}" ] || continue
      grep -h '^\[stop=' "${err}" | sed -E 's/.*prefill=([0-9]+)tok\/([0-9.]+)s new=([0-9]+)tok.*tok\/s=([0-9.]+).*/\1 \2 \3 \4/'
      awk '/maximum resident set size/ { printf "RSS %.0f\n", $1/1048576 }' "${err}"
    done)
    echo "${stats}" | awk -v c="${case_id}" '
      $1=="RSS" { if ($2>rss) rss=$2; next }
      NF==4 { p=$1; pf+=$2; g=$3; v[++n]=$4 }
      END {
        if (n==0) { printf "%-18s %8s\n", c, "NO DATA"; exit }
        for (i=1; i<=n; i++) for (j=1; j<=n-i; j++) if (v[j]>v[j+1]) { t=v[j]; v[j]=v[j+1]; v[j+1]=t }
        med = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
        printf "%-18s %8d %8d %10.2f %10.2f %10.2f %10.2f %8d\n", c, p, g, pf/n, med, v[1], v[n], rss
      }'
  done
} 2>&1 | tee "${root}/summary.txt"

# Reject any run that did not reach a natural end of turn.
bad=$(grep -L 'stop=endOfTurn' "${root}"/measured/*.stderr || true)
if [ -n "${bad}" ]; then
  echo "WARNING: runs without stop=endOfTurn:" >&2
  echo "${bad}" >&2
fi
echo "[${label}] complete" >&2
