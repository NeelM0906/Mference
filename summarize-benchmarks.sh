#!/bin/bash
# Recompute one authoritative table per model from every measured run's stderr,
# independent of how many run-benchmark.sh invocations produced them.
# usage: ./summarize-benchmarks.sh
set -uo pipefail
cd "$(dirname "$0")"

for dir in benchmark-results/*/; do
  label=$(basename "${dir}")
  [ -d "${dir}/measured" ] || continue
  echo "## ${label}"
  printf '%-18s %7s %7s %9s %9s %9s %9s %8s %5s\n' \
    case prompt gen prefill_s median min max peakMiB runs
  for case_id in short-explanation medium-review long-synthesis; do
    files=$(ls "${dir}"/measured/${case_id}*.stderr 2>/dev/null || true)
    [ -n "${files}" ] || continue
    {
      for err in ${files}; do
        grep -h '^\[stop=' "${err}" | sed -E \
          's/.*prefill=([0-9]+)tok\/([0-9.]+)s new=([0-9]+)tok.*tok\/s=([0-9.]+).*/D \1 \2 \3 \4/'
        awk '/maximum resident set size/ { printf "RSS %.0f\n", $1/1048576 }' "${err}"
      done
    } | awk -v c="${case_id}" '
      $1=="RSS" { if ($2>rss) rss=$2; next }
      $1=="D"  { p=$2; pf+=$3; g=$4; v[++n]=$5 }
      END {
        if (n==0) next
        for (i=1;i<=n;i++) for (j=1;j<=n-i;j++) if (v[j]>v[j+1]) { t=v[j];v[j]=v[j+1];v[j+1]=t }
        med = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
        printf "%-18s %7d %7d %9.2f %9.2f %9.2f %9.2f %8d %5d\n", c,p,g,pf/n,med,v[1],v[n],rss,n
      }'
  done
  # Protocol gate: every measured run must reach a natural end of turn.
  total=$(ls "${dir}"/measured/*.stderr 2>/dev/null | wc -l | tr -d ' ')
  ok=$(grep -l 'stop=endOfTurn' "${dir}"/measured/*.stderr 2>/dev/null | wc -l | tr -d ' ')
  echo "stop=endOfTurn: ${ok}/${total} measured runs"
  echo
done
