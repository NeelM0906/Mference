#!/usr/bin/env bash
# Decide how much of the Swift suite a pull request needs (engine program spec
# section 12, change 2). Reads changed paths, one per line, on stdin and prints
# one word:
#
#   docs  every path is documentation; only the docs job runs.
#   t0    every path is engine, server, llama.cpp, script or documentation
#         code; the suite runs without MferenceTestsCore.
#   full  anything else (runtime, kernels, installer, CLI, Package.swift,
#         workflows, vendored packages, test support); the default suite runs.
#
# Why t0 may skip MferenceTestsCore: that target tests Mference,
# MferenceValidationSupport, MferenceRepackCore and MferenceCLICore. None of
# them depends on MferenceEngine, LlamaCpp or MferenceServerCore, so a change
# confined to those targets cannot change what MferenceTestsCore exercises.
# The exception is MetalContextDeviceCreationTests, which scans every Swift
# file under Sources/; the t0 step in ci.yml runs that suite as well.
# Documentation counts as docs only when it is Markdown outside the code trees
# and nothing that runs in CI names it or one of its parent directories (tests
# read docs/FAMILY_CONTRACT.md, benchmark prompt sets under docs/, and so on).
# Run it from the repository root so that check can search the tree. Every
# other file under docs/ or site/ counts as code.
# An empty list prints "full", so a failed file listing never narrows CI.

set -euo pipefail

# Paths whose files the test job executes or reads. .github is left out: a
# change there is always full, and the docs job runs on every pull request.
reference_roots=(Tests Scripts Sources Vendor Package.swift bringup-check.sh mference-ui.sh)

# True when CI code names the path, or one of its parent directories below the
# top level (docs/ alone is named everywhere and says nothing). A failed search
# counts as a reference, so it can only widen CI.
referenced() {
  local candidate="$1" status
  while :; do
    status=0
    git grep -q -F -e "$candidate" -- "${reference_roots[@]}" 2>/dev/null || status=$?
    if [[ "$status" -ne 1 ]]; then
      return 0
    fi
    [[ "$candidate" == */*/* ]] || return 1
    candidate="${candidate%/*}"
  done
}

scope=docs
seen=0
while IFS= read -r path || [[ -n "$path" ]]; do
  [[ -z "$path" ]] && continue
  seen=1
  case "$path" in
    Sources/*|Tests/*|Vendor/*|Scripts/*)
      kind=code ;;
    *.md|.github/ISSUE_TEMPLATE/*)
      kind=docs ;;
    *)
      kind=code ;;
  esac
  if [[ "$kind" == docs ]] && referenced "$path"; then
    kind=code
  fi
  if [[ "$kind" == docs ]]; then
    continue
  fi
  case "$path" in
    Sources/MferenceEngine/*|Tests/MferenceEngine/*|\
    Sources/LlamaCpp/*|Tests/LlamaCpp/*|\
    Sources/MferenceServer/*|Tests/MferenceServer/*|\
    mference-ui.sh)
      kind=t0 ;;
    Scripts/test.sh)
      kind=full ;;
    Scripts/*)
      kind=t0 ;;
    *)
      kind=full ;;
  esac
  if [[ "$kind" == full ]]; then
    scope=full
    break
  fi
  scope=t0
done

if [[ "$seen" -eq 0 ]]; then
  scope=full
fi
echo "$scope"
