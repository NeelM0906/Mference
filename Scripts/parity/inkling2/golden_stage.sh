#!/bin/bash
# golden_stage.sh: one stage of the K0 weight-loading goldens (greedy
# continuations, top-20 log-probabilities, the 48-visit trace), through
# guarded_run.sh (swap guard; not a timing run, the power source is recorded).
#
#   golden_stage.sh <build-dir> <goldens-dir> <stage>
#
# Stages, each its own lock hold:
#   metal      ink-golden generate, -ngl 99 (the default GPU offload): the goldens
#   metal-r2   the same again in a fresh process: the reference's run-to-run noise
#   metal5     trace02's prompt on Metal without the trace (inputs/generate_prompts_5.json,
#              log-probs at every position): does the trace's eval callback change
#              Metal's numbers? Writes generate5/metal
#   cpu        the same with -ngl 0: ink-golden then loads no GPU device, turns
#              op offload and CPU weight repacking off, so every op runs on the
#              CPU from the mapped file: the backend noise
#   metal2     the second prompt set (inputs/generate_prompts_2.json: more log-prob
#              prompts, ~1,300 and ~2,600 tokens, a chat-formatted trace) on Metal
#   cpu2       the second set CPU-only (the G6a reference for trace02)
#   metal3/cpu3 trace of the lp05 tool prompt (300 tokens) on each backend, to
#              locate where Metal and CPU-only diverge on its tool JSON
#   tf-metal/tf-cpu teacher-forced: every log-prob golden's prompt + generated
#              ids as one prompt (inputs/generate_prompts_tf.json, from
#              make_tf_prompts.py), so the noise covers every position
#   cpu4       trace01b CPU-only (inputs/generate_prompts_4.json): trace01's
#              prompt ids plus its greedy token 152717 (" Paris") as one ids
#              prompt, traced, log-probs at every position: the G6a record of
#              position 5, the position a runner reaches by one decode step
# Set 1 writes generate/<stage>, set 2 generate2/<stage without the 2>.
# All use n_ctx 4096, n_ubatch 512, 8 threads.
set -uo pipefail
B=${1:?build dir}; G=${2:?goldens dir}; STAGE=${3:?stage}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/paths.sh"; GGUF=$INKLING_GGUF
IN=$HERE/inputs/generate_prompts.json; OUTD=$G/generate/$STAGE
case "$STAGE" in
  metal|metal-r2) ngl=99 ;;
  cpu) ngl=0 ;;
  metal5) ngl=99; IN=$HERE/inputs/generate_prompts_5.json; OUTD=$G/generate5/metal ;;
  metal2) ngl=99; IN=$HERE/inputs/generate_prompts_2.json; OUTD=$G/generate2/metal ;;
  cpu2) ngl=0; IN=$HERE/inputs/generate_prompts_2.json; OUTD=$G/generate2/cpu ;;
  metal3) ngl=99; IN=$HERE/inputs/generate_prompts_3.json; OUTD=$G/generate3/metal ;;
  cpu3) ngl=0; IN=$HERE/inputs/generate_prompts_3.json; OUTD=$G/generate3/cpu ;;
  cpu4) ngl=0; IN=$HERE/inputs/generate_prompts_4.json; OUTD=$G/generate4/cpu ;;
  tf-metal) ngl=99; IN=$HERE/inputs/generate_prompts_tf.json; OUTD=$G/generate-tf/metal ;;
  tf-cpu) ngl=0; IN=$HERE/inputs/generate_prompts_tf.json; OUTD=$G/generate-tf/cpu ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
exec "$HERE/guarded_run.sh" "$OUTD/log" "$GGUF" -- \
  "$B/ink-golden" generate --model "$GGUF" --in "$IN" \
  --out "$OUTD" --ngl "$ngl" --threads 8 --n-ctx 4096 --ubatch 512 --top 20
