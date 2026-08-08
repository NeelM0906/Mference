# Maple parity command

`MferenceMapleParity` is the narrow command-line surface for the parity core.
Acceptance exports use the wrapper so the recorded binary is rebuilt from the
same clean checkout revision immediately before the model run:

```text
Scripts/run_maple_mference_teacher_forcing.sh preflight --model scratch/maple.gturbo --repository .
Scripts/run_maple_mference_teacher_forcing.sh export --model scratch/maple.gturbo --corpus raven.txt --output trace.jsonl --repository .
MferenceMapleParity compare mlx-trace.jsonl mference-trace.jsonl
```

`export --max-positions N` is limited to a diagnostic prefix below the fixed
1,639-position acceptance trace and cannot produce an acceptable result. The
repository is resolved with Git before the clean-check requirement is applied.
The command rejects duplicate or unknown options and never overwrites a trace,
downloads, deletes, or terminates a model process.
