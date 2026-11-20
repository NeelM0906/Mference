# Speech models and SDK (Notes)

Nothing here is bundled in the app. The app downloads these files on the
person's request (Notes › setup card, Settings › Speech), from huggingface.co
only, verifies each file's size and SHA-256 against
`Sources/MferenceAppCore/Speech/SpeechModelPins.swift` (generated from the
verified download of 2026-11-20), and keeps them under
`~/Library/Application Support/Mference/Speech/<repo>/<revision>/`.

## SDK

- FluidAudio (FluidInference/FluidAudio), Apache 2.0, pinned `exact: "0.18.4"`
  (tag commit `e0c6103bc87a88d25276cad491c86e757f58c865`), traits
  `["Diarizer"]` (no TTS, no prebuilt NemoTextProcessing binary). Spike S1
  showed it loads Fluid Speech Pro as `.v3` with `AsrModels.loadLocal`.

## Speech package (dictation, meetings, files): 653,388,064 bytes

| Model | Repository @ revision | Licence | Bytes |
| --- | --- | --- | --- |
| Fluid Speech Pro, Core ML | `FluidVoice-Labs/fluid-speech-pro-coreml` @ `3b9d38a93d606f84011e5a0a669b7ea146375a22` | CC BY 4.0 | 652,324,639 |
| Silero VAD unified 256 ms v6.2.1 | `FluidInference/silero-vad-coreml` @ `b419383c55c110e2c9271fa6ee0ea83d03c70d96` | MIT | 1,063,425 |

Required credit, shown in Settings › Speech and the acknowledgements:
**"Fluid Speech Pro by FluidVoice"**, built on NVIDIA Parakeet TDT 0.6B v3 and
Moondream Parakeet Ultra (both CC BY 4.0). The repository's `LICENSE` and
`NOTICE` are downloaded with the model.

Install note: the joint model `JointDecision.mlmodelc` is installed as
`JointDecisionv3.mlmodelc`, the name FluidAudio's `.v3` loader expects. It has
no top-K outputs, so the app never passes a language hint to the decoder.

## Speaker labels package (meetings, optional): 66,627,357 bytes

| Model | Repository @ revision | Licence | Bytes |
| --- | --- | --- | --- |
| Offline diarization, pyannote Community-1 set (Segmentation, FBank, Embedding, PLDA, PldaRho, PLDA parameters) | `FluidInference/speaker-diarization-coreml` @ `7eab74f3463c736418d0eb10296261ae581146c9` | CC BY 4.0 (scoped in its `NOTICE.md`) | 22,008,379 |
| Live diarization, LS-EEND AMI variant, 100 ms step (`optimized/ami/100ms/ls_eend_ami_100ms.mlmodelc`, config embedded) | `FluidInference/ls-eend-coreml` @ `28ce1b1f8ef186729df63b3886fbaae7bc10c4a1` | MIT | 44,618,978 |

Attribution for the Community-1 set (per its NOTICE.md): pyannote, WeSpeaker,
BUT Speech@FIT and Fluid Inference, CC BY 4.0, modified Core ML conversions.
The legacy `pyannote_segmentation` / `wespeaker*` files are not downloaded.
Of LS-EEND only the 100 ms AMI export and `LICENSE` are pinned (repinned
2026-11-20 after C-N1 run 1): FluidAudio 0.18.4's `LSEENDModel` loads only the
`optimized/<variant>/<step>/ls_eend_<variant>_<step>.mlmodelc` layout and reads
its config from the model metadata, and 100 ms is its default step. The older
`AMI/ls_eend_ami_step` export (no embedded config, other state tensors), its
`ls_eend_ami_step.json` sidecar and the stray `AMI/.DS_Store` are not
downloaded, nor are the other steps and variants.
