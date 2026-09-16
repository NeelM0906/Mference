# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Independent execution of installed Swift-Qwen weights, without a second model.

Run the repository's safety checks and strict install verification first.
Run alone, after QuantizerQualityMeasurement has written the native corpus:

  uv run Scripts/swift_qwen_mlx_reference.py MODEL.gturbo NATIVE_DUMP_DIRECTORY

Prints full-vocabulary KL/top-1 comparisons to stdout. No weights are downloaded
or written. This verifies execution of the installed quantized tensors; it does
NOT establish BF16-to-INT4 quality, fine-tune quality, or MTP correctness.
The reference uses MLX-LM's unmodified qwen3_5 model and FP16 compute (FP32 A_log),
matching Mference's execution precision. Already-converted norms and convolution
axes must not pass through sanitize() a second time.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model, ModelArgs
import numpy as np

MODEL_ID = 'swift-qwen3.8-27b-int4g64'
# The source INDEX fingerprint (distinct from the chat template fingerprint).
SOURCE = '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df'


def installed_weights(path):
    payload = np.memmap(path / 'model_weights.bin', mode='r', dtype=np.uint8)
    index_size, resident_size, count = struct.unpack_from('<QQQ', payload)
    if index_size + resident_size != len(payload) or 24 + count * 72 > index_size:
        raise ValueError('invalid resident index bounds')
    tensors, packed = {}, set()

    def array(offset, size, dtype, shape):
        if offset < index_size or offset + size > len(payload):
            raise ValueError('tensor outside resident payload')
        result = np.frombuffer(payload, dtype=dtype, count=size // np.dtype(dtype).itemsize,
                               offset=offset).reshape(shape)
        return result

    def bf16(offset, size, shape):
        raw = array(offset, size, '<u2', shape).astype(np.uint32)
        return mx.array((raw << 16).view(np.float32)).astype(mx.float16)

    for i in range(count):
        fields = struct.unpack_from('<I H B x Q Q 4I Q Q Q Q', payload, 24 + i * 72)
        name_offset, name_size, dtype, offset, size = fields[:5]
        if name_offset + name_size > index_size:
            raise ValueError('tensor name outside index')
        name = bytes(payload[name_offset:name_offset + name_size]).decode('utf-8')
        shape = tuple(d for d in fields[5:9] if d)
        so, ss, bo, bs = fields[9:]
        if name.startswith('mtp.'):
            continue  # plain decode only; never attach the base checkpoint's head
        if name in tensors:
            raise ValueError('duplicate tensor name')
        if dtype == 0:
            rows, columns = shape
            if columns % 64 or size != rows * columns // 2 or ss != bs or ss != rows * columns // 32:
                raise ValueError(f'{name}: expected affine INT4 group-64')
            tensors[name] = mx.array(array(offset, size, '<u4', (rows, columns // 8)))
            prefix = name.removesuffix('.weight')
            tensors[prefix + '.scales'] = bf16(so, ss, (rows, columns // 64))
            tensors[prefix + '.biases'] = bf16(bo, bs, (rows, columns // 64))
            packed.add(prefix)
        elif dtype == 1:
            tensors[name] = bf16(offset, size, shape)
        elif dtype in (2, 3):
            value = mx.array(array(offset, size, '<f2' if dtype == 2 else '<f4', shape))
            tensors[name] = value if name.endswith('.A_log') else value.astype(mx.float16)
        else:
            raise ValueError(f'{name}: unexpected dtype {dtype}')
    return tensors, packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('install', type=Path)
    parser.add_argument('native_dump', type=Path)
    args = parser.parse_args()
    manifest_bytes = (args.install / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['modelID'] != MODEL_ID or manifest['sourceSnapshotHash'].removeprefix('sha256:') != SOURCE:
        raise ValueError('expected the pinned Swift-Qwen checkpoint, not base Qwen')
    if not (args.install / 'verified-install.json').is_file():
        raise ValueError('strictly verify the completed installation first')
    corpus = json.loads((args.native_dump / 'tokens.json').read_text())['items']
    meta = json.loads((args.native_dump / 'meta.json').read_text())
    if meta['modelID'] != MODEL_ID:
        raise ValueError('native dump belongs to another checkpoint')
    by_name = {item['name']: item for item in meta['items']}
    config = json.loads((args.install / 'tokenizer/config.json').read_text())
    weights, packed = installed_weights(args.install)
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda path, module: path in packed)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    del weights
    results = []
    for item in corpus:
        info = by_name[item['name']]
        native = np.memmap(args.native_dump / info['file'], dtype='<f2', mode='r').reshape(
            info['positions'], info['vocab'])
        if len(item['sequence']) != len(native):
            raise ValueError('native dump/token sequence length mismatch')
        cache = model.make_cache()
        divergences, agreement, maximum_errors = [], [], []
        native_continuation_consistent = True
        for position, token in enumerate(item['sequence']):
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1]
            mx.eval(logits)
            reference = np.asarray(logits.astype(mx.float32))
            candidate = np.asarray(native[position], dtype=np.float32)
            if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
                raise ValueError(f'non-finite logits in {item["name"]} at {position}')
            lp = reference - reference.max()
            lp -= np.log(np.exp(lp).sum())
            lq = candidate - candidate.max()
            lq -= np.log(np.exp(lq).sum())
            divergences.append(float(np.sum(np.exp(lp) * (lp - lq))))
            agreement.append(int(reference.argmax() == candidate.argmax()))
            maximum_errors.append(float(np.max(np.abs(reference - candidate))))
            if item['continuationStart'] - 1 <= position < len(item['sequence']) - 1:
                native_continuation_consistent &= int(candidate.argmax()) == item['sequence'][position + 1]
        continuation = agreement[item['continuationStart'] - 1:-1]
        prefix_match = next((i for i, same in enumerate(continuation) if not same), len(continuation))
        result = {'name': item['name'], 'positions': len(native),
                  'mean_kl_nats': float(np.mean(divergences)), 'max_kl_nats': max(divergences),
                  'top1_agreement': float(np.mean(agreement)), 'max_abs_logit_error': max(maximum_errors),
                  'continuation_top1_agreement': float(np.mean(continuation)) if continuation else None,
                  'greedy_prefix_match_tokens': prefix_match,
                  'native_continuation_consistent': bool(native_continuation_consistent)}
        if not native_continuation_consistent:
            raise ValueError(f'{item["name"]}: native teacher-forced logits do not reproduce its own greedy continuation')
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({'modelID': MODEL_ID, 'source_index_sha256': SOURCE,
                      'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                      'mlx_lm': importlib.metadata.version('mlx-lm'),
                      'mlx': importlib.metadata.version('mlx'), 'items': results}), flush=True)


if __name__ == '__main__':
    main()
