# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Sample pinned BF16 ranges; compare installed bytes and independent MLX INT4.

No full checkpoint download. Run after strict install verification:
  uv run Scripts/swift_qwen_source_gate.py MODEL.gturbo --cache SAMPLE_DIRECTORY

Reports reconstruction error, not model quality. Does not run inference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request

import mlx.core as mx
import numpy as np

REVISION = '1b30aaaf753fe5c1cb51ada2ea0367a53445359c'
INDEX_SHA = '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df'
MODEL_ID = 'swift-qwen3.8-27b-int4g64'


def bounded_fetch(url, output, rng=None, retries=4):
    """Require exact 206 ranges; never accept a full shard in place of a range."""
    expected = None
    headers = {}
    if rng:
        first, last = map(int, rng.split('-'))
        expected = last - first + 1
        if expected <= 0 or expected > 8 * 1024 * 1024:
            raise ValueError('sample/header range exceeds 8 MiB cap')
        headers['Range'] = 'bytes=' + rng
    elif not url.endswith('.json'):
        raise ValueError('only JSON metadata may be fetched without a range')
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if rng and (response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes ' + rng + '/')):
                    raise ValueError('server did not honor the exact byte range')
                data = response.read((expected or 8 * 1024 * 1024) + 1)
                if (expected is not None and len(data) != expected) or len(data) > 8 * 1024 * 1024:
                    raise ValueError('unexpected response size')
                Path(output).write_bytes(data)
                return
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('install', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.install / 'manifest.json').read_text())
    if manifest['modelID'] != MODEL_ID or manifest['sourceSnapshotHash'].removeprefix('sha256:') != INDEX_SHA:
        raise ValueError('expected the pinned Swift-Qwen install')
    if not (args.install / 'verified-install.json').is_file():
        raise ValueError('verify the complete install first')
    args.cache.mkdir(parents=True, exist_ok=True)
    module_path = Path(__file__).with_name('quantizer-weight-gate.py')
    spec = importlib.util.spec_from_file_location('weight_gate', module_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    # Reuse the established safetensors parser and encoder transcription, but
    # enforce bounded downloads instead of that script's general curl helper.
    gate.curl = bounded_fetch
    source = gate.Repo(f'https://huggingface.co/ukisai/Swift-Qwen3.8-27b/resolve/{REVISION}',
                       str(args.cache), 'swift')
    if hashlib.sha256((args.cache / 'swift.index.json').read_bytes()).hexdigest() != INDEX_SHA:
        raise ValueError('source index fingerprint mismatch')
    payload = np.memmap(args.install / 'model_weights.bin', mode='r', dtype=np.uint8)
    index_size, resident_size, count = struct.unpack_from('<QQQ', payload)
    if index_size + resident_size != len(payload) or 24 + count * 72 > index_size:
        raise ValueError('invalid resident index')
    entries = {}
    for i in range(count):
        row = struct.unpack_from('<I H B x Q Q 4I Q Q Q Q', payload, 24 + i * 72)
        if row[0] + row[1] > index_size:
            raise ValueError('invalid name bounds')
        name = bytes(payload[row[0]:row[0] + row[1]]).decode()
        entries[name] = row

    def read(offset, size, dtype):
        if offset < index_size or offset + size > len(payload):
            raise ValueError('tensor outside resident payload')
        return np.frombuffer(payload, dtype=dtype, count=size // np.dtype(dtype).itemsize, offset=offset)

    projections = ['model.language_model.embed_tokens.weight', 'lm_head.weight', 'mtp.fc.weight']
    for layer in [0, 3, 31, 63]:
        prefix = f'model.language_model.layers.{layer}.'
        projections += [prefix + f'mlp.{kind}_proj.weight' for kind in ['gate', 'up', 'down']]
        projections += [prefix + suffix for suffix in (
            ['linear_attn.in_proj_qkv.weight', 'linear_attn.out_proj.weight'] if layer == 0 else
            ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight'])]
    vectors = ['model.language_model.norm.weight', 'mtp.norm.weight', 'mtp.pre_fc_norm_hidden.weight',
               'model.language_model.layers.0.input_layernorm.weight',
               'model.language_model.layers.3.self_attn.q_norm.weight',
               'model.language_model.layers.0.linear_attn.norm.weight',
               'model.language_model.layers.0.linear_attn.conv1d.weight',
               'model.language_model.layers.0.linear_attn.A_log']
    results = []
    for name in projections + vectors:
        target = name.replace('model.language_model.', 'language_model.model.')
        if name == 'lm_head.weight':
            target = 'language_model.lm_head.weight'
        row = entries[target]
        _, _, dtype, offset, size, *rest = row
        shape = tuple(d for d in rest[:4] if d)
        so, ss, bo, bs = rest[4:]
        tag = re.sub('[^a-zA-Z0-9]', '_', name)
        if name in projections:
            info = source.info(name)
            if info['dtype'] != 'BF16' or tuple(info['shape']) != shape or dtype != 0:
                raise ValueError(f'{name}: source/target shape or dtype mismatch')
            start, rows, columns = 17, 8, shape[1]
            path, _, _, _ = source.rows(name, start, rows, tag)
            values = gate.bf16_to_f32(np.fromfile(path, dtype='<u2')).reshape(rows, columns)
            packed, scales, biases, _ = gate.encode_ours(values.reshape(-1, 64))
            actual = read(offset + start * columns // 2, rows * columns // 2, 'u1')
            actual_s = read(so + start * columns // 32, rows * columns // 32, '<u2')
            actual_b = read(bo + start * columns // 32, rows * columns // 32, '<u2')
            exact = (np.array_equal(actual, packed.ravel()) and np.array_equal(actual_s, scales.ravel())
                     and np.array_equal(actual_b, biases.ravel()))
            if not exact:
                raise ValueError(f'{name}: installed bytes differ from the source-row encoder')
            q = mx.array(actual.view('<u4').reshape(rows, columns // 8))
            s = mx.array(gate.bf16_to_f32(actual_s).reshape(rows, columns // 64))
            b = mx.array(gate.bf16_to_f32(actual_b).reshape(rows, columns // 64))
            ours = np.asarray(mx.dequantize(q, s, b, group_size=64, bits=4))
            independent = mx.quantize(mx.array(values).astype(mx.bfloat16), group_size=64, bits=4)
            # Compare both quantizers' grids in FP32, without charging only
            # the independent side for a BF16-rounded dequantized output.
            control = np.asarray(mx.dequantize(independent[0], independent[1].astype(mx.float32),
                                              independent[2].astype(mx.float32), group_size=64, bits=4))
            denom = float(np.linalg.norm(values))
            result = {'name': name, 'rows': rows, 'source_row_start': start, 'bytes_exact': exact,
                      'relative_error_ours': float(np.linalg.norm(ours - values) / denom),
                      'relative_error_mlx': float(np.linalg.norm(control - values) / denom)}
        else:
            info = source.info(name)
            if int(np.prod(info['shape'])) * gate.DTYPE_BYTES[info['dtype']] > 1024 * 1024:
                raise ValueError('vector/conv sample exceeds cap')
            original = source.tensor(name, tag).reshape(-1)
            fold = ((name.endswith('norm.weight') and '.linear_attn.' not in name)
                    or name in ('mtp.pre_fc_norm_hidden.weight', 'mtp.pre_fc_norm_embedding.weight'))
            expected = gate.bf16_bits(gate.bf16_to_f32(original) + 1) if fold else original
            expected_bytes = expected.tobytes()
            if bytes(payload[offset:offset + size]) != expected_bytes:
                raise ValueError(f'{name}: passthrough/norm-fold mismatch')
            result = {'name': name, 'bytes_exact': True, 'folded_plus_one': fold}
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({'modelID': MODEL_ID, 'revision': REVISION, 'samples': len(results),
                      'all_installed_bytes_exact': True, 'items': results}), flush=True)


if __name__ == '__main__':
    main()
