# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Tiny resident-format/reference-reader tests; no model download or inference."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

script = Path(__file__).resolve().parents[1] / 'swift_qwen_mlx_reference.py'
spec = importlib.util.spec_from_file_location('swift_qwen_reference', script)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
source_spec = importlib.util.spec_from_file_location('source_gate', script.with_name('swift_qwen_source_gate.py'))
source_gate = importlib.util.module_from_spec(source_spec)
source_spec.loader.exec_module(source_gate)


def fixture(folder):
    index_size = 4096
    names = ['language_model.lm_head.weight', 'language_model.model.norm.weight',
             'language_model.model.layers.0.linear_attn.A_log', 'mtp.norm.weight']
    blob = bytearray(index_size)
    offset = 24 + 72 * len(names)
    entries = []
    for name in names:
        encoded = name.encode()
        blob[offset:offset + len(encoded)] = encoded
        entries.append((offset, len(encoded)))
        offset += len(encoded)
    words = np.full((2, 8), 0x76543210, dtype='<u4').tobytes()
    scales, biases = struct.pack('<HH', 0x3f80, 0x3f80), bytes(4)
    norms, a_log, mtp = struct.pack('<HH', 0x3f80, 0xbf00), struct.pack('<f', 1.00001), bytes(2)
    blob.extend(words + scales + biases + norms + a_log + mtp)
    struct.pack_into('<QQQ', blob, 0, index_size, len(blob) - index_size, len(names))
    geometry = [
        (0, 4096, 64, (2, 64, 0, 0), 4160, 4, 4164, 4),
        (1, 4168, 4, (2, 0, 0, 0), 0, 0, 0, 0),
        (3, 4172, 4, (1, 0, 0, 0), 0, 0, 0, 0),
        (1, 4176, 2, (1, 0, 0, 0), 0, 0, 0, 0),
    ]
    for i, (name_offset, name_size) in enumerate(entries):
        dtype, start, size, shape, so, ss, bo, bs = geometry[i]
        struct.pack_into('<I H B x Q Q 4I Q Q Q Q', blob, 24 + 72 * i,
                         name_offset, name_size, dtype, start, size, *shape, so, ss, bo, bs)
    (folder / 'model_weights.bin').write_bytes(blob)


class ReferenceReaderTests(unittest.TestCase):
    def test_source_ranges_never_accept_full_shard_responses(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'sample'
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaises(ValueError):
                    source_gate.bounded_fetch('https://huggingface.co/model.safetensors', output, rng='0-7')
            response.read.assert_not_called()
            self.assertFalse(output.exists())

    def test_source_fetch_caps_ranges_before_network_access(self):
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(ValueError):
                source_gate.bounded_fetch('https://huggingface.co/model.safetensors', '/unused', rng='0-999999999')
            with self.assertRaises(ValueError):
                source_gate.bounded_fetch('https://huggingface.co/model.safetensors', '/unused')
            network.assert_not_called()

    def test_packed_nibbles_companions_precision_and_mtp_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture(folder)
            weights, packed = reference.installed_weights(folder)
            prefix = 'language_model.lm_head'
            self.assertEqual(packed, {prefix})
            unpacked = mx.dequantize(weights[prefix + '.weight'], weights[prefix + '.scales'],
                                    weights[prefix + '.biases'], group_size=64, bits=4)
            np.testing.assert_array_equal(np.asarray(unpacked), np.tile(np.arange(8), (2, 8)))
            np.testing.assert_array_equal(np.asarray(weights['language_model.model.norm.weight']), [1, -.5])
            self.assertEqual(weights['language_model.model.layers.0.linear_attn.A_log'].dtype, mx.float32)
            self.assertNotIn('mtp.norm.weight', weights)

    def test_truncated_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture(folder)
            path = folder / 'model_weights.bin'
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                reference.installed_weights(folder)


if __name__ == '__main__':
    unittest.main()
