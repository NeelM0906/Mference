# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.31.3", "mlx==0.32.2", "mlx-metal==0.32.2", "numpy==2.5.3"]
# ///
"""Check the storage adapter against the complete upstream toy Gemma graph."""
import importlib.util
from pathlib import Path
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4_text as gemma
import numpy as np

spec = importlib.util.spec_from_file_location("reference", Path(__file__).parents[1] / "gemma_qat_mlx_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceTests(unittest.TestCase):
    def test_bounded_adapter_preserves_upstream_graph(self):
        config = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=32,
                      num_attention_heads=2, num_key_value_heads=2, num_global_key_value_heads=1,
                      head_dim=32, global_head_dim=64, vocab_size=64,
                      num_kv_shared_layers=0, hidden_size_per_layer_input=0,
                      sliding_window=128, layer_types=["sliding_attention", "full_attention"],
                      enable_moe_block=True, num_experts=4, top_k_experts=2,
                      moe_intermediate_size=32, attention_k_eq_v=True)
        mx.random.seed(42)
        source = gemma.Model(gemma.ModelArgs.from_dict(config))
        nn.quantize(source, group_size=32, bits=4,
                    class_predicate=lambda name, obj: hasattr(obj, "to_quantized") and not name.endswith("router.proj"))
        source.set_dtype(mx.float16)
        source.final_logit_softcapping = None
        mx.eval(source.parameters())

        class Reader:
            def __init__(self):
                self.entries = {"language_model." + key for key, _ in tree_flatten(source.parameters())}
                self.reads = []

            def resident(self, name, embedding=False):
                fields = name.removeprefix("language_model.").split(".")
                value = source
                for field in fields[:-1]:
                    value = value[int(field)] if isinstance(value, list) else value[field]
                if isinstance(value, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                    return value
                return value[fields[-1]]

            def expert(self, layer, expert):
                self.reads.append((layer, expert))
                switch = source.model.layers[layer].experts.switch_glu
                return {role: reference.module(nn.QuantizedLinear,
                            weight=switch[role + "_proj"].weight[expert],
                            scales=switch[role + "_proj"].scales[expert],
                            biases=switch[role + "_proj"].biases[expert],
                            group_size=32, bits=4, mode="affine")
                        for role in ("gate", "up", "down")}

        reader = Reader()
        adapter, softcap = reference.installed_model(reader, {"text_config": config})
        self.assertEqual(softcap, 30)
        expected_cache, actual_cache = source.make_cache(), adapter.make_cache()
        for token in [2, 17, 35]:
            expected = source(mx.array([[token]]), cache=expected_cache)
            actual = adapter(mx.array([[token]]), cache=actual_cache)
            mx.eval(expected, actual)
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.002, rtol=0.002)
        self.assertEqual(len(reader.reads), 3 * 2 * 2)
        self.assertTrue(all(0 <= expert < 4 for _, expert in reader.reads))

    def test_comparison_rejects_wrong_logits(self):
        values = np.array([1, 2, -3, 4], dtype=np.float16)
        self.assertTrue(reference.compare(values, values, 30)["passed"])
        self.assertFalse(reference.compare(values, -values, 30)["passed"])
        with self.assertRaises(ValueError):
            reference.compare(values, values * np.nan, 30)


if __name__ == "__main__":
    unittest.main()
