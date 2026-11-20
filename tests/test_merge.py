import unittest
from unittest.mock import patch
import numpy as np
from app.generator import Generator
from app.train import ARTIFACT


class MergeTests(unittest.TestCase):
    def test_runtime_merge(self):
        with np.load(ARTIFACT) as data:
            rng = np.random.default_rng(42)
            x = rng.normal(size=(2, 768)).astype("f")
            with patch("app.generator.GPT2") as base:
                base.return_value.w = {"wte.weight": np.broadcast_to(np.float32(0.25), (50257, 768))}
                model = Generator()
            reference = x @ model.weight + float(data["scale"]) * ((x @ data["a"]) @ data["b"])
            np.testing.assert_allclose(model.logits(x), reference, rtol=1e-4, atol=1e-4)

