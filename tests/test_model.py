import unittest
import numpy as np
from app.adapter import LoRA
from app.data import examples, HELD_OUT
from app.train import ARTIFACT


class SportsTests(unittest.TestCase):
    def test_data_split(self):
        train = {event for _, event, _ in examples()}
        test = {event for _, event, _ in examples(HELD_OUT)}
        self.assertEqual((len(train), len(test)), (48, 12))
        self.assertTrue(train.isdisjoint(test))

    def test_learning(self):
        x = np.eye(4, dtype="f")
        w = np.zeros((4, 4), dtype="f")
        model = LoRA(w, rank=4)
        before = model.logits(x)
        model.optimize(x, np.arange(4), epochs=1000)
        np.testing.assert_array_equal(w, 0)
        self.assertTrue(np.all(np.diag(model.logits(x)) > np.diag(before)))
        np.testing.assert_array_equal(model.logits(x).argmax(1), np.arange(4))

    def test_saved_adapter(self):
        with np.load(ARTIFACT) as saved:
            self.assertEqual(saved["a"].shape, (768, 8))
            self.assertEqual(saved["b"].shape, (8, len(saved["vocab"])))
            self.assertGreater(np.linalg.norm(saved["b"]), 0)

