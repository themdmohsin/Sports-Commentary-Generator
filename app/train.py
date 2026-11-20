from pathlib import Path
import numpy as np

from .adapter import LoRA
from .base import GPT2
from .data import examples, prompt

ARTIFACT = Path(__file__).with_name("sports_adapter.npz")
EOS = 50256


def train():
    model = GPT2()
    features, targets = [], []
    for _, event, line in examples():
        prefix = model.encode(prompt(event))
        target = model.encode(" " + line) + [EOS]
        ids = prefix + target[:-1]
        features.append(model.hidden(ids)[len(prefix)-1:])
        targets.extend(target)
    vocab, labels = np.unique(targets, return_inverse=True)
    adapter = LoRA(model.w["wte.weight"][vocab].T)
    adapter.optimize(np.concatenate(features), labels)
    np.savez_compressed(ARTIFACT, a=adapter.a, b=adapter.b, vocab=vocab, scale=adapter.scale)


if __name__ == "__main__":
    train()

