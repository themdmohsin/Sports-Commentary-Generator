import re
import numpy as np

from .base import GPT2
from .data import ACTIONS, prompt
from .train import ARTIFACT, EOS


class Generator:
    def __init__(self):
        self.base = GPT2()
        with np.load(ARTIFACT) as saved:
            self.vocab = saved["vocab"]
            a, b, scale = saved["a"], saved["b"], saved["scale"]
        self.weight = self.base.w["wte.weight"][self.vocab].T.copy()

        self.merged = self.weight + float(scale) * (a @ b)

    def logits(self, x):
        return x @ self.merged

    def generate(self, event):
        event = event.strip()
        if not event or len(event) > 240:
            raise ValueError("Enter an event between 1 and 240 characters.")
        prefix = self.base.encode(prompt(event))
        # Ground known plays in the supplied player's name. Score both calls
        # token by token with the adapted model, including unseen name pieces.
        for _, action, first, second in ACTIONS:
            match = re.fullmatch(r"(.{1,60}?)\s+" + re.escape(action) + r"\b(.*)", event, re.I)
            if match:
                calls = [line.format(p=match[1]) for line in (first, second)]
                line = max(calls, key=lambda c: self.score(prefix, c))
                score = re.search(r"\b(\d{1,3}\s*[-:]\s*\d{1,3})\b", match[2])
                return line + (f" Score: {score[1]}." if score else "")
        result = []
        for _ in range(28):
            x = self.base.hidden(prefix + result)[-1]
            token = int(self.vocab[np.argmax(self.logits(x))])
            if token == EOS:
                break
            result.append(token)
            line = self.base.tokenizer.decode(result).strip()
            if line.endswith((".", "!", "?")):
                break
        return self.base.tokenizer.decode(result).strip()

    def score(self, prefix, line, adapted=True):
        target = self.base.encode(" " + line) + [EOS]
        x = self.base.hidden(prefix + target[:-1])[len(prefix)-1:]
        logits = self.logits(x) if adapted else x @ self.weight
        missing = sorted(set(target) - set(self.vocab))
        vocab = self.vocab.tolist() + missing
        if missing:
            logits = np.column_stack((logits, x @ self.base.w["wte.weight"][missing].T))
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(log_probs[np.arange(len(target)), [vocab.index(t) for t in target]].mean())

