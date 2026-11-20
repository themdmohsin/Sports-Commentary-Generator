"""Frozen GPT-2 small forward pass. Weights are memory-mapped read-only."""
import json
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

MODEL = "gpt2"
REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class GPT2:
    def __init__(self):
        options = dict(repo_id=MODEL, revision=REV, allow_patterns=["tokenizer.json", "model.safetensors"])
        try:
            root = Path(snapshot_download(**options, local_files_only=True))
            if not all((root / name).is_file() for name in options["allow_patterns"]):
                raise FileNotFoundError
        except (OSError, FileNotFoundError):
            root = Path(snapshot_download(**options))
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        path = root / "model.safetensors"
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        self.w = {name: np.memmap(path, dtype="<f4", mode="r", shape=tuple(info["shape"]),
                                  offset=8 + size + info["data_offsets"][0])
                  for name, info in header.items() if name != "__metadata__" and info["dtype"] == "F32"}

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def norm(self, x, key):
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + 1e-5) * self.w[key+".weight"] + self.w[key+".bias"]

    def linear(self, x, key):
        return x @ self.w[key+".weight"] + self.w[key+".bias"]

    def hidden(self, ids):
        n = len(ids)
        x = self.w["wte.weight"][ids] + self.w["wpe.weight"][:n]
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        for layer in range(12):
            key = f"h.{layer}."
            qkv = self.linear(self.norm(x, key+"ln_1"), key+"attn.c_attn")
            q, k, v = [m.reshape(n, 12, 64).transpose(1, 0, 2) for m in np.split(qkv, 3, axis=-1)]
            attention = softmax(np.where(mask, -1e9, q @ k.transpose(0, 2, 1) / 8))
            x = x + self.linear((attention @ v).transpose(1, 0, 2).reshape(n, 768), key+"attn.c_proj")
            h = self.linear(self.norm(x, key+"ln_2"), key+"mlp.c_fc")
            h = 0.5 * h * (1 + np.tanh(np.sqrt(np.float32(2 / np.pi)) * (h + 0.044715 * h**3)))
            x = x + self.linear(h, key+"mlp.c_proj")
        return self.norm(x, "ln_f")

