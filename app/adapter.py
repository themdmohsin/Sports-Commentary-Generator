import numpy as np

from .base import softmax


class LoRA:
    """W_eff = W + (alpha/r) AB. Only A and B receive gradients."""
    def __init__(self, weight, rank=8):
        rng = np.random.default_rng(14)
        self.weight = weight
        self.a = rng.normal(0, .01, (weight.shape[0], rank)).astype("f")
        self.b = np.zeros((rank, weight.shape[1]), dtype="f")
        self.scale = 2.0

    def logits(self, x):
        return x @ self.weight + self.scale * ((x @ self.a) @ self.b)

    def optimize(self, x, y, epochs=180):
        # Full-batch Adam on the commentary tokens only.
        params = [self.a, self.b]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        base = x @ self.weight
        for step in range(1, epochs + 1):
            hidden = x @ self.a
            error = softmax(base + self.scale * (hidden @ self.b))
            error[np.arange(len(y)), y] -= 1
            error *= self.scale / len(y)
            gradients = [x.T @ (error @ self.b.T), hidden.T @ error]
            for i, grad in enumerate(gradients):
                m[i] = .9 * m[i] + .1 * grad
                v[i] = .999 * v[i] + .001 * grad**2
                params[i] -= .003 * (m[i] / (1-.9**step)) / (np.sqrt(v[i] / (1-.999**step)) + 1e-8)

