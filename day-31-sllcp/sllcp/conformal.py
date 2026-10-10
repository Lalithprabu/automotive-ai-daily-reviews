"""Split label-conditional conformal prediction, with and without scene localization.

Nonconformity score for (x, y):  s(x, y) = 1 - p_y(x)     (p = predictor's softmax)

Label-conditional: a separate calibration set per true label y, so the guarantee is
    P( y in C(X) | Y = y ) >= 1 - alpha            (per class; vital when 'unsafe' is rare)

Localized: calibration points are re-weighted by similarity of their scene embedding to the
test scene, w_i(x) = exp(-||z(x) - z_i||^2 / (2 h^2)), and the (1-alpha) quantile is taken of the
WEIGHTED empirical score distribution, with the test point's own unit weight placed at +inf
(the standard weighted/localized-CP construction). Weights are normalised:
    p_i(x) = w_i / (sum_j w_j + 1),      p_inf(x) = 1 / (sum_j w_j + 1)
"""
import torch


class SplitLabelConditionalCP:
    """Standard (non-localized) label-conditional split CP -- the 'old way'."""

    def __init__(self, alpha: float = 0.1, n_classes: int = 2):
        self.alpha, self.K = alpha, n_classes
        self.q = None

    def fit(self, probs: torch.Tensor, y: torch.Tensor):
        # probs (N, K), y (N,)
        self.q = torch.full((self.K,), float("inf"))
        for k in range(self.K):
            s = 1.0 - probs[y == k, k]  # (n_k,)
            n = s.numel()
            level = min(1.0, (n + 1) * (1 - self.alpha) / n)  # finite-sample correction
            self.q[k] = torch.quantile(s, level) if n > 0 else float("inf")
        return self

    def predict_sets(self, probs: torch.Tensor) -> torch.Tensor:
        # returns bool (B, K): label k in prediction set iff s(x, k) <= q_k
        return (1.0 - probs) <= self.q[None]


class SplitLabelLocalizedCP:
    """SLLCP-style: label-conditional + scene-localized weighted quantiles."""

    def __init__(self, alpha: float = 0.1, bandwidth: float = 0.5, n_classes: int = 2, chunk: int = 2048):
        self.alpha, self.h, self.K, self.chunk = alpha, bandwidth, n_classes, chunk
        self.cal = []  # per class: (sorted scores (n_k,), embeddings (n_k, D))

    def fit(self, probs: torch.Tensor, emb: torch.Tensor, y: torch.Tensor):
        self.cal = []
        for k in range(self.K):
            m = y == k
            s = 1.0 - probs[m, k]
            order = torch.argsort(s)
            self.cal.append((s[order], emb[m][order]))
        return self

    def _local_quantile(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """Weighted (1-alpha) quantile of class-k calibration scores around each test embedding z (B, D)."""
        s, ez = self.cal[k]
        if s.numel() == 0:  # no calibration data for this label -> fail-safe: always include it
            return torch.full((z.shape[0],), float("inf"))
        out = torch.empty(z.shape[0])
        for i in range(0, z.shape[0], self.chunk):
            zz = z[i:i + self.chunk]
            d2 = torch.cdist(zz, ez) ** 2  # (b, n_k)
            w = torch.exp(-d2 / (2 * self.h ** 2))  # (b, n_k) kernel weights
            p = w / (w.sum(1, keepdim=True) + 1.0)  # +1 = test point's mass at +inf
            cum = torch.cumsum(p, 1)  # CDF over ascending scores
            ok = cum >= (1 - self.alpha)
            first = ok.float().argmax(1)  # first index reaching level
            has = ok.any(1)
            q = s[first]
            out[i:i + self.chunk] = torch.where(has, q, torch.full_like(q, float("inf")))
        return out

    def quantiles(self, emb: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._local_quantile(emb, k) for k in range(self.K)], 1)  # (B, K)

    def predict_sets(self, probs: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return (1.0 - probs) <= self.quantiles(emb)  # bool (B, K)
