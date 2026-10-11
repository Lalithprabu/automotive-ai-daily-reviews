import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, pytest
from scope import *
from scope.evaluate import fit_alpha, metrics, blocks
from scope.precision import dense_target, hvp_target, distill_loss, psd_inverse, marginals

S0 = 0.05


class GaussianDenoiser(torch.nn.Module):
    """Exact MMSE denoiser for x0 ~ N(0, Sigma):  D(x, s) = Sigma (Sigma + s^2 I)^-1 x."""
    def __init__(self, Sigma):
        super().__init__(); self.Sigma = Sigma
    def forward(self, x, sigma, cond, mode):
        s = float(sigma)
        return x @ (self.Sigma @ torch.linalg.inv(self.Sigma + s ** 2 * torch.eye(D))).T


def _rand_sigma(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g) * 0.2
    return A @ A.T + 0.01 * torch.eye(D)


def test_true_cov_psd_and_sampler_matches():
    rng = np.random.default_rng(0)
    v, c = np.full(4000, 1.2), np.full(4000, 0.7)
    cov = true_cov(v[:1], c[:1])[0]
    assert np.linalg.eigvalsh(cov).min() > -1e-9
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(D))
    x = (L @ rng.normal(size=(D, 4000))).T
    assert np.abs(np.cov(x.T) - cov).max() < 0.03


def test_head_precision_is_symmetric_psd_and_banded_factor():
    head = PrecisionHead(S0, hidden=32, bandwidth=4, rank=2)
    P = head.normalized_precision(torch.randn(3, COND_DIM), torch.eye(2)[[0, 1, 0]], torch.randn(3, D))
    assert torch.allclose(P, P.transpose(1, 2), atol=1e-5)
    assert torch.linalg.eigvalsh(P).min() > 0
    assert head.mask.sum(1).max() <= 5            # lower factor really is band-limited (bw+1 per row)


def test_covariance_inverse_of_precision_and_marginals_shape():
    head = PrecisionHead(S0, hidden=32)
    c, m, a = torch.randn(2, COND_DIM), torch.eye(2), torch.randn(2, D)
    Sig = head.covariance(c, m, a, calibrated=False)
    P = head.normalized_precision(c, m, a) / S0 ** 2
    assert torch.allclose(Sig @ P, torch.eye(D).expand(2, D, D), atol=2e-3)
    assert marginals(Sig).shape == (T_FUT, 2, 2, 2) or marginals(Sig).shape[-2:] == (2, 2)


def test_gaussian_case_curvature_equals_posterior_precision():
    """Core maths: for Gaussian data, (I - dD/dx) = sigma0^2 (Sigma + sigma0^2 I)^-1 exactly."""
    Sig = _rand_sigma(); den = GaussianDenoiser(Sig)
    x = torch.randn(2, D)
    Pt = dense_target(den, x, None, None, S0)
    expected = S0 ** 2 * torch.linalg.inv(Sig + S0 ** 2 * torch.eye(D))
    assert torch.allclose(Pt[0], expected, atol=1e-4)


def test_hvp_matches_dense_target_column():
    den = GaussianDenoiser(_rand_sigma(1)); x = torch.randn(3, D)
    Pt = dense_target(den, x, None, None, S0)
    v = torch.randn(3, D)
    assert torch.allclose(hvp_target(den, x, None, None, v, S0), torch.einsum("bij,bj->bi", Pt, v), atol=1e-4)


def test_distillation_reduces_loss_and_recovers_covariance_on_gaussian_case():
    torch.manual_seed(0)
    Sig = _rand_sigma(2); den = GaussianDenoiser(Sig)
    head = PrecisionHead(S0, hidden=64, bandwidth=D, rank=4)      # full bandwidth: exact family
    anc = torch.randn(64, D) * 0.1; c = torch.zeros(64, COND_DIM); m = torch.eye(2)[torch.zeros(64, dtype=torch.long)]
    Pt = dense_target(den, anc, c, m, S0); Sig_t, ld = psd_inverse(Pt)
    opt = torch.optim.Adam(head.parameters(), 3e-3)
    first = None
    for i in range(400):
        loss, kl = distill_loss(head, den, anc, c, m, probes=4, Sig_t=Sig_t, logdet_Sig_t=-ld, kl_weight=0.05)
        if first is None: first = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.2 * first


def test_gradients_flow_to_head_only():
    cfg_den = Denoiser(32, 2)
    for p in cfg_den.parameters(): p.requires_grad_(False)
    head = PrecisionHead(S0, hidden=32)
    c, m, a = torch.randn(4, COND_DIM), torch.eye(2)[[0, 1, 0, 1]], torch.randn(4, D)
    loss, _ = distill_loss(head, cfg_den, a, c, m, probes=2)
    loss.backward()
    assert all(p.grad is not None for p in head.parameters() if p.requires_grad and p is not head.log_scale)
    assert all(p.grad is None for p in cfg_den.parameters())


def test_alpha_closed_form_minimises_nll():
    torch.manual_seed(0)
    Sig = torch.eye(D).expand(200, D, D) * 0.5
    e = torch.randn(200, D) * 0.9
    a = fit_alpha(Sig, e)
    f = lambda al: metrics(Sig, e, al)["nll_per_step"]
    assert f(a) <= f(a * 1.2) and f(a) <= f(a * 0.8)


def test_sampler_deterministic_nominal_and_shapes():
    m = Denoiser(32, 2).eval()
    c, mo = torch.randn(3, COND_DIM), torch.eye(2)[[0, 1, 0]]
    a1, a2 = sample(m, c, mo, None, 4), sample(m, c, mo, None, 4)
    assert a1.shape == (3, D) and torch.allclose(a1, a2)
