"""Uncertainty propagation (MC dropout / deep ensemble) and split-conformal calibration."""
import math, numpy as np, torch

@torch.no_grad()
def propagate_samples(models, batch, n_samples=16, mc_passes=1):
    """Run each (model, pass), draw n_samples aleatoric samples through the kinematic layer.
    returns dict: mean_traj (B,Tp,2), cov_ale, cov_epi, cov_total (B,Tp,2,2), all_samples (B,S,Tp,2).
    epistemic = spread of per-pass/member mean trajectories; aleatoric = mean within-pass sample cov."""
    ale_covs, means, allS = [], [], []
    for m in models:
        m.eval()
        if mc_passes > 1: m.set_mc(True)
        for _ in range(mc_passes):
            mu, lv = m.motion_dist(batch["hist"], batch["adj"])
            s = m.sample_traj(mu, lv, batch["v0"], batch["psi0"], n_samples)   # (B,n,Tp,2)
            mm = s.mean(1); d = s - mm.unsqueeze(1)
            ale_covs.append(torch.einsum("bnti,bntj->btij", d, d) / (n_samples - 1))
            means.append(mm); allS.append(s)
        m.set_mc(False)
    ale = torch.stack(ale_covs).mean(0)                                  # (B,Tp,2,2)
    M = torch.stack(means)                                               # (P,B,Tp,2)
    mean = M.mean(0); dm = M - mean
    epi = torch.einsum("pbti,pbtj->btij", dm, dm) / max(M.shape[0] - 1, 1)
    return dict(mean_traj=mean, cov_ale=ale, cov_epi=epi, cov_total=ale + epi, all_samples=torch.cat(allS, 1))

def total_covariance(out): return out["cov_total"]

def mahalanobis(err, cov, eps=1e-8):
    c = cov + eps * torch.eye(2, device=cov.device)
    sol = torch.linalg.solve(c, err.unsqueeze(-1)).squeeze(-1)
    return torch.sqrt((err * sol).sum(-1).clamp_min(0))

class ConformalCalibrator:
    """Split conformal prediction on the Mahalanobis nonconformity score r_t = sqrt(e^T (S+eps I)^-1 e),
    one quantile per horizon step, finite-sample rank j = ceil((n+1)(1-delta)).
    Calibrated covariance: q_t^2 (S + eps I). Marginal (per-horizon) coverage under exchangeability only."""
    def __init__(self, delta=0.05, eps=1e-8): self.delta, self.eps, self.q = delta, eps, None
    def fit(self, pos, mean, cov):
        r = mahalanobis(pos - mean, cov, self.eps)                       # (S,Tp)
        n = r.shape[0]; j = min(math.ceil((n + 1) * (1 - self.delta)), n)
        self.q = torch.sort(r, dim=0).values[j - 1]                       # (Tp,)
        return self
    def calibrate(self, cov):
        return (self.q ** 2).view(1, -1, 1, 1) * (cov + self.eps * torch.eye(2, device=cov.device))
    def coverage(self, pos, mean, cov, calibrated=True):
        r = mahalanobis(pos - mean, cov, self.eps)
        thr = self.q.view(1, -1) if calibrated else math.sqrt(-2 * math.log(self.delta))  # 2D chi: sqrt(chi2_2(0.95))
        return (r <= thr).float()

def ellipse_params(cov):
    """cov (...,2,2) -> semi-axes (a,b) and angle (rad) of the 1-sigma ellipse."""
    w, V = torch.linalg.eigh(cov)
    a = w[..., 1].clamp_min(0).sqrt(); b = w[..., 0].clamp_min(0).sqrt()
    ang = torch.atan2(V[..., 1, 1], V[..., 0, 1])
    return a, b, ang
