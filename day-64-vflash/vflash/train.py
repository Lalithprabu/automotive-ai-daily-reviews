"""Train drafter vs AR baseline on the synthetic task; report ADE and latency."""
import time, torch, torch.nn as nn
from .model import VFlashDrafter, ARBaseline
from .data import batch, T

class Enc(nn.Module):
    def __init__(s, d=128):
        super().__init__(); s.l = nn.Linear(16, d); s.p = nn.Parameter(torch.randn(1, 8, d) * .02)
        s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, batch_first=True, norm_first=True), 2)
    def forward(s, x): return s.t(s.l(x) + s.p)

def fit(model, enc, kind, steps, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(list(model.parameters()) + list(enc.parameters()), 2e-3)
    for i in range(steps):
        c, y = batch(64, seed * 10000 + i)
        ctx = enc(c)
        pred = model(ctx, "traj2d", T) if kind == "flash" else model(ctx, target=y)
        loss = nn.functional.smooth_l1_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, enc, kind):
    c, y = batch(512, 999999); model.eval(); enc.eval()
    ctx = enc(c)
    t0 = time.perf_counter()
    pred = model(ctx, "traj2d", T) if kind == "flash" else model(ctx, length=T)
    dt = time.perf_counter() - t0
    ade = (pred - y).norm(dim=-1).mean().item()
    return ade, dt

if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    res = {}
    for kind in ("flash", "ar"):
        enc = Enc(); m = VFlashDrafter() if kind == "flash" else ARBaseline()
        fit(m, enc, kind, steps)
        res[kind] = evaluate(m, enc, kind)
        torch.save({"m": m.state_dict(), "e": enc.state_dict()}, f"assets/{kind}.pt")
        print(f"{kind:5s} ADE={res[kind][0]:.4f}  latency(512 scenes)={res[kind][1]*1e3:.1f} ms")
    print(f"speedup x{res['ar'][1]/res['flash'][1]:.1f}")
