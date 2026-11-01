import numpy as np


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def aurc(prob, y):
    """Area under the risk-coverage curve (selective classification, confidence = max prob).  Lower is better."""
    conf, wrong = prob.max(1), (prob.argmax(1) != y).astype(float)
    r = np.cumsum(wrong[np.argsort(-conf)]) / np.arange(1, len(y) + 1)
    return float(r.mean())


def report(prob, y):
    return dict(acc=float((prob.argmax(1) == y).mean()), ece=ece(prob, y), aurc=aurc(prob, y))
