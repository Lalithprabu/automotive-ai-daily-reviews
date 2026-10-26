"""Synthetic 3-lane scenes for a SafeDriveVLA-style toy: ego + K agents + a language-like navigation instruction.

Frame: ego-centred longitudinal axis s (m, + = ahead), agents move with constant RELATIVE speed v (m/s).
Lanes 0,1,2 (0 = leftmost). Instruction: 0 KEEP, 1 LEFT (lane-1), 2 RIGHT (lane+1).
Strict-following rollout = the ego executes the instruction literally (the thing the paper's world model 'dreams').
Labels (all computed analytically, no learning involved):
  collision flag c_t, min-gap g_t per dream step, 3-level compliance mode, and an expert lateral profile.
"""
import numpy as np

H_DEFAULT, DT, LANE_W, COLLIDE_GAP = 8, 0.5, 3.5, 7.0
KEEP, LEFT, RIGHT = 0, 1, 2
FOLLOW, ADAPT, REFUSE = 0, 1, 2
MODE_NAMES = ["FOLLOW", "ADAPT", "REFUSE"]
NAV_NAMES = ["KEEP", "LEFT", "RIGHT"]
# lateral progress of the ego toward the target lane (0..1) per step for each mode (expert behaviour)
PROFILES = {
    FOLLOW: np.array([0, .25, .6, 1, 1, 1, 1, 1], np.float32),   # execute now
    ADAPT:  np.array([0, .1, .2, .3, .4, .4, .4, .4], np.float32),  # signal/nudge, hold, re-check (never commits)
    REFUSE: np.zeros(8, np.float32),                              # stay in lane
}


def sample_scenes(n, rng, K=4, H=H_DEFAULT):
    """Returns dict of numpy arrays for n scenes."""
    ego_lane = rng.integers(0, 3, n)
    nav = rng.integers(0, 3, n)
    n_ag = rng.integers(1, K + 1, n)
    lane = rng.integers(0, 3, (n, K))
    s = rng.uniform(-30, 40, (n, K))
    v = rng.uniform(-6, 6, (n, K))
    valid = (np.arange(K)[None] < n_ag[:, None])
    # keep agents out of the ego's own lane overlap at t=0 (no pre-existing crash)
    clash = (lane == ego_lane[:, None]) & (np.abs(s) < COLLIDE_GAP + 3)
    s = np.where(clash, s + np.sign(s + 1e-3) * 14, s)
    return dict(ego_lane=ego_lane, nav=nav, lane=lane, s=s, v=v, valid=valid.astype(np.float32))


def strict_follow_rollout(sc, H=H_DEFAULT):
    """Ground-truth consequences of literally obeying the instruction. Returns collision[n,H], gap[n,H], offroad[n]."""
    n, K = sc["lane"].shape
    delta = np.array([0, -1, 1])[sc["nav"]]
    target = sc["ego_lane"] + delta
    offroad = (target < 0) | (target > 2)
    t = np.arange(H)[None, :]                                          # [1,H]
    occ = np.where(t[:, :] >= 2, target[:, None], sc["ego_lane"][:, None])        # lane occupied at step t  [n,H]
    s_t = sc["s"][:, :, None] + sc["v"][:, :, None] * DT * np.arange(H)[None, None]   # [n,K,H]
    same = (sc["lane"][:, :, None] == occ[:, None, :]) & (sc["valid"][:, :, None] > 0)
    d = np.abs(s_t)
    gap = np.where(same, d, 99.0).min(1)                                # [n,H]
    gap = np.clip(gap / 20.0, 0, 2.0).astype(np.float32)
    coll = (gap * 20.0 < COLLIDE_GAP)
    coll[offroad] = True
    return coll.astype(np.float32), gap, offroad


def label_mode(coll, offroad, H=H_DEFAULT):
    any_c = coll.max(1) > 0
    tc = np.where(any_c, coll.argmax(1), H)
    mode = np.full(len(coll), FOLLOW)
    mode[any_c & (tc >= 4)] = ADAPT
    mode[(any_c & (tc < 4)) | offroad] = REFUSE
    return mode


def build_batch(sc, H=H_DEFAULT):
    coll, gap, off = strict_follow_rollout(sc, H)
    mode = label_mode(coll, off, H)
    prof = np.stack([PROFILES[m][:H] for m in mode]).astype(np.float32)          # expert lateral progress
    n, K = sc["lane"].shape
    lane_oh = np.eye(3, dtype=np.float32)[sc["lane"]]                             # [n,K,3]
    agents = np.concatenate([lane_oh, (sc["s"] / 30.0)[..., None], (sc["v"] / 6.0)[..., None],
                             sc["valid"][..., None]], -1).astype(np.float32)      # [n,K,6]
    ego = np.eye(3, dtype=np.float32)[sc["ego_lane"]]                             # [n,3]
    return dict(agents=agents, ego=ego, nav=sc["nav"].astype(np.int64), coll=coll, gap=gap,
                mode=mode.astype(np.int64), profile=prof, valid=sc["valid"])


def executed_collision(sc, profile, H=H_DEFAULT):
    """Run an arbitrary lateral profile in the toy world: ego is 'in the target lane' once progress >= 0.5.
    Returns collision-any flag per scene (used to score what the policy actually does)."""
    delta = np.array([0, -1, 1])[sc["nav"]]
    target = sc["ego_lane"] + delta
    off = (target < 0) | (target > 2)
    in_t = profile >= 0.5                                                          # [n,H]
    occ = np.where(in_t, target[:, None], sc["ego_lane"][:, None])
    s_t = sc["s"][:, :, None] + sc["v"][:, :, None] * DT * np.arange(H)[None, None]
    same = (sc["lane"][:, :, None] == occ[:, None, :]) & (sc["valid"][:, :, None] > 0)
    hit = (same & (np.abs(s_t) < COLLIDE_GAP)).any((1, 2))
    return hit | (off & (profile.max(1) >= 0.5))
