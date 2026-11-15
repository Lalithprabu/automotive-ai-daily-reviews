"""
Synthetic driving-scene event-frame dataset.

DISCLOSURE: real event-camera datasets (the paper evaluates on 1 Mpx
automotive driving data, FRED, and EV-Flying) are not available inside this
sandboxed session. Every number produced by `train.py` / `simulate.py` in
this repo comes from this synthetic generator, not from the paper's real
data — see SOURCING.md and the README's sourcing table.

Scene model: 1-3 agents (vehicles and pedestrians) move through a bird's-eye
event-sensor frame with a per-agent constant-acceleration-plus-jitter motion
model (not pure constant velocity, so the "predict ahead" task is not
trivially solvable by copying the last observed velocity). Each observed
timestep is rendered as a 2-channel synthetic "event frame"
(ON-polarity / OFF-polarity activity, gaussian-blobbed at each agent's
current position) plus low-amplitude background sensor noise.

For every sample, a random `latency_steps` (how many frames ahead the model
must predict for) is drawn, and the ground-truth target is each agent's
*true* position at `t_obs_last + latency_steps` — computed by continuing
the same motion model forward, not observed directly. This is exactly the
"predict for when the output becomes available" setup the paper describes.
"""

import math
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["vehicle", "pedestrian"]

# Per-class kinematic + rendered-size priors (this project's own synthetic
# defaults, chosen to be qualitatively realistic — vehicles faster/bigger,
# pedestrians slower/smaller — not measured from any real dataset).
CLASS_PRIORS = {
    0: dict(speed_range=(4.0, 9.0), size_range=(10.0, 16.0)),   # vehicle, px/frame, box px
    1: dict(speed_range=(0.8, 2.2), size_range=(4.0, 6.0)),     # pedestrian
}


@dataclass
class Agent:
    cls: int
    pos0: torch.Tensor       # [2] (x, y) at the first observed frame
    vel0: torch.Tensor       # [2] initial velocity, px/frame
    accel: torch.Tensor      # [2] constant acceleration, px/frame^2
    size: torch.Tensor       # [2] (w, h) in px

    def position_at(self, t: float) -> torch.Tensor:
        # Constant-acceleration kinematics: x(t) = x0 + v0*t + 0.5*a*t^2
        return self.pos0 + self.vel0 * t + 0.5 * self.accel * (t ** 2)


def _draw_agent(h: int, w: int, generator: torch.Generator) -> Agent:
    cls = int(torch.randint(0, 2, (1,), generator=generator).item())
    priors = CLASS_PRIORS[cls]

    margin = 12.0
    pos0 = torch.tensor(
        [
            torch.empty(1).uniform_(margin, w - margin, generator=generator).item(),
            torch.empty(1).uniform_(margin, h - margin, generator=generator).item(),
        ]
    )
    speed = torch.empty(1).uniform_(*priors["speed_range"], generator=generator).item()
    heading = torch.empty(1).uniform_(0, 2 * math.pi, generator=generator).item()
    vel0 = torch.tensor([speed * math.cos(heading), speed * math.sin(heading)])

    # Small lateral/longitudinal acceleration jitter -- enough that pure
    # constant-velocity extrapolation is measurably suboptimal.
    accel = torch.empty(2).uniform_(-0.35, 0.35, generator=generator)

    size = torch.tensor(
        [
            torch.empty(1).uniform_(*priors["size_range"], generator=generator).item(),
            torch.empty(1).uniform_(*priors["size_range"], generator=generator).item(),
        ]
    )
    return Agent(cls=cls, pos0=pos0, vel0=vel0, accel=accel, size=size)


def render_event_frame(agents, t: float, h: int, w: int, noise_level: float = 0.02,
                        generator: torch.Generator = None) -> torch.Tensor:
    """Renders a synthetic 2-channel (ON/OFF polarity) event frame at time t.

    Shape: [2, H, W]
    """
    frame = torch.zeros(2, h, w)
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    for agent in agents:
        cx, cy = agent.position_at(t).tolist()
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        sigma = max(agent.size.mean().item() / 3.0, 1.5)
        gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        # Leading-edge motion direction -> ON channel; trailing -> OFF channel,
        # offset slightly along the velocity vector (a simplified stand-in
        # for real leading/trailing-edge event polarity structure).
        speed = agent.vel0.norm().item() + 1e-6
        heading = agent.vel0 / speed
        on_shift = heading * sigma * 0.5
        off_shift = -heading * sigma * 0.5
        on_gauss = torch.exp(-((xx - (cx + on_shift[0])) ** 2 + (yy - (cy + on_shift[1])) ** 2) / (2 * sigma ** 2))
        off_gauss = torch.exp(-((xx - (cx + off_shift[0])) ** 2 + (yy - (cy + off_shift[1])) ** 2) / (2 * sigma ** 2))
        frame[0] += on_gauss
        frame[1] += off_gauss

    if generator is not None:
        frame += noise_level * torch.rand(frame.shape, generator=generator)
    frame.clamp_(0.0, 1.0)
    return frame


@dataclass
class SyntheticSample:
    event_seq: torch.Tensor       # [T, 2, H, W]
    latency_steps: torch.Tensor   # scalar float
    future_centers: torch.Tensor  # [num_agents, 2] (x, y) at t_obs_last + latency
    stale_centers: torch.Tensor   # [num_agents, 2] (x, y) at t_obs_last (the "naive" baseline target)
    sizes: torch.Tensor           # [num_agents, 2]
    classes: torch.Tensor         # [num_agents]


class SyntheticDrivingEventDataset(Dataset):
    """Synthetic event-camera driving dataset for latency-compensated
    detection. See module docstring for the full disclosure."""

    def __init__(
        self,
        num_samples: int = 512,
        height: int = 128,
        width: int = 128,
        t_obs: int = 5,
        max_agents: int = 3,
        max_latency_steps: int = 4,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.h = height
        self.w = width
        self.t_obs = t_obs
        self.max_agents = max_agents
        self.max_latency_steps = max_latency_steps
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> SyntheticSample:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)

        n_agents = int(torch.randint(1, self.max_agents + 1, (1,), generator=gen).item())
        agents = [_draw_agent(self.h, self.w, gen) for _ in range(n_agents)]

        # Observed frames at t = -(T-1), ..., -1, 0 (0 == "last observed").
        frames = [
            render_event_frame(agents, t=float(i - (self.t_obs - 1)), h=self.h, w=self.w, generator=gen)
            for i in range(self.t_obs)
        ]
        event_seq = torch.stack(frames, dim=0)  # [T, 2, H, W]

        latency_steps = float(torch.randint(1, self.max_latency_steps + 1, (1,), generator=gen).item())

        stale_centers = torch.stack([a.position_at(0.0) for a in agents], dim=0)
        future_centers = torch.stack([a.position_at(latency_steps) for a in agents], dim=0)
        sizes = torch.stack([a.size for a in agents], dim=0)
        classes = torch.tensor([a.cls for a in agents], dtype=torch.long)

        return SyntheticSample(
            event_seq=event_seq,
            latency_steps=torch.tensor(latency_steps),
            future_centers=future_centers,
            stale_centers=stale_centers,
            sizes=sizes,
            classes=classes,
        )


def collate_samples(batch):
    """Custom collate: pads the variable agent-count fields with a ragged
    Python list (kept as list-of-tensors) since heatmap targets are built
    per-sample downstream, not batched as a dense tensor here."""
    return {
        "event_seq": torch.stack([s.event_seq for s in batch], dim=0),          # [B, T, 2, H, W]
        "latency_steps": torch.stack([s.latency_steps for s in batch], dim=0),  # [B]
        "future_centers": [s.future_centers for s in batch],
        "stale_centers": [s.stale_centers for s in batch],
        "sizes": [s.sizes for s in batch],
        "classes": [s.classes for s in batch],
    }
