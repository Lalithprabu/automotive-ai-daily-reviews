import numpy as np


def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0


def episode_metrics(lat, speeds, steers, progress, crashed, dt=0.1):
    acc = np.diff(speeds) / dt
    jerk = np.diff(acc) / dt
    return dict(lat_rms=rms(lat), jerk_rms=rms(jerk), steer_rate_rms=rms(np.diff(steers) / dt),
                progress=float(progress), off_road=bool(crashed))
