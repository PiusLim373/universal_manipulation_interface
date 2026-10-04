"""Scene pose track -> uniform 60 Hz grid. Shared by episode_prep, build_zarr and the server.

Grid points are chosen BY TIME, never by frame index: the wrist drops frames, so
striding every Nth frame stops being uniform the moment one is lost. Each grid
point takes the interpolated pose and the nearest wrist frame, and is rejected if
that frame is too far away in time.
"""

import os

import cv2
import numpy as np

DERIVED = 'derived'
TCP_NPZ = 'scene_tcp.npz'

# The TRACKED board on the gripper -- NOT the 7x5 30 mm calibration target.
TRACK_BOARD = {'size': (4, 4), 'square': 0.025, 'marker': 0.018, 'dict': 'DICT_4X4_50'}
# Override for testing on other footage, e.g. "7x5,0.030,0.022,DICT_4X4_50".
if os.environ.get('ROBOSPEC_TRACK_BOARD'):
    _s, _sq, _mk, _d = os.environ['ROBOSPEC_TRACK_BOARD'].split(',')
    TRACK_BOARD = {'size': tuple(int(v) for v in _s.split('x')),
                   'square': float(_sq), 'marker': float(_mk), 'dict': _d}

# Training crop of the wrist frame: a full-height square from this left edge, then
# resized to 224. The D455's colour camera sits off the gripper's centre line, so its
# crop is shifted left to keep both fingers in at full opening. Other sizes centre.
WRIST_CROP_X = {(640, 360): 106}


def wrist_crop(w, h):
    """-> (x0, size) of the square training crop for a w x h wrist frame."""
    return WRIST_CROP_X.get((w, h), (w - h) // 2), h


# Board centre -> TCP, then body-fixed turns. Fixed by how the board is mounted.
TCP_OFFSET = (0.0, -0.155, 0.0725)
TCP_ROTATION = ('x', '180', 'z', '90')

# Board <- D455 IMU rotation, and the offset added to gyro stamps to land on the
# scene clock. Fixed by how the D455 and the board are mounted; measured on
# 20261004_213326. After remounting either, re-run:
#   episode_prep.py --calibrate-imu data/capture/<session>/ep*
IMU_ROTATION = ((1.0000, 0.0011, 0.0014),
                (0.0012, 0.1644, -0.9864),
                (-0.0013, 0.9864, 0.1644))
IMU_TIME_OFFSET = -0.0025   # s

GRID_HZ = 60.0          # matches umi.yaml; the policy is trained at this rate
MIN_VALID = 0.90        # drop an episode below this fraction of usable grid points
MIN_SEGMENT = 1.5       # s -- drop contiguous runs shorter than this
MAX_FRAME_DIST = 0.012  # s -- ~one dropped wrist frame (11.1 ms period)
MAX_SPIKE = 0.020       # m -- per-frame jump treated as a depth-ambiguity flip
MAX_POSE_GAP = 0.0625   # s -- pose-track hole wider than this invalidates
DELTA = 0.0             # extra latency to shift the wrist stream by
MAX_GRIPPER_GAP = 0.1   # s -- gripper-sensor hole wider than this invalidates

# per scene frame, stored in scene_tcp.npz
OK, LOST, SPIKE, OUTSIDE, POSE_GAP, NO_WRIST = range(6)
STATUS = ('ok', 'lost', 'spike', 'outside', 'pose-gap', 'no-wrist')


def board_args(legacy=False):
    b = TRACK_BOARD
    return (b['size'], b['square'], b['marker'], b['dict'], legacy)


def board_key():
    b = TRACK_BOARD
    return f"{b['size'][0]}x{b['size'][1]},{b['square']},{b['marker']},{b['dict']}"


# ------------------------------------------------------------------ SE(3)
def interp_T(T0, T1, u):
    """Constant-speed SE(3) interpolation: geodesic on rotation, straight on position."""
    R0 = T0[:3, :3]
    rvec, _ = cv2.Rodrigues(R0.T @ T1[:3, :3])
    R = R0 @ cv2.Rodrigues(rvec * u)[0]
    return np.block([[R, ((1 - u) * T0[:3, 3] + u * T1[:3, 3]).reshape(3, 1)],
                     [np.zeros((1, 3)), np.ones((1, 1))]])


def pose6(T):
    """4x4 -> (6,) position + axis-angle, the representation the zarr stores."""
    return np.concatenate([T[:3, 3], cv2.Rodrigues(T[:3, :3])[0].flatten()])


# ------------------------------------------------------------- pose track
def drop_spikes(pose_t, poses, thresh):
    """Remove isolated out-and-back excursions (planar PnP depth flips).

    Detected as |P[i] - midpoint(P[i-1], P[i+1])|: smooth motion contributes only
    a*dt^2 (<1 mm at 120 fps), a spike roughly twice its own size.
    Returns (pose_t, poses, keep_mask).
    """
    if len(poses) < 3 or thresh <= 0:
        return pose_t, poses, np.ones(len(poses), bool)
    P = np.array([T[:3, 3] for T in poses])
    keep = np.ones(len(P), bool)
    dev = np.linalg.norm(P[1:-1] - (P[:-2] + P[2:]) / 2, axis=1)
    keep[1:-1] = dev <= thresh
    return pose_t[keep], [p for p, k in zip(poses, keep) if k], keep


def build_grid(pose_t, wrist_t, rate, max_pose_gap, max_frame_dist, delta, window=None):
    """Uniform-in-time grid -> (grid_times, wrist_idx, gap_ok, near_ok).

    gap_ok and near_ok stay separate: a pose gap means the scene camera lost the
    board, a missing wrist frame means the D405 dropped frames. `window` (lo, hi)
    in seconds narrows the grid, which is how a trim is applied.
    """
    lo = max(pose_t[0] - delta, wrist_t[0])
    hi = min(pose_t[-1] - delta, wrist_t[-1])
    if window is not None:
        lo, hi = max(lo, window[0]), min(hi, window[1])
    if hi <= lo:
        return (np.empty(0), np.empty(0, int),
                np.empty(0, bool), np.empty(0, bool))

    grid = np.arange(lo, hi, 1.0 / rate)

    j = np.searchsorted(pose_t, grid + delta)
    j = np.clip(j, 1, len(pose_t) - 1)
    gap_ok = (pose_t[j] - pose_t[j - 1]) <= max_pose_gap

    k = np.searchsorted(wrist_t, grid)
    k = np.clip(k, 1, len(wrist_t) - 1)
    left = np.abs(wrist_t[k - 1] - grid) <= np.abs(wrist_t[k] - grid)
    widx = np.where(left, k - 1, k)
    near_ok = np.abs(wrist_t[widx] - grid) <= max_frame_dist

    return grid, widx, gap_ok, near_ok


def segments(valid, min_len):
    """All contiguous True runs of at least min_len, as [(start, end), ...].

    A zarr episode cannot carry holes, so a recording with a real gap becomes
    more than one episode rather than being trimmed to its longest run.
    """
    out, start = [], None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def poses_on_grid(grid, pose_t, poses, delta):
    """Interpolate the TCP pose onto each grid time."""
    out = np.empty((len(grid), 6), dtype=np.float32)
    for i, g in enumerate(grid):
        t = g + delta
        j = int(np.clip(np.searchsorted(pose_t, t), 1, len(pose_t) - 1))
        t0, t1 = pose_t[j - 1], pose_t[j]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out[i] = pose6(interp_T(poses[j - 1], poses[j], float(np.clip(u, 0, 1))))
    return out


def frame_status(t, tracked, spiked, grid, gap_ok, near_ok):
    """Status code per scene frame. A frame's grid verdict is found by time, via
    the nearest grid point, never by index arithmetic."""
    st = np.full(len(t), OK, np.uint8)
    if len(grid):
        j = np.clip(np.searchsorted(grid, t), 1, max(len(grid) - 1, 1))
        g = np.where(np.abs(grid[j - 1] - t) <= np.abs(grid[np.minimum(j, len(grid) - 1)] - t),
                     j - 1, np.minimum(j, len(grid) - 1))
        st[~near_ok[g]] = NO_WRIST
        st[~gap_ok[g]] = POSE_GAP
        st[(t < grid[0]) | (t > grid[-1])] = OUTSIDE
    else:
        st[:] = OUTSIDE
    st[spiked] = SPIKE
    st[~tracked] = LOST
    return st


# ------------------------------------------------------------------ episodes
def ep_path(ep_dir, *parts):
    return os.path.join(ep_dir, DERIVED, *parts)


def wrist_times(ep_dir):
    return np.load(os.path.join(ep_dir, 'wrist', 'wrist_ts.npz'))['t_ns'].astype(np.float64) / 1e9


def gripper_track(ep_dir):
    """-> (t_s, width_m) of the recorded opening, or None if the episode has no
    gripper recording. The one place the signal comes from: the edit UI plots it
    and export resamples it. Converted with the mapping saved beside the samples.
    """
    p = os.path.join(ep_dir, 'gripper', 'gripper_ts.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if len(z['t_ns']) < 2:
        return None
    w = np.clip(z['raw'] / float(z['raw_open']), 0.0, 1.0) * float(z['max_width_m'])
    return z['t_ns'].astype(np.float64) / 1e9, w


def imu_track(ep_dir):
    """-> {tg, w, ta, a}: gyro (s, rad/s) and accel (s, m/s^2) of the wrist IMU on
    its own clock (IMU_TIME_OFFSET not applied), or None without a recording."""
    p = os.path.join(ep_dir, 'imu', 'imu_ts.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if len(z['gyro_t_ns']) < 2:
        return None
    return {'tg': z['gyro_t_ns'] / 1e9, 'w': z['gyro'].astype(np.float64),
            'ta': z['accel_t_ns'] / 1e9, 'a': z['accel'].astype(np.float64)}


def gripper_on_grid(ep_dir, grid):
    """Gripper width at each grid time, (n, 1) float32 as the zarr stores it."""
    t, w = gripper_track(ep_dir)
    return np.interp(grid, t, w).astype(np.float32).reshape(-1, 1)


def load_track(ep_dir):
    """scene_tcp.npz -> dict(t_ns, T_tcp, tracked, rms, status)."""
    z = np.load(ep_path(ep_dir, TCP_NPZ))
    T = z['T_tcp']
    return {'t_ns': z['t_ns'], 'T_tcp': T, 'tracked': np.isfinite(T[:, 0, 0]),
            'rms': z['rms'], 'status': z['status']}


def plan_episode(ep_dir, trim=None, with_poses=True, track=None):
    """One recording -> the segments that go into the zarr.

    trim is (t_in, t_out) in seconds on the episode clock (t=0 at the first scene
    frame); either end may be None. Returns a dict: segments (grid, wrist_idx,
    pose), spans on the episode clock, usable, kept_s, note, and reason (None if
    the episode is kept). with_poses=False skips the pose interpolation, for the
    live preview.
    """
    tr = track or load_track(ep_dir)
    t0 = tr['t_ns'][0] / 1e9
    out = {'segments': [], 'spans': [], 'usable': 0.0, 'grid_n': 0, 'kept_s': 0.0}
    grip = gripper_track(ep_dir)
    if grip is None:
        return {**out, 'note': 'no gripper recording', 'reason': 'no gripper recording'}
    good = np.nonzero(tr['tracked'])[0]
    if len(good) < 2:
        return {**out, 'note': 'no tracked poses', 'reason': 'no tracked poses'}
    n_tracked = len(good)
    # spikes are dropped on the whole track, so a trim cannot change which go
    pose_t, poses, spike_keep = drop_spikes(
        tr['t_ns'][good].astype(np.float64) / 1e9, [tr['T_tcp'][i] for i in good], MAX_SPIKE)
    n_spike = int((~spike_keep).sum())

    window = None
    if trim is not None and (trim[0] is not None or trim[1] is not None):
        window = (t0 + (trim[0] if trim[0] is not None else -1e9),
                  t0 + (trim[1] if trim[1] is not None else 1e9))
    grid, widx, gap_ok, near_ok = build_grid(
        pose_t, wrist_times(ep_dir), GRID_HZ, MAX_POSE_GAP, MAX_FRAME_DIST, DELTA, window)
    if len(grid) == 0:
        why = 'no temporal overlap between the two cameras'
        return {**out, 'note': why, 'reason': why}
    # like a pose gap: a point is only as good as the two openings around it
    gt = grip[0]
    j = np.clip(np.searchsorted(gt, grid), 1, len(gt) - 1)
    grip_ok = (grid >= gt[0]) & (grid <= gt[-1]) & (gt[j] - gt[j - 1] <= MAX_GRIPPER_GAP)
    valid = gap_ok & near_ok & grip_ok

    segs = segments(valid, int(round(MIN_SEGMENT * GRID_HZ)))
    kept = sum(e - s for s, e in segs)
    n_grip = int((~grip_ok).sum())
    note = (f'tracked {n_tracked/len(tr["t_ns"]):.1%} '
            f'({n_spike} spike{"" if n_spike == 1 else "s"} dropped), grid {len(grid)}, '
            + (f'{n_grip} without a gripper reading, ' if n_grip else '')
            + f'valid {valid.mean():.1%}, {len(segs)} segment(s), '
            f'kept {kept}/{len(grid)} = {kept/len(grid):.1%}')
    out.update(usable=float(valid.mean()), grid_n=int(len(grid)), kept_s=kept / GRID_HZ,
               spans=[(float(grid[s] - t0), float(grid[e - 1] - t0)) for s, e in segs])

    # the gate is on how much of the recording is USABLE, not on what survives
    # segmentation
    if valid.mean() < MIN_VALID:
        return {**out, 'spans': [], 'kept_s': 0.0, 'note': note,
                'reason': f'usable {valid.mean():.0%} below {MIN_VALID:.0%}'}
    if not segs:
        return {**out, 'note': note,
                'reason': f'no run reached {MIN_SEGMENT:.1f} s'}

    if with_poses:
        # demo start/end are scoped to the whole (trimmed) recording, not to each
        # segment -- a segment boundary is a dropped-frame gap, not a new demo
        vi = np.nonzero(valid)[0]
        ends = poses_on_grid(grid[[vi[0], vi[-1]]], pose_t, poses, DELTA)
        for i, (s, e) in enumerate(segs):
            out['segments'].append({
                'dir': ep_dir, 'seg': i, 'grid': grid[s:e], 'wrist_idx': widx[s:e],
                'pose': poses_on_grid(grid[s:e], pose_t, poses, DELTA),
                'demo_start': ends[0], 'demo_end': ends[1]})
    return {**out, 'note': note, 'reason': None}
