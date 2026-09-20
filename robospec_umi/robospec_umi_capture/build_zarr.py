#!/usr/bin/env python3
"""Turn capture.py sessions into the dataset.zarr.zip that UmiDataset trains from.

The SLAM pipeline cannot do this. scripts_slam_pipeline/00-06 are GoPro/SLAM/IMU
specific end to end, and 07 wants a tag_detection.pkl beside every video, a
demos/-relative layout, and pulls images by frame-index ranges assuming frames are
evenly spaced in time. Ours are not: the wrist camera drops ~3.4% of frames, in bursts
of up to 167 ms. So this emits 07's schema directly from the capture-session layout.

Poses are written ABSOLUTE, never pre-subtracted. UmiDataset.__getitem__ converts to
relative at training time -- against the current observation and against
demo_start_pose -- so subtracting here would convert twice. A useful consequence of
everything reaching the policy as a relative pose is that a global frame change
cancels, (G*base)^-1 * (G*pose) = base^-1 * pose, which makes the scene camera's own
frame a perfectly good world frame.

Timing is the whole point of this file. Grid points are chosen BY TIME, never by frame
index: striding every Nth frame silently stops being uniform the moment a frame is
dropped. Each grid point takes the interpolated pose and the nearest wrist frame, and
is rejected if that frame is more than half a tick away -- pairing a stale image with
a fresh pose is exactly the corruption the timestamp sidecars exist to prevent, and it
is invisible afterwards.

Usage:
    python3 robospec_umi/robospec_umi_capture/build_zarr.py \
        data/capture/20260919_150000 -o data/dataset.zarr.zip
"""

import argparse
import json
import os
import sys
from fractions import Fraction

import av
import cv2
import numpy as np
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_capture
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

# The repo root carries diffusion_policy/ and umi/, which are not installed.
sys.path.append(REPO)

from diffusion_policy.common.replay_buffer import ReplayBuffer          # noqa: E402
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl  # noqa: E402
from umi.common.cv_util import get_image_transform                      # noqa: E402

# Sits beside this file. Imported rather than driven through its CLI: find_video()
# globs *.mp4 and ours are .mkv, and write_csv() emits time_s = i / fps -- the
# uniform-grid assumption this file exists to replace. The solver and the
# seed-poisoning guards in track() are worth reusing as-is.
import video_processor_charuco as V                                     # noqa: E402

register_codecs()

# The scene calibration solve writes here, and capture records the focus/zoom it
# ran at, so the two can be cross-checked before any pose is baked in.
CALIB_JSON = os.path.join(DATA, 'calibration', 'scene_intrinsics.json')

# ---------------------------------------------------------------- fixed geometry
# The TRACKED board carried through the scene -- NOT the 7x5 30 mm calibration
# target. Two different physical boards; swapping them silently wrecks every pose.
TRACK_BOARD = {
    'size': (4, 4),
    'square': 0.020,        # m
    'marker': 0.015,        # m
    'dict': 'DICT_4X4_50',
}

# Board centre -> TCP, then body-fixed turns. Fixed by how the board is mounted
# on the gripper, so it is hardware, not a tuning knob.
TCP_OFFSET = (0.0, -0.200, 0.055)
TCP_ROTATION = ('x', '180', 'z', '90')

# ------------------------------------------------------------------ grid + gates
GRID_HZ = 60.0          # matches umi.yaml; the policy is trained at this rate
OUT_RES = 224           # UmiDataset's camera0_rgb edge
GRIPPER_WIDTH = 0.0     # no gripper-width sensor on this rig
MIN_VALID = 0.90        # warn below this fraction of usable grid points
MIN_SEGMENT = 1.5       # s -- drop contiguous runs shorter than this
MAX_FRAME_DIST = 0.012  # s -- half a 60 Hz tick past which pose/image are stale
MAX_SPIKE = 0.020       # m -- per-tick jump treated as a depth-ambiguity flip
MAX_POSE_GAP = 0.0625   # s -- pose-track hole wider than this invalidates
DELTA = 0.0             # extra latency to shift the wrist stream by
COMPRESSION_LEVEL = 99  # JpegXl quality for camera0_rgb


def scene_resolution(ep_dir):
    """(w, h) of the scene video, so intrinsics are never assumed to be 1080p."""
    with av.open(os.path.join(ep_dir, 'scene', 'scene.mkv')) as c:
        st = c.streams.video[0]
        return int(st.width), int(st.height)


def resolve_intrinsics(session_dir, explicit):
    """Check the calibration actually describes this session. -> (path, session_meta)

    This pipeline is rectilinear only. The guard is still worth its lines because
    a calibration that does not describe the footage produces poses that look
    entirely plausible and are wrong, with no error anywhere downstream -- it
    silently corrupts every label in the dataset rather than failing.
    """
    meta = {}
    p = os.path.join(session_dir, 'session.json')
    if os.path.exists(p):
        with open(p) as f:
            meta = json.load(f)
    scene = meta.get('scene', {})

    path = explicit or CALIB_JSON
    if not os.path.exists(path):
        sys.exit(f'{path} not found -- run calibrate_scene_cam.py solve, or pass '
                 f'--intrinsics explicitly')
    with open(path) as f:
        intr = json.load(f)
    kind = intr.get('intrinsic_type', '?')
    print(f'  intrinsics: {os.path.basename(path)} ({kind})')

    if kind != 'PINHOLE':
        sys.exit(
            f'\nCALIBRATION DOES NOT MATCH THE FOOTAGE.\n'
            f'  {path} is {kind}, expected PINHOLE.\n'
            f'  Projection models are not interchangeable -- the pose solve would\n'
            f'  return plausible, wrong depths and nothing downstream would notice.')

    # Focus and zoom are geometry: a session recorded at a different focus has a
    # different focal length and is not described by this calibration. Same check
    # verify.py makes, repeated here because build_zarr is what bakes it in.
    locked = intr.get('locked_controls') or {}
    for key in ('focus_absolute', 'zoom_absolute'):
        wanted, got = locked.get(key), scene.get(key)
        if wanted is None or got is None or int(got) < 0:
            continue
        if int(wanted) != int(got):
            sys.exit(f'\n  scene {key}={got} but the calibration was shot at '
                     f'{wanted}.\n  Focus and zoom change the focal length, so these '
                     f'intrinsics do not\n  apply to this footage. Re-calibrate, or '
                     f'pass the matching --intrinsics.')
    return path, meta


# ------------------------------------------------------------------ SE(3)
def interp_T(T0, T1, u):
    """Constant-speed SE(3) interpolation: geodesic on rotation, straight on position."""
    R0 = T0[:3, :3]
    rvec, _ = cv2.Rodrigues(R0.T @ T1[:3, :3])
    R = R0 @ cv2.Rodrigues(rvec * u)[0]
    return np.block([[R, ((1 - u) * T0[:3, 3] + u * T1[:3, 3]).reshape(3, 1)],
                     [np.zeros((1, 3)), np.ones((1, 1))]])


def pose6(T):
    """4x4 -> (6,) position + axis-angle, the representation the zarr stores.

    Straight off the matrix via Rodrigues rather than through the quaternion the CSV
    writes, to avoid a lossy round trip.
    """
    return np.concatenate([T[:3, 3], cv2.Rodrigues(T[:3, :3])[0].flatten()])


# ------------------------------------------------------------- scene tracking
def scene_poses(ep_dir, V, cam, board_args, tcp_args, cache=True, keep_frames=False):
    """(times, poses, n_scene_frames) of the TCP for every frame the board was tracked in.

    Detection is cached: it costs ~90 s per episode and grid settings get re-tuned.

    keep_frames additionally returns (ts, dets, rows, good_idx) for --annotate-video.
    good_idx[j] is the SCENE FRAME index that produced pose j, which is what lets a
    later verdict be attributed back to a frame -- the poses list alone has already
    forgotten which frames failed to solve. Off by default so a normal build does not
    hold every frame's detections alive for no reason.
    """
    vid = os.path.join(ep_dir, 'scene', 'scene.mkv')
    ts = np.load(os.path.join(ep_dir, 'scene', 'scene_ts.npz'))['t_ns']
    cpath = os.path.join(ep_dir, 'scene', 'scene_charuco_dets.npz')

    if cache and os.path.exists(cpath):
        dets = list(np.load(cpath, allow_pickle=True)['dets'])
    else:
        board = V.make_board(*board_args)
        detect = V.make_detector(board)
        c = av.open(vid)
        st = c.streams.video[0]
        st.thread_count = 4
        dets = []
        for i, f in enumerate(c.decode(st)):
            m, cc, ci = detect(f.to_ndarray(format='gray'))
            dets.append({'markers': m, 'corners': cc, 'ids': ci})
            if (i + 1) % 200 == 0:
                print(f'      detected {i+1} frames ...', end='\r', flush=True)
        c.close()
        if cache:
            np.savez_compressed(cpath, dets=np.array(dets, dtype=object),
                                n_frames=len(dets))

    # the index invariant verify.py enforces is what makes this pairing safe
    n = min(len(dets), len(ts))
    if len(dets) != len(ts):
        print(f'      WARNING: {len(dets)} detections vs {len(ts)} timestamps, '
              f'using the first {n}')
    dets, ts = dets[:n], ts[:n]

    board = V.make_board(*board_args)
    chess, markers = V.board_geometry(board)
    rows = V.track(dets, chess, markers, cam, V.board_centre(chess), tcp_args)
    good_idx = np.array([i for i, r in enumerate(rows) if r[1] is not None], dtype=int)
    frames = (ts.astype(np.float64) / 1e9, dets, rows, good_idx) if keep_frames else None
    if len(good_idx) == 0:
        return np.empty(0), [], len(rows), frames
    return (ts[good_idx].astype(np.float64) / 1e9,
            [rows[i][1] for i in good_idx], len(rows), frames)


def drop_spikes(pose_t, poses, thresh):
    """Remove isolated out-and-back excursions from the pose track.

    The board is planar, so PnP has a depth ambiguity that the solver occasionally
    resolves to the wrong branch for a frame or two. It reprojects fine -- these
    survive the solver's own rms gate -- but the position jumps out and returns, which
    real hand motion never does. Observed on this data: a 27 mm round trip over two
    frames, implying 2.7 m/s.

    Detected as the second difference, |P[i] - midpoint(P[i-1], P[i+1])|. Smooth motion
    contributes only a*dt^2 -- ~0.8 mm at 120 fps, since the term falls with the square
    of the frame period -- while an out-and-back spike contributes roughly twice its own
    size, so the two separate cleanly. MAX_SPIKE was measured on 48 fps footage, where
    the same smooth term is ~5 mm, so it is conservative here: the safe direction.

    Preferred over simply tightening the solver's max_rms, which is a blunt instrument:
    reaching the same peak speed that way cost 15% of the frames, where this costs 2.6%
    and leaves an already-clean episode untouched.
    """
    if len(poses) < 3 or thresh <= 0:
        return pose_t, poses, np.ones(len(poses), bool)
    P = np.array([T[:3, 3] for T in poses])
    keep = np.ones(len(P), bool)
    dev = np.linalg.norm(P[1:-1] - (P[:-2] + P[2:]) / 2, axis=1)
    keep[1:-1] = dev <= thresh
    # the MASK, not just a count: --annotate-video needs to say which frames went
    return pose_t[keep], [p for p, k in zip(poses, keep) if k], keep


# --------------------------------------------------------------- resampling
def build_grid(pose_t, wrist_t, rate, max_pose_gap, max_frame_dist, delta):
    """Uniform-in-time grid, plus which wrist frame serves each point.

    Returns (grid_times, wrist_idx, gap_ok, near_ok). A point is usable only if both
    masks hold: the pose track must not have too wide a hole around it to interpolate
    honestly, and a wrist frame must land close enough in time.

    The two are returned SEPARATELY rather than pre-combined because they are
    different faults with different fixes -- a pose gap means the scene camera lost
    the board, a missing wrist frame means the D405 dropped frames to USB contention.
    Collapsing them loses the only clue about which to go and fix.

    max_frame_dist is deliberately tied to the wrist camera's frame period rather than
    to the grid rate: the wrist samples every ~11.1 ms, so any grid time is within
    5.6 ms of a frame unless one was dropped. A 12 ms bound therefore means "at most
    one dropped frame away", which stays meaningful if the grid rate changes.
    """
    lo = max(pose_t[0] - delta, wrist_t[0])
    hi = min(pose_t[-1] - delta, wrist_t[-1])
    if hi <= lo:
        return (np.empty(0), np.empty(0, int),
                np.empty(0, bool), np.empty(0, bool))

    grid = np.arange(lo, hi, 1.0 / rate)
    tol = max_frame_dist

    j = np.searchsorted(pose_t, grid + delta)
    j = np.clip(j, 1, len(pose_t) - 1)
    gap_ok = (pose_t[j] - pose_t[j - 1]) <= max_pose_gap

    k = np.searchsorted(wrist_t, grid)
    k = np.clip(k, 1, len(wrist_t) - 1)
    # nearest of the two bracketing wrist frames
    left = np.abs(wrist_t[k - 1] - grid) <= np.abs(wrist_t[k] - grid)
    widx = np.where(left, k - 1, k)
    near_ok = np.abs(wrist_t[widx] - grid) <= tol

    return grid, widx, gap_ok, near_ok


def segments(valid, min_len):
    """All contiguous True runs of at least min_len, as [(start, end), ...].

    A zarr episode is a contiguous block and cannot carry holes, so a recording with a
    real gap in the middle has to become more than one episode. Trimming to the single
    longest run instead would be wasteful: on this data 97.5% of grid points are valid,
    yet one 167 ms wrist gap caps the longest run at 79% of the recording. Splitting
    keeps both sides. UMI's 06_generate_dataset_plan.py segments demos the same way.
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


# ------------------------------------------------------------------ episodes
def plan_episode(ep_dir, V, cam, board_args, tcp_args, args, annotate=False):
    """One recording -> a list of contiguous segments, a report line, and (when
    annotate) everything --annotate-video needs to explain each frame's fate."""
    pose_t, poses, n_scene, frames = scene_poses(
        ep_dir, V, cam, board_args, tcp_args, not args.no_cache, annotate)
    if len(pose_t) < 2:
        return [], 'no tracked poses', None
    pose_t, poses, spike_keep = drop_spikes(pose_t, poses, MAX_SPIKE)
    n_spike = int((~spike_keep).sum())
    wrist_t = np.load(os.path.join(ep_dir, 'wrist',
                                   'wrist_ts.npz'))['t_ns'].astype(np.float64) / 1e9
    grid, widx, gap_ok, near_ok = build_grid(
        pose_t, wrist_t, GRID_HZ, MAX_POSE_GAP, MAX_FRAME_DIST, DELTA)
    if len(grid) == 0:
        return [], 'no temporal overlap between the two cameras', None
    valid = gap_ok & near_ok

    min_len = int(round(MIN_SEGMENT * GRID_HZ))
    segs = segments(valid, min_len)
    kept = sum(e - s for s, e in segs)
    note = (f'tracked {len(pose_t)/max(n_scene,1):.1%} '
            f'({n_spike} spike{"" if n_spike == 1 else "s"} dropped), grid {len(grid)}, '
            f'valid {valid.mean():.1%}, {len(segs)} segment(s), '
            f'kept {kept}/{len(grid)} = {kept/len(grid):.1%}')
    # the gate is on how much of the recording is USABLE, not on how much survives
    # segmentation -- a recording can be 99% valid yet fragmented into runs too short
    # to train on, and that is a note about the split, not a reason to bin the episode
    # annotation payload is built even when the episode is rejected -- a rejected
    # episode is exactly the one you want to look at
    def ann(names):
        if not annotate or frames is None:
            return None
        ts, dets, rows, good_idx = frames
        return {'ts': ts, 'dets': dets, 'rows': rows, 'good_idx': good_idx,
                'spike_keep': spike_keep, 'grid': grid, 'gap_ok': gap_ok,
                'near_ok': near_ok, 'segs': segs, 'seg_names': names}

    if valid.mean() < MIN_VALID:
        return ([], note + f' -- valid below MIN_VALID {MIN_VALID:.0%}',
                ann([]))
    if not segs:
        return ([], note + f' -- no run reached MIN_SEGMENT {MIN_SEGMENT:.1f}s',
                ann([]))

    # demo_start/end are scoped to the whole RECORDING, not to each segment, matching
    # 06_generate_dataset_plan.py: it takes first_valid_step/last_valid_step over the
    # entire demo and hands the same pair to every segment cut from it. A segment
    # boundary is an artefact of a dropped-frame gap, not the start of a new
    # demonstration, so "where the demo began" must not move when a recording splits.
    vi = np.nonzero(valid)[0]
    ends_pose = poses_on_grid(grid[[vi[0], vi[-1]]], pose_t, poses, DELTA)

    base = os.path.basename(ep_dir)
    out = []
    for i, (s, e) in enumerate(segs):
        out.append({'dir': ep_dir, 'seg': i, 'grid': grid[s:e],
                    'wrist_idx': widx[s:e],
                    'pose': poses_on_grid(grid[s:e], pose_t, poses, DELTA),
                    'demo_start': ends_pose[0], 'demo_end': ends_pose[1]})
    names = [f'{base}_s{i}' if len(segs) > 1 else base for i in range(len(segs))]
    return out, note, ann(names)


# -------------------------------------------------------------- annotation
GREEN, AMBER, RED, GREY = (90, 210, 90), (40, 190, 245), (60, 60, 235), (150, 150, 150)


def frame_verdicts(a):
    """What became of every scene frame. -> list of (key, text, colour), per frame.

    key is a short tag for the end-of-episode tally; text is what goes on screen.
    They are separate so the two INVALID reasons stay distinguishable when counted --
    collapsing them would throw away exactly what splitting gap_ok/near_ok bought.

    Bridging two different indexings, which is the whole difficulty here:
      * dets/rows are per SCENE FRAME, at the camera's ~120 Hz
      * grid/valid/segments are per 60 Hz TIME step, with no frame correspondence

    So a frame's grid verdict is found by time, via the nearest grid index -- never
    by index arithmetic, which would silently desynchronise the moment a scene frame
    was dropped. Spike membership goes the other way: good_idx[~spike_keep] maps the
    filtered pose array back onto frame numbers.
    """
    ts, rows, grid = a['ts'], a['rows'], a['grid']
    n = len(rows)
    spiked = set(int(i) for i in a['good_idx'][~a['spike_keep']])

    # which segment, if any, owns each grid index
    seg_of = np.full(len(grid), -1, dtype=int)
    for k, (s, e) in enumerate(a['segs']):
        seg_of[s:e] = k

    out = []
    for i in range(n):
        if rows[i][0] is None:
            out.append(('lost', 'LOST - board not tracked', RED))
            continue
        if i in spiked:
            out.append(('spike', 'SPIKE DROPPED - depth-ambiguity flip', RED))
            continue
        t = ts[i]
        if len(grid) == 0 or t < grid[0] or t > grid[-1]:
            out.append(('outside', "OUTSIDE GRID - beyond the cameras' overlap", GREY))
            continue
        j = int(np.clip(np.searchsorted(grid, t), 1, len(grid) - 1))
        g = j - 1 if abs(grid[j - 1] - t) <= abs(grid[j] - t) else j
        if not a['gap_ok'][g]:
            out.append(('pose-gap', 'INVALID - pose gap too wide to interpolate', AMBER))
        elif not a['near_ok'][g]:
            out.append(('no-wrist', 'INVALID - no wrist frame close enough', AMBER))
        elif seg_of[g] < 0:
            out.append(('trimmed', 'TRIMMED - run too short to keep', AMBER))
        else:
            out.append(('kept', f'KEPT -> {a["seg_names"][seg_of[g]]}', GREEN))
    return out


def write_annotated(ep_dir, V, cam, a, out_fps, crf=23):
    """Render scene_annotated.mp4: tracking overlay + what build_zarr did with it.

    Encoded with libx264 through PyAV rather than cv2's mp4v. That is not stylistic:
    mp4v produced 243 MB for a 6-second episode here (321 KB/frame, worse than raw
    noise), which over a 20-episode session is several gigabytes of debug video.
    x264 at crf 23 gives the same picture in a fraction of the space, and av is
    already this file's decoder so it adds no dependency.

    Decoding is also PyAV, which sidesteps a trap: cv2 reports CAP_PROP_FPS=1000 and
    an 8x-inflated frame count on these files, because capture.py rebases PTS onto a
    1 ms timebase. The playback rate here comes from the timestamp sidecar instead --
    the only honest source.
    """
    src = os.path.join(ep_dir, 'scene', 'scene.mkv')
    dst = os.path.join(ep_dir, 'scene', 'scene_annotated.mp4')
    verdicts = frame_verdicts(a)
    ts = a['ts']
    real_fps = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else out_fps

    tally, i = {}, 0
    inp = av.open(src)
    ist = inp.streams.video[0]
    ist.thread_count = 4
    w, h = int(ist.width), int(ist.height)
    out = av.open(dst, 'w')
    ost = out.add_stream('libx264', rate=Fraction(out_fps).limit_denominator(1000))
    ost.width, ost.height, ost.pix_fmt = w, h, 'yuv420p'
    ost.options = {'crf': str(crf), 'preset': 'veryfast'}
    try:
        for frame in inp.decode(ist):
            if i >= len(verdicts):
                break
            img = V.draw_frame(frame.to_ndarray(format='bgr24'),
                               a['dets'][i], a['rows'][i], cam)
            key, text, colour = verdicts[i]
            tally[key] = tally.get(key, 0) + 1

            cv2.rectangle(img, (0, h - 92), (w, h), (18, 18, 18), -1)
            cv2.rectangle(img, (0, h - 92), (14, h), colour, -1)
            for txt, y, sc, col in (
                    (text, h - 54, 1.0, colour),
                    (f'frame {i}/{len(verdicts)}   t={ts[i]-ts[0]:6.3f}s   '
                     f'captured {real_fps:.1f} fps, playing {out_fps:.0f} fps '
                     f'({real_fps/max(out_fps,1e-6):.1f}x slow motion)',
                     h - 20, 0.62, GREY)):
                cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            (0, 0, 0), 4 if sc > 0.8 else 3, cv2.LINE_AA)
                cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                            2 if sc > 0.8 else 1, cv2.LINE_AA)
            for p in ost.encode(av.VideoFrame.from_ndarray(img, format='bgr24')):
                out.mux(p)
            i += 1
            if i % 100 == 0:
                print(f'      rendering {i}/{len(verdicts)} ...', end='\r', flush=True)
        for p in ost.encode(None):          # flush the encoder
            out.mux(p)
    finally:
        inp.close()
        out.close()

    size = os.path.getsize(dst) / 1e6 if os.path.exists(dst) else 0
    print(f'      wrote scene_annotated.mp4  {i} frames, {size:.0f} MB     ')
    print('        ' + '  '.join(f'{k} {v}' for k, v in sorted(
        tally.items(), key=lambda kv: -kv[1])))
    return dst


def select_episodes(names, ep_dirs):
    """Resolve --annotate-episode values against the episodes actually found.

    A bare number is accepted ('19' -> ep019) because typing the zero padding is a
    nuisance. An unmatched name is fatal rather than skipped: a typo would otherwise
    be indistinguishable from a finished render, and only noticed after the run.
    """
    if not names:
        return None                      # None means "all"
    have = {os.path.basename(d) for d, _ in ep_dirs}
    want, bad = set(), []
    for n in names:
        cand = n if n in have else f'ep{int(n):03d}' if n.isdigit() else n
        if cand in have:
            want.add(cand)
        else:
            bad.append(n)
    if bad:
        sys.exit(f'--annotate-episode: no such episode {", ".join(bad)}\n'
                 f'  available: {", ".join(sorted(have))}')
    return want


def episode_lowdim(ep, gripper_width):
    """The six low-dim arrays for one segment.

    demo_start/end come from the recording, not from this segment -- see plan_episode.
    Only demo_start is actually read by UmiDataset (to build the *_wrt_start rotation
    feature, with noise added); demo_end is carried for schema parity with 07.
    """
    p = ep['pose']
    n = len(p)
    start = np.broadcast_to(ep['demo_start'], (n, 6)).astype(np.float32)
    end = np.broadcast_to(ep['demo_end'], (n, 6)).astype(np.float32)
    return {
        'robot0_eef_pos': p[:, :3].copy(),
        'robot0_eef_rot_axis_angle': p[:, 3:].copy(),
        'robot0_gripper_width': np.full((n, 1), gripper_width, dtype=np.float32),
        'robot0_demo_start_pose': start.copy(),
        'robot0_demo_end_pose': end.copy(),
    }


def fill_images(ep, img_array, buffer_start, tf):
    """Decode the wrist video once, writing the selected frames to their buffer slots.

    Sequential decode with selection, not seeking: h264 random access is unreliable,
    and at 60 Hz off an ~87 fps source most frames are wanted anyway.
    """
    want = {}
    for i, fi in enumerate(ep['wrist_idx']):
        want.setdefault(int(fi), []).append(buffer_start + i)
    c = av.open(os.path.join(ep['dir'], 'wrist', 'wrist.mkv'))
    st = c.streams.video[0]
    st.thread_count = 4
    written = 0
    for idx, frame in enumerate(c.decode(st)):
        slots = want.pop(idx, None)
        if slots is None:
            continue
        img = tf(frame.to_ndarray(format='rgb24'))
        for s in slots:
            img_array[s] = img
        written += len(slots)
        if not want:
            break
    c.close()
    return written


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sessions', nargs='+', help='capture.py session directories')
    ap.add_argument('-o', '--output', required=True, help='dataset.zarr.zip path')
    ap.add_argument('--intrinsics', default=None,
                    help=f'scene-camera calibration (default: {CALIB_JSON}). It is '
                         f'cross-checked against each session\'s recorded focus and '
                         f'zoom either way.')
    ap.add_argument('--no-cache', action='store_true',
                    help='re-run ChArUco detection instead of using the cache')
    ap.add_argument('--annotate-video', action='store_true',
                    help='render <ep>/scene/scene_annotated.mp4: the ChArUco tracking '
                         'overlay plus what this script did with each frame -- kept '
                         'into which segment, or dropped as a spike / pose gap / no '
                         'wrist frame / too-short run. Off by default; costs roughly '
                         '30-60 s and 30-60 MB per episode.')
    ap.add_argument('--annotate-episode', nargs='*', default=None, metavar='NAME',
                    help='limit --annotate-video to these episodes (default: all). '
                         'Accepts ep019 or a bare 19. Matches on the directory name, '
                         'so with several sessions it selects that episode in each.')
    ap.add_argument('--annotate-fps', type=float, default=30.0,
                    help='playback rate of the annotated video (default: 30). Well '
                         'below the 120 fps capture rate on purpose -- at 8 ms per '
                         'frame the overlay is unreadable. The banner states the real '
                         'capture rate and the slow-motion factor.')
    args = ap.parse_args()

    if args.annotate_episode is not None and not args.annotate_video:
        print('  warning: --annotate-episode has no effect without --annotate-video; '
              'nothing will be rendered\n')

    board_args = (TRACK_BOARD['size'], TRACK_BOARD['square'],
                  TRACK_BOARD['marker'], TRACK_BOARD['dict'], False)
    tcp_args = V.make_T(V.parse_rotation(TCP_ROTATION), TCP_OFFSET)
    out_res = (OUT_RES, OUT_RES)

    # Intrinsics are resolved PER SESSION, not once globally: each session records
    # the focus and zoom it ran at, and that is what the calibration is checked
    # against. Mixing sessions in one dataset is legitimate -- the scene camera
    # only ever produces poses, and a solved pose carries no trace of the lens.
    ep_dirs = []
    for s in args.sessions:
        s = os.path.abspath(os.path.expanduser(s))
        print(f'{os.path.basename(s)}')
        ipath, _ = resolve_intrinsics(s, args.intrinsics)
        eps = [os.path.join(s, d) for d in sorted(os.listdir(s))
               if d.startswith('ep') and os.path.isdir(os.path.join(s, d))]
        if not eps:
            print('  no episodes\n')
            continue
        w, h = scene_resolution(eps[0])
        cam = V.load_intrinsics(ipath, w, h)
        print(f'  scene video {w}x{h}\n')
        ep_dirs += [(d, cam) for d in eps]
    print(f'{len(ep_dirs)} episodes found\n')

    want_ann = (select_episodes(args.annotate_episode, ep_dirs)
                if args.annotate_video else set())

    kept, skipped = [], []
    for d, cam in ep_dirs:
        name = os.path.basename(d)
        print(f'  {name} ...', flush=True)
        annotate = args.annotate_video and (want_ann is None or name in want_ann)
        segs, note, ann = plan_episode(d, V, cam, board_args, tcp_args, args, annotate)
        print(f'      {note}')
        if annotate and ann is not None:
            write_annotated(d, V, cam, ann, args.annotate_fps)
        if not segs:
            skipped.append((name, note))
            continue
        for s in segs:
            s['name'] = f'{name}_s{s["seg"]}' if len(segs) > 1 else name
            print(f'      -> {s["name"]}: {len(s["grid"])} steps '
                  f'({len(s["grid"])/GRID_HZ:.1f}s)')
            kept.append(s)

    if not kept:
        sys.exit('\nno segments passed -- nothing to write')

    # low-dim first, then one image array sized to the total, filled by buffer offset
    rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    starts = []
    total = 0
    for ep in kept:
        starts.append(total)
        rb.add_episode(data=episode_lowdim(ep, GRIPPER_WIDTH), compressors=None)
        total += len(ep['grid'])

    img_array = rb.data.require_dataset(
        name='camera0_rgb', shape=(total,) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=JpegXl(level=COMPRESSION_LEVEL, numthreads=1),
        dtype=np.uint8)

    tf = None
    print()
    for ep, buf in zip(kept, starts):
        if tf is None:
            c = av.open(os.path.join(ep['dir'], 'wrist', 'wrist.mkv'))
            st = c.streams.video[0]
            tf = get_image_transform((st.width, st.height), out_res)
            print(f'image transform: {st.width}x{st.height} -> centre crop -> '
                  f'{out_res[0]}x{out_res[1]}')
            c.close()
        n = fill_images(ep, img_array, buf, tf)
        print(f'  {ep["name"]}: {n}/{len(ep["grid"])} frames written')

    print(f'\nsaving {total} steps across {len(kept)} episodes -> {args.output}')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with zarr.ZipStore(args.output, mode='w') as zs:
        rb.save_to_store(store=zs)

    print(f'\n{"="*60}')
    print(f'included {len(kept)} episodes, {total} steps at {GRID_HZ:.0f} Hz '
          f'({total/GRID_HZ:.1f} s)')
    for name, why in skipped:
        print(f'  excluded {name}: {why}')
    print(f'wrote {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
