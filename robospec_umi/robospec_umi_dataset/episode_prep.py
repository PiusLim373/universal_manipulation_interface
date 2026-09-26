#!/usr/bin/env python3
"""Per-episode derived files for the edit UI and build_zarr, cached in <ep>/derived/.

  wrist.mp4               wrist video the browser can play and seek
  scene_charuco_dets.npz  ChArUco detections (independent of the intrinsics)
  scene_tcp.npz           TCP pose + status per scene frame
  scene_annotated.mp4     tracking overlay, on the real capture timestamps
  prep.json               written last; the cache is valid while version, board
                          and intrinsics all match

videos=False builds only what export needs (detections + pose track); the two
mp4s are added later, cheaply, if the episode is opened in the trimmer.

Each video starts at its own first frame; prep.json records wrist_offset_s, the
wrist's start on the scene clock.

Usage:
    python3 robospec_umi/robospec_umi_dataset/episode_prep.py data/capture/<stamp>/ep000 [...]
"""

import argparse
import hashlib
import json
import os
import time
from fractions import Fraction

import av
import cv2
import numpy as np

import timeline as TL
import video_processor_charuco as V

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_dataset
REPO = os.path.dirname(os.path.dirname(HERE))
CALIB_JSON = os.path.join(REPO, 'data', 'calibration', 'scene_intrinsics.json')

PREP_VERSION = 1
PREVIEW_WIDTH = 1280
GOP = 30    # keyframe interval, so a browser seek does not decode from the start
CRF = 23

GREEN, AMBER, RED, GREY = (90, 210, 90), (40, 190, 245), (60, 60, 235), (150, 150, 150)
BANNER = {
    TL.OK: ('OK', GREEN),
    TL.LOST: ('LOST - board not tracked', RED),
    TL.SPIKE: ('SPIKE - depth-ambiguity flip', RED),
    TL.OUTSIDE: ("OUTSIDE - beyond the cameras' overlap", GREY),
    TL.POSE_GAP: ('INVALID - pose gap too wide to interpolate', AMBER),
    TL.NO_WRIST: ('INVALID - no wrist frame close enough', AMBER),
}


def sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def cache_key(intr_path=CALIB_JSON):
    return {'version': PREP_VERSION, 'board': TL.board_key(),
            'intrinsics_sha1': sha1(intr_path)}


def read_prep(ep_dir):
    try:
        with open(TL.ep_path(ep_dir, 'prep.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_prepped(ep_dir, key, videos=True):
    p = read_prep(ep_dir)
    return (bool(p) and all(p.get(k) == v for k, v in key.items())
            and (not videos or p.get('videos', True)))


def summary(m):
    """One-line result of a prep.json dict."""
    u = m['untrimmed']
    kept = (f'kept {u["kept_s"]:.1f} s' if u['reason'] is None
            else f'dropped: {u["reason"]}')
    return f'tracked {m["tracked"]:.0%}, usable {u["usable"]:.0%}, {kept}'


def _save_npz(path, **arrays):
    tmp = path[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _ts(ep_dir, cam):
    return np.load(os.path.join(ep_dir, cam, f'{cam}_ts.npz'))['t_ns']


# ------------------------------------------------------------------ encoding
class Mp4:
    """x264 mp4 with per-frame timestamps in ms, written to a temp name first."""

    def __init__(self, dst, w, h):
        self.dst, self.tmp = dst, dst[:-4] + '.tmp.mp4'
        self.out = av.open(self.tmp, 'w', options={'movflags': '+faststart'})
        self.st = self.out.add_stream('libx264', rate=120)
        self.st.width, self.st.height, self.st.pix_fmt = w, h, 'yuv420p'
        self.st.codec_context.time_base = Fraction(1, 1000)
        self.st.codec_context.gop_size = GOP
        self.st.options = {'crf': str(CRF), 'preset': 'veryfast'}
        self.last = -1

    def put(self, frame, t_ms):
        if isinstance(frame, np.ndarray):
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        self.last = max(int(round(t_ms)), self.last + 1)
        frame.pts, frame.time_base = self.last, Fraction(1, 1000)
        for p in self.st.encode(frame):
            self.out.mux(p)

    def close(self):
        for p in self.st.encode(None):
            self.out.mux(p)
        self.out.close()
        os.replace(self.tmp, self.dst)


def encode_wrist(ep_dir, progress):
    ts = _ts(ep_dir, 'wrist')
    c = av.open(os.path.join(ep_dir, 'wrist', 'wrist.mkv'))
    ist = c.streams.video[0]
    ist.thread_count = 4
    out = Mp4(TL.ep_path(ep_dir, 'wrist.mp4'), ist.width, ist.height)
    i = 0
    try:
        for frame in c.decode(ist):
            if i >= len(ts):
                break
            out.put(frame, (ts[i] - ts[0]) / 1e6)
            i += 1
            if i % 100 == 0:
                progress('wrist', i, len(ts))
        out.close()
    finally:
        c.close()
    return i


# --------------------------------------------------------------- scene track
def detect(ep_dir, progress):
    path = TL.ep_path(ep_dir, 'scene_charuco_dets.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if 'board' in z.files and str(z['board']) == TL.board_key():
            return list(z['dets'])
    detect_fn = V.make_detector(V.make_board(*TL.board_args()))
    n = len(_ts(ep_dir, 'scene'))
    c = av.open(os.path.join(ep_dir, 'scene', 'scene.mkv'))
    st = c.streams.video[0]
    st.thread_count = 4
    dets = []
    try:
        for f in c.decode(st):
            m, cc, ci = detect_fn(f.to_ndarray(format='gray'))
            dets.append({'markers': m, 'corners': cc, 'ids': ci})
            if len(dets) % 50 == 0:
                progress('detect', len(dets), n)
    finally:
        c.close()
    _save_npz(path, dets=np.array(dets, dtype=object), n_frames=len(dets),
              board=TL.board_key())
    return dets


def track(ep_dir, dets, cam):
    """-> (rows, status, t_ns); writes scene_tcp.npz."""
    ts = _ts(ep_dir, 'scene')
    n = min(len(dets), len(ts))      # an interrupted recording can differ by a tail
    dets, ts = dets[:n], ts[:n]
    chess, markers = V.board_geometry(V.make_board(*TL.board_args()))
    tcp = V.make_T(V.parse_rotation(TL.TCP_ROTATION), TL.TCP_OFFSET)
    rows = V.track(dets, chess, markers, cam, V.board_centre(chess), tcp)

    tracked = np.array([r[1] is not None for r in rows], bool)
    T = np.full((n, 4, 4), np.nan)
    good = np.nonzero(tracked)[0]
    for i in good:
        T[i] = rows[i][1]

    t = ts.astype(np.float64) / 1e9
    spiked = np.zeros(n, bool)
    grid, gap_ok, near_ok = np.empty(0), np.empty(0, bool), np.empty(0, bool)
    if len(good) >= 2:
        pose_t, _, keep = TL.drop_spikes(t[good], [T[i] for i in good], TL.MAX_SPIKE)
        spiked[good[~keep]] = True
        grid, _, gap_ok, near_ok = TL.build_grid(
            pose_t, TL.wrist_times(ep_dir), TL.GRID_HZ, TL.MAX_POSE_GAP,
            TL.MAX_FRAME_DIST, TL.DELTA)
    status = TL.frame_status(t, tracked, spiked, grid, gap_ok, near_ok)

    _save_npz(TL.ep_path(ep_dir, TL.TCP_NPZ), t_ns=ts, T_tcp=T,
              rms=np.array([r[5] for r in rows], np.float64),
              n_pts=np.array([r[2] for r in rows], np.int32), status=status)
    return rows, status, ts


def render(ep_dir, dets, rows, status, ts, cam, progress):
    c = av.open(os.path.join(ep_dir, 'scene', 'scene.mkv'))
    ist = c.streams.video[0]
    ist.thread_count = 4
    w, h = int(ist.width), int(ist.height)
    ow = PREVIEW_WIDTH
    oh = int(round(h * ow / w / 2)) * 2
    out = Mp4(TL.ep_path(ep_dir, 'scene_annotated.mp4'), ow, oh)
    n = len(rows)
    try:
        for i, frame in enumerate(c.decode(ist)):
            if i >= n:
                break
            img = V.draw_frame(frame.to_ndarray(format='bgr24'), dets[i], rows[i], cam)
            text, colour = BANNER[int(status[i])]
            cv2.rectangle(img, (0, h - 92), (w, h), (18, 18, 18), -1)
            cv2.rectangle(img, (0, h - 92), (14, h), colour, -1)
            for txt, y, sc, col in (
                    (text, h - 54, 1.0, colour),
                    (f'frame {i}/{n}   t={(ts[i] - ts[0]) / 1e9:6.3f}s', h - 20, 0.62, GREY)):
                cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            (0, 0, 0), 4 if sc > 0.8 else 3, cv2.LINE_AA)
                cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                            2 if sc > 0.8 else 1, cv2.LINE_AA)
            out.put(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA),
                    (ts[i] - ts[0]) / 1e6)
            if (i + 1) % 50 == 0:
                progress('render', i + 1, n)
        out.close()
    finally:
        c.close()


# ------------------------------------------------------------------- prepare
def prepare(ep_dir, intr_path=CALIB_JSON, progress=None, force=False, videos=True):
    """Build (or reuse) <ep>/derived/. -> the prep.json dict."""
    progress = progress or (lambda *a: None)
    key = cache_key(intr_path)
    if not force and is_prepped(ep_dir, key, videos):
        return read_prep(ep_dir)
    # the track is already valid and only the videos are missing: keep the marker
    track_ok = not force and is_prepped(ep_dir, key, videos=False)

    t_start = time.time()
    os.makedirs(TL.ep_path(ep_dir), exist_ok=True)
    marker = TL.ep_path(ep_dir, 'prep.json')
    if not track_ok and os.path.exists(marker):   # never leave a stale marker behind
        os.remove(marker)

    if videos and (force or not os.path.exists(TL.ep_path(ep_dir, 'wrist.mp4'))):
        encode_wrist(ep_dir, progress)
    dets_npz = TL.ep_path(ep_dir, 'scene_charuco_dets.npz')
    if force and os.path.exists(dets_npz):
        os.remove(dets_npz)
    dets = detect(ep_dir, progress)

    with av.open(os.path.join(ep_dir, 'scene', 'scene.mkv')) as c:
        w, h = c.streams.video[0].width, c.streams.video[0].height
    cam = V.load_intrinsics(intr_path, w, h)
    progress('track', 0, 1)
    rows, status, ts = track(ep_dir, dets, cam)     # cheap once detections are cached
    if videos:
        render(ep_dir, dets, rows, status, ts, cam, progress)

    wts = _ts(ep_dir, 'wrist')
    plan = TL.plan_episode(ep_dir, with_poses=False)
    counts = np.bincount(status, minlength=len(TL.STATUS))
    meta = {
        **key,
        'videos': videos,
        'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'seconds': round(time.time() - t_start, 1),
        'scene_frames': int(len(ts)),
        'duration_s': float((ts[-1] - ts[0]) / 1e9),
        'wrist_offset_s': float((wts[0] - ts[0]) / 1e9),
        'wrist_duration_s': float((wts[-1] - wts[0]) / 1e9),
        'tracked': float((status != TL.LOST).mean()),
        'status_counts': {s: int(n) for s, n in zip(TL.STATUS, counts)},
        'untrimmed': {k: plan[k] for k in ('usable', 'kept_s', 'reason', 'note')},
    }
    tmp = marker + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, marker)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+', help='episode directories (<session>/epNNN)')
    ap.add_argument('--intrinsics', default=CALIB_JSON)
    ap.add_argument('--force', action='store_true', help='rebuild even if cached')
    ap.add_argument('--no-videos', action='store_true',
                    help='only what export needs: detections and the pose track')
    args = ap.parse_args()
    av.logging.set_level(av.logging.PANIC)   # MJPEG APP-marker and pix-fmt noise
    cv2.setNumThreads(4)

    for ep in args.episodes:
        def progress(stage, i, n):
            print(f'PROGRESS {stage} {i} {n}', flush=True)
        m = prepare(os.path.abspath(ep), args.intrinsics, progress, args.force,
                    not args.no_videos)
        print(f'{ep}: {summary(m)}  ({m["seconds"]} s)', flush=True)


def prepare_tracks(ep_dir, intr_path=CALIB_JSON):
    """prepare(videos=False) in a worker process (build_zarr's pool)."""
    av.logging.set_level(av.logging.PANIC)
    cv2.setNumThreads(4)
    return prepare(ep_dir, intr_path, videos=False)


if __name__ == '__main__':
    main()
