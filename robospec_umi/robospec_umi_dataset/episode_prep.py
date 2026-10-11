#!/usr/bin/env python3
"""Per-episode derived files for the edit UI and build_zarr, cached in <ep>/derived/.

  wrist.mp4               wrist video the browser can play and seek
  scene_charuco_dets.npz  ChArUco detections (independent of the intrinsics)
  scene_tcp.npz           TCP pose + status per scene frame
  scene_annotated.mp4     tracking overlay, on the real capture timestamps
  prep.json               written last; the cache is valid while the board and
                          intrinsics match

videos=False builds only what export needs (detections + pose track); the two
mp4s are added later, cheaply, if the episode is opened in the trimmer.

Each video starts at its own first frame; prep.json records wrist_offset_s, the
wrist's start on the scene clock.

Detection and the annotated render run on a pool of worker processes sized to
the machine (usable_cores); videos are encoded with NVENC when the GPU has it,
else x264. The detections are identical whatever the worker count.

Usage:
    python3 robospec_umi/robospec_umi_dataset/episode_prep.py data/capture/<stamp>/ep000 [...]
    python3 robospec_umi/robospec_umi_dataset/episode_prep.py --calibrate-imu data/capture/<stamp>/ep*
"""

import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import deque
from fractions import Fraction

import av
import cv2
import numpy as np

import timeline as TL
import video_processor_charuco as V

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_dataset
REPO = os.path.dirname(os.path.dirname(HERE))
CALIB_JSON = os.path.join(REPO, 'data', 'calibration', 'scene_intrinsics.json')

PREVIEW_WIDTH = 1280
GOP = 30    # keyframe interval, so a browser seek does not decode from the start
CRF = 23

MAX_WORKERS = 8     # per prep; detection barely gains past 8 (6.9 s vs 6.6 s at 12)
RESERVE_CORES = 2   # left for the server, the live streams and the desktop
PREP_NICE = 10      # capture and the server keep priority over prep
RESULT_TIMEOUT = 30  # s for one frame's result (~20 ms normally); only a dead worker hits it

GREEN, AMBER, RED, GREY = (90, 210, 90), (40, 190, 245), (60, 60, 235), (150, 150, 150)
BANNER = {
    TL.OK: ('OK', GREEN),
    TL.LOST: ('LOST - board not tracked', RED),
    TL.SPIKE: ('SPIKE - pose rejected (flip or bad corners)', RED),
    TL.OUTSIDE: ("OUTSIDE - beyond the cameras' overlap", GREY),
    TL.POSE_GAP: ('INVALID - pose gap too wide to interpolate', AMBER),
    TL.NO_WRIST: ('INVALID - no wrist frame close enough', AMBER),
}


def sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def cache_key(intr_path=CALIB_JSON):
    return {'board': TL.board_key(), 'intrinsics_sha1': sha1(intr_path)}


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
    imu = ', imu fused' if (m.get('imu') or {}).get('fused') else ''
    return f'tracked {m["tracked"]:.0%}, usable {u["usable"]:.0%}, {kept}{imu}'


def _save_npz(path, **arrays):
    tmp = path[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _ts(ep_dir, cam):
    return np.load(os.path.join(ep_dir, cam, f'{cam}_ts.npz'))['t_ns']


# ------------------------------------------------------------------- workers
def _cgroup_cpus():
    """A cgroup CPU quota in cores (docker --cpus), or None."""
    for files in (('/sys/fs/cgroup/cpu.max',),
                  ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')):
        vals = []
        try:
            for f in files:
                with open(f) as fh:
                    vals += fh.read().split()
        except OSError:
            continue
        if vals[0] in ('max', '-1'):
            return None
        return int(vals[0]) / int(vals[1])
    return None


def usable_cores():
    """Cores this process may run on: its affinity, capped by a cgroup quota."""
    n = len(os.sched_getaffinity(0))
    q = _cgroup_cpus()
    return max(1, min(n, int(q)) if q else n)


def prep_workers(share):
    return max(1, min(MAX_WORKERS, int(share)))


def default_workers():
    """For a prep running on its own."""
    return prep_workers(usable_cores() - RESERVE_CORES)


def lower_priority():
    """Absolute, so calling it again does not keep lowering it."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, max(os.getpriority(os.PRIO_PROCESS, 0), PREP_NICE))
    except OSError:
        pass


_W = {}     # per worker process: the detector, a JPEG decoder and the camera


def _worker_init(cam):
    lower_priority()
    cv2.setNumThreads(1)
    av.logging.set_level(av.logging.PANIC)
    _W.update(detect=V.make_detector(V.make_board(*TL.board_args())),
              jpeg=av.CodecContext.create('mjpeg', 'r'), cam=cam)


def _decode(packet, fmt):
    """One scene JPEG through PyAV, as the serial path decodes it. cv2.imdecode
    gives slightly different pixels, and moved corners on 5 of 652 frames."""
    return _W['jpeg'].decode(av.Packet(packet))[0].to_ndarray(format=fmt)


def _detect_job(packet):
    m, cc, ci = _W['detect'](_decode(packet, 'gray'))
    return {'markers': m, 'corners': cc, 'ids': ci}


def _render_job(job):
    packet, det, row, status, label, size = job
    img = annotate(_decode(packet, 'bgr24'), det, row, status, label, _W['cam'])
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return av.VideoFrame.from_ndarray(img, format='bgr24').reformat(format='yuv420p').to_ndarray()


class Workers:
    """Prep worker processes. map() yields results in order with at most 4 jobs
    per worker in flight, so frames never pile up when the encoder lags.

    Pool replaces a worker that dies, but the job it held never returns, so each
    result has a timeout: a crash fails the prep rather than hanging it. (Not
    ProcessPoolExecutor: on 3.9 its shutdown deadlocks after a worker dies with
    packets still queued for it.)
    """

    def __init__(self, n, cam):
        self.n = n
        self.pool = mp.get_context('spawn').Pool(n, _worker_init, (cam,))

    def map(self, fn, items):
        pending = deque()
        for x in items:
            pending.append(self.pool.apply_async(fn, (x,)))
            if len(pending) >= 4 * self.n:
                yield self._get(pending.popleft())
        while pending:
            yield self._get(pending.popleft())

    @staticmethod
    def _get(r):
        try:
            return r.get(RESULT_TIMEOUT)
        except mp.TimeoutError:
            raise RuntimeError(f'a prep worker returned nothing for {RESULT_TIMEOUT} s '
                               '(it crashed or could not start)') from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.terminate()


def _packets(path):
    """The scene video's JPEG packets, demuxed lazily (no decode)."""
    c = av.open(path)
    try:
        for p in c.demux(c.streams.video[0]):
            if p.size:
                yield bytes(p)
    finally:
        c.close()


# ------------------------------------------------------------------ encoding
class Mp4:
    """H.264 mp4 with per-frame timestamps in ms, written to a temp name first.

    NVENC when the GPU offers it -- 3.5x faster than x264, and at cq 35 the same
    size and quality (3.8 MB / 33.4 dB vs 3.5 MB / 33.1 dB) -- else x264.
    bf 0: B-frames reorder the stream and the muxer rejects our pts.
    """

    CODECS = (('h264_nvenc', {'preset': 'p4', 'rc': 'vbr', 'cq': '35', 'bf': '0'}),
              ('libx264', {'crf': str(CRF), 'preset': 'veryfast'}))
    no_nvenc = False    # set after a failed open, so it is not retried per video

    def __init__(self, dst, w, h):
        self.dst, self.tmp = dst, dst[:-4] + '.tmp.mp4'
        for codec, opts in self.CODECS[1 if Mp4.no_nvenc else 0:]:
            self.out = av.open(self.tmp, 'w', options={'movflags': '+faststart'})
            try:
                self.st = self.out.add_stream(codec, rate=120)
                self.st.width, self.st.height, self.st.pix_fmt = w, h, 'yuv420p'
                self.st.codec_context.time_base = Fraction(1, 1000)
                self.st.codec_context.gop_size = GOP
                self.st.options = opts
                self.st.codec_context.open()      # no usable NVENC fails here
                self.codec = codec
                break
            except Exception:                     # noqa: BLE001
                self.out.close()
                if codec == 'libx264':
                    raise
                Mp4.no_nvenc = True
        self.last = -1

    def put(self, frame, t_ms):
        if isinstance(frame, np.ndarray):        # 2-D: a worker's ready yuv420p
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24' if frame.ndim == 3
                                               else 'yuv420p')
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
    return out.codec


# --------------------------------------------------------------- scene track
def load_dets(ep_dir):
    """Cached detections for the current board, or None."""
    path = TL.ep_path(ep_dir, 'scene_charuco_dets.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if 'board' in z.files and str(z['board']) == TL.board_key():
            return list(z['dets'])
    return None


def detect(ep_dir, progress, workers=None):
    dets = load_dets(ep_dir)
    if dets is not None:
        return dets
    n = len(_ts(ep_dir, 'scene'))
    src = os.path.join(ep_dir, 'scene', 'scene.mkv')
    if workers is not None:
        frames = workers.map(_detect_job, _packets(src))
    else:
        detect_fn = V.make_detector(V.make_board(*TL.board_args()))

        def serial():
            c = av.open(src)
            st = c.streams.video[0]
            st.thread_count = 4
            try:
                for f in c.decode(st):
                    m, cc, ci = detect_fn(f.to_ndarray(format='gray'))
                    yield {'markers': m, 'corners': cc, 'ids': ci}
            finally:
                c.close()
        frames = serial()
    dets = []
    for d in frames:
        dets.append(d)
        if len(dets) % 50 == 0:
            progress('detect', len(dets), n)
    _save_npz(TL.ep_path(ep_dir, 'scene_charuco_dets.npz'), dets=np.array(dets, dtype=object),
              n_frames=len(dets), board=TL.board_key())
    return dets


def gyro_input(ep_dir, t):
    """The episode's gyro + rig constants for V.track, or None without an IMU."""
    imu = TL.imu_track(ep_dir)
    if imu is None:
        return None
    return {'t': t, 'tg': imu['tg'], 'w': imu['w'],
            'R_board_imu': np.array(TL.IMU_ROTATION), 'offset': TL.IMU_TIME_OFFSET}


def track(ep_dir, dets, cam):
    """-> (rows, status, t_ns, imu info); writes scene_tcp.npz."""
    ts = _ts(ep_dir, 'scene')
    n = min(len(dets), len(ts))      # an interrupted recording can differ by a tail
    dets, ts = dets[:n], ts[:n]
    t = ts.astype(np.float64) / 1e9
    chess, markers = V.board_geometry(V.make_board(*TL.board_args()))
    tcp = V.make_T(V.parse_rotation(TL.TCP_ROTATION), TL.TCP_OFFSET)
    rows = V.track(dets, chess, markers, cam, V.board_centre(chess), tcp,
                   gyro=gyro_input(ep_dir, t))
    imu = V.track.imu or {'fused': False, 'reason': 'no imu recording'}

    tracked = np.array([r[1] is not None for r in rows], bool)
    # corners the fused rotation disagrees with: the board is there, the pose is not
    rejected = np.array([r[4] == 'imu rejected' for r in rows], bool)
    T = np.full((n, 4, 4), np.nan)
    good = np.nonzero(tracked)[0]
    for i in good:
        T[i] = rows[i][1]

    spiked = rejected.copy()
    grid, gap_ok, near_ok = np.empty(0), np.empty(0, bool), np.empty(0, bool)
    if len(good) >= 2:
        pose_t, _, keep = TL.drop_spikes(t[good], [T[i] for i in good], TL.MAX_SPIKE)
        spiked[good[~keep]] = True
        grid, _, gap_ok, near_ok = TL.build_grid(
            pose_t, TL.wrist_times(ep_dir), TL.GRID_HZ, TL.MAX_POSE_GAP,
            TL.MAX_FRAME_DIST, TL.DELTA)
    status = TL.frame_status(t, tracked | rejected, spiked, grid, gap_ok, near_ok)

    _save_npz(TL.ep_path(ep_dir, TL.TCP_NPZ), t_ns=ts, T_tcp=T,
              rms=np.array([r[5] for r in rows], np.float64),
              n_pts=np.array([r[2] for r in rows], np.int32), status=status)
    return rows, status, ts, imu


def annotate(img, det, row, status, label, cam):
    """The tracking overlay plus the status banner, drawn in place."""
    img = V.draw_frame(img, det, row, cam)
    h, w = img.shape[:2]
    text, colour = BANNER[int(status)]
    cv2.rectangle(img, (0, h - 92), (w, h), (18, 18, 18), -1)
    cv2.rectangle(img, (0, h - 92), (14, h), colour, -1)
    for txt, y, sc, col in ((text, h - 54, 1.0, colour), (label, h - 20, 0.62, GREY)):
        cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (0, 0, 0), 4 if sc > 0.8 else 3, cv2.LINE_AA)
        cv2.putText(img, txt, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                    2 if sc > 0.8 else 1, cv2.LINE_AA)
    return img


def render(ep_dir, dets, rows, status, ts, cam, progress, workers=None):
    """-> the codec scene_annotated.mp4 was encoded with."""
    src = os.path.join(ep_dir, 'scene', 'scene.mkv')
    with av.open(src) as c:
        w, h = int(c.streams.video[0].width), int(c.streams.video[0].height)
    ow = PREVIEW_WIDTH
    oh = int(round(h * ow / w / 2)) * 2
    n = len(rows)

    def label(i):
        return f'frame {i}/{n}   t={(ts[i] - ts[0]) / 1e9:6.3f}s'

    if workers is not None:
        jobs = ((p, dets[i], rows[i], status[i], label(i), (ow, oh))
                for i, p in zip(range(n), _packets(src)))
        frames = workers.map(_render_job, jobs)
    else:
        def serial():
            c = av.open(src)
            st = c.streams.video[0]
            st.thread_count = 4
            try:
                for i, f in zip(range(n), c.decode(st)):
                    img = annotate(f.to_ndarray(format='bgr24'), dets[i], rows[i], status[i],
                                   label(i), cam)
                    yield cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            finally:
                c.close()
        frames = serial()
    out = Mp4(TL.ep_path(ep_dir, 'scene_annotated.mp4'), ow, oh)
    for i, img in enumerate(frames):
        out.put(img, (ts[i] - ts[0]) / 1e6)
        if (i + 1) % 50 == 0:
            progress('render', i + 1, n)
    out.close()
    return out.codec


# ------------------------------------------------------------------- prepare
def prepare(ep_dir, intr_path=CALIB_JSON, progress=None, force=False, videos=True,
            workers=None):
    """Build (or reuse) <ep>/derived/. -> the prep.json dict.
    workers: processes for detection and the render; None sizes it to the machine."""
    progress = progress or (lambda *a: None)
    key = cache_key(intr_path)
    if not force and is_prepped(ep_dir, key, videos):
        return read_prep(ep_dir)
    # the track is already valid and only the videos are missing: keep the marker
    track_ok = not force and is_prepped(ep_dir, key, videos=False)

    lower_priority()
    workers = default_workers() if workers is None else max(1, workers)
    t_start = time.time()
    os.makedirs(TL.ep_path(ep_dir), exist_ok=True)
    marker = TL.ep_path(ep_dir, 'prep.json')
    if not track_ok and os.path.exists(marker):   # never leave a stale marker behind
        os.remove(marker)

    stage, encoder = {}, None

    def lap(name, t0):
        stage[name] = round(time.time() - t0, 2)
        return time.time()

    t = time.time()
    if videos and (force or not os.path.exists(TL.ep_path(ep_dir, 'wrist.mp4'))):
        encoder = encode_wrist(ep_dir, progress)
        t = lap('wrist', t)
    dets_npz = TL.ep_path(ep_dir, 'scene_charuco_dets.npz')
    if force and os.path.exists(dets_npz):
        os.remove(dets_npz)
    dets = load_dets(ep_dir)

    with av.open(os.path.join(ep_dir, 'scene', 'scene.mkv')) as c:
        w, h = c.streams.video[0].width, c.streams.video[0].height
    cam = V.load_intrinsics(intr_path, w, h)
    # a pool only when there is frame work for it: starting one costs ~1 s
    pool = Workers(workers, cam) if workers > 1 and (dets is None or videos) else None
    with pool or contextlib.nullcontext():
        if dets is None:
            dets = detect(ep_dir, progress, pool)
            t = lap('detect', t)
        progress('track', 0, 1)
        rows, status, ts, imu = track(ep_dir, dets, cam)   # cheap once detections are cached
        t = lap('track', t)
        if videos:
            encoder = render(ep_dir, dets, rows, status, ts, cam, progress, pool)
            t = lap('render', t)

    wts = _ts(ep_dir, 'wrist')
    plan = TL.plan_episode(ep_dir, with_poses=False)
    counts = np.bincount(status, minlength=len(TL.STATUS))
    meta = {
        **key,
        'videos': videos,
        'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'seconds': round(time.time() - t_start, 1),
        'workers': workers if pool is not None else 1,
        'encoder': encoder,
        'stage_s': stage,
        'scene_frames': int(len(ts)),
        'duration_s': float((ts[-1] - ts[0]) / 1e9),
        'wrist_offset_s': float((wts[0] - ts[0]) / 1e9),
        'wrist_duration_s': float((wts[-1] - wts[0]) / 1e9),
        'tracked': float((status != TL.LOST).mean()),
        'status_counts': {s: int(n) for s, n in zip(TL.STATUS, counts)},
        'imu': imu,
        'untrimmed': {k: plan[k] for k in ('usable', 'kept_s', 'reason', 'note')},
    }
    tmp = marker + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, marker)
    return meta


def calibrate_imu(ep_dirs, intr_path):
    """Pool the episodes' vision tracks and gyros -> print the IMU rig constants."""
    chess, markers = V.board_geometry(V.make_board(*TL.board_args()))
    T_c = V.board_centre(chess)
    tcp = V.make_T(V.parse_rotation(TL.TCP_ROTATION), TL.TCP_OFFSET)
    eps, cam = [], None
    for d in ep_dirs:
        imu = TL.imu_track(d)
        if imu is None:
            print(f'{d}: no imu recording, skipped')
            continue
        dets = detect(d, lambda *a: None)
        ts = _ts(d, 'scene')
        n = min(len(dets), len(ts))
        dets, t = dets[:n], ts[:n].astype(np.float64) / 1e9
        if cam is None:
            with av.open(os.path.join(d, 'scene', 'scene.mkv')) as c:
                cam = V.load_intrinsics(intr_path, c.streams.video[0].width,
                                        c.streams.video[0].height)
        rows = V.track(dets, chess, markers, cam, T_c, tcp)
        R, sig = V.vision_rotations(dets, rows, chess, markers, cam, T_c)
        ok = np.isfinite(sig)
        good = np.nonzero(ok)[0]
        if len(good) >= 2:      # vision spikes stay out of the fit
            _, _, keep = TL.drop_spikes(t[good], [rows[i][1] for i in good], TL.MAX_SPIKE)
            ok[good[~keep]] = False
        eps.append({'t': t, 'R': R, 'sig': sig, 'ok': ok, 'tg': imu['tg'], 'w': imu['w']})
    if not eps:
        raise SystemExit('no episode with an imu recording')
    off, R_bi, b, st = V.calibrate_gyro(eps)
    moved = np.degrees(np.linalg.norm(V.so3_log(R_bi.T @ np.array(TL.IMU_ROTATION))))
    print(f'calibration over {len(eps)} episodes: speed corr {st["speed_corr"]:.3f}, '
          f'increments {st["increments"]}, residual {st["residual_deg"]:.2f} deg '
          f'per {V.INC_S} s')
    print(f'gyro bias {np.degrees(b).round(3)} deg/s (fitted per episode in the pipeline)')
    print(f'{moved:.2f} deg from the current IMU_ROTATION, '
          f'{(off - TL.IMU_TIME_OFFSET) * 1000:+.1f} ms from IMU_TIME_OFFSET\n')
    rows = ',\n                '.join('(' + ', '.join(f'{v:.4f}' for v in r) + ')' for r in R_bi)
    print(f'IMU_ROTATION = ({rows})\nIMU_TIME_OFFSET = {off:.4f}   # s')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episodes', nargs='+', help='episode directories (<session>/epNNN)')
    ap.add_argument('--intrinsics', default=CALIB_JSON)
    ap.add_argument('--force', action='store_true', help='rebuild even if cached')
    ap.add_argument('--no-videos', action='store_true',
                    help='only what export needs: detections and the pose track')
    ap.add_argument('--calibrate-imu', action='store_true',
                    help='print IMU_ROTATION / IMU_TIME_OFFSET for timeline.py from '
                         'these episodes, and prep nothing')
    ap.add_argument('--workers', type=int, default=None,
                    help='processes for detection and the render (default: sized to '
                         'this machine, 1 = serial)')
    args = ap.parse_args()
    av.logging.set_level(av.logging.PANIC)   # MJPEG APP-marker and pix-fmt noise
    cv2.setNumThreads(4)
    if args.calibrate_imu:
        return calibrate_imu([os.path.abspath(e) for e in args.episodes], args.intrinsics)

    for ep in args.episodes:
        def progress(stage, i, n):
            print(f'PROGRESS {stage} {i} {n}', flush=True)
        m = prepare(os.path.abspath(ep), args.intrinsics, progress, args.force,
                    not args.no_videos, args.workers)
        print(f'{ep}: {summary(m)}  ({m["seconds"]} s, {m.get("workers", 1)} worker(s), '
              f'{m.get("encoder") or "no video"})', flush=True)


def prepare_tracks(ep_dir, intr_path=CALIB_JSON, workers=None):
    """prepare(videos=False) in a worker process (build_zarr's pool)."""
    av.logging.set_level(av.logging.PANIC)
    cv2.setNumThreads(4)
    return prepare(ep_dir, intr_path, videos=False, workers=workers)


if __name__ == '__main__':
    main()
