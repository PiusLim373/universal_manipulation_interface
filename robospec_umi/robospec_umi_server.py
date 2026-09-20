#!/usr/bin/env python3
"""robospec_umi control server -- serves the web UI and drives the pipeline.

    python3 robospec_umi/robospec_umi_server.py
    python3 robospec_umi/robospec_umi_server.py --host 0.0.0.0 --port 8080

Built on aiohttp because it is already in the conda env (FastAPI is not), and
because web.StreamResponse handles both of the streaming shapes this needs --
multipart/x-mixed-replace for camera frames and text/event-stream for state --
without any extra machinery.

THREE TRANSPORTS, ONE EACH FOR WHAT IT IS GOOD AT
  frames   multipart/x-mixed-replace on a plain GET. Binary-native, and <img src>
           renders it with no JavaScript. NOT SSE: text/event-stream is UTF-8
           only, so every JPEG would need base64 -- a third more bytes plus an
           encode and a decode on a path that is otherwise a straight copy.
  state    SSE. Small JSON at ~8 Hz is exactly what it is for.
  commands POST, because they need an acknowledgement.

WHY THE CAMERA WORK RUNS IN THREADS
cap.read() blocks. Run it on the event loop and every other request stalls behind
it. So each session owns a worker thread that steps the camera and drops the
newest (jpeg, state) into a single overwritten slot; the HTTP handlers only ever
read that slot. A slow or absent client therefore misses frames instead of
applying back-pressure to the camera, which is the correct trade for a preview.
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from aiohttp import web

# OpenCV/TBB defaults to one worker per core (16 here). Only the resize
# parallelises -- decode, encode and the mask are serial -- so the rest of the
# pool just spins, costing 1300% CPU and running SLOWER than 4 threads.
# Must precede any parallel region. OPENCV_FOR_THREADS_NUM is ignored by this build.
cv2.setNumThreads(4)

# Shared by the worker (paces its decode) and the multipart writer (paces its
# sends). Keep them equal: a faster worker is wasted, a faster writer resends.
PUBLISH_FPS = 30

# One session key for the whole capture flow, because ONE session spans both the
# Preview-and-Lock stage and the recording stage. The cameras are opened once, on
# entry to stage 1, and stay open: repeated rs.pipeline() start/stop demonstrably
# stalls the D405 (it needs dev.hardware_reset() to come back), and streaming
# continuously also pays the 3 s warm-up once, while the operator is tuning,
# rather than again on the way into recording.
CAPTURE_KEY = 'capture.session'

# ------------------------------------------------------------------- logging
# Nothing here called basicConfig before, so aiohttp's access logger emitted at
# INFO into a logger with no handler and was silently dropped -- the terminal
# showed six print()s and nothing else, while eleven worker-thread failures
# latched into a `.err` string that nobody ever read. Configured in main().
log = logging.getLogger('robospec')
log_cam = log.getChild('camera')      # leases, sessions, worker threads
log_cap = log.getChild('capture')     # recording lifecycle
log_job = log.getChild('job')         # subprocesses


class QuietLibav(logging.Filter):
    """Hide three libav messages that fire every single session and mean nothing.

    Configuring logging also routes PyAV's ffmpeg messages here, and all three of
    these arrive at ERROR:

      unable to decode APP fields      the DECXIN's MJPEG carries a non-standard
                                       APP marker. The frames themselves are
                                       fine -- verified decoding one in PIL and
                                       in Chrome, DHT present, baseline JPEG.
      buffers still owned on close     V4L2 teardown, after the last frame.
      ioctl(VIDIOC_QBUF)               the same teardown, one line later.

    Left in, they train you to ignore ERROR lines, which costs more than they
    are worth. Visible again under -v.
    """

    BENIGN = ('unable to decode APP fields',
              'Some buffers are still owned by the caller',
              'ioctl(VIDIOC_QBUF)')

    def filter(self, record):
        if record.levelno > logging.DEBUG and log.level > logging.DEBUG:
            msg = record.getMessage()
            return not any(b in msg for b in self.BENIGN)
        return True


class QuietPaths(logging.Filter):
    """Drop access-log lines for endpoints the UI polls.

    useCamera() hits /api/camera every 2.5 s, so without this the terminal is
    24 lines a minute of nothing and the events that matter scroll away.
    Streaming endpoints need no special case: aiohttp logs a request when the
    response COMPLETES, so an MJPEG stream logs once on disconnect, which is
    worth seeing.
    """

    NOISY = ('/api/camera', '/api/capture/session', '/api/calibration/session',
             '/assets/', '/api/health')

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.NOISY)

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi
REPO = os.path.dirname(HERE)                        # repo root
DATA = os.path.join(REPO, 'data')
UI_DIST = os.path.join(HERE, 'robospec_umi_ui', 'dist')
CALIB_ROOT = os.path.join(DATA, 'calibration')
CALIB_DIR = os.path.join(HERE, 'robospec_umi_calibration')
CAPTURE_DIR = os.path.join(HERE, 'robospec_umi_capture')
CAPTURE_ROOT = os.path.join(DATA, 'capture')
ACTIVE_JSON = os.path.join(CALIB_ROOT, 'scene_intrinsics.json')

sys.path.insert(0, CALIB_DIR)
sys.path.insert(0, CAPTURE_DIR)
import calibrate_scene_cam as CS                        # noqa: E402
import scene_cam_charuco_detector as DT                 # noqa: E402
import preview as PV                                    # noqa: E402
import capture as CAP                                   # noqa: E402

JPEG_QUALITY = 80
# Downscaled from 1080p before encoding: the operator is aiming a board, not
# inspecting pixels, and the full frame costs bandwidth for detail no one reads.
# 1280 rather than 960 because the browser will not scale an <img> UP past its
# natural size -- at 960 the preview simply sat at 960x540 in a much larger
# panel. Encoding cost is ~1.8x the pixels, which is a millisecond or so.
PREVIEW_WIDTH = 1280
IDLE_RELEASE_S = 600     # a lease with no subscribers self-releases after this


# ============================================================ camera ownership
class Busy(Exception):
    """Raised when a device is already leased. Carries the current owner so the
    UI can name it instead of showing a bare failure."""

    def __init__(self, device, owner):
        super().__init__(f'{device} is in use by {owner}')
        self.device, self.owner = device, owner


class Lease:
    def __init__(self, device, owner):
        self.device, self.owner = device, owner
        self.since = time.time()
        self.subscribers = 0
        self.idle_since = time.time()
        self.recording = False      # an episode is armed on this device


class CameraBroker:
    """One streaming owner per device, because V4L2 says so.

    Measured on this rig: a second VideoCapture on the same node returns
    isOpened() False. But v4l2-ctl reads AND WRITES controls while another
    process streams -- so the kernel will happily let you change focus in the
    middle of a recording, silently invalidating every pose derived from it.
    That is what may_write_controls() exists to prevent; the OS will not.
    """

    DEVICES = ('scene', 'wrist')

    def __init__(self):
        self._lock = threading.Lock()
        self._leases = {d: None for d in self.DEVICES}

    def acquire(self, device, owner):
        return self.acquire_all((device,), owner)[device]

    def acquire_all(self, devices, owner):
        """All or nothing, under ONE pass of the lock.

        Two sequential acquire() calls would not be atomic against a competing
        request -- the lock is taken and dropped per call, so another session can
        slip in between them and both end up half-owning a camera. Checking every
        device before taking any also means there is nothing to roll back.
        """
        with self._lock:
            for d in devices:
                cur = self._leases.get(d)
                if cur is not None:
                    raise Busy(d, cur.owner)
            for d in devices:
                self._leases[d] = Lease(d, owner)
            log_cam.info('acquired %s -> %s', '+'.join(devices), owner)
            return {d: self._leases[d] for d in devices}

    def release(self, device, owner=None, force=False):
        with self._lock:
            cur = self._leases.get(device)
            if cur is None:
                return False
            if not force and owner is not None and cur.owner != owner:
                return False
            self._leases[device] = None
            log_cam.info('released %s (was %s, held %.0fs)', device, cur.owner,
                         time.time() - cur.since)
            return True

    def release_all(self, devices, owner=None, force=False):
        return [d for d in devices if self.release(d, owner, force)]

    def set_recording(self, devices, on):
        """Mark leases as mid-episode. Drives may_write_controls and keeps the
        idle reaper from killing a long take with the browser tab closed."""
        with self._lock:
            for d in devices:
                lz = self._leases.get(d)
                if lz is not None:
                    lz.recording = bool(on)

    def recording(self, devices):
        return any(getattr(self._leases.get(d), 'recording', False)
                   for d in devices)

    def owner(self, device):
        cur = self._leases.get(device)
        return None if cur is None else cur.owner

    def lease(self, device):
        return self._leases.get(device)

    def may_write_controls(self, device):
        """Control writes are gated by POLICY, not capability.

        Refused only while an EPISODE IS ARMED on that device. Changing focus or
        zoom mid-recording changes the focal length and nothing downstream can
        detect it -- the poses stay smooth, plausible and wrong -- and changing
        exposure mid-episode is the time-varying photometric noise the fixed
        settings exist to avoid.

        Keyed on the armed flag rather than on the owner's name: one capture
        session spans tuning AND recording, so refusing by owner would make the
        tuning stage unable to tune.
        """
        cur = self._leases.get(device)
        if cur is None:
            return True, None
        if cur.recording:
            return False, cur.owner
        return True, None

    def status(self):
        out = {}
        for d in self.DEVICES:
            lz = self._leases[d]
            out[d] = None if lz is None else {
                'owner': lz.owner,
                'since': lz.since,
                'held_s': round(time.time() - lz.since, 1),
                'subscribers': lz.subscribers,
                'recording': lz.recording,
            }
        return out


BROKER = CameraBroker()


# =================================================================== sessions
class StreamWorker:
    """Runs a session's step() on a thread, publishing the newest frame + state.

    One overwritten slot, never a queue: a queue would either grow without bound
    or push back on the camera. A slow client simply misses frames.
    """

    def __init__(self, session, device_key, owner):
        self.session = session
        self.device_key = device_key
        self.device_keys = (device_key,)   # CapturePreview holds two; see below
        self.owner = owner
        self.jpeg = None
        self.state = None
        self.err = None
        self.frames = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        # Grab at camera rate so the V4L2 queue stays drained (no stale frames),
        # but decode/annotate/encode only at PUBLISH_FPS. The camera runs at 120
        # and nothing downstream reads faster than 30, so the other 90 frames a
        # second are dropped before the decode, which is where the cost is.
        period = 1.0 / PUBLISH_FPS
        nxt = time.perf_counter()
        try:
            while not self._stop.is_set():
                if not self.session.grab():      # blocks on the camera; no busy-wait
                    time.sleep(0.005)
                    continue
                now = time.perf_counter()
                if now < nxt:
                    continue
                # Advance the deadline by exactly one period. Re-basing it on
                # `now` instead would fold in the wait for the next grab (up to
                # a camera frame, 8.3 ms) every time, compounding into ~26 fps.
                nxt += period
                if nxt < now:            # fell behind: resync, do not burst
                    nxt = now + period
                r = self.session.step(grabbed=True)
                if r is None:
                    continue
                _, view, state = r
                h, w = view.shape[:2]
                if w > PREVIEW_WIDTH:
                    s = PREVIEW_WIDTH / w
                    view = cv2.resize(view, (PREVIEW_WIDTH, int(h * s)),
                                      interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', view,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self._lock:
                        self.jpeg = buf.tobytes()
                        self.state = state
                        self.frames += 1
        except Exception as e:                       # noqa: BLE001
            self.err = f'{type(e).__name__}: {e}'
            log_cam.exception('%s worker died', self.owner)
        finally:
            try:
                self.session.close()
            except Exception:                        # noqa: BLE001
                log_cam.exception('%s failed to close its session', self.owner)

    def snapshot(self, cam=None):
        with self._lock:
            return self.jpeg, self.state

    def stop(self):
        self._stop.set()
        self._t.join(timeout=3.0)
        BROKER.release_all(self.device_keys, self.owner, force=True)


class CapturePreview:
    """Publishes both of a RecordSession's cameras. Same interface as
    StreamWorker -- device_keys, snapshot(cam), stop() -- so the streaming
    endpoints do not care which one they are serving.

    ONE object owning two threads, not two StreamWorkers. A StreamWorker closes
    its session and force-releases its lease when its thread ends; with two of
    them over one RecordSession, whichever died first would tear down the session
    the other was still using and half-release the pair. Lease lifetime is
    one-to-one with the session, so the object that owns it must be too.

    Two threads rather than one loop because the cameras cost wildly different
    amounts -- a scene frame is a ~10 ms JPEG decode, a wrist frame arrives
    already decoded -- and sharing a loop couples the slow one to the fast one.
    """

    # Scene, annotated: decode + stats + mask. Only ever while idle, where a
    # dropped wrist frame costs nothing. Raw passthrough otherwise: the demuxed
    # MJPEG packet is a complete baseline JPEG (DHT present, verified), so it
    # goes to the browser with no decode and no re-encode at all.
    SCENE_ANNOTATED_FPS = 6.0
    SCENE_RAW_FPS = 4.0
    SCENE_REC_FPS = 2.0          # mirrors the CLI's REC_SCENE_PREVIEW_FPS
    WRIST_FPS = 10.0
    WRIST_REC_FPS = 4.0

    def __init__(self, session, owner=CAPTURE_KEY):
        self.session, self.owner = session, owner
        self.device_keys = tuple(session.recs)
        self.err = None
        self.annotate, self.mask = True, False
        self._slots = {k: {'jpeg': None, 'stats': None, 'lock': threading.Lock(),
                           'frames': 0} for k in self.device_keys}
        self._stop = threading.Event()
        self._threads = []
        if 'scene' in self.device_keys:
            self._threads.append(threading.Thread(target=self._scene_loop,
                                                  daemon=True))
        if 'wrist' in self.device_keys:
            self._threads.append(threading.Thread(target=self._wrist_loop,
                                                  daemon=True))
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------- publishing
    def _put(self, cam, jpeg, stats=None):
        s = self._slots[cam]
        with s['lock']:
            s['jpeg'] = jpeg
            if stats is not None:
                s['stats'] = stats
            s['frames'] += 1

    def _pace(self, nxt, period):
        """Fixed-increment deadline. Re-basing on now() folds the wait for the
        next frame into every period and compounds into a slower rate."""
        now = time.perf_counter()
        if now < nxt:
            return nxt, False
        nxt += period
        if nxt < now:
            nxt = now + period
        return nxt, True

    def _scene_loop(self):
        rec = self.session.recs['scene']
        nxt = time.perf_counter()
        try:
            while not self._stop.is_set():
                # Derived every iteration, never stored: a mode flag and the arm
                # state could disagree, this cannot.
                recording = self.session.recording
                annotate = self.annotate and not recording
                period = 1.0 / (self.SCENE_ANNOTATED_FPS if annotate else
                                self.SCENE_REC_FPS if recording else
                                self.SCENE_RAW_FPS)
                nxt, due = self._pace(nxt, period)
                if not due:
                    time.sleep(0.01)
                    continue
                raw = rec.preview.take()
                if raw is None:
                    continue
                if not annotate:
                    self._put('scene', raw)          # zero decode, zero re-encode
                    continue
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean, clip, dark, _, verdict, colour = CS.exposure_stats(gray)
                if self.mask:
                    img[gray >= 250] = (0, 0, 255)
                    img[gray <= 8] = (255, 60, 0)
                cv2.putText(img, f'mean {mean:.0f}   clipped {clip:.1f}%   '
                            f'crushed {dark:.1f}%   {verdict}', (14, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                h, w = img.shape[:2]
                if w > PREVIEW_WIDTH:
                    img = cv2.resize(img, (PREVIEW_WIDTH,
                                           int(h * PREVIEW_WIDTH / w)),
                                     interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.jpg', img,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    self._put('scene', buf.tobytes(),
                              {'mean': round(mean, 1), 'clipped': round(clip, 2),
                               'crushed': round(dark, 2), 'verdict': verdict})
        except Exception as e:                       # noqa: BLE001
            self.err = f'scene preview: {type(e).__name__}: {e}'
            log_cam.exception('scene preview thread died')

    def _wrist_loop(self):
        rec = self.session.recs['wrist']
        nxt = time.perf_counter()
        try:
            while not self._stop.is_set():
                period = 1.0 / (self.WRIST_REC_FPS if self.session.recording
                                else self.WRIST_FPS)
                nxt, due = self._pace(nxt, period)
                if not due:
                    time.sleep(0.01)
                    continue
                img = rec.preview.take()
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean, clip, dark, _, verdict, _ = CS.exposure_stats(gray)
                ok, buf = cv2.imencode('.jpg', img,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    self._put('wrist', buf.tobytes(),
                              {'mean': round(mean, 1), 'clipped': round(clip, 2),
                               'crushed': round(dark, 2), 'verdict': verdict})
        except Exception as e:                       # noqa: BLE001
            self.err = f'wrist preview: {type(e).__name__}: {e}'
            log_cam.exception('wrist preview thread died')

    # ------------------------------------------------------------- interface
    @property
    def frames(self):
        """Total published frames, so camera_status can treat this like a
        StreamWorker without knowing which it has."""
        return sum(s['frames'] for s in self._slots.values())

    def snapshot(self, cam=None):
        cam = cam or ('scene' if 'scene' in self._slots else self.device_keys[0])
        s = self._slots.get(cam)
        if s is None:
            return None, None
        with s['lock']:
            return s['jpeg'], s['stats']

    def state(self):
        st = self.session.state()
        st['annotate'], st['mask'] = self.annotate, self.mask
        st['preview_error'] = self.err
        st['stats'] = {k: self._slots[k]['stats'] for k in self.device_keys}
        return st

    def stop(self):
        """Ends the preview AND the recording session -- they are one lifetime."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        try:
            self.session.finish()
        except Exception:                            # noqa: BLE001
            pass
        BROKER.release_all(self.device_keys, self.owner, force=True)


SESSIONS = {}        # 'calibration.capture' | 'capture.session' -> worker
JOBS = {}            # job_id -> {proc, log[], status, rc}


def _stop_session(name):
    w = SESSIONS.pop(name, None)
    if w is not None:
        w.stop()
    return w


# ===================================================================== helpers
def json_err(status, msg, **extra):
    return web.json_response({'error': msg, **extra}, status=status)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:                                # noqa: BLE001
        return None


def run_summary(d):
    """One calibration run directory -> a row for the UI."""
    name = os.path.basename(d)
    frames = len(glob.glob(os.path.join(d, '[0-9]*.png')))
    j = read_json(os.path.join(d, 'scene_intrinsics.json'))
    row = {'name': name, 'path': d, 'frames': frames,
           'mtime': os.path.getmtime(d), 'solved': j is not None}
    if j:
        row.update({
            'reproj': j.get('final_reproj_error'),
            'holdout': j.get('holdout_reproj_error'),
            'n_images': j.get('nr_calib_images'),
            'model': j.get('distortion_model'),
            'fov': j.get('fov_deg'),
            'solved_at': j.get('solved_at'),
        })
    return row


def active_summary():
    j = read_json(ACTIVE_JSON)
    if not j:
        return None
    return {
        'source_run': j.get('source_run'),
        'reproj': j.get('final_reproj_error'),
        'holdout': j.get('holdout_reproj_error'),
        'n_images': j.get('nr_calib_images'),
        'model': j.get('distortion_model'),
        'fov': j.get('fov_deg'),
        'solved_at': j.get('solved_at'),
        'locked_controls': j.get('locked_controls'),
        'image_size': [j.get('image_width'), j.get('image_height')],
    }


# ==================================================================== handlers
async def health(request):
    return web.json_response({'ok': True, 'ui': ui_built(), 'data': DATA})


async def camera_status(request):
    return web.json_response({
        'devices': BROKER.status(),
        'sessions': {k: {'frames': w.frames, 'error': w.err}
                     for k, w in SESSIONS.items()},
    })


async def camera_release(request):
    body = await request.json() if request.can_read_body else {}
    device = body.get('device', 'scene')
    lease = BROKER.lease(device)
    # "Take over" from another tab must not be able to kill a take in progress.
    if lease is not None and lease.recording:
        return json_err(409, f'{lease.owner} is recording an episode; stop it '
                             f'there before taking the camera', owner=lease.owner)
    owner = BROKER.owner(device)
    if owner and owner in SESSIONS:
        _stop_session(owner)       # releases every device the session held
    BROKER.release(device, force=True)
    return web.json_response({'ok': True, 'released': owner})


# ------------------------------------------------------------ run management
async def calib_runs(request):
    runs = []
    if os.path.isdir(CALIB_ROOT):
        for d in sorted(os.listdir(CALIB_ROOT)):
            p = os.path.join(CALIB_ROOT, d)
            if os.path.isdir(p):
                runs.append(run_summary(p))
    runs.sort(key=lambda r: r['mtime'], reverse=True)
    return web.json_response({'runs': runs, 'active': active_summary()})


async def calib_delete_run(request):
    name = request.match_info['name']
    d = os.path.join(CALIB_ROOT, name)
    if os.path.basename(d) != name or not os.path.isdir(d):
        return json_err(404, f'no such run: {name}')
    shutil.rmtree(d)
    return web.json_response({'ok': True, 'deleted': name})


async def calib_activate(request):
    body = await request.json()
    name = body.get('run', '')
    src = os.path.join(CALIB_ROOT, name, 'scene_intrinsics.json')
    if os.path.basename(os.path.dirname(src)) != name or not os.path.exists(src):
        return json_err(404, f'{name} has no solved scene_intrinsics.json')
    shutil.copyfile(src, ACTIVE_JSON)
    return web.json_response({'ok': True, 'active': active_summary()})


async def calib_active(request):
    return web.json_response(active_summary())


# --------------------------------------------------------------------- lock
async def calib_lock(request):
    """Lock GEOMETRY only -- autofocus, focus, zoom -- and assert the frame rate.

    Photometric controls are deliberately untouched: measured flat at 119.4-119.8
    fps across exposure 1..10000, gain 0..1023, gamma 0..255, WB 2800..6500, and
    under auto exposure and auto WB. None of them changes the projection, so
    freezing them only costs the operator the ability to work in another room.
    """
    ok_policy, holder = BROKER.may_write_controls('scene')
    if not ok_policy:
        return json_err(409, f'{holder} is recording; changing focus or zoom now '
                             f'would silently invalidate that footage', owner=holder)
    body = await request.json() if request.can_read_body else {}
    device = body.get('device', CS.DEVICE)
    ok, got, dyn_ok = await asyncio.to_thread(
        CS.lock_geometry, device, int(body.get('focus', 0)), int(body.get('zoom', 100)))
    wanted = {'focus_automatic_continuous': 0,
              'focus_absolute': int(body.get('focus', 0)),
              'zoom_absolute': int(body.get('zoom', 100))}
    rows = [{'control': c, 'wanted': wanted[c], 'got': got.get(c),
             'stuck': got.get(c) == wanted[c]} for c in wanted]
    return web.json_response({
        'ok': bool(ok) and dyn_ok, 'geometry_ok': bool(ok), 'device': device,
        'controls': rows,
        'dynamic_framerate': {
            'value': got.get('exposure_dynamic_framerate'), 'ok': dyn_ok,
            # Never written, only checked. With it at 1 the camera trades frame
            # rate for light: measured 78 fps at exposure 800, 10 fps at 10000.
            'note': 'must be 0 or the camera drops below 120 fps under long exposure',
        },
    })


# ------------------------------------------------------- photometric tuning
def _wrist_session():
    """The capture session, if one is live -- the only holder of the D405."""
    w = SESSIONS.get(CAPTURE_KEY)
    return None if w is None else w.session


async def camera_controls_get(request):
    """Controls for either camera.

    `device` is a LOGICAL camera name now ('scene' or 'wrist'), not a /dev path,
    so one API call serves both cameras and both flows. Scene is answered from
    v4l2-ctl whether or not anything is streaming; wrist needs the live session,
    because its options live on the started pipeline's sensor.
    """
    cam = request.query.get('device', 'scene')
    if cam == 'wrist':
        sess = _wrist_session()
        if sess is None:
            return json_err(409, 'no capture session; the wrist camera is not open')
        return web.json_response(await asyncio.to_thread(sess.wrist_controls))
    return web.json_response(await asyncio.to_thread(PV.scene_controls, CS.DEVICE))


async def camera_controls_set(request):
    body = await request.json()
    cam = body.pop('device', 'scene')
    ok_policy, holder = BROKER.may_write_controls(cam)
    if not ok_policy:
        return json_err(409, f'{holder} is recording; the picture is frozen for '
                             f'the duration of the episode', owner=holder)
    if cam == 'wrist':
        sess = _wrist_session()
        if sess is None:
            return json_err(409, 'no capture session; the wrist camera is not open')
        setter = sess.set_wrist_control
    else:
        def setter(ctrl, val):
            return PV.scene_set(CS.DEVICE, ctrl, val)
    results = {}
    for ctrl, val in body.items():
        ok, detail = await asyncio.to_thread(setter, ctrl, val)
        results[ctrl] = {'ok': ok, 'detail': detail}
    return web.json_response({'ok': all(r['ok'] for r in results.values()),
                              'results': results})


async def calib_tune_start(request):
    if 'calibration.tune' in SESSIONS:
        return web.json_response({'ok': True, 'existing': True})
    body = await request.json() if request.can_read_body else {}
    device = body.get('device', CS.DEVICE)
    try:
        BROKER.acquire('scene', 'calibration.tune')
    except Busy as b:
        return json_err(409, str(b), owner=b.owner, device=b.device)
    try:
        sess = await asyncio.to_thread(PV.PreviewSession, device,
                                       bool(body.get('mask', False)))
    except Exception as e:                           # noqa: BLE001
        BROKER.release('scene', 'calibration.tune', force=True)
        return json_err(400, str(e))
    SESSIONS['calibration.tune'] = StreamWorker(sess, 'scene', 'calibration.tune')
    return web.json_response({'ok': True})


async def calib_tune_stop(request):
    _stop_session('calibration.tune')
    return web.json_response({'ok': True})


async def calib_tune_mask(request):
    w = SESSIONS.get('calibration.tune')
    if w is None:
        return json_err(409, 'no preview session')
    body = await request.json()
    w.session.mask = bool(body.get('on', False))
    return web.json_response({'ok': True, 'mask': w.session.mask})


async def tune_preview(request):
    return await _mjpeg(request, 'calibration.tune')


async def tune_state(request):
    return await _sse_state(request, 'calibration.tune')


# ------------------------------------------------------------ capture session
async def calib_session_start(request):
    if 'calibration.capture' in SESSIONS:
        w = SESSIONS['calibration.capture']
        return web.json_response({'ok': True, 'existing': True,
                                  'run_dir': w.session.out})
    body = await request.json() if request.can_read_body else {}
    device = body.get('device', CS.DEVICE)
    try:
        BROKER.acquire('scene', 'calibration.capture')
    except Busy as b:
        return json_err(409, str(b), owner=b.owner, device=b.device)
    try:
        sess = await asyncio.to_thread(CS.CaptureSession, device, None,
                                       int(body.get('target', 60)))
    except Exception as e:                           # noqa: BLE001
        BROKER.release('scene', 'calibration.capture', force=True)
        return json_err(400, str(e))
    SESSIONS['calibration.capture'] = StreamWorker(sess, 'scene',
                                                   'calibration.capture')
    return web.json_response({'ok': True, 'run_dir': sess.out,
                              'locked': sess.locked})


async def calib_session_status(request):
    """What has actually happened, with NO lease taken.

    This is what lets the wizard's guarded URLs survive a browser refresh: the
    stage a user may open is derived from server truth rather than from React
    state that a reload throws away. Without it, refreshing mid-capture would
    bounce you to stage one while the session kept running behind you.
    """
    w = SESSIONS.get('calibration.capture')
    if w is None:
        got = await asyncio.to_thread(CS.v4l2_get, CS.DEVICE, CS.GEOMETRY_CTRLS)
        return web.json_response({
            'active': False, 'run_dir': None, 'saved': 0,
            'locked': got.get('focus_automatic_continuous') == 0,
        })
    _, state = w.snapshot()
    return web.json_response({
        'active': True,
        'run_dir': w.session.out,
        'run': os.path.basename(w.session.out),
        'saved': w.session.saved,
        'locked': True,
        'auto': w.session.auto,
        'coverage_done': bool(state['coverage']['done']) if state else False,
    })


async def calib_session_stop(request):
    """RESET: stop the session, release the camera, delete the run directory."""
    w = SESSIONS.get('calibration.capture')
    if w is None:
        return web.json_response({'ok': True, 'existed': False})
    run_dir = w.session.out
    _stop_session('calibration.capture')
    keep = request.query.get('keep') == '1'
    if not keep and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    return web.json_response({'ok': True, 'run_dir': run_dir,
                              'deleted': not keep})


async def calib_auto(request):
    w = SESSIONS.get('calibration.capture')
    if w is None:
        return json_err(409, 'no capture session')
    body = await request.json()
    w.session.auto = bool(body.get('on', True))
    return web.json_response({'ok': True, 'auto': w.session.auto})


async def calib_keep(request):
    w = SESSIONS.get('calibration.capture')
    if w is None:
        return json_err(409, 'no capture session')
    ok, why = await asyncio.to_thread(w.session.keep)
    return web.json_response({'ok': ok, 'detail': str(why)})


async def calib_undo(request):
    w = SESSIONS.get('calibration.capture')
    if w is None:
        return json_err(409, 'no capture session')
    ok, why = await asyncio.to_thread(w.session.undo)
    return web.json_response({'ok': ok, 'detail': str(why)})


# ------------------------------------------------------------------- solve
async def calib_solve(request):
    body = await request.json() if request.can_read_body else {}
    run = body.get('run')
    frames = os.path.join(CALIB_ROOT, run) if run else None
    if frames and not os.path.isdir(frames):
        return json_err(404, f'no such run: {run}')

    # A SUBPROCESS, not an in-process call: solve is tens of seconds to minutes
    # of blocking calibrateCameraExtended, which would freeze the event loop and
    # with it every other request, including the one that would cancel it.
    # -u because stdout to a pipe is block-buffered otherwise and the log would
    # arrive in one lump at the end, looking like a hang.
    cmd = [sys.executable, '-u',
           os.path.join(CALIB_DIR, 'calibrate_scene_cam.py'), 'solve']
    if frames:
        cmd.append(frames)
    jid = uuid.uuid4().hex[:8]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=REPO)
    job = JOBS[jid] = {'proc': proc, 'log': [], 'status': 'running',
                       'rc': None, 'run': run}

    def drain():
        # Must run unconditionally, not only while a client is watching: the
        # pipe buffer is 64 KB and solve would block forever on a full one.
        for line in proc.stdout:
            job['log'].append(line.rstrip('\n'))
        job['rc'] = proc.wait()
        job['status'] = 'done' if job['rc'] == 0 else 'failed'
        (log_job.info if job['rc'] == 0 else log_job.error)(
            'job %s %s (rc=%s, %d log lines)', jid, job['status'], job['rc'],
            len(job['log']))

    threading.Thread(target=drain, daemon=True).start()
    log_job.info('job %s spawned: %s', jid, ' '.join(cmd))
    return web.json_response({'ok': True, 'job_id': jid})


async def job_log(request):
    jid = request.match_info['jid']
    job = JOBS.get(jid)
    if job is None:
        return json_err(404, f'no such job: {jid}')
    resp = web.StreamResponse(headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })
    await resp.prepare(request)
    i = int(request.query.get('frm', 0))
    try:
        while True:
            while i < len(job['log']):
                await resp.write(f'data: {json.dumps({"i": i, "line": job["log"][i]})}\n\n'
                                 .encode())
                i += 1
            if job['status'] != 'running':
                await resp.write(f'data: {json.dumps({"done": job["status"], "rc": job["rc"]})}\n\n'
                                 .encode())
                break
            await asyncio.sleep(0.2)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


# --------------------------------------------------------------- test stage
async def calib_test_start(request):
    if 'calibration.test' in SESSIONS:
        return web.json_response({'ok': True, 'existing': True})
    body = await request.json() if request.can_read_body else {}
    device = body.get('device', CS.DEVICE)
    intr = body.get('intrinsics') or ACTIVE_JSON
    if body.get('run'):
        intr = os.path.join(CALIB_ROOT, body['run'], 'scene_intrinsics.json')
    if not os.path.exists(intr):
        return json_err(404, f'no calibration at {intr}')
    try:
        BROKER.acquire('scene', 'calibration.test')
    except Busy as b:
        return json_err(409, str(b), owner=b.owner, device=b.device)
    try:
        sess = await asyncio.to_thread(DT.DetectorSession, device, intr)
    except Exception as e:                           # noqa: BLE001
        BROKER.release('scene', 'calibration.test', force=True)
        return json_err(400, str(e))
    SESSIONS['calibration.test'] = StreamWorker(sess, 'scene', 'calibration.test')
    return web.json_response({'ok': True, 'intrinsics': intr,
                              'geometry_ok': bool(sess.geometry_ok)})


async def calib_test_stop(request):
    _stop_session('calibration.test')
    return web.json_response({'ok': True})


# =========================================================== capture (record)
def _episodes_on_disk(d):
    """Episode dirs holding real video, read from the filesystem.

    The count must not come from an in-memory counter: this is also what answers
    for a session the server was killed during, and the sidecars are checkpointed
    about once a second, so those episodes are genuinely recoverable.
    """
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not (name.startswith('ep') and os.path.isdir(p)):
            continue
        vids = [f for _, _, fs in os.walk(p) for f in fs if f.endswith('.mkv')]
        if vids:
            out.append(name)
    return out


def _latest_session_dir():
    if not os.path.isdir(CAPTURE_ROOT):
        return None
    dirs = [d for d in sorted(os.listdir(CAPTURE_ROOT))
            if os.path.isdir(os.path.join(CAPTURE_ROOT, d))]
    return os.path.join(CAPTURE_ROOT, dirs[-1]) if dirs else None


async def capture_status(request):
    """What has actually happened, with NO lease taken.

    Same contract as calib_session_status: it is what lets the wizard's guarded
    URLs survive a refresh, and it answers from durable evidence -- the
    filesystem -- when no session is live.
    """
    w = SESSIONS.get(CAPTURE_KEY)
    if w is not None:
        st = w.state()
        st['active'] = True
        return web.json_response(st)
    d = _latest_session_dir()
    eps = _episodes_on_disk(d) if d else []
    return web.json_response({
        'active': False, 'session_dir': d,
        'session': os.path.basename(d) if d else None,
        'n_episodes': len(eps), 'episodes': eps,
        'complete': bool(d and os.path.exists(os.path.join(d, 'session.json'))),
        'orphan': bool(eps and d
                       and not os.path.exists(os.path.join(d, 'session.json'))),
        'scene_locked': (await asyncio.to_thread(
            CS.v4l2_get, CS.DEVICE, CS.GEOMETRY_CTRLS)
        ).get('focus_automatic_continuous') == 0,
    })


async def capture_start(request):
    if CAPTURE_KEY in SESSIONS:
        w = SESSIONS[CAPTURE_KEY]
        st = w.state()
        st.update({'ok': True, 'existing': True})
        return web.json_response(st)
    body = await request.json() if request.can_read_body else {}
    want_scene = bool(body.get('scene', True))
    want_wrist = bool(body.get('wrist', True))
    devices = tuple(d for d, on in (('scene', want_scene), ('wrist', want_wrist))
                    if on)
    if not devices:
        return json_err(400, 'nothing to record (both cameras disabled)')
    try:
        BROKER.acquire_all(devices, CAPTURE_KEY)
    except Busy as b:
        return json_err(409, str(b), owner=b.owner, device=b.device)
    try:
        sess = await asyncio.to_thread(
            CAP.RecordSession, CAPTURE_ROOT, want_scene, want_wrist,
            int(body.get('scene_exposure', 800)), int(body.get('scene_wb', 4600)),
            int(body.get('scene_gamma', 128)), int(body.get('scene_gain', 100)),
            int(body.get('wrist_exposure', 6000)), int(body.get('wrist_gain', 248)),
            int(body.get('wrist_wb', 4600)), bool(body.get('wrist_auto_wb', False)))
    except Exception as e:                           # noqa: BLE001
        BROKER.release_all(devices, CAPTURE_KEY, force=True)
        log_cap.exception('session failed to start')
        return json_err(400, str(e))
    SESSIONS[CAPTURE_KEY] = CapturePreview(sess)
    log_cap.info('session %s started (%s) -> %s',
                 sess.stamp, '+'.join(sess.recs), sess.dir)
    st = SESSIONS[CAPTURE_KEY].state()
    st['ok'] = True
    return web.json_response(st)


async def capture_stop(request):
    """Ends the session. Episodes are KEPT unless ?discard=1 -- the inverse of
    the calibration default, because episodes are not cheap to re-shoot."""
    w = SESSIONS.get(CAPTURE_KEY)
    if w is None:
        return web.json_response({'ok': True, 'existed': False})
    if w.session.recording:
        return json_err(409, 'an episode is recording; stop it first')
    d = w.session.dir
    n = len(w.session.episodes)
    _stop_session(CAPTURE_KEY)          # finishes the session and writes metadata
    discard = request.query.get('discard') in ('1', 'true', 'yes')
    if discard:
        shutil.rmtree(d, ignore_errors=True)
    return web.json_response({'ok': True, 'session_dir': d, 'n_episodes': n,
                              'discarded': discard})


async def capture_episode(request):
    """Explicit start/stop, never a toggle.

    With ~0.6 s of arm/disarm latency and a keyboard driving it, a toggle
    endpoint desyncs: two presses 100 ms apart would start an episode and
    immediately stop it. The client knows the state from the SSE; the server
    makes the stated intent idempotent.
    """
    w = SESSIONS.get(CAPTURE_KEY)
    if w is None:
        return json_err(409, 'no capture session')
    body = await request.json() if request.can_read_body else {}
    action = body.get('action')
    if action not in ('start', 'stop'):
        return json_err(400, "action must be 'start' or 'stop'")
    sess = w.session
    try:
        if action == 'start':
            if sess.recording:
                return web.json_response({'ok': True, 'already': True,
                                          **w.state()})
            i = await asyncio.to_thread(sess.start_episode)
            BROKER.set_recording(w.device_keys, True)
            log_cap.info('ep%03d armed', i)
            return web.json_response({'ok': True, 'episode_index': i,
                                      **w.state()})
        if not sess.recording:
            return web.json_response({'ok': True, 'already': True, **w.state()})
        ep = await asyncio.to_thread(sess.stop_episode)
        BROKER.set_recording(w.device_keys, False)
        log_cap.info('ep%03d saved  %s  (%.1fs)', ep['index'],
                     '  '.join(f'{k} {ep[k]["frames"]}@{ep[k]["fps"]:.1f}fps'
                               + (f' {ep[k]["queue_drops"]}drop'
                                  if ep[k]['queue_drops'] else '')
                               for k in sess.recs if k in ep),
                     ep['duration_s'])
        return web.json_response({'ok': True, 'episode': ep, **w.state()})
    except CAP.RecordError as e:
        BROKER.set_recording(w.device_keys, sess.recording)
        log_cap.warning('episode %s refused: %s', action, e)
        return json_err(409, str(e), warmup_left=round(sess.warmup_left, 1))


async def capture_finish(request):
    w = SESSIONS.get(CAPTURE_KEY)
    if w is None:
        return json_err(409, 'no capture session')
    if w.session.recording:
        return json_err(409, 'an episode is recording; stop it first')
    sess = w.session
    d, stamp = sess.dir, sess.stamp
    meta = await asyncio.to_thread(sess.finish)
    _stop_session(CAPTURE_KEY)
    log_cap.info('session %s finished: %d episode(s) -> %s',
                 stamp, len(meta['episodes']), d)
    return web.json_response({'ok': True, 'session': stamp, 'session_dir': d,
                              'n_episodes': len(meta['episodes']),
                              'episodes': meta['episodes']})


async def capture_preview_opts(request):
    w = SESSIONS.get(CAPTURE_KEY)
    if w is None:
        return json_err(409, 'no capture session')
    body = await request.json() if request.can_read_body else {}
    if 'annotate' in body:
        w.annotate = bool(body['annotate'])
    if 'mask' in body:
        w.mask = bool(body['mask'])
    return web.json_response({'ok': True, 'annotate': w.annotate, 'mask': w.mask})


async def capture_stream(request):
    cam = request.match_info['cam']
    if cam not in ('scene', 'wrist'):
        return json_err(404, f'no such camera: {cam}')
    return await _mjpeg(request, CAPTURE_KEY, cam)


async def capture_state(request):
    return await _sse_state(request, CAPTURE_KEY)


# ------------------------------------------------------- streaming endpoints
def _subscribe(w):
    """Every lease the worker holds, so a wrist viewer is not credited to the
    scene lease -- which would leave the wrist at zero subscribers forever and
    hand it to the idle reaper while somebody was watching it."""
    return [lz for lz in (BROKER.lease(d) for d in w.device_keys) if lz]


async def _mjpeg(request, name, cam=None):
    w = SESSIONS.get(name)
    if w is None:
        return json_err(409, 'no session', session=name)
    leases = _subscribe(w)
    resp = web.StreamResponse(headers={
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })
    await resp.prepare(request)
    for lz in leases:
        lz.subscribers += 1
    # No rate limit here: the worker already produces at PUBLISH_FPS, so this
    # just forwards each new frame. Pacing in both places makes the two 30 Hz
    # loops beat against each other and silently drops ~12% of frames.
    last = None
    try:
        while name in SESSIONS:
            jpeg, _ = w.snapshot(cam)
            if jpeg is None or jpeg is last:
                await asyncio.sleep(0.002)
                continue
            last = jpeg
            await resp.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                             b'Content-Length: ' + str(len(jpeg)).encode() +
                             b'\r\n\r\n' + jpeg + b'\r\n')
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        for lz in leases:
            lz.subscribers = max(0, lz.subscribers - 1)
            lz.idle_since = time.time()
    return resp


async def _sse_state(request, name):
    w = SESSIONS.get(name)
    if w is None:
        return json_err(409, 'no session', session=name)
    leases = _subscribe(w)
    resp = web.StreamResponse(headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })
    await resp.prepare(request)
    for lz in leases:
        lz.subscribers += 1
    # A worker with its own state() (the capture session) reports far more than
    # the per-frame stats in the slot -- episodes, warm-up, arm state.
    get_state = getattr(w, 'state', None)
    try:
        while name in SESSIONS:
            state = get_state() if get_state else w.snapshot()[1]
            if state is not None:
                payload = dict(state)
                payload['worker_error'] = w.err
                await resp.write(f'data: {json.dumps(payload)}\n\n'.encode())
            await asyncio.sleep(0.125)
        await resp.write(b'data: {"ended": true}\n\n')
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        for lz in leases:
            lz.subscribers = max(0, lz.subscribers - 1)
            lz.idle_since = time.time()
    return resp


async def calib_preview(request):
    return await _mjpeg(request, 'calibration.capture')


async def calib_state(request):
    return await _sse_state(request, 'calibration.capture')


async def test_preview(request):
    return await _mjpeg(request, 'calibration.test')


async def test_state(request):
    return await _sse_state(request, 'calibration.test')


# --------------------------------------------------------- run-dir file serve
async def calib_run_file(request):
    """Serve undistort_preview.png and friends out of a run directory."""
    name, fname = request.match_info['name'], request.match_info['file']
    p = os.path.join(CALIB_ROOT, name, fname)
    if os.path.basename(p) != fname or not os.path.isfile(p):
        return json_err(404, 'not found')
    return web.FileResponse(p)


# ==================================================================== the UI
def ui_built():
    return os.path.isfile(os.path.join(UI_DIST, 'index.html'))


PLACEHOLDER = """<!doctype html><title>robospec_umi</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:34rem;margin:12vh auto;
padding:0 1.5rem;background:#111;color:#eee}}code{{background:#2a2a2a;padding:.1em .35em;
border-radius:3px}}a{{color:#6b9bff}}</style>
<h1>robospec_umi</h1>
<p>The server is running. The web UI has not been built yet.</p>
<p>Expected at <code>{dist}</code>. Build it with <code>npm install &amp;&amp;
npm run build</code> in <code>robospec_umi_ui/</code> &mdash; that directory is
bind-mounted, so no container rebuild is needed.</p>
<p>For development, <code>npm run dev</code> proxies <code>/api</code> here.</p>
<p><a href="/api/health">/api/health</a> &middot;
   <a href="/api/camera">/api/camera</a></p>
"""


async def index(request):
    if ui_built():
        return web.FileResponse(os.path.join(UI_DIST, 'index.html'))
    return web.Response(text=PLACEHOLDER.format(dist=UI_DIST),
                        content_type='text/html')


async def spa_fallback(request):
    """Any unmatched non-/api path is a client-side route -> index.html."""
    if request.path.startswith('/api/'):
        return json_err(404, f'no such endpoint: {request.path}')
    if ui_built():
        asset = os.path.normpath(os.path.join(UI_DIST, request.path.lstrip('/')))
        if asset.startswith(UI_DIST) and os.path.isfile(asset):
            return web.FileResponse(asset)
        return web.FileResponse(os.path.join(UI_DIST, 'index.html'))
    return await index(request)


# ==================================================================== plumbing
async def idle_reaper(app):
    """Release a lease nobody is watching.

    A session deliberately outlives the browser so a refresh reconnects mid
    capture. Without this, closing the tab would pin the camera until the
    server restarted.
    """
    try:
        while True:
            await asyncio.sleep(30)
            for name, w in list(SESSIONS.items()):
                leases = [lz for lz in (BROKER.lease(d) for d in w.device_keys)
                          if lz]
                if not leases:
                    continue
                # Never reap mid-episode. A ten-minute take with the tab closed
                # has no subscribers and would otherwise be killed at 600 s.
                if any(lz.recording for lz in leases):
                    for lz in leases:
                        lz.idle_since = time.time()
                    continue
                if any(lz.subscribers > 0 for lz in leases):
                    for lz in leases:
                        lz.idle_since = time.time()
                    continue
                idle = min(time.time() - lz.idle_since for lz in leases)
                if idle > IDLE_RELEASE_S:
                    log_cam.info('releasing %s after %ss with no subscribers',
                                 name, IDLE_RELEASE_S)
                    _stop_session(name)
    except asyncio.CancelledError:
        pass


async def on_startup(app):
    app['reaper'] = asyncio.create_task(idle_reaper(app))


async def on_cleanup(app):
    app['reaper'].cancel()
    for name in list(SESSIONS):
        _stop_session(name)


def build_app():
    app = web.Application()
    r = app.router
    r.add_get('/api/health', health)
    r.add_get('/api/camera', camera_status)
    r.add_post('/api/camera/release', camera_release)

    r.add_get('/api/calibration/runs', calib_runs)
    r.add_delete('/api/calibration/runs/{name}', calib_delete_run)
    r.add_post('/api/calibration/activate', calib_activate)
    r.add_get('/api/calibration/active', calib_active)
    r.add_get('/api/calibration/runs/{name}/file/{file}', calib_run_file)

    r.add_get('/api/camera/controls', camera_controls_get)
    r.add_post('/api/camera/controls', camera_controls_set)
    r.add_post('/api/calibration/tune', calib_tune_start)
    r.add_delete('/api/calibration/tune', calib_tune_stop)
    r.add_post('/api/calibration/tune/mask', calib_tune_mask)
    r.add_get('/api/calibration/tune/stream', tune_preview)
    r.add_get('/api/calibration/tune/state', tune_state)

    # Nothing in lock is calibration-specific -- both flows pin the same three
    # geometry controls. The old path stays registered so an already-built
    # dist/ keeps working.
    r.add_post('/api/camera/lock', calib_lock)
    r.add_post('/api/calibration/lock', calib_lock)
    r.add_get('/api/calibration/session', calib_session_status)
    r.add_post('/api/calibration/session', calib_session_start)
    r.add_delete('/api/calibration/session', calib_session_stop)
    r.add_get('/api/calibration/preview', calib_preview)
    r.add_get('/api/calibration/state', calib_state)
    r.add_post('/api/calibration/auto', calib_auto)
    r.add_post('/api/calibration/keep', calib_keep)
    r.add_post('/api/calibration/undo', calib_undo)

    r.add_post('/api/calibration/solve', calib_solve)
    r.add_get('/api/jobs/{jid}/log', job_log)

    r.add_post('/api/calibration/test', calib_test_start)
    r.add_delete('/api/calibration/test', calib_test_stop)
    r.add_get('/api/calibration/test/preview', test_preview)
    r.add_get('/api/calibration/test/state', test_state)

    r.add_get('/api/capture/session', capture_status)
    r.add_post('/api/capture/session', capture_start)
    r.add_delete('/api/capture/session', capture_stop)
    r.add_post('/api/capture/episode', capture_episode)
    r.add_post('/api/capture/finish', capture_finish)
    r.add_post('/api/capture/preview', capture_preview_opts)
    r.add_get('/api/capture/stream/{cam}', capture_stream)
    r.add_get('/api/capture/state', capture_state)

    r.add_get('/', index)
    if ui_built():
        assets = os.path.join(UI_DIST, 'assets')
        if os.path.isdir(assets):
            r.add_static('/assets', assets)
    r.add_route('*', '/{tail:.*}', spa_fallback)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1',
                    help='0.0.0.0 in the container so the port mapping reaches it')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='DEBUG level: per-request lines and every control write')
    args = ap.parse_args()

    # threadName is not decoration: every interesting failure here happens on a
    # worker thread (_scene_loop, _wrist_loop, SceneRecorder._capture,
    # WristRecorder._capture), and a traceback without it does not say which
    # camera died.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s  %(message)s',
        datefmt='%H:%M:%S')
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('aiohttp.access').addFilter(QuietPaths())
    # On the HANDLER, not on logging.getLogger('libav'): these records come from
    # child loggers (libav.mjpeg, libav.generic), and a filter on a logger only
    # sees records made by that logger -- propagation to an ancestor's handlers
    # skips the ancestor's filters.
    for h in logging.getLogger().handlers:
        h.addFilter(QuietLibav())

    log.info('repo   %s', REPO)
    log.info('data   %s', DATA)
    log.info('camera %s%s', CS.DEVICE,
             '' if os.path.exists(CS.DEVICE)
             else '   [MISSING - install 99-decxin-cam.rules]')
    log.info('ui     %s  (%s)', UI_DIST,
             'built' if ui_built() else 'NOT BUILT - placeholder')
    log.info('serving on http://%s:%s', args.host, args.port)
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == '__main__':
    main()
