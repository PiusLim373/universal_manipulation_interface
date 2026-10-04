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
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import av
import cv2
import numpy as np
import yaml
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
log_edit = log.getChild('edit')       # dataset editing + prep queue
log_cal = log.getChild('calibration')  # activate / import / delete
log_train = log.getChild('train')     # checkpoint downloads


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
             '/assets/', '/api/health', '/api/edit/', '/api/train/')

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
DATASET_DIR = os.path.join(HERE, 'robospec_umi_dataset')
DATASET_ROOT = os.path.join(DATA, 'dataset')
OUTPUTS_ROOT = os.path.join(DATA, 'outputs')
TRAIN_CONFIG = os.path.join(REPO, 'diffusion_policy', 'config',
                            'train_diffusion_unet_timm_umi_workspace.yaml')
ACTIVE_JSON = os.path.join(CALIB_ROOT, 'scene_intrinsics.json')

sys.path.insert(0, CALIB_DIR)
sys.path.insert(0, CAPTURE_DIR)
sys.path.insert(0, DATASET_DIR)
import calibrate_scene_cam as CS                        # noqa: E402
import scene_cam_charuco_detector as DT                 # noqa: E402
import preview as PV                                    # noqa: E402
import capture as CAP                                   # noqa: E402
import episode_prep as EP                               # noqa: E402
import timeline as TL                                   # noqa: E402
import verify as VF                                     # noqa: E402

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
                # the training crop, on the preview's own copy of the frame
                h, w = img.shape[:2]
                x0, side = TL.wrist_crop(w, h)
                cv2.rectangle(img, (x0, 0), (x0 + side - 1, h - 1), (0, 255, 255), 1)
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
# Set on shutdown. Every long-lived stream checks it, or an open browser tab
# holds Ctrl+C for aiohttp's full 60 s shutdown timeout.
STOPPING = threading.Event()


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
    bound = await asyncio.to_thread(_bound_runs)
    if os.path.isdir(CALIB_ROOT):
        for d in sorted(os.listdir(CALIB_ROOT)):
            p = os.path.join(CALIB_ROOT, d)
            if os.path.isdir(p):
                runs.append({**run_summary(p), 'datasets': bound.get(d, 0)})
    runs.sort(key=lambda r: r['mtime'], reverse=True)
    return web.json_response({'runs': runs, 'active': active_summary()})


async def calib_delete_run(request):
    """Deleting the active run deactivates it: datasets bound to it lock."""
    name = request.match_info['name']
    d = os.path.join(CALIB_ROOT, name)
    if os.path.basename(d) != name or not os.path.isdir(d):
        return json_err(404, f'no such run: {name}')
    busy = [pid for pid in EXPORT_JOBS if _export_running(pid)
            and ((read_json(_proj_path(pid)) or {}).get('intrinsics') or {}).get('run') == name]
    if busy:
        return json_err(409, f'dataset {busy[0]} is exporting with {name}')
    act = read_json(ACTIVE_JSON) or {}
    active = act.get('source_run') == name or (
        _sha1(ACTIVE_JSON) is not None and _sha1(ACTIVE_JSON) == _sha1(_run_json(name)))
    shutil.rmtree(d)
    if active:
        os.remove(ACTIVE_JSON)
    log_cal.info('deleted run %s%s', name, ' (was active: nothing is active now)' if active else '')
    return web.json_response({'ok': True, 'deleted': name, 'deactivated': active})


async def calib_activate(request):
    body = await request.json()
    name = body.get('run', '')
    src = os.path.join(CALIB_ROOT, name, 'scene_intrinsics.json')
    if os.path.basename(os.path.dirname(src)) != name or not os.path.exists(src):
        return json_err(404, f'{name} has no solved scene_intrinsics.json')
    shutil.copyfile(src, ACTIVE_JSON)
    log_cal.info('activated %s', name)
    return web.json_response({'ok': True, 'active': active_summary()})


INTR_REQUIRED = ('source_run', 'intrinsic_type', 'k', 'd', 'image_width', 'image_height')


def _check_intrinsics_upload(raw):
    """-> (json, warnings). Raises ValueError saying what is wrong."""
    try:
        j = json.loads(raw)
    except ValueError as e:
        raise ValueError(f'not valid JSON ({e})') from None
    if not isinstance(j, dict):
        raise ValueError('not a JSON object')
    missing = [k for k in INTR_REQUIRED if k not in j]
    if missing:
        raise ValueError('missing ' + ', '.join(f"'{k}'" for k in missing))
    if not isinstance(j['source_run'], str) or not SESS_RE.match(j['source_run']):
        raise ValueError("'source_run' must be the calibration's datetime, "
                         "like 20260924_234214")
    if j['intrinsic_type'] != 'PINHOLE':
        raise ValueError(f"'intrinsic_type' is {j['intrinsic_type']!r}, expected 'PINHOLE'")

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)
    if not (isinstance(j['k'], list) and len(j['k']) == 9 and all(map(num, j['k']))):
        raise ValueError("'k' must be 9 numbers (the 3x3 camera matrix)")
    if not (isinstance(j['d'], list) and len(j['d']) >= 4 and all(map(num, j['d']))):
        raise ValueError("'d' must be at least 4 distortion coefficients")
    for k in ('image_width', 'image_height'):
        if not (isinstance(j[k], int) and not isinstance(j[k], bool) and j[k] > 0):
            raise ValueError(f"'{k}' must be a positive integer")
    lc = j.get('locked_controls')
    warnings = []
    if not isinstance(lc, dict) or None in (lc.get('focus_absolute'), lc.get('zoom_absolute')):
        warnings.append('no locked focus/zoom: sessions cannot be checked against it')
    return j, warnings


async def calib_upload(request):
    """Import a scene_intrinsics.json as calibration/<source_run>/. Stored
    byte-for-byte so its sha1 (what datasets bind to) matches the original."""
    raw = await request.read()
    try:
        j, warnings = _check_intrinsics_upload(raw)
    except ValueError as e:
        return json_err(400, f'malformed intrinsic file: {e}')
    run = j['source_run']
    d = os.path.join(CALIB_ROOT, run)
    if os.path.exists(d):
        have = _sha1(_run_json(run))
        return json_err(409, f'{run} is already imported' if have == hashlib.sha1(raw).hexdigest()
                        else f'a different calibration run named {run} already exists')
    os.makedirs(d)
    dst = _run_json(run)
    with open(dst + '.tmp', 'wb') as f:
        f.write(raw)
    os.replace(dst + '.tmp', dst)
    activated = not os.path.exists(ACTIVE_JSON)
    if activated:
        shutil.copyfile(dst, ACTIVE_JSON)
    log_cal.info('imported %s%s', run, ' and activated it' if activated else '')
    return web.json_response({'ok': True, 'run': run, 'activated': activated,
                              'warnings': warnings})


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
    return web.json_response({'ok': True, 'job_id': _spawn_job(cmd, 'solve', run=run)})


def _spawn_job(cmd, tag, **info):
    """Run cmd as a subprocess in JOBS; its output goes to the UI and the log."""
    jid = uuid.uuid4().hex[:8]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=REPO)
    job = JOBS[jid] = {'proc': proc, 'log': [], 'status': 'running',
                       'rc': None, **info}

    def drain():
        # Must run unconditionally, not only while a client is watching: the
        # pipe buffer is 64 KB and the child would block forever on a full one.
        for line in proc.stdout:
            line = line.rstrip('\n')
            job['log'].append(line)
            log_job.info('%s | %s', tag, line)
        job['rc'] = proc.wait()
        job['status'] = 'done' if job['rc'] == 0 else 'failed'
        (log_job.info if job['rc'] == 0 else log_job.error)(
            '%s %s (job %s, rc=%s)', tag, job['status'], jid, job['rc'])

    threading.Thread(target=drain, daemon=True).start()
    log_job.info('%s started (job %s): %s', tag, jid, ' '.join(cmd))
    return jid


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
        while not STOPPING.is_set():
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
    # wrist exposure / gain / auto exposure: the camera model's defaults unless set
    opt = lambda k, f: None if body.get(k) is None else f(body[k])   # noqa: E731
    try:
        sess = await asyncio.to_thread(
            CAP.RecordSession, CAPTURE_ROOT, want_scene, want_wrist,
            int(body.get('scene_exposure', 800)), int(body.get('scene_wb', 4600)),
            int(body.get('scene_gamma', 128)), int(body.get('scene_gain', 100)),
            opt('wrist_exposure', int), opt('wrist_gain', int),
            int(body.get('wrist_wb', 4600)), bool(body.get('wrist_auto_wb', True)),
            opt('wrist_auto_exposure', bool))
    except Exception as e:                           # noqa: BLE001
        BROKER.release_all(devices, CAPTURE_KEY, force=True)
        log_cap.exception('session failed to start')
        return json_err(400, str(e))
    SESSIONS[CAPTURE_KEY] = CapturePreview(sess)
    log_cap.info('session %s started (%s) -> %s',
                 sess.stamp, '+'.join(k for k, _ in sess._streams()), sess.dir)
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
        while name in SESSIONS and not STOPPING.is_set():
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
    # CapturePreview has a state() method (episodes, warm-up, arm state);
    # StreamWorker has a `state` dict attribute. Test callable, not presence.
    get_state = getattr(w, 'state', None)
    if not callable(get_state):
        get_state = None
    try:
        while name in SESSIONS and not STOPPING.is_set():
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


# ============================================================ edit (dataset)
SESS_RE = re.compile(r'^\d{8}_\d{6}$')
EP_RE = re.compile(r'^ep\d{3}$')
MEDIA = ('scene_annotated.mp4', 'wrist.mp4')
STAGES = ('select', 'verify', 'edit', 'export')
PREP_WORKERS = 2
EXPORT_JOBS = {}     # project id -> job id


def _http(cls, msg):
    return cls(text=json.dumps({'error': msg}), content_type='application/json')


def _recording_dir():
    w = SESSIONS.get(CAPTURE_KEY)
    d = getattr(getattr(w, 'session', None), 'dir', None)
    return os.path.abspath(d) if d else None


def _ep_dir(sess, ep):
    if not (SESS_RE.match(sess) and EP_RE.match(ep)):
        raise _http(web.HTTPBadRequest, 'bad session or episode name')
    return os.path.join(CAPTURE_ROOT, sess, ep)


def _prep_key():
    try:
        return EP.cache_key(ACTIVE_JSON)
    except OSError:
        return None


# ------------------------------------------------- dataset <-> intrinsic
def _run_json(run):
    return os.path.join(CALIB_ROOT, os.path.basename(run), 'scene_intrinsics.json')


def _sha1(path):
    try:
        return EP.sha1(path)
    except OSError:
        return None


def _active_binding():
    """The active intrinsic as a dataset binds to it, or None."""
    j, sha1 = read_json(ACTIVE_JSON), _sha1(ACTIVE_JSON)
    if not j or not sha1:
        return None
    return {'run': j.get('source_run'), 'sha1': sha1,
            'solved_at': j.get('solved_at'), 'reproj': j.get('final_reproj_error')}


def _migrate(p):
    """Older projects stored only the sha1: name the run it came from."""
    b = p.get('intrinsics') or {}
    if b.get('sha1') and not b.get('run'):
        act = _active_binding()
        if act and act['sha1'] == b['sha1']:
            b['run'] = act['run']
        elif os.path.isdir(CALIB_ROOT):
            b['run'] = next((n for n in sorted(os.listdir(CALIB_ROOT))
                             if _sha1(_run_json(n)) == b['sha1']), None)
        b.pop('path', None)
    return p


def _intr_state(p, act=None):
    """ok: bound intrinsic is active. inactive: its run still exists.
    missing: deleted, re-solved, or never bound."""
    b = p.get('intrinsics') or {}
    act = act or _active_binding()
    if b.get('sha1') and act and act['sha1'] == b['sha1']:
        status = 'ok'
    elif b.get('run') and b.get('sha1') and _sha1(_run_json(b['run'])) == b['sha1']:
        status = 'inactive'
    else:
        status = 'missing'
    return {'run': b.get('run'), 'sha1': b.get('sha1'), 'solved_at': b.get('solved_at'),
            'status': status, 'active': act}


def _require_bound(p):
    st = _intr_state(p)
    if st['status'] != 'ok':
        raise _http(web.HTTPConflict, f'intrinsic {st["run"] or "?"} is not active: '
                                      f'activate it, or rebind this dataset')


def _bound_runs():
    """run -> number of datasets bound to it."""
    out = {}
    if os.path.isdir(DATASET_ROOT):
        for f in os.listdir(DATASET_ROOT):
            p = re.match(r'^\d{8}_\d{6}_dataset\.json$', f) and read_json(os.path.join(DATASET_ROOT, f))
            run = p and _migrate(p).get('intrinsics', {}).get('run')
            if run:
                out[run] = out.get(run, 0) + 1
    return out


def _export_running(pid):
    jid = EXPORT_JOBS.get(pid)
    return bool(jid) and JOBS.get(jid, {}).get('status') == 'running'


def _session_row(name, key):
    d = os.path.join(CAPTURE_ROOT, name)
    meta = read_json(os.path.join(d, 'session.json')) or {}
    eps = _episodes_on_disk(d)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    scene = meta.get('scene', {})
    return {
        'name': name, 'episodes': len(eps), 'complete': bool(meta),
        'duration_s': round(sum(e.get('duration_s', 0) for e in meta.get('episodes', [])), 1),
        'size_mb': round(size / 1e6, 1),
        'prepped': sum(bool(key) and EP.is_prepped(os.path.join(d, e), key) for e in eps),
        'focus': scene.get('focus_absolute'), 'zoom': scene.get('zoom_absolute'),
    }


async def edit_sessions(request):
    """Capture sessions to pick from; the one still recording is hidden."""
    if not os.path.isdir(CAPTURE_ROOT):
        return web.json_response([])
    rec, key = _recording_dir(), _prep_key()
    names = [n for n in sorted(os.listdir(CAPTURE_ROOT), reverse=True)
             if SESS_RE.match(n) and os.path.join(CAPTURE_ROOT, n) != rec]
    rows = await asyncio.to_thread(lambda: [_session_row(n, key) for n in names])
    return web.json_response([r for r in rows if r['episodes']])


# ------------------------------------------------------------------ projects
def _proj_path(pid):
    if not SESS_RE.match(pid or ''):
        raise _http(web.HTTPBadRequest, 'bad project id')
    return os.path.join(DATASET_ROOT, f'{pid}_dataset.json')


def _load_proj(pid):
    p = read_json(_proj_path(pid))
    if p is None:
        raise _http(web.HTTPNotFound, f'no such project: {pid}')
    return _migrate(p)


def _save_proj(p):
    p['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    os.makedirs(DATASET_ROOT, exist_ok=True)
    path = _proj_path(p['id'])
    with open(path + '.tmp', 'w') as f:
        json.dump(p, f, indent=2)
    os.replace(path + '.tmp', path)
    return p


def _proj_episodes(p):
    """[(session, ep, 'session/ep')] for every episode on disk."""
    return [(s, e, f'{s}/{e}') for s in p['sessions']
            for e in _episodes_on_disk(os.path.join(CAPTURE_ROOT, s))]


def _verified_ok(p, sess, ep):
    return (p.get('verify') or {}).get(sess, {}).get('episodes', {}).get(ep, {}).get('ok', False)


def _check_sessions(v):
    if not isinstance(v, list) or not v:
        raise _http(web.HTTPBadRequest, 'pick at least one session')
    rec = _recording_dir()
    for s in v:
        d = os.path.join(CAPTURE_ROOT, str(s))
        if not SESS_RE.match(str(s)) or not os.path.isdir(d):
            raise _http(web.HTTPBadRequest, f'no such session: {s}')
        if d == rec:
            raise _http(web.HTTPConflict, f'{s} is still recording')
    return sorted(set(v))


def _clean_trim(t):
    if not isinstance(t, (list, tuple)) or len(t) != 2:
        return None
    a, b = (None if x is None else round(float(x), 4) for x in t)
    if a is None and b is None:
        return None
    if a is not None and b is not None and b <= a:
        return None
    return [a, b]


def _proj_view(p):
    """The project plus what only the server knows (a running export, whether
    its intrinsic is active)."""
    running = EXPORT_JOBS[p['id']] if _export_running(p['id']) else None
    return {**p, 'export_job': running, 'intrinsics_state': _intr_state(p)}


async def edit_projects(request):
    rows, act = [], _active_binding()
    if os.path.isdir(DATASET_ROOT):
        for f in sorted(os.listdir(DATASET_ROOT), reverse=True):
            m = re.match(r'^(\d{8}_\d{6})_dataset\.json$', f)
            p = m and read_json(os.path.join(DATASET_ROOT, f))
            if not p:
                continue
            _migrate(p)
            eps = _proj_episodes(p)
            z = os.path.join(DATASET_ROOT, f'{p["id"]}_dataset.zarr.zip')
            rows.append({
                'id': p['id'], 'created': p.get('created'), 'updated': p.get('updated'),
                'stage': p.get('stage'), 'sessions': p['sessions'], 'episodes': len(eps),
                'included': sum(_verified_ok(p, s, e) and not p['episodes'].get(k, {}).get('excluded')
                                for s, e, k in eps) if p.get('verify') else None,
                'export': p.get('export'), 'intrinsics': _intr_state(p, act),
                'local': all(os.path.isdir(os.path.join(CAPTURE_ROOT, s)) for s in p['sessions']),
                'uploaded': p.get('uploaded'),
                'zip_mb': round(os.path.getsize(z) / 1e6, 1) if os.path.isfile(z) else None,
            })
    files = [d for d in _datasets() if not d['project']]
    return web.json_response({'projects': rows, 'active': act, 'files': files})


async def edit_project_create(request):
    body = await request.json()
    sessions = _check_sessions(body.get('sessions'))
    act = _active_binding()
    if act is None:
        raise _http(web.HTTPConflict, 'no scene intrinsic is active: calibrate or upload one first')
    pid = time.strftime('%Y%m%d_%H%M%S')
    while os.path.exists(_proj_path(pid)):
        await asyncio.sleep(1)
        pid = time.strftime('%Y%m%d_%H%M%S')
    p = _save_proj({
        'id': pid, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'stage': 'verify',
        'sessions': sessions, 'intrinsics': act,
        'verify': {}, 'episodes': {}, 'export': None,
    })
    log_edit.info('project %s created with intrinsic %s: %s', pid, act['run'], ', '.join(sessions))
    return web.json_response(_proj_view(p))


async def edit_project_get(request):
    return web.json_response(_proj_view(_load_proj(request.match_info['pid'])))


async def edit_project_put(request):
    """Autosave. Only the editable fields; verify and export are server-owned."""
    body = await request.json()
    p = _load_proj(request.match_info['pid'])
    _require_bound(p)
    if 'sessions' in body:
        new = _check_sessions(body['sessions'])
        if new != p['sessions']:
            p['sessions'] = new
            p['verify'] = {s: v for s, v in (p.get('verify') or {}).items() if s in new}
            p['episodes'] = {k: v for k, v in p['episodes'].items()
                             if k.split('/')[0] in new}
            p['stage'] = 'verify'
    for k, v in (body.get('episodes') or {}).items():
        sess, _, ep = k.partition('/')
        if sess not in p['sessions'] or not EP_RE.match(ep):
            continue
        e = {'excluded': bool(v.get('excluded')), 'trim': _clean_trim(v.get('trim'))}
        if e['excluded'] or e['trim']:
            p['episodes'][k] = e
        else:
            p['episodes'].pop(k, None)
    st = body.get('stage')
    if st in STAGES and STAGES.index(st) > STAGES.index(p.get('stage', 'select')):
        if st != 'verify' and any(s not in (p.get('verify') or {}) for s in p['sessions']):
            raise _http(web.HTTPConflict, 'verify every session first')
        p['stage'] = st
    return web.json_response(_proj_view(_save_proj(p)))


async def edit_project_delete(request):
    pid = request.match_info['pid']
    p = _load_proj(pid)
    if p.get('export'):
        raise _http(web.HTTPConflict, 'this project has an exported dataset; '
                                      'delete the dataset instead')
    os.remove(_proj_path(pid))
    log_edit.info('project %s deleted', pid)
    return web.json_response({'ok': True})


async def edit_project_verify(request):
    pid = request.match_info['pid']
    p = _load_proj(pid)
    _require_bound(p)
    out = {}
    for s in p['sessions']:
        try:
            r = await asyncio.to_thread(VF.verify_session,
                                        os.path.join(CAPTURE_ROOT, s), ACTIVE_JSON)
        except Exception as e:                       # noqa: BLE001
            log_edit.exception('verify %s crashed', s)
            r = {'session': s, 'ok': False, 'warnings': [], 'episodes': {}, 'lines': [],
                 'session_fails': [f'verify crashed: {type(e).__name__}: {e}']}
        r['ran_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        out[s] = r
    p = _load_proj(pid)          # re-read: the UI may have saved meanwhile
    p['verify'] = out
    _save_proj(p)
    eps = _proj_episodes(p)
    log_edit.info('project %s verified: %d/%d episodes pass', pid,
                  sum(_verified_ok(p, s, e) for s, e, _ in eps), len(eps))
    return web.json_response(_proj_view(p))


async def edit_project_rebind(request):
    """Bind to the active intrinsic. Episodes re-prep as they are viewed or
    exported; verify reruns, since its focus/zoom check depends on it."""
    pid = request.match_info['pid']
    p = _load_proj(pid)
    act = _active_binding()
    if act is None:
        raise _http(web.HTTPConflict, 'no scene intrinsic is active')
    if _export_running(pid):
        raise _http(web.HTTPConflict, 'this dataset is exporting')
    old = (p.get('intrinsics') or {}).get('run')
    p['intrinsics'], p['verify'] = act, {}
    _save_proj(p)
    log_edit.info('project %s rebound: intrinsic %s -> %s', pid, old, act['run'])
    return web.json_response(_proj_view(p))


# ---------------------------------------------------------------- prep queue
class PrepQueue:
    """episode_prep.py subprocesses in the background, the viewed episode first.

    Starts nothing while a capture is recording or an export is running:
    detection is CPU-heavy and would cost the recording frames. A project's
    queued episodes are dropped once nobody has its Edit page open; ones already
    running finish, since their result is cached. So are queued episodes whose
    project's intrinsic is no longer the active one.
    """

    SCRIPT = os.path.join(DATASET_DIR, 'episode_prep.py')
    SPAN = {'start': (0, 0), 'wrist': (0, 5), 'detect': (5, 70),
            'track': (70, 72), 'render': (72, 100)}

    def __init__(self):
        self.jobs = {}       # 'sess/ep' -> {status, stage, i, n, err, meta, proc}
        self.queue = []
        self.owners = {}     # 'sess/ep' -> {project ids that asked for it}
        self.watchers = {}   # project id -> open Edit pages
        self.focus = None    # the episode someone is looking at
        self.need = {}       # 'sess/ep' -> intrinsic sha1 it was queued for
        self.lock = threading.Lock()

    def paused(self):
        if BROKER.recording(BROKER.DEVICES):
            return 'recording'
        if any(JOBS.get(j, {}).get('status') == 'running' for j in EXPORT_JOBS.values()):
            return 'exporting'
        return None

    @staticmethod
    def _summary(meta):
        return {k: meta.get(k) for k in ('tracked', 'untrimmed', 'duration_s', 'prepared_at')}

    def want(self, keys, focus=None, pid=None, sha1=None):
        key = _prep_key()
        with self.lock:
            for k in keys:
                j = self.jobs.get(k)
                if j and j['status'] in ('queued', 'running'):
                    self.owners.setdefault(k, set()).add(pid)
                    continue
                d = os.path.join(CAPTURE_ROOT, k)
                if key and EP.is_prepped(d, key):
                    if not j or j['status'] != 'done':
                        self.jobs[k] = {'status': 'done', 'meta': self._summary(EP.read_prep(d))}
                    continue
                if j and j['status'] == 'failed' and k != focus:
                    continue             # retried only when looked at again
                self.jobs[k] = {'status': 'queued'}
                self.queue.append(k)
                self.need[k] = sha1
                self.owners.setdefault(k, set()).add(pid)
            if focus in keys:
                # the project's queue restarts at the focus: it, the episodes
                # after it, then wrap back to where it left off
                self.focus = focus
                at = keys.index(focus)
                rank = {k: (i - at) % len(keys) for i, k in enumerate(keys)}
                mine = sorted((k for k in self.queue if k in rank), key=rank.get)
                self.queue = mine + [k for k in self.queue if k not in rank]
        self.pump()

    def pump(self):
        key = _prep_key()
        with self.lock:
            stale = [k for k in self.queue if key is None
                     or self.need.get(k) != key['intrinsics_sha1']]
            for k in stale:
                self.queue.remove(k)
                self.jobs.pop(k, None)
                self.need.pop(k, None)
        if stale:
            log_edit.info('active intrinsic changed: dropped %d queued prep(s)', len(stale))
        if key is None or self.paused():
            return
        with self.lock:
            while self.queue:
                running = sum(j['status'] == 'running' for j in self.jobs.values())
                # the viewed episode may take one extra slot rather than wait
                if running < PREP_WORKERS or (running == PREP_WORKERS
                                              and self.queue[0] == self.focus):
                    self._spawn(self.queue.pop(0))
                else:
                    break

    def _spawn(self, k):
        self.need.pop(k, None)
        cmd = [sys.executable, '-u', self.SCRIPT, os.path.join(CAPTURE_ROOT, k),
               '--intrinsics', ACTIVE_JSON]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=REPO)
        job = self.jobs[k] = {'status': 'running', 'stage': 'start', 'i': 0, 'n': 1,
                              'proc': proc, 'tail': [], 'started': time.time()}
        log_edit.info('prep %s started', k)
        threading.Thread(target=self._drain, args=(k, job), daemon=True).start()

    def _drain(self, k, job):
        proc = job['proc']
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line.startswith('PROGRESS '):
                _, stage, i, n = line.split()
                job.update(stage=stage, i=int(i), n=int(n))
            else:
                job['tail'] = (job['tail'] + [line])[-20:]
                log_edit.debug('prep %s | %s', k, line)
        rc = proc.wait()
        with self.lock:
            job['proc'] = None
            if rc == 0:
                job['status'] = 'done'
                job['meta'] = self._summary(EP.read_prep(os.path.join(CAPTURE_ROOT, k)) or {})
                log_edit.info('prep %s done in %.0fs: %s', k, time.time() - job['started'],
                              EP.summary(job['meta']) if job['meta'].get('untrimmed') else '?')
            else:
                job['status'] = 'failed'
                job['err'] = next((ln for ln in reversed(job['tail']) if ln.strip()),
                                  f'exit code {rc}')
                log_edit.error('prep %s failed (rc=%s):\n%s', k, rc, '\n'.join(job['tail'][-8:]))
        self.pump()

    def watch(self, pid):
        with self.lock:
            self.watchers[pid] = self.watchers.get(pid, 0) + 1

    def unwatch(self, pid):
        with self.lock:
            self.watchers[pid] -= 1
            if self.watchers[pid] > 0:
                return
            del self.watchers[pid]
            dropped = []
            for k in list(self.owners):
                self.owners[k].discard(pid)
                if self.owners[k]:
                    continue
                del self.owners[k]
                if k in self.queue:
                    self.queue.remove(k)
                    self.jobs.pop(k, None)
                    self.need.pop(k, None)
                    dropped.append(k)
            running = sum(j['status'] == 'running' for j in self.jobs.values())
        if dropped:
            log_edit.info('project %s closed: dropped %d queued prep(s), %d running '
                          'will finish', pid, len(dropped), running)

    def view(self, keys):
        out = {}
        for k in keys:
            j = self.jobs.get(k)
            if j is None:
                continue
            v = {'status': j['status']}
            if j['status'] == 'running':
                lo, hi = self.SPAN.get(j['stage'], (0, 100))
                v.update(stage=j['stage'], pct=round(lo + (hi - lo) * j['i'] / max(j['n'], 1)))
            elif j['status'] == 'failed':
                v['err'] = j.get('err')
            elif j['status'] == 'done':
                v['meta'] = j.get('meta')
            out[k] = v
        return out

    def stop(self):
        with self.lock:
            self.queue.clear()
            for j in self.jobs.values():
                if j.get('proc') is not None:
                    j['proc'].terminate()


PREP = PrepQueue()


async def edit_prep(request):
    """Queue prep for the project's verified episodes; `focus` jumps the queue."""
    p = _load_proj(request.match_info['pid'])
    _require_bound(p)
    body = await request.json() if request.can_read_body else {}
    keys = [k for s, e, k in _proj_episodes(p) if _verified_ok(p, s, e)]
    focus = body.get('focus')
    if focus:
        _ep_dir(*focus.partition('/')[::2])
        if focus not in keys:
            keys.append(focus)
    await asyncio.to_thread(PREP.want, keys, focus, p['id'], p['intrinsics']['sha1'])
    return web.json_response({'ok': True, 'paused': PREP.paused()})


async def edit_prep_state(request):
    p = _load_proj(request.match_info['pid'])
    keys = [k for _, _, k in _proj_episodes(p)]
    resp = web.StreamResponse(headers={
        'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'})
    await resp.prepare(request)
    PREP.watch(p['id'])
    last = None
    try:
        # aiohttp does not cancel a handler whose client left, and this only
        # writes on change, so check the connection itself
        while (request.transport is not None and not request.transport.is_closing()
               and not STOPPING.is_set()):
            # re-read: a rebind or an activation elsewhere changes the status
            cur = _migrate(read_json(_proj_path(p['id'])) or p)
            txt = json.dumps({'paused': PREP.paused(), 'episodes': PREP.view(keys),
                              'intrinsics': _intr_state(cur)['status']})
            if txt != last:
                await resp.write(f'data: {txt}\n\n'.encode())
                last = txt
            await asyncio.sleep(0.5)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        PREP.unwatch(p['id'])
    return resp


async def edit_pump():
    """Restart the queue once a recording or export ends."""
    try:
        while True:
            await asyncio.sleep(1.0)
            PREP.pump()
    except asyncio.CancelledError:
        pass


# ------------------------------------------------------------- per episode
def _require_prepped(d, videos=True):
    key = _prep_key()
    if not key or not EP.is_prepped(d, key, videos):
        raise _http(web.HTTPConflict, 'episode is not prepared yet')


def _rpy_deg(R):
    """(n,3,3) -> (n,3) extrinsic XYZ roll/pitch/yaw in degrees."""
    return np.degrees(np.stack([np.arctan2(R[:, 2, 1], R[:, 2, 2]),
                                np.arcsin(np.clip(-R[:, 2, 0], -1, 1)),
                                np.arctan2(R[:, 1, 0], R[:, 0, 0])], 1))


def _decimate(n, most=600):
    return np.arange(0, n, max(1, -(-n // most)))


def _rows(a):
    return [np.round(r, 2).tolist() if np.isfinite(r).all() else None for r in a]


def _timeline(d):
    """Per-frame status plus the absolute TCP pose (scene-camera frame) and the
    gripper width, decimated for the graphs. Lost and spike frames are left out
    of the pose, as export leaves them out."""
    tr = TL.load_track(d)
    meta = EP.read_prep(d)
    t0 = tr['t_ns'][0] / 1e9
    t = tr['t_ns'] / 1e9 - t0
    use = tr['tracked'] & (tr['status'] != TL.SPIKE)
    xyz = np.full((len(t), 3), np.nan)
    rpy = np.full((len(t), 3), np.nan)
    if use.any():
        T = tr['T_tcp'][use]
        xyz[use] = T[:, :3, 3] * 1000
        # unwrapped over the kept frames: roll sits near +-180 on this rig
        rpy[use] = np.degrees(np.unwrap(np.radians(_rpy_deg(T[:, :3, :3])), axis=0))
    idx = _decimate(len(t))
    grip = TL.gripper_track(d)
    if grip is not None:
        gt, gw = grip
        gi = _decimate(len(gt))
        grip = {'t': np.round(gt[gi] - t0, 4).tolist(), 'width': np.round(gw[gi] * 1000, 2).tolist()}
    return {
        't': np.round(t, 4).tolist(), 'status': tr['status'].tolist(),
        'status_names': TL.STATUS,
        'tcp': {'t': np.round(t[idx], 4).tolist(),
                'xyz': _rows(xyz[idx]), 'rpy': _rows(rpy[idx])},
        'gripper': grip,
        'wrist_crop': _wrist_crop(d),
        'duration_s': float(t[-1]), 'wrist_offset_s': meta['wrist_offset_s'],
        'wrist_duration_s': meta['wrist_duration_s'], 'meta': meta,
    }


def _wrist_crop(d):
    """The training crop as fractions of the wrist frame, for the edit overlay."""
    with av.open(os.path.join(d, 'wrist', 'wrist.mkv')) as c:
        st = c.streams.video[0]
        w, h = st.width, st.height
    x0, side = TL.wrist_crop(w, h)
    return {'x0': x0, 'size': side, 'w': w, 'h': h}


async def edit_timeline(request):
    d = _ep_dir(request.match_info['sess'], request.match_info['ep'])
    _require_prepped(d)
    return web.json_response(await asyncio.to_thread(_timeline, d))


async def edit_plan(request):
    """What export would keep of this episode, for a given trim."""
    d = _ep_dir(request.match_info['sess'], request.match_info['ep'])
    _require_prepped(d, videos=False)
    q = request.query
    trim = _clean_trim([q.get('in') or None, q.get('out') or None])
    r = await asyncio.to_thread(TL.plan_episode, d, trim, False)
    return web.json_response({k: r[k] for k in
                              ('usable', 'kept_s', 'spans', 'reason', 'note', 'grid_n')})


async def edit_media(request):
    d = _ep_dir(request.match_info['sess'], request.match_info['ep'])
    f = request.match_info['file']
    p = os.path.join(d, TL.DERIVED, f)
    if f not in MEDIA or not os.path.exists(p):
        raise _http(web.HTTPNotFound, f'no {f} for this episode')
    return web.FileResponse(p, headers={'Cache-Control': 'no-cache'})


async def edit_export(request):
    pid = request.match_info['pid']
    p = _load_proj(pid)
    if _export_running(pid):
        return web.json_response({'ok': True, 'job_id': EXPORT_JOBS[pid], 'existing': True})
    _require_bound(p)
    if BROKER.recording(BROKER.DEVICES):
        return json_err(409, 'a capture is recording -- export after it stops')
    if not any(_verified_ok(p, s, e) and not p['episodes'].get(k, {}).get('excluded')
               for s, e, k in _proj_episodes(p)):
        return json_err(400, 'no episodes are included')
    p['stage'] = 'export'
    _save_proj(p)
    # the run's own copy: activating another one mid-export cannot mix them
    b = p['intrinsics']
    intr = _run_json(b['run']) if b.get('run') else ACTIVE_JSON
    if _sha1(intr) != b['sha1']:
        intr = ACTIVE_JSON
    cmd = [sys.executable, '-u', os.path.join(DATASET_DIR, 'build_zarr.py'),
           '--project', _proj_path(pid), '--intrinsics', intr]
    jid = EXPORT_JOBS[pid] = _spawn_job(cmd, f'export {pid}', project=pid)
    log_edit.info('project %s export started (job %s)', pid, jid)
    return web.json_response({'ok': True, 'job_id': jid})


# ------------------------------------------------------------ dataset files
ZIP_RE = re.compile(r'^[\w-][\w.-]*\.zarr\.zip$')
PROJ_RE = re.compile(r'^(\d{8}_\d{6})_dataset\.json$')
UPLOAD_JSON_MAX = 10 << 20


def _ds_path(name):
    if not (ZIP_RE.match(name or '') or PROJ_RE.match(name or '')):
        raise _http(web.HTTPBadRequest, f'bad dataset file name: {name}')
    return os.path.join(DATASET_ROOT, name)


def _zip_id(name):
    m = re.match(r'^(\d{8}_\d{6})_dataset\.zarr\.zip$', name)
    return m.group(1) if m else None


def _is_zarr_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False
    return ('.zgroup' in names and 'meta/episode_ends/.zarray' in names
            and any(n.startswith('data/') for n in names))


def _datasets():
    """Every .zarr.zip, newest first, with its project's export summary."""
    out = []
    if not os.path.isdir(DATASET_ROOT):
        return out
    for f in os.listdir(DATASET_ROOT):
        if not ZIP_RE.match(f):
            continue
        st = os.stat(os.path.join(DATASET_ROOT, f))
        pid = _zip_id(f)
        p = read_json(_proj_path(pid)) if pid else None
        e = (p or {}).get('export') or {}
        b = e.get('intrinsics') or (_migrate(p).get('intrinsics') if p else None) or {}
        out.append({'file': f, 'id': pid, 'project': bool(p), 'uploaded': (p or {}).get('uploaded'),
                    'size_mb': round(st.st_size / 1e6, 1), 'mtime': st.st_mtime,
                    'episodes': e.get('episodes'), 'steps': e.get('steps'),
                    'duration_s': e.get('duration_s'), 'intrinsic': b.get('run')})
    return sorted(out, key=lambda d: d['mtime'], reverse=True)


async def datasets_list(request):
    return web.json_response(await asyncio.to_thread(_datasets))


async def datasets_upload(request):
    """Raw-body upload into data/dataset/: a .zarr.zip (streamed, so only disk
    limits it), or the <id>_dataset.json that goes with one already here."""
    name = request.query.get('name', '')
    path = _ds_path(name)
    if os.path.exists(path):
        return json_err(409, f'{name} already exists here')
    os.makedirs(DATASET_ROOT, exist_ok=True)
    if name.endswith('.json'):
        return await _upload_project(request, name)

    need = request.content_length or 0
    free = shutil.disk_usage(DATASET_ROOT).free
    if need + (1 << 30) > free:
        return json_err(507, f'not enough disk: {need / 1e9:.1f} GB upload, '
                             f'{free / 1e9:.1f} GB free')
    part = os.path.join(DATASET_ROOT, f'.{name}.part')
    size, done = 0, False
    try:
        try:
            with open(part, 'wb') as f:
                async for chunk in request.content.iter_chunked(1 << 20):
                    f.write(chunk)
                    size += len(chunk)
        except ConnectionResetError:
            log_edit.warning('upload of %s dropped at %.0f MB', name, size / 1e6)
            return json_err(400, 'connection lost')
        if need and size != need:
            return json_err(400, f'{name}: upload cut short at {size / 1e6:.0f} MB')
        if not await asyncio.to_thread(_is_zarr_zip, part):
            return json_err(400, f'{name} is not a dataset .zarr.zip')
        os.replace(part, path)
        done = True
    finally:
        if not done and os.path.exists(part):
            os.remove(part)
    log_edit.info('uploaded %s (%.1f MB)', name, size / 1e6)
    return web.json_response({'ok': True, 'file': name, 'size_mb': round(size / 1e6, 1)})


async def _upload_project(request, name):
    raw = b''
    async for chunk in request.content.iter_chunked(1 << 16):
        raw += chunk
        if len(raw) > UPLOAD_JSON_MAX:
            return json_err(413, f'{name} is too big for a project file')
    pid = PROJ_RE.match(name).group(1)
    try:
        p = json.loads(raw)
    except ValueError as e:
        return json_err(400, f'{name} is not valid JSON ({e})')
    if not isinstance(p, dict) or p.get('id') != pid or not isinstance(p.get('sessions'), list):
        return json_err(400, f'{name} is not a dataset project file for {pid}')
    if not os.path.exists(os.path.join(DATASET_ROOT, f'{pid}_dataset.zarr.zip')):
        return json_err(409, f'upload {pid}_dataset.zarr.zip first')
    p.setdefault('episodes', {})
    p['uploaded'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    _save_proj(p)
    log_edit.info('uploaded %s', name)
    return web.json_response({'ok': True, 'file': name})


async def datasets_download(request):
    name = request.match_info['file']
    path = _ds_path(name)
    if not os.path.isfile(path):
        raise _http(web.HTTPNotFound, f'no {name}')
    return web.FileResponse(path, headers={
        'Content-Disposition': f'attachment; filename="{name}"'})


async def datasets_delete(request):
    """The whole dataset: the .zarr.zip and its project json, if any."""
    name = request.match_info['file']
    if not ZIP_RE.match(name):
        raise _http(web.HTTPBadRequest, 'name the .zarr.zip')
    pid = _zip_id(name)
    if pid and _export_running(pid):
        raise _http(web.HTTPConflict, f'{pid} is exporting')
    gone = [f for f in (_ds_path(name), pid and _proj_path(pid)) if f and os.path.exists(f)]
    if not gone:
        raise _http(web.HTTPNotFound, f'no {name}')
    for f in gone:
        os.remove(f)
    log_edit.info('deleted dataset %s', ', '.join(os.path.basename(f) for f in gone))
    return web.json_response({'ok': True, 'deleted': [os.path.basename(f) for f in gone]})


# ------------------------------------------------------------------ training
CONTAINER = 'robospec_umi'   # container_name in robospec_umi_compose.yaml
_YAML = {}                   # path -> (mtime, parsed)


def _read_yaml(path):
    try:
        mt = os.path.getmtime(path)
        if _YAML.get(path, (None,))[0] != mt:
            with open(path) as f:
                _YAML[path] = (mt, yaml.safe_load(f))
        return _YAML[path][1]
    except (OSError, yaml.YAMLError):
        return None


async def train_info(request):
    cfg = _read_yaml(TRAIN_CONFIG) or {}
    tr = cfg.get('training') or {}
    return web.json_response({
        'in_container': os.path.exists('/.dockerenv'), 'container': CONTAINER, 'repo': REPO,
        'config': os.path.splitext(os.path.basename(TRAIN_CONFIG))[0],
        'defaults': {'num_epochs': tr.get('num_epochs'), 'checkpoint_every': tr.get('checkpoint_every'),
                     'lr': (cfg.get('optimizer') or {}).get('lr')},
    })


async def train_gpu(request):
    try:
        proc = await asyncio.create_subprocess_exec(
            'nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return web.json_response({'error': 'nvidia-smi not found'})
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 3)
    except asyncio.TimeoutError:
        proc.kill()
        return web.json_response({'error': 'nvidia-smi timed out'})

    def num(v):
        try:
            return float(v)
        except ValueError:
            return None
    gpus = []
    for line in out.decode().splitlines():
        f = [x.strip() for x in line.split(',')]
        if len(f) == 4:
            gpus.append({'name': f[0], 'mem_used_mb': num(f[1]), 'mem_total_mb': num(f[2]),
                         'util': num(f[3])})
    return web.json_response({'gpus': gpus} if gpus else {'error': 'no GPU reported'})


def _last_json_line(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode(errors='replace').splitlines()
    except OSError:
        return None
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except ValueError:
            continue
    return None


def _runs():
    """Training runs under data/outputs, newest activity first. A checkpoint
    belongs to its directory, or the run above a checkpoints/ directory."""
    runs = {}
    now = time.time()
    for root, dirs, files in os.walk(OUTPUTS_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'wandb']
        run = os.path.dirname(root) if os.path.basename(root) == 'checkpoints' else root
        ckpts = [f for f in files if f.endswith('.ckpt')]
        if ckpts or 'logs.json.txt' in files:
            runs.setdefault(run, [])
        for f in ckpts:
            st = os.stat(os.path.join(root, f))
            runs[run].append({'path': os.path.relpath(os.path.join(root, f), OUTPUTS_ROOT),
                              'name': f, 'size_mb': round(st.st_size / 1e6, 1),
                              'mtime': st.st_mtime, 'saving': now - st.st_mtime < 30})
    out = []
    for run, ckpts in runs.items():
        logp = os.path.join(run, 'logs.json.txt')
        cfg = _read_yaml(os.path.join(run, '.hydra', 'config.yaml')) or {}
        ov = _read_yaml(os.path.join(run, '.hydra', 'overrides.yaml')) or []
        last = _last_json_line(logp)
        seen = [c['mtime'] for c in ckpts] + ([os.path.getmtime(logp)] if last else [])
        out.append({
            'run': os.path.relpath(run, OUTPUTS_ROOT),
            'dataset': next((o.split('=', 1)[1] for o in ov if isinstance(o, str)
                             and o.startswith('task.dataset_path=')), None),
            'num_epochs': (cfg.get('training') or {}).get('num_epochs'),
            'progress': last and {k: last.get(k) for k in ('epoch', 'global_step', 'train_loss')},
            'running': bool(last) and now - os.path.getmtime(logp) < 120,
            'updated': max(seen, default=os.path.getmtime(run)),
            'checkpoints': sorted(ckpts, key=lambda c: c['mtime'], reverse=True),
        })
    return sorted(out, key=lambda r: r['updated'], reverse=True)


async def train_runs(request):
    return web.json_response(await asyncio.to_thread(_runs))


async def train_ckpt(request):
    rel = request.query.get('path', '')
    root = os.path.realpath(OUTPUTS_ROOT)
    path = os.path.realpath(os.path.join(root, rel))
    if not (path.startswith(root + os.sep) and path.endswith('.ckpt') and os.path.isfile(path)):
        raise _http(web.HTTPNotFound, f'no checkpoint {rel}')
    # the run in the name, so latest.ckpt from two runs cannot collide
    name = os.path.relpath(path, root).replace(os.sep + 'checkpoints' + os.sep, os.sep)
    name = name.replace(os.sep, '_')
    log_train.info('download %s (%.0f MB)', rel, os.path.getsize(path) / 1e6)
    return web.FileResponse(path, headers={
        'Content-Disposition': f'attachment; filename="{name}"'})


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


# index.html must revalidate, or a rebuilt UI keeps loading the old bundle;
# the hashed files under assets/ can cache freely
NO_CACHE = {'Cache-Control': 'no-cache'}


async def index(request):
    if ui_built():
        return web.FileResponse(os.path.join(UI_DIST, 'index.html'), headers=NO_CACHE)
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
        return web.FileResponse(os.path.join(UI_DIST, 'index.html'), headers=NO_CACHE)
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
    app['edit_pump'] = asyncio.create_task(edit_pump())


async def on_shutdown(app):
    STOPPING.set()


async def on_cleanup(app):
    app['reaper'].cancel()
    app['edit_pump'].cancel()
    PREP.stop()
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
    r.add_post('/api/calibration/upload', calib_upload)
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

    r.add_get('/api/edit/sessions', edit_sessions)
    r.add_get('/api/edit/projects', edit_projects)
    r.add_post('/api/edit/projects', edit_project_create)
    r.add_get('/api/edit/projects/{pid}', edit_project_get)
    r.add_put('/api/edit/projects/{pid}', edit_project_put)
    r.add_delete('/api/edit/projects/{pid}', edit_project_delete)
    r.add_post('/api/edit/projects/{pid}/verify', edit_project_verify)
    r.add_post('/api/edit/projects/{pid}/rebind', edit_project_rebind)
    r.add_post('/api/edit/projects/{pid}/prep', edit_prep)
    r.add_get('/api/edit/projects/{pid}/prep/state', edit_prep_state)
    r.add_post('/api/edit/projects/{pid}/export', edit_export)
    r.add_get('/api/edit/episode/{sess}/{ep}/timeline', edit_timeline)
    r.add_get('/api/edit/episode/{sess}/{ep}/plan', edit_plan)
    r.add_get('/api/edit/media/{sess}/{ep}/{file}', edit_media)

    r.add_get('/api/datasets', datasets_list)
    r.add_post('/api/datasets/upload', datasets_upload)
    r.add_get('/api/datasets/{file}', datasets_download)
    r.add_delete('/api/datasets/{file}', datasets_delete)
    r.add_get('/api/train/info', train_info)
    r.add_get('/api/train/gpu', train_gpu)
    r.add_get('/api/train/runs', train_runs)
    r.add_get('/api/train/ckpt', train_ckpt)

    r.add_get('/', index)
    if ui_built():
        assets = os.path.join(UI_DIST, 'assets')
        if os.path.isdir(assets):
            r.add_static('/assets', assets)
    r.add_route('*', '/{tail:.*}', spa_fallback)

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
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
    if not args.verbose:
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
