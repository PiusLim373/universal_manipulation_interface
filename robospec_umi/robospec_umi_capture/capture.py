#!/usr/bin/env python3
"""Synchronised dual-camera episode recorder: scene (ChArUco tracking) + RealSense
D405 or D455 (wrist).

Both cameras stream continuously from launch and are *gated* per episode: frames are
pulled and discarded until you arm an episode, then written until you stop. Nothing is
opened or closed between episodes.

That matters for more than convenience. A camera needs about a second after streaming
starts before it behaves -- the first frames arrive in a burst with drops, and auto
white balance and sensor warm-up shift the colour. Restarting per episode would put
that transient at the *start of every episode*, which is the part you least want
corrupted. Streaming continuously pays it once, during warm-up, while nothing is
being recorded.

Exposure AND white balance are locked on the scene camera. Auto exposure moves the
exposure midpoint frame to frame, which turns the constant camera-to-camera offset
into time-varying noise; auto white balance makes the same scene change colour between
episodes, which is a spurious signal for the policy to latch onto. The D455 wrist runs
both on auto by default, by choice (see WRIST_MODELS).

Sync is established after the fact: every frame carries its own timestamp on a shared
CLOCK_MONOTONIC timeline, and post-processing interpolates the pose track onto wrist
frame times. On this rig 1 ms of timing error costs ~1 mm of TCP label error at p90
hand speed, which is the budget the whole design is built around.

Why timestamps sit in a sidecar and not in the video file: neither container can carry
them. MJPEG will not mux into MP4 at all ("unspecified pixel format"), `-copyts` into
MKV writes a file that no longer opens, and MKV without it restarts timestamps at zero
and rounds them to 1 ms -- the entire error budget spent on rounding. So each camera
writes an int64-nanosecond array next to its video. int64 rather than float:
CLOCK_MONOTONIC is a ~475000 s number, and float32 collapses an 8.3 ms step to 0.

That leaves container choice to robustness alone, so both are MKV -- a plain MP4
killed mid-recording loses every frame, not just the tail.

Scene frames are passed through as the camera's own JPEGs: PyAV demuxes the V4L2
device and returns the compressed packet with the kernel's 1 us CLOCK_MONOTONIC
timestamp, so nothing is decoded or re-encoded and corner sharpness is exactly what
the sensor produced. Wrist frames arrive decoded and are encoded with NVENC.


THE PREVIEW WINDOW IS DELIBERATELY LOSSY -- READ THIS
-----------------------------------------------------
The preview shows a *sample* of the stream, not the stream. It skips frames freely and
that is by design: recording must never wait for the display.

  * the display holds ONE slot per camera, overwritten by the newest frame -- never a
    queue, which would grow or push back on the producer
  * the display raises a flag when it wants a frame; the capture thread only copies
    into the slot when that flag is set, so the copy happens a few times a second
    rather than at full frame rate
  * recording work runs BEFORE display work in every code path, and the display store
    is a non-blocking lock acquire -- if it cannot be taken instantly it is skipped
  * only the frames actually shown are ever decoded

The on-screen "skipped" counter is the proof it is sampling: it should be large. A
large number is correct, not a problem.

What it actually costs, measured on this rig
--------------------------------------------
Those precautions stop the display from ever *blocking* a capture thread, but they
cannot remove GIL contention. Decoding one 1080p scene JPEG takes ~10 ms (and decoding
it at reduced scale barely helps -- 10.0 ms vs 12.0 ms, since the entropy decode
dominates). The wrist camera has to be serviced every 11.1 ms or librealsense drops
frames, so a burst of cv2 work on the main thread can make it miss its slot.

Measured over a 10 s episode, wrist frame rate:

    no display                       87.7 fps      device drops 22
    preview, throttled while REC     87.7 fps      device drops 22   <- the default
    preview at full rate while REC   86.4 fps      device drops 35
    preview at 30 fps while REC      74.7 fps      device drops 150

The scene camera is unaffected in every case (48.6 fps throughout) -- it has a 20 ms
service interval and its demux releases the GIL.

So the preview backs off automatically once an episode starts (REC_DISPLAY_FPS,
REC_SCENE_PREVIEW_FPS), which is why the default column matches no-display exactly.
You need the preview for framing the shot, not during the take. --no-display removes
the window entirely.

Losses of this kind are never silent: they show up as gaps in the wrist `seq` counter,
and verify.py reports them per episode.

Usage:
    python3 robospec_umi/robospec_umi_capture/capture.py
      SPACE / ENTER   start / stop an episode
      Q / ESC         finish the session
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from fractions import Fraction

import av
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
try:
    import serial
except ImportError:
    serial = None

# Errors in the capture/write threads used to latch into `self.err` and be read
# by nobody, so a camera dying mid-session showed up only as a frame counter that
# stopped moving. With no handler configured -- the CLI case -- Python's
# last-resort handler still puts WARNING and above on stderr, so this costs the
# CLI nothing and gains it the reason.
log = logging.getLogger('robospec.capture')

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_capture
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

# ------------------------------------------------------------ fixed hardware
# A udev symlink, not a /dev/videoN number: USB enumeration order is not stable,
# and opening the wrong node hands you a different camera rather than failing.
# See robospec_umi/99-decxin-cam.rules.
SCENE_DEVICE = '/dev/scene_cam'
SCENE_SIZE = (1920, 1080)

# Geometry, not brightness: focusing changes the focal length and digital zoom
# moves f and the principal point, so both must match what the calibration was
# shot at. Fixed here rather than exposed as flags -- every rig uses these, and
# verify.py cross-checks them against the calibration's locked_controls.
SCENE_FOCUS = 0
SCENE_ZOOM = 100
# The DECXIN delivers a true 120 fps at 1920x1080 MJPG, measured, and does NOT
# trade frame rate for exposure the way the D405 does -- it holds 120 fps even at
# the top of the exposure range, because exposure_dynamic_framerate is 0.
SCENE_FPS = 120

ENCODER = 'h264_nvenc'

# Per wrist camera, picked by device name at session start. Exposure here is in us
# whatever the camera's own unit; the recorder converts on write. auto_exposure is
# its default, or None where it is not offered at all.
WRIST_MODELS = {
    # Colour comes off the stereo module. MEASURED cliff: 10000 us holds 89.9 fps,
    # 11000 gives exactly 60.0, 20000 gives 30.0 -- with ZERO dropped frames, it
    # silently renegotiates to a slower divisor of 90, so the slider stops before
    # it. No auto exposure: it pins gain at the sensor floor (16) and lands at mean
    # luma ~10 against ~62 for manual 6000 us / gain 248 -- tuned for the stereo
    # depth pipeline, not RGB appearance.
    'D405': {'size': (848, 480), 'fps': 90, 'exposure_unit_us': 1,
             'exposure_us': 6000, 'exposure_max_us': 10000, 'gain': 248,
             'auto_exposure': None, 'imu': False},
    # A separate RGB sensor, exposure in 100 us, gain 0..128. 848x480 tops out at
    # 60 fps and is broken on this unit, hence 640x360. Manual 1..109 (10.9 ms)
    # holds 89.9 fps; 110-111 give black frames, longer lowers the rate. Auto
    # exposure holds 90 fps with auto_exposure_priority 0, so it is the default.
    # See d455_test/d455_documentation.md.
    'D455': {'size': (640, 360), 'fps': 90, 'exposure_unit_us': 100,
             'exposure_us': 6000, 'exposure_max_us': 10900, 'gain': 64,
             'auto_exposure': True, 'imu': True},
}

# D455 IMU, gyro and accel both at this rate (accel actually runs ~203 Hz, so
# always go by the timestamps). Factory intrinsics are identity: uncalibrated.
IMU_HZ = 200
IMU_STALE = 0.5             # s without a sample before the UI calls it stalled

# What the wrist offers a UI; enable_auto_exposure only where the model allows it.
# Auto white balance is fine on both -- no frame-rate or brightness cost.
WRIST_TUNABLE = ('enable_auto_exposure', 'exposure', 'gain', 'white_balance',
                 'enable_auto_white_balance', 'gamma')


def wrist_model():
    """-> the WRIST_MODELS key of the connected RealSense. Reads its name only;
    options are only ever written through the started pipeline (see _configure)."""
    devs = rs.context().query_devices()
    if len(devs) == 0:
        raise RecordError('no RealSense camera found')
    name = devs[0].get_info(rs.camera_info.name)
    for k in WRIST_MODELS:
        if k in name:
            return k
    raise RecordError(f'{name} is not a supported wrist camera ({", ".join(WRIST_MODELS)})')

# Arduino on the gripper (arduino/UMI_Force_Angle.ino): '<opening>_<force>' lines at
# ~40 Hz. The opening is a pot reading, linear in finger gap from 0 (closed) to
# GRIPPER_RAW_OPEN at GRIPPER_MAX_WIDTH. See robospec_umi/99-gripper-sensor.rules.
GRIPPER_DEVICE = '/dev/gripper_sensor'
GRIPPER_BAUD = 115200
GRIPPER_RAW_OPEN = 1146
GRIPPER_MAX_WIDTH = 0.115   # m
GRIPPER_STALE = 0.5         # s without a line before the UI calls it stalled
GRIPPER_TRACE_S = 5.0       # s of opening history in the UI trace
GRIPPER_TRACE_HZ = 10.0     # its sample rate, as the wrist preview: idle
GRIPPER_TRACE_REC_HZ = 4.0  # and while recording

# Seconds of streaming before the first episode may start, so the startup burst
# and colour settling land in the bin rather than in episode 0.
WARMUP = 3.0
# Seconds the scene camera is armed before, and disarmed after, the wrist -- so
# every wrist frame has scene poses on both sides to interpolate between. Timing
# critical: shortening this leaves wrist frames at the episode edges unlabelled.
LEAD = 0.2

# ------------------------------------------------------------------- preview
# Sampled and deliberately lossy. A 1080p scene JPEG costs ~10 ms to decode and
# that decode holds the GIL long enough to starve the wrist capture thread of its
# 11.1 ms service interval, so the scene pane runs on its own slower clock and
# both rates back off again once recording starts.
DISPLAY_FPS = 10.0
SCENE_PREVIEW_FPS = 4.0
REC_DISPLAY_FPS = 4.0
REC_SCENE_PREVIEW_FPS = 1.0
PANE_HEIGHT = 420


# ------------------------------------------------------------------- clocks
def mono():
    """The shared timeline. CLOCK_MONOTONIC cannot step; CLOCK_REALTIME can (NTP,
    suspend), and a step mid-recording would silently corrupt one stream."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def frame_rate(t_ns):
    """Rate measured first stamp to last, not over wall time -- the cameras take about
    a second to start delivering, and dividing by wall time folds that dead period in
    and understates the real rate."""
    if len(t_ns) < 2:
        return 0.0
    span = (int(t_ns[-1]) - int(t_ns[0])) / 1e9
    return len(t_ns) / span if span > 0 else 0.0


def mono_to_epoch_offset(n=21):
    """Seconds to add to CLOCK_MONOTONIC to get CLOCK_REALTIME.

    librealsense reports global time as epoch milliseconds while V4L2 stamps are
    CLOCK_MONOTONIC, so one has to be converted. Reading both back to back brackets the
    offset by the read latency; the median of several rejects scheduling outliers.
    """
    return float(np.median([time.clock_gettime(time.CLOCK_REALTIME) - mono()
                            for _ in range(n)]))


# ------------------------------------------------------- preview frame slot
class LatestFrame:
    """One slot holding the newest frame for the preview, and nothing else.

    A queue is the wrong structure here: it would either grow without bound or apply
    backpressure to the capture thread. A single overwritten slot cannot do either --
    a slow display simply misses frames, which is the intended behaviour.

    `want` gates the producer so the copy only happens when the display is actually
    ready for another frame.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.item = None
        self.want = True
        self.skipped = 0

    def offer(self, make_item):
        """Called from a capture thread. Returns immediately if the display is busy or
        does not want a frame; `make_item` is only invoked when it will be used, so the
        cost of copying is not paid on skipped frames."""
        if not self.want:
            self.skipped += 1
            return
        if not self.lock.acquire(blocking=False):
            self.skipped += 1        # never wait on the display from a capture thread
            return
        try:
            self.item = make_item()
            self.want = False
        finally:
            self.lock.release()

    def take(self):
        """Called from the display thread."""
        with self.lock:
            item, self.item = self.item, None
            self.want = True
        return item


class RateMeter:
    """Live fps over a short sliding window, for the overlay."""

    def __init__(self, n=60):
        self.t = deque(maxlen=n)

    def tick(self, t=None):
        self.t.append(mono() if t is None else t)

    def fps(self):
        if len(self.t) < 2:
            return 0.0
        span = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / span if span > 0 else 0.0


# --------------------------------------------------------------- base gating
def discard_pending(pending):
    """Close and delete a pre-opened container that was never recorded into, so a
    session never ends up with empty stub files for an episode that did not happen."""
    if pending is None:
        return
    _, path, container, _ = pending
    try:
        container.close()
    except Exception:
        pass
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


class GatedRecorder:
    """Capture always, write only between arm() and disarm().

    Control messages travel through the same queue as frames, so an episode boundary
    lands exactly between two frames rather than wherever a lock happened to be held.
    """

    def __init__(self, qsize):
        self.q = queue.Queue(maxsize=qsize)
        self.stop_evt = threading.Event()
        self.dropped, self.err, self.threads = 0, None, []
        self.live = 0          # frames seen since launch, for the warm-up readout
        self.written = 0       # frames written in the current episode
        self.preview = LatestFrame()
        self.meter = RateMeter()
        self.armed = False

    def prepare(self, out_dir):
        """Open the next episode's container ahead of time.

        Creating a container -- and for the wrist an NVENC session -- takes tens of
        milliseconds and holds the GIL, which stalls the capture thread long enough
        for the driver to drop frames. Measured: a burst of 6-14 lost frames at frame
        index 1 of every episode. Doing it while idle moves that cost off the
        recording path entirely, so ARM only has to flip a flag.
        """
        self.q.put(('PREPARE', out_dir))

    def arm(self, out_dir):
        self.armed = True
        self.q.put(('ARM', out_dir))

    def disarm(self):
        self.armed = False
        self.q.put(('DISARM', None))

    def start(self):
        self.threads = [threading.Thread(target=self._write, daemon=True),
                        threading.Thread(target=self._capture, daemon=True)]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_evt.set()
        for t in self.threads:
            t.join(timeout=20)


# -------------------------------------------------------------- scene camera
def v4l2_ranges(device):
    """-> {ctrl: (min, max)} parsed from --list-ctrls.

    UVC clamps an out-of-range write silently and still reports success, so a
    value carried over from a different camera looks applied and is not. The
    DECXIN's backlight_compensation has min=16, where the old scene camera took
    0 -- exactly that failure.
    """
    r = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if ':' not in line or '0x' not in line:
            continue
        head, tail = line.split(':', 1)
        name = head.split()[0]
        if '(bool)' in head:
            out[name] = (0, 1)
            continue
        kv = dict(t.split('=', 1) for t in tail.split() if '=' in t)
        try:
            out[name] = (int(kv['min']), int(kv['max']))
        except (KeyError, ValueError):
            pass
    return out


def set_scene_controls(device, exposure_units, wb_temp, gamma=None, gain=None):
    """Lock exposure, gain, gamma, white balance, focus and zoom via v4l2.

    exposure_units is nominally UVC's 100 us ticks, but the DECXIN does not honour
    that linearly -- brightness is not proportional to the setting and there is a
    step change around 200 -- so treat it as a dial, not milliseconds.

    Focus and zoom are pinned to SCENE_FOCUS / SCENE_ZOOM because they are
    geometry. The DECXIN ships with focus_automatic_continuous=1, and recording
    with autofocus on makes the ChArUco solve return plausible, wrong depths
    with no error anywhere.

    Ordering matters. The *_absolute controls report flags=inactive and ignore
    writes until their auto counterpart is off, so auto_exposure and
    focus_automatic_continuous are cleared first.

    On brightness: prefer buying it with exposure rather than gamma. Gamma is a
    tone curve applied after digitisation, so it amplifies noise along with
    signal, while real photons improve SNR. Measured on the DECXIN at matched
    brightness, exposure 800 / gamma 128 has roughly half the temporal noise of
    exposure 200 / gamma 160. Its gain control does something between 0 and ~200
    (mean 2.0 at gain 0, 6.4 at gain 200) and essentially nothing above that
    (6.5 at gain 1023).
    """
    wanted = [('focus_automatic_continuous', 0), ('focus_absolute', SCENE_FOCUS),
              ('auto_exposure', 1), ('exposure_time_absolute', exposure_units),
              ('white_balance_automatic', 0), ('white_balance_temperature', wb_temp),
              ('backlight_compensation', 0), ('zoom_absolute', SCENE_ZOOM)]
    if gamma is not None:
        wanted.append(('gamma', gamma))
    if gain is not None:
        wanted.append(('gain', gain))

    rng = v4l2_ranges(device)
    for ctrl, val in wanted:
        if ctrl in rng:
            lo, hi = rng[ctrl]
            if not lo <= val <= hi:
                print(f'  note: {ctrl}={val} outside this camera\'s {lo}..{hi}, '
                      f'using {min(max(val, lo), hi)}')
                val = min(max(val, lo), hi)
        r = subprocess.run(['v4l2-ctl', '-d', device, '-c', f'{ctrl}={val}'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  warning: could not set {ctrl}={val}: {r.stderr.strip()}')

    got = read_scene_controls(device)
    print(f'  scene locked: {got}')
    if 'focus_automatic_continuous: 0' not in got:
        print('  WARNING: autofocus did NOT turn off -- poses will be wrong with '
              'no other symptom.')
    return got


# The controls session.json records, in the order they are reported. verify.py
# does substring matching on the resulting string, so the shape must not drift.
SCENE_REPORTED = ('auto_exposure', 'exposure_time_absolute', 'gain', 'gamma',
                  'white_balance_automatic', 'white_balance_temperature',
                  'focus_automatic_continuous', 'focus_absolute', 'zoom_absolute')


def read_scene_controls(device):
    """-> the one-line 'name: value, name: value' string session.json stores."""
    r = subprocess.run(['v4l2-ctl', '-d', device, '--get-ctrl',
                        ','.join(SCENE_REPORTED)], capture_output=True, text=True)
    return r.stdout.strip().replace('\n', ', ')


def v4l2_get_values(device, names):
    """-> {name: int} for the named controls, skipping any the camera lacks."""
    r = subprocess.run(['v4l2-ctl', '-d', device, '--get-ctrl', ','.join(names)],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        try:
            out[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return out


class SceneRecorder(GatedRecorder):
    """Demux MJPEG packets from V4L2 and re-mux them unchanged."""

    def __init__(self, device, width, height, fps, qsize=256):
        super().__init__(qsize)
        self.device, self.size, self.fps = device, (width, height), fps
        self.t_ns, self.out_dir, self.path = [], None, None
        self.checkpoint = max(fps, 1)

    def _capture(self):
        try:
            inp = av.open(self.device, format='v4l2', options={
                'input_format': 'mjpeg',
                'video_size': f'{self.size[0]}x{self.size[1]}',
                'framerate': str(self.fps)})
        except Exception as e:
            self.err = f'scene open failed: {e}'
            log.exception('scene camera failed to open')
            self.q.put(None)
            return
        ist = inp.streams.video[0]
        self.q.put(('STREAM', (ist, ist.time_base)))
        try:
            for packet in inp.demux(ist):
                if self.stop_evt.is_set():
                    break
                if packet.pts is None or packet.size == 0:
                    continue
                self.live += 1
                self.meter.tick()
                # RECORDING FIRST -- the display must never delay this
                try:
                    self.q.put_nowait(('F', (packet.pts, packet)))
                except queue.Full:
                    self.dropped += 1     # never block the driver's buffer queue
                # then, only if the display is idle and wants one, hand it the JPEG
                # bytes. Still compressed: decoding is the display thread's job.
                self.preview.offer(lambda p=packet: bytes(p))
        except Exception as e:
            self.err = f'scene capture failed: {e}'
            log.exception('scene capture thread died')
        finally:
            inp.close()
            self.q.put(None)

    def _write(self):
        ist = tb = out = ost = None
        pending = None                 # (out_dir, path, container, stream)
        pts0 = None
        while True:
            item = self.q.get()
            if item is None:
                break
            kind, payload = item
            if kind == 'STREAM':
                ist, tb = payload
                continue
            if kind == 'PREPARE':
                if ist is None or pending is not None:
                    continue
                p = os.path.join(payload, 'scene.mkv')
                c = av.open(p, 'w')
                pending = (payload, p, c, c.add_stream(template=ist))  # no re-encode
                continue
            if kind == 'ARM':
                if pending is None or pending[0] != payload:
                    if pending is not None:
                        discard_pending(pending)       # stale, e.g. episode renumbered
                        pending = None
                    p = os.path.join(payload, 'scene.mkv')
                    c = av.open(p, 'w')
                    pending = (payload, p, c, c.add_stream(template=ist))
                self.out_dir, self.path, out, ost = pending
                pending = None
                self.t_ns, pts0, self.written = [], None, 0
                continue
            if kind == 'DISARM':
                if out is not None:
                    self.save()
                    try:
                        out.close()
                    except Exception:
                        pass
                out = ost = None
                continue
            if out is None:
                continue                                # idle: discard the frame
            pts, packet = payload
            # PTS here are absolute CLOCK_MONOTONIC (~478000 s). Muxing those verbatim
            # is what produced unreadable MKVs, so rebase for the container and keep
            # the absolute value in the sidecar instead.
            if pts0 is None:
                pts0 = pts
            packet.pts = pts - pts0
            packet.dts = packet.pts
            packet.stream = ost
            try:
                out.mux(packet)
            except Exception as e:
                self.err = self.err or f'scene write failed: {e}'
                log.exception('scene write failed')
                continue
            self.t_ns.append(int(pts * tb * 1_000_000_000))
            self.written += 1
            # MKV survives a kill -9, but a sidecar written only at episode end does
            # not -- and video without timestamps cannot be synced, so the crash
            # resilience would be worthless. Checkpoint about once a second.
            if len(self.t_ns) % self.checkpoint == 0:
                self.save()
        discard_pending(pending)       # never leave a stub for an unrecorded episode

    def save(self):
        if self.out_dir is None:
            return 0, 0.0
        t = np.array(self.t_ns, dtype=np.int64)
        np.savez(os.path.join(self.out_dir, 'scene_ts.npz'), t_ns=t)
        return len(t), frame_rate(t)


# -------------------------------------------------------------- wrist camera
class WristRecorder(GatedRecorder):
    """RealSense colour (a WRIST_MODELS entry) via librealsense, encoded with NVENC.

    Exposure is in us everywhere outside this class -- the UI, session.json -- and
    converted to the model's own unit only when written to the camera.

    Timestamps come from the device, not from arrival in Python: the device stamp
    jitters by ~0.02 ms where arrival jitters by ~0.09 ms and lags by ~8 ms. That needs
    global_time_enabled, which maps the camera's internal clock onto the host epoch and
    keeps re-fitting it, so the camera's clock drift is tracked instead of accumulating.
    """

    def __init__(self, model, mono2epoch, encoder='h264_nvenc', exposure_us=None,
                 gain=None, wb_temp=4600, auto_wb=True, auto_exposure=None,
                 imu=None, queue_size=32, qsize=256):
        super().__init__(qsize)
        m = WRIST_MODELS[model]
        self.model, self.spec, self.imu = model, m, imu
        self.size, self.fps, self.mono2epoch = m['size'], m['fps'], mono2epoch
        self.unit = m['exposure_unit_us']
        self.encoder = encoder
        self.exposure_us = m['exposure_us'] if exposure_us is None else exposure_us
        self.gain = m['gain'] if gain is None else gain
        self.wb_temp, self.queue_size = wb_temp, queue_size
        self.auto_wb = bool(auto_wb)
        self.ae_offered = m['auto_exposure'] is not None
        self.auto_exp = self.ae_offered and bool(
            m['auto_exposure'] if auto_exposure is None else auto_exposure)
        self.rows, self.out_dir, self.path = [], None, None
        self.domains = set()
        self.settings = {}          # filled in once the pipeline is up
        self.checkpoint = max(self.fps, 1)
        # Live control surface, all populated in _configure on the started
        # pipeline. `wanted` is a shadow of every value we have written, and it
        # is what controls() reports -- see the comment there.
        self.sensor = None
        self.ranges = {}
        self.wanted = {}
        self._opt_lock = threading.Lock()

    def _configure(self, prof):
        """Write every option on the STARTED pipeline's sensor.

        Never on a bare device handle from query_devices(). Doing that opened the
        device twice -- once to configure, again for the pipeline -- and wedged the
        D405 three times: it stopped responding, query_devices() then returned 0 while
        lsusb still showed it, and the process blocked in an uninterruptible kernel USB
        wait (state D) that no signal could clear, not even SIGKILL. Only a physical
        replug released it. Configuring through the running pipeline opens the device
        exactly once and has been exercised repeatedly without trouble.
        """
        dev = prof.get_device()
        # the stereo module on the D405, a separate RGB camera on the D455 --
        # where writes to sensors[0] land on the depth module, silently
        s = dev.first_color_sensor()
        out = {'name': dev.get_info(rs.camera_info.name),
               'serial': dev.get_info(rs.camera_info.serial_number),
               'firmware': dev.get_info(rs.camera_info.firmware_version)}

        def put(opt, val):
            if not s.supports(opt):
                return None
            s.set_option(opt, val)
            return s.get_option(opt)

        # global time maps the camera's clock onto the host epoch and keeps re-fitting
        # it, so its drift is tracked rather than accumulating. On by default, but it is
        # per-sensor and the stamps are only comparable to the scene camera's if it is
        # actually on.
        put(rs.option.global_time_enabled, 1)
        # Every option is written: they persist on the device across sessions.
        # Manual values go in before auto is switched on, so turning auto off
        # later has something to return to.
        put(rs.option.enable_auto_exposure, 0)
        put(rs.option.enable_auto_white_balance, 0)
        put(rs.option.auto_exposure_priority, 0)    # auto exposure may never lower fps
        put(rs.option.frames_queue_size, self.queue_size)
        self.exposure_us = round(self.exposure_us / self.unit) * self.unit
        out['exposure_us'] = put(rs.option.exposure, self.exposure_us / self.unit) * self.unit
        r = s.get_option_range(rs.option.gain)       # 248 is past the D455's 128
        self.gain = float(np.clip(self.gain, r.min, r.max))
        out['gain'] = put(rs.option.gain, self.gain)
        out['white_balance'] = put(rs.option.white_balance, self.wb_temp)
        put(rs.option.enable_auto_exposure, 1 if self.auto_exp else 0)
        put(rs.option.enable_auto_white_balance, 1 if self.auto_wb else 0)
        out['global_time_enabled'] = s.get_option(rs.option.global_time_enabled)
        out['auto_exposure'] = s.get_option(rs.option.enable_auto_exposure)
        out['auto_white_balance'] = s.get_option(rs.option.enable_auto_white_balance)

        # Cache the ranges ONCE, for the tunable options only. Walking the full
        # rs.option list on a live stream stalls this camera -- reading
        # hdr_enabled / sequence_* left it enumerating but delivering no frames,
        # recoverable only with dev.hardware_reset(). A fixed tuple read once
        # makes that impossible by construction rather than by discipline.
        self.sensor = s
        for name in self._tunable():
            opt = getattr(rs.option, name, None)
            if opt is None or not s.supports(opt):
                continue
            r = s.get_option_range(opt)
            k = self.unit if name == 'exposure' else 1    # held in us
            self.ranges[name] = {'min': r.min * k, 'max': r.max * k,
                                 'step': r.step * k, 'default': r.default * k}
            self.wanted[name] = s.get_option(opt) * k
        # what we just wrote, so the shadow starts truthful
        self.wanted['exposure'] = float(self.exposure_us)
        self.wanted['gain'] = float(self.gain)
        self.wanted['white_balance'] = float(self.wb_temp)
        self.wanted['enable_auto_white_balance'] = 1.0 if self.auto_wb else 0.0
        if self.ae_offered:
            self.wanted['enable_auto_exposure'] = 1.0 if self.auto_exp else 0.0
        out['model'] = self.model
        return out

    def _tunable(self):
        return [n for n in WRIST_TUNABLE if n != 'enable_auto_exposure' or self.ae_offered]

    # ----------------------------------------------------------- live controls
    def controls(self):
        """Current values and ranges, shaped exactly like PreviewSession.controls()
        so one UI component serves both cameras.

        `value` comes from the WRITE SHADOW, not from get_option. Measured on this
        D405: with auto white balance running, white_balance keeps reporting the
        last manually written number, and it does not update when auto is switched
        back off. The same is true of exposure and gain under auto exposure. A
        slider bound to that readback would sit on a stale number and jump when
        the user touched it. The shadow is authoritative because _configure writes
        every one of these at start-up, so it IS the device state -- except where
        the camera is driving the control itself, which is what `stale` marks.
        """
        if self.sensor is None:
            return {}
        auto_wb = bool(self.wanted.get('enable_auto_white_balance', 0))
        auto_exp = bool(self.wanted.get('enable_auto_exposure', 0))
        out = {}
        for name in self._tunable():
            if name not in self.ranges:
                continue
            r = self.ranges[name]
            e = {'type': 'bool' if r['max'] == 1 and r['step'] == 1
                         and name.startswith('enable_') else 'int',
                 'min': int(r['min']), 'max': int(r['max']),
                 'step': int(r['step']) or 1, 'default': int(r['default']),
                 'value': int(self.wanted.get(name, r['default'])),
                 'inactive': False, 'adjustable': True}
            if name == 'exposure':
                e['useful_max'] = self.spec['exposure_max_us']
            if name in ('exposure', 'gain') and auto_exp:
                # driven by auto exposure, which the camera does not report
                e['adjustable'] = False
                e['stale'] = True
            elif name == 'white_balance':
                # driven by the camera, and the number we hold is not what it is
                # actually using -- say so rather than showing a plausible lie
                e['adjustable'] = not auto_wb
                e['stale'] = auto_wb
            out[name] = e
        out['_auto_exposure'] = auto_exp
        out['_auto_wb'] = auto_wb
        out['_model'] = self.model      # picks the UI's wrist schema
        return out

    def set(self, ctrl, value):
        """-> (ok, detail). Clamps, snaps to step, writes, updates the shadow."""
        if self.sensor is None:
            return False, 'wrist camera is not up yet'
        if ctrl not in self._tunable() or ctrl not in self.ranges:
            return False, f'{ctrl} is not tunable on the wrist camera'
        # writing exposure OR gain under auto exposure silently switches it off
        # (measured on the D455: AE 1 -> 0 on a gain write), so refuse both
        if ctrl in ('exposure', 'gain') and self.wanted.get('enable_auto_exposure'):
            return False, 'auto exposure is on; switch it off to set this'
        r = self.ranges[ctrl]
        hi = r['max']
        if ctrl == 'exposure':
            hi = min(hi, self.spec['exposure_max_us'])
        v = float(np.clip(float(value), r['min'], hi))
        # Snap to the step before writing. White balance steps by 10; an unsnapped
        # write is rounded by the driver and the readback then disagrees with the
        # slider, which looks like the control being ignored.
        step = r['step'] or 1
        v = r['min'] + round((v - r['min']) / step) * step
        v = float(np.clip(v, r['min'], hi))
        try:
            with self._opt_lock:
                self.sensor.set_option(getattr(rs.option, ctrl),
                                       v / self.unit if ctrl == 'exposure' else v)
                self.wanted[ctrl] = v
                if ctrl == 'enable_auto_exposure' and not v:
                    # back to manual: re-assert what the sliders show
                    self.sensor.set_option(rs.option.exposure,
                                           self.wanted['exposure'] / self.unit)
                    self.sensor.set_option(rs.option.gain, self.wanted['gain'])
                if ctrl == 'enable_auto_white_balance' and not v:
                    # Coming off auto, the device keeps reporting -- and using --
                    # something we cannot read. Re-assert the shadow so "off"
                    # lands on the number the slider is showing.
                    self.sensor.set_option(rs.option.white_balance,
                                           self.wanted['white_balance'])
        except Exception as e:                       # noqa: BLE001
            return False, f'{type(e).__name__}: {e}'
        self._sync_settings()
        return True, str(int(v))

    def _sync_settings(self):
        """Keep `settings` current so session.json records what was TUNED.

        It used to be a snapshot from _configure, which is a lie the moment an
        operator moves a slider -- and session.json is the only record of what
        the footage was actually shot at.
        """
        auto_wb = bool(self.wanted.get('enable_auto_white_balance', 0))
        auto_exp = bool(self.wanted.get('enable_auto_exposure', 0))
        self.settings.update({
            # null under auto exposure: the camera does not report what it uses
            'exposure_us': None if auto_exp else self.wanted.get('exposure'),
            'gain': None if auto_exp else self.wanted.get('gain'),
            'auto_white_balance': 1.0 if auto_wb else 0.0,
            'auto_exposure': 1.0 if auto_exp else 0.0,
            # Under auto WB the device will not tell us what it settled on, so
            # null is the honest answer -- see controls().
            'white_balance': None if auto_wb else self.wanted.get('white_balance'),
        })
        if auto_wb:
            self.settings['white_balance_note'] = (
                'auto white balance was on; the camera does not report the value '
                'it chose, so none is recorded')

    def _capture(self):
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, self.size[0], self.size[1],
                          rs.format.bgr8, self.fps)
        try:
            prof = pipe.start(cfg)
            self.settings = self._configure(prof)
            self._sync_settings()
        except Exception as e:
            self.err = f'wrist start failed: {e}'
            log.exception('wrist camera failed to start')
            self.q.put(None)
            return
        if self.imu is not None:
            self.imu.attach(prof, self.mono2epoch)
        try:
            while not self.stop_evt.is_set():
                try:
                    frames = pipe.wait_for_frames(2000)
                except Exception:
                    continue
                f = frames.get_color_frame()
                if not f:
                    continue
                arrival = mono()
                self.domains.add(str(f.get_frame_timestamp_domain()))
                t_dev_s = f.get_timestamp() / 1000.0            # epoch seconds
                row = (int((t_dev_s - self.mono2epoch) * 1e9),  # -> monotonic ns
                       int(t_dev_s * 1e9), int(arrival * 1e9),
                       int(f.get_frame_number()))
                img = np.asanyarray(f.get_data())
                self.live += 1
                self.meter.tick(arrival)
                # RECORDING FIRST -- the copy is needed because librealsense reuses
                # the buffer once this frame goes out of scope
                try:
                    self.q.put_nowait(('F', (row, img.copy())))
                except queue.Full:
                    self.dropped += 1
                self.preview.offer(lambda i=img: i.copy())
        except Exception as e:
            self.err = f'wrist capture failed: {e}'
            log.exception('wrist capture thread died')
        finally:
            if self.imu is not None:
                self.imu.detach()          # the motion sensor before the pipeline
            try:
                pipe.stop()
            except Exception:
                pass
            self.q.put(None)

    def _open(self, out_dir):
        p = os.path.join(out_dir, 'wrist.mkv')
        c = av.open(p, 'w')
        st = c.add_stream(self.encoder, rate=self.fps)
        st.width, st.height, st.pix_fmt = self.size[0], self.size[1], 'yuv420p'
        st.time_base = Fraction(1, 1000)
        # bf=0 disables B-frames: with them the muxed order is DTS order and no longer
        # matches the order rows were appended, which would silently break the
        # frame-index-to-timestamp mapping the sidecar depends on.
        st.options = {'preset': 'p4', 'tune': 'ull', 'rc': 'vbr', 'cq': '23', 'bf': '0'}
        # Open NVENC now. PyAV otherwise opens it on the first encode, and that
        # ~100 ms GIL hold dropped 6-16 wrist frames at the start of every episode.
        st.codec_context.open()
        return (out_dir, p, c, st)

    def _write(self):
        out = st = None
        pending = None
        t0 = None
        while True:
            item = self.q.get()
            if item is None:
                break
            kind, payload = item
            if kind == 'PREPARE':
                if pending is None:
                    pending = self._open(payload)   # NVENC session built while idle
                continue
            if kind == 'ARM':
                if pending is None or pending[0] != payload:
                    discard_pending(pending)
                    pending = self._open(payload)
                self.out_dir, self.path, out, st = pending
                pending = None
                self.rows, t0, self.written = [], None, 0
                continue
            if kind == 'DISARM':
                if out is not None:
                    try:
                        for p in st.encode(None):       # flush the encoder
                            out.mux(p)
                    except Exception:
                        pass
                    self.save()
                    try:
                        out.close()
                    except Exception:
                        pass
                out = st = None
                continue
            if out is None:
                continue
            row, img = payload
            if t0 is None:
                t0 = row[0]
            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
            # container PTS is for playback only; the sidecar is the real clock
            frame.pts = int((row[0] - t0) / 1_000_000)
            frame.time_base = st.time_base
            try:
                for p in st.encode(frame):
                    out.mux(p)
            except Exception as e:
                self.err = self.err or f'wrist write failed: {e}'
                log.exception('wrist write failed')
                continue
            self.rows.append(row)
            self.written += 1
            if len(self.rows) % self.checkpoint == 0:
                self.save()
        discard_pending(pending)

    def save(self):
        if self.out_dir is None:
            return 0, 0.0
        a = np.array(self.rows, dtype=np.int64).reshape(-1, 4)
        np.savez(os.path.join(self.out_dir, 'wrist_ts.npz'),
                 t_ns=a[:, 0], t_device_ns=a[:, 1],
                 t_arrival_ns=a[:, 2], seq=a[:, 3])
        return len(a), frame_rate(a[:, 0])


# ------------------------------------------------------------------- gripper
class GripperRecorder:
    """The gripper sensor as one more stream: each line stamped on arrival with
    CLOCK_MONOTONIC, the clock both cameras use, and kept only while armed.

    Not a GatedRecorder: 40 lines a second need no writer thread. Reads block in
    read(1), which releases the GIL; pyserial's readline() would instead take
    it once per byte, ~500 times a second, next to the wrist's 11.1 ms deadline.
    """

    def __init__(self, device=None):
        self.device = device or GRIPPER_DEVICE
        # opened here so a missing or busy device fails the session up front
        self.port = serial.Serial(self.device, GRIPPER_BAUD, timeout=0.2, exclusive=True)
        self.err, self.live, self.written, self.bad, self.late = None, 0, 0, 0, 0
        self.late_at_arm = 0
        self.armed, self.out_dir, self.rows = False, None, []
        self.latest = None                        # (t_ns, raw, force_raw)
        self.recent = deque(maxlen=int(GRIPPER_TRACE_S * 2 * 50))
        self.meter = RateMeter()
        self.checkpoint = 40                      # ~1 s, like the camera sidecars
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.port.close()

    def _read(self):
        buf = b''
        try:
            self.port.reset_input_buffer()
            while not self.stop_evt.is_set():
                first = self.port.read(1)
                if not first:
                    continue
                t = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                buf += first + self.port.read(self.port.in_waiting)
                *lines, buf = buf.split(b'\n')
                rows = []
                for ln in lines:
                    try:
                        raw, force = (int(x) for x in ln.split(b'_'))
                    except ValueError:
                        self.bad += 1          # the partial first line, or noise
                        continue
                    rows.append((t, raw, force))
                if not rows:
                    continue
                # several lines in one read queued while this thread was held up;
                # only the newest one's stamp is true, so the rest are dropped
                self.late += len(rows) - 1
                row = rows[-1]
                self.latest = row
                self.recent.append(row)
                self.live += 1
                self.meter.tick(t / 1e9)
                with self.lock:
                    if self.armed:
                        self.rows.append(row)
                        self.written += 1
                        if self.written % self.checkpoint == 0:
                            self._write()
        except (serial.SerialException, OSError) as e:
            self.err = f'{self.device}: {e}'
            log.error('gripper: %s', self.err)

    def prepare(self, out_dir):
        pass

    def arm(self, out_dir):
        with self.lock:
            self.out_dir, self.rows, self.written = out_dir, [], 0
            self.late_at_arm = self.late
            self.armed = True

    def disarm(self):
        with self.lock:
            self.armed = False

    def _write(self):
        a = np.array(self.rows, dtype=np.int64).reshape(-1, 3)
        np.savez(os.path.join(self.out_dir, 'gripper_ts.npz'),
                 t_ns=a[:, 0], raw=a[:, 1].astype(np.int16),
                 force_raw=a[:, 2].astype(np.int32),
                 raw_open=GRIPPER_RAW_OPEN, max_width_m=GRIPPER_MAX_WIDTH)
        return a

    def save(self):
        with self.lock:
            if self.out_dir is None:
                return 0, 0.0
            a = self._write()
        return len(a), frame_rate(a[:, 0])

    @staticmethod
    def width(raw):
        return min(max(raw / GRIPPER_RAW_OPEN, 0.0), 1.0) * GRIPPER_MAX_WIDTH

    def state(self, trace_hz):
        """Latest reading plus the last GRIPPER_TRACE_S of width (mm), decimated
        to trace_hz so the UI refresh costs what a camera preview does."""
        last, now = self.latest, time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        trace, due = [], now - GRIPPER_TRACE_S * 1e9
        for t, raw, _ in list(self.recent):
            if t >= due:
                trace.append(round(self.width(raw) * 1000, 1))
                due = t + 1e9 / trace_hz
        return {
            'width_m': None if last is None else round(self.width(last[1]), 4),
            'raw': None if last is None else last[1],
            'force_raw': None if last is None else last[2],
            'hz': round(self.meter.fps(), 1),
            'age_s': None if last is None else round((now - last[0]) / 1e9, 2),
            'stale_s': GRIPPER_STALE, 'max_width_m': GRIPPER_MAX_WIDTH,
            'written': self.written, 'late': self.late, 'armed': self.armed, 'error': self.err,
            'trace_hz': trace_hz, 'trace': trace,
        }


# ----------------------------------------------------------------------- imu
class ImuRecorder:
    """The D455's gyro and accel, stamped on the wrist frames' clock and kept
    only while armed.

    On the motion sensor's own callback, not in the colour pipeline:
    wait_for_frames pairs one IMU sample with each colour frame and drops the
    rest (measured: 85 of 200 gyro samples a second). Opened on the started
    pipeline's device, so the camera is still opened once -- see _configure.
    """

    STREAMS = ('gyro', 'accel')

    def __init__(self):
        self.err, self.live, self.written, self.armed = None, 0, 0, False
        self.out_dir, self.sensor, self.extr = None, None, None
        self.mono2epoch = 0.0
        self.domains = set()
        n = int(GRIPPER_TRACE_S * IMU_HZ * 1.5)
        self.recent = {k: deque(maxlen=n) for k in self.STREAMS}
        self.meter = {k: RateMeter(n=IMU_HZ) for k in self.STREAMS}
        self.rows = {k: [] for k in self.STREAMS}
        self.lock = threading.Lock()

    def attach(self, prof, mono2epoch):
        """Start gyro + accel at IMU_HZ on the wrist pipeline's device."""
        self.mono2epoch = mono2epoch
        try:
            s = next((x for x in prof.get_device().query_sensors() if x.is_motion_sensor()),
                     None)
            if s is None:
                raise RuntimeError('the wrist camera has no motion sensor')
            want = {}
            for p in s.get_stream_profiles():
                if p.fps() == IMU_HZ and p.stream_type() in (rs.stream.gyro, rs.stream.accel):
                    want.setdefault(p.stream_type(), p)
            if len(want) < 2:
                raise RuntimeError(f'no {IMU_HZ} Hz gyro and accel profiles')
            e = prof.get_stream(rs.stream.color).get_extrinsics_to(want[rs.stream.gyro])
            self.extr = (np.array(e.rotation, np.float64).reshape(3, 3),
                         np.array(e.translation, np.float64))
            s.open(list(want.values()))
            s.start(self._on_frame)
            self.sensor = s
        except Exception as e:                       # noqa: BLE001
            self.err = f'imu start failed: {e}'
            log.exception('imu failed to start')

    def detach(self):
        s, self.sensor = self.sensor, None
        if s is None:
            return
        for fn in (s.stop, s.close):
            try:
                fn()
            except Exception:                        # noqa: BLE001
                pass

    def _on_frame(self, f):
        """librealsense's thread: stamp, keep, nothing else."""
        try:
            k = 'gyro' if f.get_profile().stream_type() == rs.stream.gyro else 'accel'
            v = f.as_motion_frame().get_motion_data()
            t = int((f.get_timestamp() / 1000.0 - self.mono2epoch) * 1e9)
            row = (t, v.x, v.y, v.z)
            self.domains.add(f.get_frame_timestamp_domain())
            self.recent[k].append(row)
            self.meter[k].tick(t / 1e9)
            self.live += 1
            if self.armed:
                with self.lock:
                    if self.armed:
                        self.rows[k].append(row)
                        self.written += 1
        except Exception as e:                       # noqa: BLE001
            self.err = self.err or f'imu: {e}'

    def arm(self, out_dir):
        with self.lock:
            self.out_dir, self.written = out_dir, 0
            self.rows = {k: [] for k in self.STREAMS}
            self.armed = True

    def disarm(self):
        with self.lock:
            self.armed = False

    def save(self):
        """Written once, after disarm: a checkpoint would rebuild a growing array
        under the GIL, next to the wrist's 11.1 ms deadline."""
        with self.lock:
            if self.out_dir is None:
                return {}
            rows = {k: list(v) for k, v in self.rows.items()}
        arrays, out = {}, {}
        for k, r in rows.items():
            t = np.array([x[0] for x in r], np.int64)
            arrays[f'{k}_t_ns'] = t
            arrays[k] = np.array([x[1:] for x in r], np.float32).reshape(-1, 3)
            out[k], out[f'{k}_hz'] = len(t), frame_rate(t)
        R, tr = self.extr if self.extr is not None else (np.eye(3), np.zeros(3))
        np.savez(os.path.join(self.out_dir, 'imu_ts.npz'), **arrays, rate_hz=IMU_HZ,
                 color_to_imu_R=R, color_to_imu_t=tr)
        return out

    def state(self, trace_hz):
        """Latest gyro (deg/s) and accel (m/s^2), plus the last GRIPPER_TRACE_S of
        rotation speed: the peak per 1/trace_hz bin, so a quick flick still shows."""
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        g = np.array(list(self.recent['gyro']), np.float64).reshape(-1, 4)
        acc = self.recent['accel'][-1] if self.recent['accel'] else None
        trace = []
        w = g[g[:, 0] >= now - GRIPPER_TRACE_S * 1e9]
        if len(w):
            speed = np.degrees(np.linalg.norm(w[:, 1:], axis=1))
            b = ((w[:, 0] - w[0, 0]) * trace_hz // 1e9).astype(np.int64)
            trace = np.round(np.maximum.reduceat(
                speed, np.flatnonzero(np.diff(b, prepend=-1))), 1).tolist()
        last = g[-1] if len(g) else None
        return {
            'speed_dps': None if last is None else round(float(np.degrees(np.linalg.norm(last[1:]))), 1),
            'gyro_dps': None if last is None else np.round(np.degrees(last[1:]), 1).tolist(),
            'accel': None if acc is None else [round(v, 2) for v in acc[1:]],
            'gyro_hz': round(self.meter['gyro'].fps(), 1),
            'accel_hz': round(self.meter['accel'].fps(), 1),
            'age_s': None if last is None else round((now - last[0]) / 1e9, 2),
            'stale_s': IMU_STALE, 'written': self.written, 'armed': self.armed, 'error': self.err,
            'trace_hz': trace_hz, 'trace': trace,
        }

    def meta(self):
        return {
            'present': True, 'rate_hz': IMU_HZ, 'units': {'gyro': 'rad/s', 'accel': 'm/s^2'},
            'axes': 'D455 IMU frame (librealsense)',
            'intrinsics': 'factory identity: uncalibrated',
            'color_to_imu': None if self.extr is None else {
                'R': self.extr[0].round(6).tolist(), 't_m': self.extr[1].round(6).tolist()},
            'timestamp_domains': sorted(str(d) for d in self.domains),
        }


# ------------------------------------------------------------------- session
class RecordError(RuntimeError):
    """A refusal or a hardware failure. Raised rather than sys.exit so a server
    can turn it into a 400 and the CLI into a SystemExit."""


class RecordSession:
    """A recording session: both cameras streaming, episodes gated on top.

    Everything main() used to own in local variables and closures. The CLI drives
    it from a cv2/stdin loop; the web server drives it from request handlers. The
    two must not each grow their own copy of the episode state machine, because
    the parts that matter here are the ones that look like padding -- the LEAD
    ordering, the 0.4 s settle, the exact ep-dir string -- and a second
    implementation is where they quietly go missing.

    Every method that arms or disarms BLOCKS for a few hundred ms. Call them off
    the event loop.
    """

    def __init__(self, out_root=None, scene=True, wrist=True,
                 scene_exposure=800, scene_wb=4600, scene_gamma=128,
                 scene_gain=100, wrist_exposure_us=None, wrist_gain=None,
                 wrist_wb=4600, wrist_auto_wb=True, wrist_auto_exposure=None):
        if not scene and not wrist:
            raise RecordError('nothing to record (both cameras disabled)')
        if wrist and rs is None:
            raise RecordError('pyrealsense2 is not installed')
        if serial is None:
            raise RecordError('pyserial is not installed')
        # opened first: a missing gripper fails in milliseconds, not after the
        # cameras have spun up
        try:
            self.gripper = GripperRecorder()
        except (serial.SerialException, OSError) as e:
            raise RecordError(f'gripper sensor {GRIPPER_DEVICE}: {e}') from None

        self.gripper.start()
        self.recs, self.imu = {}, None
        try:
            self.stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.dir = os.path.join(os.path.abspath(out_root or os.path.join(DATA, 'capture')),
                                    self.stamp)
            os.makedirs(self.dir, exist_ok=True)
            self.mono2epoch = mono_to_epoch_offset()

            self.scene_ctrls = ''
            self.scene_args = {'exposure': scene_exposure, 'wb': scene_wb,
                               'gamma': scene_gamma, 'gain': scene_gain}
            if scene:
                self.scene_ctrls = set_scene_controls(
                    SCENE_DEVICE, scene_exposure, scene_wb, scene_gamma,
                    None if scene_gain < 0 else scene_gain)

            # insertion order is load-bearing: it drives the order cameras appear in
            # the saved-episode line and in session.json
            if scene:
                self.recs['scene'] = SceneRecorder(SCENE_DEVICE, *SCENE_SIZE, SCENE_FPS)
            if wrist:
                # configured inside the recorder, on its own started pipeline -- never
                # through a second handle from query_devices(). See _configure().
                model = wrist_model()
                if WRIST_MODELS[model]['imu']:
                    self.imu = ImuRecorder()
                self.recs['wrist'] = WristRecorder(model,
                                                   self.mono2epoch, ENCODER,
                                                   exposure_us=wrist_exposure_us,
                                                   gain=wrist_gain, wb_temp=wrist_wb,
                                                   auto_wb=wrist_auto_wb,
                                                   auto_exposure=wrist_auto_exposure,
                                                   imu=self.imu)
            for r in self.recs.values():
                r.start()
        except BaseException:
            self.close_cameras()     # releases the gripper port too
            raise

        self.episodes, self.ep_i = [], 0
        self.recording, self.t_ep = False, None
        self.transition = None            # 'arming' | 'stopping' -- the UI's WAITING
        self._lock = threading.RLock()
        self._prepared = set()
        self._closed = False

        try:
            self.wait_ready()
        except Exception:
            self.close_cameras()
            raise

        # Warm-up is timed from the moment both cameras are actually delivering,
        # not from construction: the wrist pipeline alone can take a second to
        # come up, and starting the clock before that would let the gate expire
        # while the cameras were still settling.
        self.t_warm = mono()
        self._house = threading.Thread(target=self._housekeep, daemon=True)
        self._house.start()

    # ------------------------------------------------------------- bring-up
    def wait_ready(self, timeout=10.0):
        """Block until every camera has delivered a frame, or raise.

        main() used to wait only for the wrist's settings dict, so a scene node
        held by another process produced a running session with a dead stream and
        no complaint until the metadata was written. A web UI cannot show that --
        it looks like a working page with a blank image.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            for k, r in self._streams():
                if r.err:
                    raise RecordError(f'{k}: {r.err}')
            if all(r.live > 0 for _, r in self._streams()) and (
                    'wrist' not in self.recs or self.recs['wrist'].settings):
                return
            time.sleep(0.05)
        dead = [k for k, r in self.recs.items() if r.live == 0]
        if self.gripper.live == 0:
            dead.append(f'gripper (no lines from {GRIPPER_DEVICE}; is the sketch running?)')
        if self.imu is not None and self.imu.live == 0:
            dead.append('imu (no samples from the wrist camera)')
        raise RecordError(f'camera(s) delivered no frames within {timeout:.0f}s: '
                          + ', '.join(dead or ['(settings never arrived)']))

    def _housekeep(self):
        """Open ep000's containers the moment warm-up lifts.

        Creating a container and an NVENC session costs tens of milliseconds of
        held GIL, which drops 6-14 frames if it happens once recording is under
        way. Doing it while idle is the whole point of prepare(); this thread is
        just what replaces the poll that used to live in main()'s event loop.
        """
        time.sleep(max(0.0, WARMUP - (mono() - self.t_warm)))
        with self._lock:
            if not self._closed and not self.recording:
                self._prep_ep(0)

    # ---------------------------------------------------------------- state
    def _sensors(self):
        """The streams armed around the cameras: the gripper, and the IMU if any."""
        return [('gripper', self.gripper), *([('imu', self.imu)] if self.imu else [])]

    def _streams(self):
        return [*self.recs.items(), *self._sensors()]

    def ep_dir(self, i, cam=None):
        """The ONE place an episode path is composed.

        SceneRecorder/WristRecorder match a prepared container by exact string
        equality against this path, so two call sites that differ by a trailing
        slash silently discard the prepared container and reopen it inline --
        paying back the frames prepare() exists to save, with no error anywhere.
        """
        d = os.path.join(self.dir, f'ep{i:03d}')
        return d if cam is None else os.path.join(d, cam)

    @property
    def warmup_left(self):
        return max(0.0, WARMUP - (mono() - self.t_warm))

    @property
    def ready(self):
        return self.warmup_left <= 0.0

    def errors(self):
        return {k: r.err for k, r in self.recs.items()}

    def state(self):
        """The SSE payload. elapsed_s is computed HERE, not in the browser:
        CLOCK_MONOTONIC is a ~475000 s number that means nothing to JS."""
        trace_hz = GRIPPER_TRACE_REC_HZ if self.recording else GRIPPER_TRACE_HZ
        return {
            'session': self.stamp, 'session_dir': self.dir,
            'cams': list(self.recs),
            'ready': self.ready, 'warmup_left': round(self.warmup_left, 1),
            'recording': self.recording, 'transition': self.transition,
            'episode_index': self.ep_i, 'n_episodes': len(self.episodes),
            'elapsed_s': round(mono() - self.t_ep, 2) if self.recording else 0.0,
            'episodes': self.episodes,
            'cameras': {k: {'fps': round(r.meter.fps(), 1), 'written': r.written,
                            'live': r.live, 'queue_drops': r.dropped,
                            'armed': r.armed, 'error': r.err}
                        for k, r in self.recs.items()},
            # trace rate follows the wrist preview: 10 Hz idle, 4 Hz recording
            'gripper': self.gripper.state(trace_hz),
            'imu': self.imu.state(trace_hz) if self.imu else None,
        }

    # --------------------------------------------------------------- controls
    # Only the WRIST is served from here. Scene controls go through v4l2-ctl,
    # which needs no handle on the camera at all -- v4l2-ctl reads and writes the
    # node while SceneRecorder streams it via PyAV (measured: exposure 100 gives
    # mean luma 4.4 and exposure 800 gives 52.5, live, mid-demux). So the server
    # calls preview.scene_controls/scene_set directly and capture.py keeps no
    # dependency on the calibration package.
    def wrist_controls(self):
        if 'wrist' not in self.recs:
            raise RecordError('this session has no wrist camera')
        return self.recs['wrist'].controls()

    def set_wrist_control(self, ctrl, value):
        """Refused while an episode is armed.

        The server gates this too, via the broker, but the guard belongs here as
        well: changing the picture mid-episode is exactly the time-varying
        photometric noise the whole fixed-exposure design exists to avoid, and
        the session should not depend on a caller remembering to ask first.
        """
        if 'wrist' not in self.recs:
            raise RecordError('this session has no wrist camera')
        if self.recording:
            raise RecordError('an episode is recording; controls are frozen')
        return self.recs['wrist'].set(ctrl, value)

    # ------------------------------------------------------------- episodes
    def _prep_ep(self, i):
        if i in self._prepared:
            return
        for k, r in self.recs.items():
            os.makedirs(self.ep_dir(i, k), exist_ok=True)
            r.prepare(self.ep_dir(i, k))
        for k, _ in self._sensors():
            os.makedirs(self.ep_dir(i, k), exist_ok=True)
        self._prepared.add(i)

    def start_episode(self):
        """-> the episode index. Blocks for LEAD seconds."""
        with self._lock:
            if self._closed:
                raise RecordError('session is finished')
            if self.recording:
                return self.ep_i
            if not self.ready:
                raise RecordError(f'still warming up ({self.warmup_left:.1f}s left)')
            self.transition = 'arming'
            i = self.ep_i
            try:
                for k, _ in self._streams():
                    os.makedirs(self.ep_dir(i, k), exist_ok=True)
                # scene, gripper and imu lead so every wrist frame is bracketed by
                # poses, openings and imu samples. Shortening or inverting this
                # leaves the episode's edge frames unlabelled -- see LEAD.
                for k, r in self._sensors():
                    r.arm(self.ep_dir(i, k))
                if 'scene' in self.recs:
                    self.recs['scene'].arm(self.ep_dir(i, 'scene'))
                    time.sleep(LEAD)
                if 'wrist' in self.recs:
                    self.recs['wrist'].arm(self.ep_dir(i, 'wrist'))
                self.t_ep = mono()
                self.recording = True
            finally:
                self.transition = None
            return i

    def stop_episode(self):
        """-> the episode row. Blocks for LEAD + 0.4 s."""
        with self._lock:
            if not self.recording:
                return None
            self.transition = 'stopping'
            i = self.ep_i
            try:
                if 'wrist' in self.recs:
                    self.recs['wrist'].disarm()
                    time.sleep(LEAD)
                if 'scene' in self.recs:
                    self.recs['scene'].disarm()
                for _, r in self._sensors():
                    r.disarm()
                # Not padding: save() reads the row lists from THIS thread while
                # the writer thread may still be draining its queue toward DISARM.
                time.sleep(0.4)
                ep = {'index': i, 'dir': f'ep{i:03d}',
                      'duration_s': mono() - self.t_ep}
                for k, r in self.recs.items():
                    n, fps = r.save()
                    ep[k] = {'frames': n, 'fps': fps, 'queue_drops': r.dropped}
                n, hz = self.gripper.save()
                ep['gripper'] = {'samples': n, 'hz': hz, 'bad': self.gripper.bad,
                                 'late': self.gripper.late - self.gripper.late_at_arm}
                if self.imu:
                    ep['imu'] = self.imu.save()
                self.episodes.append(ep)
                self.recording = False
                self.ep_i = i + 1
                self._prep_ep(self.ep_i)   # ready the next one while idle
            finally:
                self.transition = None
            return ep

    # -------------------------------------------------------------- teardown
    def close_cameras(self):
        for r in [*self.recs.values(), self.gripper]:
            try:
                r.stop()
            except Exception:                        # noqa: BLE001
                pass

    def _prune_empty(self):
        """Drop the pre-opened next episode's tree if it was never recorded, so
        it is not mistaken for a real (and broken) episode."""
        for d in sorted(os.listdir(self.dir)):
            p = os.path.join(self.dir, d)
            if d.startswith('ep') and os.path.isdir(p) and not any(
                    fs for _, _, fs in os.walk(p)):
                # topdown=False yields children before parents, exactly the order
                # rmdir needs -- do not sort this, it would undo that
                for sub, _, _ in os.walk(p, topdown=False):
                    try:
                        os.rmdir(sub)
                    except OSError:
                        pass

    def meta(self, display=None):
        """The session.json contents.

        Scene controls are RE-READ rather than reused from construction: the
        whole point of the tuning stage is that they change afterwards, and this
        file is the only record of what the footage was actually shot at. Focus
        and zoom likewise come from the readback, not from the constants --
        verify.py compares exactly those two fields against the calibration, so
        sourcing them from what we intended would make a silently failed lock
        assert success.
        """
        scene_ctrls, focus, zoom = self.scene_ctrls, SCENE_FOCUS, SCENE_ZOOM
        if 'scene' in self.recs:
            scene_ctrls = read_scene_controls(SCENE_DEVICE) or self.scene_ctrls
            got = v4l2_get_values(SCENE_DEVICE, ('focus_absolute', 'zoom_absolute'))
            focus = got.get('focus_absolute', SCENE_FOCUS)
            zoom = got.get('zoom_absolute', SCENE_ZOOM)

        m = {'session': self.stamp, 'mono_to_epoch_offset_s': self.mono2epoch,
             'clock': 'CLOCK_MONOTONIC (t_ns in every sidecar)',
             'warmup_s': WARMUP, 'scene_lead_s': LEAD,
             'display': display if display is not None else {
                 'enabled': False, 'fps': DISPLAY_FPS,
                 'frames_skipped': {k: r.preview.skipped
                                    for k, r in self.recs.items()},
                 'note': 'preview samples the stream; skipping is by design '
                         'and does not affect recorded files'},
             'scene': {'device': SCENE_DEVICE, 'size': list(SCENE_SIZE),
                       'fps_requested': SCENE_FPS,
                       'exposure_units_100us': self.scene_args['exposure'],
                       'white_balance_k': self.scene_args['wb'],
                       'gamma': self.scene_args['gamma'],
                       'gain': self.scene_args['gain'],
                       # geometry: verify.py checks these against the calibration
                       'focus_absolute': focus, 'zoom_absolute': zoom,
                       'controls': scene_ctrls,
                       'codec': 'mjpeg (passthrough, not re-encoded)'},
             'wrist': dict(self.recs['wrist'].settings, size=list(self.recs['wrist'].size),
                           fps=self.recs['wrist'].fps, encoder=ENCODER,
                           timestamp_domains=sorted(self.recs['wrist'].domains))
             if 'wrist' in self.recs else {'encoder': ENCODER},
             'gripper': {'device': GRIPPER_DEVICE, 'raw_open': GRIPPER_RAW_OPEN,
                         'max_width_m': GRIPPER_MAX_WIDTH,
                         'format': '<opening raw>_<force raw> per line',
                         'bad_lines': self.gripper.bad, 'late_lines': self.gripper.late},
             'imu': self.imu.meta() if self.imu else {'present': False},
             'episodes': self.episodes}
        for k, r in self._streams():
            if r.err:
                m[k]['error'] = r.err
        return m

    def finish(self, display=None):
        """Stop everything, prune, write session.json. -> the meta dict."""
        with self._lock:
            if self._closed:
                return self.meta(display)
            if self.recording:
                self.stop_episode()
            self._closed = True
        self.close_cameras()
        self._prune_empty()
        m = self.meta(display)
        with open(os.path.join(self.dir, 'session.json'), 'w') as f:
            json.dump(m, f, indent=2)
        return m


# --------------------------------------------------------------------- view
WIN = 'umi capture   [SPACE] episode   [Q] quit'
GREY, RED, GREEN, WHITE, AMBER = ((60, 60, 60), (40, 40, 235), (90, 200, 90),
                                  (245, 245, 245), (40, 190, 245))


def label(img, text, xy, colour=WHITE, scale=0.55, thick=1):
    """Outlined text stays readable over any camera image."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def pane(img, h, title, sub):
    """One camera pane, letterboxed to a common height so the two can sit side by side
    despite different aspect ratios."""
    if img is None:
        p = np.full((h, int(h * 16 / 9), 3), 28, np.uint8)
        label(p, 'no frame yet', (16, h // 2), AMBER, 0.7, 2)
    else:
        s = h / img.shape[0]
        p = cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    label(p, title, (12, 26), WHITE, 0.62, 2)
    label(p, sub, (12, 48), AMBER, 0.48, 1)
    cv2.rectangle(p, (0, 0), (p.shape[1] - 1, p.shape[0] - 1), (80, 80, 80), 1)
    return p


def compose(panes, recording, ep_i, n_eps, elapsed, warm, skipped, disp_fps, grip=''):
    body = np.hstack(panes) if panes else np.full((240, 640, 3), 28, np.uint8)
    W = body.shape[1]
    bar = np.full((78, W, 3), 24, np.uint8)

    if warm is not None:
        label(bar, f'WARMING UP {warm:4.1f}s   not recording', (14, 30), AMBER, 0.75, 2)
        label(bar, 'cameras settling (exposure, white balance, startup burst)',
              (14, 56), GREY if False else (170, 170, 170), 0.5, 1)
    elif recording:
        cv2.circle(bar, (26, 26), 10, RED, -1)
        label(bar, f'REC  ep{ep_i:03d}   {elapsed:5.1f}s', (46, 33), RED, 0.8, 2)
        label(bar, '[SPACE] stop episode', (14, 62), (200, 200, 200), 0.5, 1)
    else:
        cv2.circle(bar, (26, 26), 10, GREY, -1)
        label(bar, 'IDLE   not recording', (46, 33), (190, 190, 190), 0.8, 2)
        label(bar, '[SPACE] start episode    [Q] quit and save session',
              (14, 62), (200, 200, 200), 0.5, 1)

    right = f'episodes saved: {n_eps}'
    label(bar, right, (W - 250, 30), GREEN if n_eps else (150, 150, 150), 0.62, 2)
    label(bar, grip, (W - 700, 30), (200, 200, 200), 0.55, 1)
    label(bar, f'preview {disp_fps:.0f} fps, {skipped} frames skipped (by design)',
          (W - 430, 60), (140, 140, 140), 0.45, 1)

    out = np.vstack([bar, body])
    if recording:      # unmistakable at a glance from across the room
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), RED, 6)
    return out


# ----------------------------------------------------------------------- main
def stdin_keys(cmd_q):
    for line in sys.stdin:
        cmd_q.put('q' if line.strip().lower() == 'q' else ' ')
    cmd_q.put('q')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', default=os.path.join(DATA, 'capture'))
    ap.add_argument('--scene-exposure', type=int, default=800, metavar='UNITS',
                    help='brightness dial, nominally UVC 100us ticks but NOT linear '
                         'on this sensor (default: 800). MOTION BLUR LIVES HERE: this '
                         'is far longer than the 50 the old camera used, and it was '
                         'chosen while shooting a stationary calibration board. If '
                         'ChArUco corner counts drop while the gripper is moving, '
                         'lower this first and add light to the scene rather than '
                         'raising it back')
    ap.add_argument('--scene-wb', type=int, default=4600, metavar='K')
    ap.add_argument('--scene-gamma', type=int, default=128, metavar='G',
                    help='0-255, default 128 (the sensor default, near-neutral). '
                         'Gamma is applied after digitisation so it amplifies noise; '
                         'buy brightness with --scene-exposure instead where you can. '
                         'Measured at matched brightness, exposure 800/gamma 128 has '
                         'about half the temporal noise of exposure 200/gamma 160.')
    ap.add_argument('--scene-gain', type=int, default=100, metavar='G',
                    help='0-1023, default 100. Weak above ~200 on this sensor: mean '
                         '2.0 at gain 0, 6.4 at gain 200, 6.5 at gain 1023. Pass -1 '
                         'to leave it alone.')
    ap.add_argument('--wrist-exposure', type=int, default=None, metavar='US',
                    help='microseconds; default per camera (6000). Raise it only after '
                         'gain is maxed, since it is the axis that costs motion blur. '
                         'Must stay under the cap in WRIST_MODELS (D405 10000, D455 '
                         '10900) or the frame rate drops. Ignored under --wrist-ae 1.')
    ap.add_argument('--wrist-gain', type=int, default=None, metavar='G',
                    help='default per camera: D405 248 (its maximum, range 16-248; '
                         'gain is the free axis there -- no blur, no frame-rate cost), '
                         'D455 64 (range 0-128)')
    ap.add_argument('--wrist-ae', type=int, choices=(0, 1), default=None,
                    help='auto exposure, where the camera offers it (D455, on by '
                         'default); the D405 has none')
    ap.add_argument('--wrist-awb', type=int, choices=(0, 1), default=1,
                    help='auto white balance, default on')
    ap.add_argument('--wrist-wb', type=int, default=4600, metavar='K')
    ap.add_argument('--no-display', action='store_true',
                    help='headless; episodes are driven from stdin (ENTER / q)')
    ap.add_argument('--no-scene', action='store_true')
    ap.add_argument('--no-wrist', action='store_true')
    args = ap.parse_args()

    try:
        sess = RecordSession(args.output,
                             scene=not args.no_scene, wrist=not args.no_wrist,
                             scene_exposure=args.scene_exposure,
                             scene_wb=args.scene_wb, scene_gamma=args.scene_gamma,
                             scene_gain=args.scene_gain,
                             wrist_exposure_us=args.wrist_exposure,
                             wrist_gain=args.wrist_gain, wrist_wb=args.wrist_wb,
                             wrist_auto_wb=bool(args.wrist_awb),
                             wrist_auto_exposure=None if args.wrist_ae is None
                             else bool(args.wrist_ae))
    except RecordError as e:
        sys.exit(str(e))
    wr = sess.recs.get('wrist')
    if wr and args.wrist_exposure and args.wrist_exposure > wr.spec['exposure_max_us']:
        print(f'  WARNING: exposure {args.wrist_exposure}us is past the {wr.model} cap of '
              f'{wr.spec["exposure_max_us"]}us -- the frame rate will drop below {wr.fps}')

    session, recs = sess.dir, sess.recs
    print(f'session: {session}')
    print(f'  gripper: {GRIPPER_DEVICE}, raw 0..{GRIPPER_RAW_OPEN} = '
          f'0..{GRIPPER_MAX_WIDTH * 1000:.0f} mm')
    print(f'clock: CLOCK_MONOTONIC + {sess.mono2epoch:.6f} s = CLOCK_REALTIME')
    if 'wrist' in recs:
        w = recs['wrist'].settings
        # None under auto -- the camera will not report the value it chose
        gain = f'gain={w["gain"]:.0f} ' if w.get('gain') is not None else 'gain=auto '
        exp = f'{w["exposure_us"]:.0f}us' if w.get('exposure_us') is not None else 'auto'
        wb = f'{w["white_balance"]:.0f}K' if w.get('white_balance') is not None else 'auto'
        print(f'  wrist: {w["name"]} sn={w["serial"]} {recs["wrist"].size[0]}x'
              f'{recs["wrist"].size[1]}@{recs["wrist"].fps} exp={exp} {gain}'
              f'wb={wb} auto_exp={w["auto_exposure"]:.0f} '
              f'auto_wb={w["auto_white_balance"]:.0f} '
              f'global_time={w["global_time_enabled"]:.0f}')

    print('\nThe preview samples the stream and skips frames on purpose -- decoding '
          'every\nscene frame would cost 57% of a CPU core. Recording always runs '
          'first and the\ndisplay never blocks it, so a high "skipped" count is '
          'correct, not a problem.\n')

    show = not args.no_display
    if show:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    # keys come from the window AND from stdin, so the tool is usable interactively
    # and drivable from a script or a foot pedal that types
    cmd_q = queue.Queue()
    threading.Thread(target=stdin_keys, args=(cmd_q,), daemon=True).start()
    print('ENTER (here or in the window) = start/stop an episode,  q = finish')

    last_img = {k: None for k in recs}
    next_draw = 0.0
    disp_meter = RateMeter(30)

    next_scene = [0.0]

    def periods():
        """Preview rates, throttled hard while an episode is running.

        cv2 work on the main thread contends for the GIL with the wrist capture
        thread, which has to be serviced every 11.1 ms or librealsense drops frames.
        Measured worst case with a full-rate preview: wrist 88.5 -> 74.7 fps. You need
        the preview most while framing the shot and least while recording, so it backs
        off automatically once an episode starts.
        """
        if sess.recording:
            return (1.0 / REC_DISPLAY_FPS, 1.0 / REC_SCENE_PREVIEW_FPS)
        return (1.0 / DISPLAY_FPS, 1.0 / SCENE_PREVIEW_FPS)

    def render():
        _, scene_period = periods()
        panes = []
        for k, r in recs.items():
            # The two panes cost wildly different amounts: a wrist frame arrives
            # already decoded (~0 ms) while a scene frame is a 1080p JPEG that costs
            # ~10 ms to decode, and that decode holds the GIL long enough to starve
            # the wrist capture thread of its 11.1 ms service interval. So the scene
            # pane refreshes on its own, slower clock.
            if k == 'scene':
                if mono() < next_scene[0]:
                    panes.append(pane(last_img[k], PANE_HEIGHT, k.upper(),
                                      f'{r.size[0]}x{r.size[1]}   {r.meter.fps():5.1f} fps'
                                      f'   {"REC " + str(r.written) if r.armed else "idle"}'))
                    continue
                next_scene[0] = mono() + scene_period
            item = r.preview.take()
            if item is not None:
                if k == 'scene':      # decode only what is actually shown
                    img = cv2.imdecode(np.frombuffer(item, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        last_img[k] = img
                else:
                    last_img[k] = item
            sub = (f'{r.size[0]}x{r.size[1]}   {r.meter.fps():5.1f} fps'
                   f'   {"REC " + str(r.written) if r.armed else "idle"}')
            panes.append(pane(last_img[k], PANE_HEIGHT, k.upper(), sub))
        warm = sess.warmup_left or None
        disp_meter.tick()
        return compose(panes, sess.recording, sess.ep_i, len(sess.episodes),
                       (mono() - sess.t_ep) if sess.recording else 0.0, warm,
                       sum(r.preview.skipped for r in recs.values()),
                       disp_meter.fps(), grip_text())

    def grip_text():
        g, m = sess.gripper, sess.imu
        if g.err:
            return f'gripper ERROR {g.err}'
        w = g.width(g.latest[1]) * 1000 if g.latest else float('nan')
        s = f'gripper {w:5.1f} mm  {g.meter.fps():4.1f} Hz'
        if m is not None:
            last = m.recent['gyro'][-1] if m.recent['gyro'] else None
            v = np.degrees(np.linalg.norm(last[1:])) if last else float('nan')
            s += (f'   imu ERROR {m.err}' if m.err else
                  f'   imu {v:5.1f} deg/s  {m.meter["gyro"].fps():5.1f} Hz')
        return s

    def start_ep():
        print(f'\n>>> ep{sess.start_episode():03d} recording')

    def stop_ep():
        i, ep = sess.ep_i, sess.stop_episode()
        if ep is None:
            return
        print('\n<<< ep%03d saved: ' % i + '  '.join(
            f'{k} {ep[k]["frames"]} frames @ {ep[k]["fps"]:.1f} fps' for k in recs)
            + f'  gripper {ep["gripper"]["samples"]} @ {ep["gripper"]["hz"]:.1f} Hz'
            + f'  ({ep["duration_s"]:.1f}s)')

    try:
        while True:
            key = None
            if show:
                now = mono()
                if now >= next_draw:
                    next_draw = now + periods()[0]
                    cv2.imshow(WIN, render())
                k = cv2.waitKey(5) & 0xFF
                if k != 255:
                    key = {32: ' ', 13: ' ', 10: ' ', ord('q'): 'q', 27: 'q'}.get(k)
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key is None:
                try:
                    key = cmd_q.get(timeout=0.0 if show else 0.2)
                except queue.Empty:
                    key = None
            if not show and sess.recording and key is None:
                print('\r  REC ep%03d  ' % sess.ep_i + '  '.join(
                    f'{k} {r.written}' for k, r in recs.items())
                    + f'  |  {grip_text()}  |  {mono()-sess.t_ep:5.1f}s', end='', flush=True)

            if key is None:
                continue
            if key == 'q':
                break
            if key == ' ':
                if not sess.recording:
                    try:
                        start_ep()
                    except RecordError as e:
                        print(f'  {e}')
                else:
                    stop_ep()
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()

    meta = sess.finish(display={
        'enabled': show, 'fps': DISPLAY_FPS,
        'frames_skipped': {k: r.preview.skipped for k, r in recs.items()},
        'note': 'preview samples the stream; skipping is by design '
                'and does not affect recorded files'})
    for k in [*recs, 'gripper']:
        if meta.get(k, {}).get('error'):
            print(f'{k} ERROR: {meta[k]["error"]}')

    print(f'\n{len(meta["episodes"])} episodes -> {session}')
    print(f'verify with:\n  python3 {os.path.join(os.path.dirname(HERE), "robospec_umi_dataset", "verify.py")} {session}')


if __name__ == '__main__':
    main()
