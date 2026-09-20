"""Pinhole intrinsics calibration for the DECXIN 1080p120 scene camera.

Emits a plumb_bob/rational calibration that video_processor_charuco.py's
load_intrinsics() reads unmodified, via the top-level "k" and "d" keys it
already accepts as a CameraInfo dump.

    lock     freeze focus/zoom/exposure/white balance and verify they stuck
    board    render the printable ChArUco board
    capture  live capture with coverage meters
    solve    cv2.calibrateCameraExtended over the frames -> intrinsics json

Typical run -- every path defaults into data/calibration/, so:
    python3 robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py lock
    python3 robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py capture
    python3 robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py solve

then record:
    python3 robospec_umi/robospec_umi_capture/capture.py

READ THIS FIRST -- autofocus voids the calibration
--------------------------------------------------
This camera ships with focus_automatic_continuous=1. Focusing physically moves
elements, which changes the focal length, so autofocus makes fx/fy drift between
calibration views and again between calibration and recording. Nothing
downstream can detect it: the ChArUco pose solve just returns quietly wrong
depths, and the poses look perfectly plausible. Always run `lock` first;
capture.py locks the same values at recording time, from its own SCENE_FOCUS /
SCENE_ZOOM constants.

The camera also has a digital zoom_absolute (100-200) which crops and rescales,
changing both f and the principal point. It is pinned at 100.

There is no `circle` command -- a rectilinear lens has no image circle. Its real
job was an independent check on the principal point, which is done better here
by cv2.calibrateCameraExtended's per-parameter standard deviations.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

CV_VER = tuple(int(v) for v in cv2.__version__.split('.')[:2])

# A udev symlink, not a /dev/videoN number: USB enumeration order is not stable,
# and opening the wrong node hands you a different camera rather than failing.
# See robospec_umi/99-decxin-cam.rules. Override with --device if the rule is not
# installed.
DEVICE = '/dev/scene_cam'

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_calibration
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

# One directory per calibration run, plus a stable path holding the active one.
# verify.py and build_zarr.py both read the stable path, so a run from anywhere
# must still land there or their focus/zoom cross-checks go quiet.
CALIB_ROOT = os.path.join(DATA, 'calibration')
DEFAULT_INTRINSICS = os.path.join(CALIB_ROOT, 'scene_intrinsics.json')

# ------------------------------------------------------------- fixed geometry
# The printed A4 intrinsics target -- NOT the small 4x4 20 mm board that
# build_zarr.py tracks during recording. Two different physical boards.
# 7x5 at 30 mm is 210x150 mm, so it prints on A4 at 100% scale and gives 24
# interior corners.
BOARD_DICT = 'DICT_4X4_50'
SQUARES_X, SQUARES_Y = 7, 5
SQUARE_LEN = 0.030      # m
MARKER_LEN = 0.022      # m
MIN_CORNERS = 10        # per view, below which a frame is not worth keeping

# ------------------------------------------------------------------- stream
WIDTH, HEIGHT = 1920, 1080
FPS = 120

# ---------------------------------------------------------- board rendering
BOARD_DPI = 300
BOARD_MARGIN_MM = 10.0

# ------------------------------------------------------------ capture tuning
DISPLAY_SCALE = 1.0     # preview downscale; the saved frames are always full res
PER_BIN = 6             # views wanted per coverage cell
GRID_X, GRID_Y = 4, 3   # coverage grid over the frame
AUTO_PERIOD = 1.0       # s between automatic keeps
CLIP_WARN = 2.0         # % of saturated pixels worth warning about
STILL_PX = 2.0          # max corner motion for the board to count as still
STILL_S = 0.25          # s it must stay that still

# -------------------------------------------------------------- solve tuning
REJECT_SIGMA = 3.0      # per-view outlier threshold, in sigmas
REJECT_ITERS = 5        # rounds of reject-and-refit
HOLDOUT = 0.25          # fraction of views held out to score candidate models
ALPHA = 0.0             # getOptimalNewCameraMatrix alpha for the preview


# --------------------------------------------------------------- run folders
def new_calib_run():
    """A fresh data/calibration/<datetime>/ for this capture run."""
    d = os.path.join(CALIB_ROOT, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(d, exist_ok=True)
    return d


def latest_calib_run():
    """The newest run directory that actually holds frames, or None.

    `solve` defaults here so the usual capture-then-solve pair needs no path
    typed twice. It is announced on its own line at the call site rather than
    used silently -- picking the wrong frame set is not something you would
    notice in the result, only in every pose built from it afterwards.
    """
    if not os.path.isdir(CALIB_ROOT):
        return None
    runs = [os.path.join(CALIB_ROOT, d) for d in sorted(os.listdir(CALIB_ROOT))]
    # Numbered frames only -- a run holding just a rendered charuco_board.png has
    # nothing to solve.
    runs = [d for d in runs
            if os.path.isdir(d) and glob.glob(os.path.join(d, '[0-9]*.png'))]
    return runs[-1] if runs else None


# ------------------------------------------------------------------ v4l2 locks
# Locked in this order deliberately: the *_absolute controls report
# flags=inactive and silently ignore writes until their auto counterpart is off.
LOCK_ORDER = ('focus_automatic_continuous', 'focus_absolute', 'zoom_absolute',
              'auto_exposure', 'exposure_time_absolute',
              'white_balance_automatic', 'white_balance_temperature',
              'backlight_compensation', 'gamma', 'gain')

# Controls whose value must survive into recording for the calibration to apply.
GEOMETRY_CTRLS = ('focus_automatic_continuous', 'focus_absolute', 'zoom_absolute')


def v4l2_set(device, ctrl, value):
    r = subprocess.run(['v4l2-ctl', '-d', device, '-c', f'{ctrl}={value}'],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def v4l2_get(device, ctrls):
    """-> {ctrl: int}. Missing or unreadable controls are simply absent."""
    r = subprocess.run(['v4l2-ctl', '-d', device, '--get-ctrl', ','.join(ctrls)],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        try:
            out[k.strip()] = int(v.strip().split()[0])
        except (ValueError, IndexError):
            pass
    return out


def v4l2_ranges(device):
    """-> {ctrl: (min, max)} parsed from --list-ctrls.

    Worth the parse because UVC clamps an out-of-range write silently and
    reports success. This camera's backlight_compensation has min=16, so the
    value 0 that works on the old scene camera would look like it applied and
    quietly sit at 16 instead.
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
        kv = dict(tok.split('=', 1) for tok in tail.split() if '=' in tok)
        try:
            out[name] = (int(kv['min']), int(kv['max']))
        except (KeyError, ValueError):
            pass
    return out


# V4L2_CTRL_FLAG_INACTIVE. In principle the driver raises this on a control
# whose automatic counterpart has taken it over.
#
# MEASURED: this camera never sets it. exposure_time_absolute stays "active"
# under auto_exposure=3, and white_balance_temperature stays active under auto
# white balance -- both remain readable and writable, and both are simply
# ignored, with the readback reporting the stale manual value. So a caller
# cannot use this flag to decide what is adjustable HERE; derive that from
# auto_exposure and white_balance_automatic instead. Parsed anyway because it
# is correct for hardware that does implement it, and costs one line.
V4L2_FLAG_INACTIVE = 0x0010


def v4l2_controls(device):
    """-> {ctrl: {type, min, max, default, value, inactive}} from --list-ctrls.

    A superset of v4l2_ranges(): same parse, but keeps the current value and the
    flags so a caller can tell an adjustable control from one the camera is
    currently driving itself.
    """
    r = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if ':' not in line or '0x' not in line:
            continue
        head, tail = line.split(':', 1)
        name = head.split()[0]
        kind = ('bool' if '(bool)' in head else
                'menu' if '(menu)' in head else
                'int' if '(int)' in head else 'other')
        if kind == 'other':
            continue
        kv = {}
        for tok in tail.split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                kv[k] = v
        try:
            flags = int(kv.get('flags', '0'), 16)
        except ValueError:
            flags = 0
        entry = {'type': kind, 'inactive': bool(flags & V4L2_FLAG_INACTIVE)}
        for key, default in (('min', 0), ('max', 1), ('default', None),
                             ('value', None)):
            try:
                entry[key] = int(kv[key])
            except (KeyError, ValueError):
                entry[key] = default
        if kind == 'bool':
            entry['min'], entry['max'] = 0, 1
        out[name] = entry
    return out


def lock_geometry(device, focus=0, zoom=100):
    """Lock only what actually affects the intrinsics. -> (ok, got, dyn_ok).

    Photometric controls are deliberately NOT touched. Measured on this camera
    at 1080p: exposure 1..10000, gain 0..1023, gamma 0..255 and white balance
    2800..6500 all hold 119.4-119.8 fps, as do auto exposure and auto white
    balance. None of them changes the projection either, so freezing them buys
    nothing and costs the operator the ability to work in a different room.

    exposure_dynamic_framerate is ASSERTED, never written. It defaults to 0 and
    nothing in this repo sets it, but it is the single flag between 120 fps and
    a camera that trades frame rate for light: with it at 1, measured 78 fps at
    exposure 800, 20 fps at 5000, 10 fps at 10000.
    """
    wanted = {
        'focus_automatic_continuous': 0,
        'focus_absolute': int(focus),
        'zoom_absolute': int(zoom),
    }
    ok, got = apply_locks(device, wanted, verbose=False)
    dyn = v4l2_get(device, ['exposure_dynamic_framerate'])
    dyn_ok = dyn.get('exposure_dynamic_framerate', 0) == 0
    got.update(dyn)
    return ok, got, dyn_ok


def apply_locks(device, wanted, verbose=True):
    """Write every control in LOCK_ORDER that appears in `wanted`, then read back.

    Returns (ok, got). ok is False if any control did not take the requested
    value -- which is the case that matters, because a rejected write is silent
    and leaves autofocus running.
    """
    rng = v4l2_ranges(device)
    wanted = dict(wanted)
    for ctrl, val in list(wanted.items()):
        if val is None or ctrl not in rng:
            continue
        lo, hi = rng[ctrl]
        if not lo <= val <= hi:
            wanted[ctrl] = int(np.clip(val, lo, hi))
            if verbose:
                print(f'  note: {ctrl}={val} is outside this camera\'s '
                      f'{lo}..{hi}, using {wanted[ctrl]}')

    for ctrl in LOCK_ORDER:
        if ctrl not in wanted or wanted[ctrl] is None:
            continue
        ok, err = v4l2_set(device, ctrl, wanted[ctrl])
        if not ok and verbose:
            print(f'  warning: could not set {ctrl}={wanted[ctrl]}: {err}')

    got = v4l2_get(device, [c for c in LOCK_ORDER if c in wanted])
    bad = []
    for ctrl, want in wanted.items():
        if want is None:
            continue
        if ctrl not in got:
            bad.append(f'{ctrl} unreadable')
        elif got[ctrl] != want:
            bad.append(f'{ctrl}={got[ctrl]} (wanted {want})')
    if verbose:
        for ctrl in LOCK_ORDER:
            if ctrl in got:
                flag = '' if wanted.get(ctrl) in (None, got[ctrl]) else '  <-- DID NOT STICK'
                print(f'  {ctrl:<32} {got[ctrl]}{flag}')
    return not bad, got


def focus_is_locked(device):
    got = v4l2_get(device, GEOMETRY_CTRLS)
    return got.get('focus_automatic_continuous') == 0, got


def cmd_lock(args):
    wanted = {
        'focus_automatic_continuous': 0,
        'focus_absolute': args.focus,
        'zoom_absolute': args.zoom,
        'auto_exposure': 1,
        'exposure_time_absolute': args.exposure,
        'white_balance_automatic': 0,
        'white_balance_temperature': args.wb,
        'backlight_compensation': 0,
        'gamma': args.gamma,
        'gain': args.gain,
    }
    print(f'locking {args.device}')
    ok, got = apply_locks(args.device, wanted)

    if not ok:
        print('\nFAILED: at least one control did not take. On most UVC cameras '
              'this means the auto counterpart is still on, or another process '
              'holds the device. Close any viewer and retry.')
        raise SystemExit(1)

    print('\nall controls locked.')
    print('\nFocus is now fixed. Do not touch the lens ring or unplug the camera '
          'between calibrating and recording -- a replug resets these to defaults, '
          'autofocus included.')
    f = int(got.get('focus_absolute', args.focus))
    z = int(got.get('zoom_absolute', args.zoom))
    print(f'\nlocked geometry: focus {f}, zoom {z}')
    # capture.py pins these from its own constants, so a lock at anything else
    # mismatches the recording and verify.py only catches it afterwards.
    if (f, z) != (0, 100):
        print('  WARNING: capture.py records at focus 0 / zoom 100 -- these '
              'intrinsics will not apply to that footage.')


# ------------------------------------------------------------------ capture dev
def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'failed to open {device}')
    # MJPG before the size, and never BUFFERSIZE=1 -- see grab_frame.py.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        cap.release()
        raise RuntimeError(f'asked for {width}x{height}, got {got[0]}x{got[1]}')
    return cap


# ---------------------------------------------------------------------- charuco
def make_board(adict, squares, square_length, marker_length):
    """A CharucoBoard on either aruco API.

    Dispatch on the version, not hasattr: cv2 4.6 exposes a CharucoBoard symbol
    that accepts the 4.7-style constructor and then segfaults on first use.
    """
    if CV_VER < (4, 7):
        return cv2.aruco.CharucoBoard_create(
            squares[0], squares[1], square_length, marker_length, adict)
    return cv2.aruco.CharucoBoard(squares, square_length, marker_length, adict)


def board_corners(board):
    return np.array(board.chessboardCorners if CV_VER < (4, 7)
                    else board.getChessboardCorners(), dtype=np.float64)


def make_params():
    """Detector parameters for a rectilinear lens.

    A rectilinear lens does not compress markers near the rim, so the default
    adaptive-threshold sweep is adequate and a narrower one is faster at 120 fps.
    Subpixel refinement still matters -- it is the whole reason to prefer
    chessboard saddle points.
    """
    p = (cv2.aruco.DetectorParameters_create() if CV_VER < (4, 7)
         else cv2.aruco.DetectorParameters())
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 5
    p.minMarkerPerimeterRate = 0.02
    return p


def detect_charuco(gray, adict, board, params):
    """-> (charuco_corners, charuco_ids) or (None, None)."""
    corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
    if ids is None or len(ids) == 0:
        return None, None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if n is None or n < 4:
        return None, None
    return cc, ci


def inner_grid(squares_x, squares_y):
    """Interior-corner grid dimensions. ChArUco id i sits at (i % nx, i // nx)."""
    return squares_x - 1, squares_y - 1


# --------------------------------------------------------------------- geometry
def board_quad(cc, ci, nx, ny):
    """The full board outline in image space, even from a partial detection.

    Fits a homography from interior-corner grid coordinates to pixels and maps
    the four grid corners through it. A homography is exactly the right model --
    a plane imaged by a pinhole camera is a homography, up to lens distortion --
    so this extrapolates honestly rather than guessing, and it lets a view that
    only shows two thirds of the board still report its shape.

    -> (4,2) quad in grid order TL, TR, BR, BL, or None.
    """
    ids = ci.flatten()
    if len(ids) < 6:
        return None
    src = np.stack([ids % nx, ids // nx], axis=1).astype(np.float64)
    dst = cc.reshape(-1, 2).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None
    grid = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]],
                    dtype=np.float64)
    return cv2.perspectiveTransform(grid.reshape(1, -1, 2), H).reshape(-1, 2)


def quad_metrics(quad, img_shape):
    """-> (fill, tilt).

    fill is the board's linear size as a fraction of the frame -- sqrt of the
    area ratio, so it behaves like 1/distance rather than 1/distance^2.

    tilt is a foreshortening proxy in [0,1): 0 when opposite edges are equal,
    i.e. fronto-parallel, rising as the board turns away. It is deliberately
    not an angle -- recovering a real angle needs the focal length, which is the
    thing we are here to measure. What matters for conditioning is only that
    some views are clearly not fronto-parallel, and this is monotonic in that.
    """
    h, w = img_shape[:2]
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    fill = float(np.sqrt(max(area, 1e-9) / (w * h)))

    e = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    tilt = 0.0
    for a, b in ((e[0], e[2]), (e[1], e[3])):
        lo, hi = min(a, b), max(a, b)
        if hi > 1e-6:
            tilt = max(tilt, 1.0 - lo / hi)
    return fill, float(tilt)


# --------------------------------------------------------------------- coverage
SCALE_BINS = (('far', 0.0, 0.22), ('mid', 0.22, 0.40), ('near', 0.40, 1.01))
TILT_BINS = (('flat', 0.0, 0.07), ('mild', 0.07, 0.17), ('strong', 0.17, 1.01))


def _bin(bins, v):
    for i, (_, lo, hi) in enumerate(bins):
        if lo <= v < hi:
            return i
    return len(bins) - 1


class Coverage:
    """Three independent meters, because a rectilinear lens fails three ways.

    Binning by radius alone is not enough here -- that suits a lens whose
    outermost corners carry all the distortion information. A rectilinear lens
    is constrained differently, on three axes at once.

      grid   distortion grows with distance from the principal point, so the
             frame corners must be visited. Tracked by where corners actually
             land, not where the board centre is.
      scale  near and far views break the correlation between focal length and
             board distance. All-far views leave f poorly determined.
      tilt   the classic pinhole failure: every view fronto-parallel leaves the
             solve ill-conditioned, converging to a low reprojection error that
             is nonetheless wrong. Tilted views are what fix it.

    Kept as three separate meters rather than one product of bins because the
    product would have hundreds of cells and never fill, which tells the user
    nothing about what is actually missing.
    """

    def __init__(self, shape, nx=4, ny=3):
        self.h, self.w = shape[:2]
        self.nx, self.ny = nx, ny
        self.grid = np.zeros((ny, nx), dtype=np.int32)
        self.scale = np.zeros(len(SCALE_BINS), dtype=np.int32)
        self.tilt = np.zeros(len(TILT_BINS), dtype=np.int32)

    def add(self, corners, fill, tilt):
        pts = corners.reshape(-1, 2)
        gx = np.clip((pts[:, 0] / self.w * self.nx).astype(int), 0, self.nx - 1)
        gy = np.clip((pts[:, 1] / self.h * self.ny).astype(int), 0, self.ny - 1)
        for x, y in zip(gx, gy):
            self.grid[y, x] += 1
        self.scale[_bin(SCALE_BINS, fill)] += 1
        self.tilt[_bin(TILT_BINS, tilt)] += 1

    def status(self, target, per_bin):
        cells = int((self.grid >= target).sum())
        return (cells, self.grid.size,
                int((self.scale >= per_bin).sum()), len(SCALE_BINS),
                int((self.tilt >= per_bin).sum()), len(TILT_BINS))

    def done(self, target, per_bin):
        return (self.grid >= target).all() and \
               (self.scale >= per_bin).all() and (self.tilt >= per_bin).all()

    def advice(self, target, per_bin):
        """The single most useful next instruction, or '' when finished."""
        weak = np.argwhere(self.grid < target)
        if len(weak):
            names_y = ('top', 'middle', 'bottom')
            names_x = ('left', 'centre-left', 'centre-right', 'right')
            y, x = weak[np.argmin(self.grid[weak[:, 0], weak[:, 1]])]
            ny_ = names_y[int(round(y / max(self.ny - 1, 1) * 2))]
            nx_ = names_x[int(round(x / max(self.nx - 1, 1) * 3))]
            return f'fill the {ny_} {nx_} of the frame'
        for arr, bins, verb in ((self.scale, SCALE_BINS, 'shoot'),
                                (self.tilt, TILT_BINS, 'shoot')):
            if (arr < per_bin).any():
                i = int(np.argmin(arr))
                return f'{verb} more {bins[i][0]} views ({arr[i]}/{per_bin})'
        return ''

    def draw(self, img, target, per_bin, y0=130):
        # grid heat map over the frame
        for y in range(self.ny):
            for x in range(self.nx):
                x0, x1 = int(x * self.w / self.nx), int((x + 1) * self.w / self.nx)
                y0, y1 = int(y * self.h / self.ny), int((y + 1) * self.h / self.ny)
                frac = min(self.grid[y, x] / target, 1.0)
                col = (0, int(60 + 195 * frac), int(180 * (1 - frac)))
                cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), col, 2)

        # Scale and tilt as labelled bars down the left edge,
        y = 130
        for arr, bins, title in ((self.scale, SCALE_BINS, 'scale'),
                                 (self.tilt, TILT_BINS, 'tilt')):
            cv2.putText(img, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            y += 28
            for i, (name, _, _) in enumerate(bins):
                frac = min(arr[i] / per_bin, 1.0)
                col = (0, 200, 0) if frac >= 1 else (60, 160, 255)
                cv2.rectangle(img, (20, y - 14), (20 + int(180 * frac), y + 2),
                              col, -1)
                cv2.rectangle(img, (20, y - 14), (200, y + 2), (120, 120, 120), 1)
                cv2.putText(img, f'{name} {arr[i]}/{per_bin}', (210, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y += 26
            y += 8


# ------------------------------------------------------------------------ board
def cmd_board(args):
    """Render the printable board.

    Deliberately a plain predefined dictionary rather than the offset one from
    umi.common.cv_util.get_charuco_board: that builds its dictionary with
    cv2.aruco.Dictionary(bytesList, markerSize), which is the 4.7 constructor.
    On 4.6 it constructs and then segfaults on first attribute access.
    """
    if args.out is None:
        run = latest_calib_run() or new_calib_run()
        args.out = os.path.join(run, 'charuco_board.png')

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    board = make_board(adict, (SQUARES_X, SQUARES_Y),
                       SQUARE_LEN, MARKER_LEN)

    mm = 1000.0
    w_mm = SQUARES_X * SQUARE_LEN * mm
    h_mm = SQUARES_Y * SQUARE_LEN * mm
    px_per_mm = BOARD_DPI / 25.4
    size = (int(round(w_mm * px_per_mm)), int(round(h_mm * px_per_mm)))
    margin = int(round(BOARD_MARGIN_MM * px_per_mm))

    img = (board.draw(size, marginSize=margin) if CV_VER < (4, 7)
           else board.generateImage(size, marginSize=margin))
    cv2.imwrite(args.out, img)

    print(f'{SQUARES_X}x{SQUARES_Y} ChArUco, {BOARD_DICT}')
    print(f'square {SQUARE_LEN * mm:.1f} mm, marker {MARKER_LEN * mm:.1f} mm')
    print(f'printed size {w_mm:.0f} x {h_mm:.0f} mm '
          f'(+{BOARD_MARGIN_MM:.0f} mm margin) at {BOARD_DPI} dpi')
    print(f'wrote {args.out}')
    print('\nPrint at 100% / "actual size" -- any fit-to-page scaling silently '
          'changes the square size.\nMeasure a printed square with calipers; if it '
          'is not 30.0 mm, correct SQUARE_LEN at the top of this file.')
    print('Mount it on something rigid and flat. A board that bows by a '
          'millimetre biases the distortion terms, and no amount of views fixes it.')


# ---------------------------------------------------------------------- capture
def exposure_stats(gray, quad=None):
    """-> (mean, clip%, dark%, board_clip%|None, verdict, bgr_colour).

    The right exposure depends on the room, not the camera, so it is reported
    rather than assumed: a dark frame loses marker detections outright, and a
    clipped one destroys the intensity gradients the subpixel corner refinement
    works from -- which costs accuracy quietly instead of costing detections
    visibly.

    When a board quad is given, clipping is scored INSIDE it rather than over
    the whole frame. A bright window or lamp in the background blows out a few
    percent of any frame and is completely harmless, so a global threshold nags
    constantly while saying nothing about the only region that matters.

    Shared by the capture overlay and the photometric preview so the two can
    never disagree about whether the exposure is acceptable.
    """
    mean = float(gray.mean())
    clip = 100.0 * float((gray >= 250).mean())
    dark = 100.0 * float((gray <= 8).mean())
    board_clip = None
    if quad is not None:
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
        px = gray[m > 0]
        if px.size:
            board_clip = 100.0 * float((px >= 250).mean())
    scored = board_clip if board_clip is not None else clip
    if mean < 40:
        msg, col = 'DARK - raise exposure or gain', (60, 160, 255)
    elif scored > CLIP_WARN:
        msg, col = (f'CLIPPED {scored:.1f}% - lower exposure', (60, 160, 255))
    elif dark > 35:
        msg, col = 'SHADOWS CRUSHED - raise exposure', (60, 160, 255)
    else:
        msg, col = 'exposure ok', (0, 255, 0)
    return mean, clip, dark, board_clip, msg, col


class CaptureSession:
    """One calibration capture run: camera, detection, coverage meters, disk.

    Extracted from the cv2 loop so the CLI and the web server drive identical
    logic. The stillness rule and the auto-keep throttle decide which frames
    reach the calibration, so two implementations would eventually disagree
    about what got saved -- and the symptom would be a worse fit with nothing
    visible to explain it.

    Nothing here touches a window or a keyboard. The caller owns the loop and
    decides how the returned overlay is shown.
    """

    def __init__(self, device, out_dir=None, target=60):
        self.device = device
        self.out = out_dir or new_calib_run()
        self.target = target
        os.makedirs(self.out, exist_ok=True)

        locked, got = focus_is_locked(device)
        if not locked:
            raise RuntimeError(
                f'autofocus is on (focus_automatic_continuous='
                f'{got.get("focus_automatic_continuous")}). Focus changes the '
                f'focal length, so views shot at different focus settings do not '
                f'share one calibration and the fit is meaningless. Run `lock` first.')
        if got.get('zoom_absolute', 100) != 100:
            raise RuntimeError(
                f'zoom_absolute={got["zoom_absolute"]}, not 100. Digital zoom '
                f'crops and rescales, changing f and the principal point.')
        self.locked = got

        self.adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
        self.board = make_board(self.adict, (SQUARES_X, SQUARES_Y),
                                SQUARE_LEN, MARKER_LEN)
        self.params = make_params()
        self.nx, self.ny = inner_grid(SQUARES_X, SQUARES_Y)
        self.cap = open_camera(device, WIDTH, HEIGHT, FPS)
        self.cov = Coverage((HEIGHT, WIDTH), GRID_X, GRID_Y)

        # [0-9]*.png, NOT *.png: charuco_board.png and undistort_preview.png also
        # live in this directory, and counting them would push the first captured
        # frame to 0001.png and leave a gap solve would silently accept.
        self.saved = len(glob.glob(os.path.join(self.out, '[0-9]*.png')))

        self.auto = False
        self.last_auto = 0.0
        self.prev_quad = None
        self.still_since = 0.0
        self._cur = None          # newest (frame, cc, ci, fill, tilt, still, n)

        # Written NOW rather than at exit. It used to be written after the loop's
        # finally, so any exception left the run without it -- and solve then
        # quietly omits locked_controls, which is the field verify.py and
        # build_zarr.py cross-check focus and zoom against.
        self._write_locked_controls()

    def _write_locked_controls(self):
        with open(os.path.join(self.out, 'locked_controls.json'), 'w') as f:
            json.dump(v4l2_get(self.device, LOCK_ORDER), f, indent=2)

    # ------------------------------------------------------------------ state
    def coverage_state(self):
        """Coverage as plain JSON types. numpy int32/bool_ do not serialise."""
        cells, tot, sok, stot, tok, ttot = self.cov.status(self.target, PER_BIN)
        return {
            'grid': self.cov.grid.tolist(),
            'grid_nx': self.cov.nx, 'grid_ny': self.cov.ny,
            'grid_done': cells, 'grid_total': tot,
            'scale': self.cov.scale.tolist(),
            'scale_names': [b[0] for b in SCALE_BINS],
            'scale_done': sok, 'scale_total': stot,
            'tilt': self.cov.tilt.tolist(),
            'tilt_names': [b[0] for b in TILT_BINS],
            'tilt_done': tok, 'tilt_total': ttot,
            'per_bin': PER_BIN, 'target': self.target,
            'advice': self.cov.advice(self.target, PER_BIN),
            'done': bool(self.cov.done(self.target, PER_BIN)),
        }

    # ------------------------------------------------------------------- loop
    def grab(self):
        """Dequeue a frame without decoding it -- ~0.1 ms against ~9 ms for the
        decode, so a caller wanting fewer frames than the camera produces can
        drop here and never pay for the JPEG."""
        return self.cap.grab()

    def step(self, grabbed=False):
        """One iteration. -> (raw_frame, overlay_frame, state) or None on a bad read.

        Auto-keep happens here because it is time-gated and has to be evaluated
        every frame. Manual keeps come in through keep().

        grabbed=True decodes the frame the caller already grab()ed instead of
        dequeuing another. Note the motion gate below compares CONSECUTIVE
        steps, so its sensitivity follows the step rate -- pace it evenly.
        """
        if not grabbed and not self.cap.grab():
            return None
        ok, frame = self.cap.retrieve()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci = detect_charuco(gray, self.adict, self.board, self.params)

        quad = None if cc is None else board_quad(cc, ci, self.nx, self.ny)
        fill = tilt = 0.0
        if quad is not None:
            fill, tilt = quad_metrics(quad, frame.shape)

        # Motion check. This is a rolling-shutter sensor, so a board moving
        # during readout is sheared, not merely blurred: corners shift by an
        # amount that grows down the frame. That is a systematic bias the
        # calibration will happily absorb into the distortion terms.
        now = time.monotonic()
        moved = 1e9
        if quad is not None and self.prev_quad is not None:
            moved = float(np.abs(quad - self.prev_quad).max())
        if moved > STILL_PX:
            self.still_since = now
        still = quad is not None and (now - self.still_since) > STILL_S
        self.prev_quad = quad

        n = 0 if cc is None else len(cc)
        enough = n >= MIN_CORNERS
        self._cur = (frame, cc, ci, fill, tilt, still, n)

        mean, clip, dark, board_clip, exp_msg, exp_col = self._exposure(gray, quad)
        view = self._overlay(frame, cc, ci, quad, n, enough, fill, tilt, still,
                             mean, clip, dark, board_clip, exp_msg, exp_col)

        kept = False
        if self.auto and enough and still and now - self.last_auto > AUTO_PERIOD:
            self.last_auto = now
            kept = self.keep()[0]

        state = {
            'run_dir': self.out, 'saved': self.saved, 'corners': n,
            'enough': enough, 'still': still, 'auto': self.auto,
            'fill': round(fill, 3), 'tilt': round(tilt, 3),
            'scale_bin': SCALE_BINS[_bin(SCALE_BINS, fill)][0],
            'tilt_bin': TILT_BINS[_bin(TILT_BINS, tilt)][0],
            'exposure': {'mean': round(mean, 1), 'clipped': round(clip, 2),
                         'dark': round(dark, 2),
                         'board_clipped': None if board_clip is None else round(board_clip, 2),
                         'verdict': exp_msg},
            'coverage': self.coverage_state(),
            'just_kept': kept,
        }
        return frame, view, state

    def keep(self):
        """Save the newest frame. -> (saved_bool, reason). The SPACE path."""
        if self._cur is None:
            return False, 'no frame yet'
        frame, cc, ci, fill, tilt, still, n = self._cur
        if n < MIN_CORNERS:
            return False, f'only {n} corners (need {MIN_CORNERS})'
        if not still:
            return False, 'board still moving (rolling shutter shears it)'
        path = os.path.join(self.out, f'{self.saved:04d}.png')
        cv2.imwrite(path, frame)
        self.cov.add(cc, fill, tilt)
        self.saved += 1
        return True, path

    def undo(self):
        """Remove the most recently saved frame. Coverage is NOT rewound."""
        if self.saved <= 0:
            return False, 'nothing to undo'
        self.saved -= 1
        path = os.path.join(self.out, f'{self.saved:04d}.png')
        if os.path.exists(path):
            os.remove(path)
        return True, path

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # --------------------------------------------------------------- internals
    def _exposure(self, gray, quad):
        return exposure_stats(gray, quad)

    def _overlay(self, frame, cc, ci, quad, n, enough, fill, tilt, still,
                 mean, clip, dark, board_clip, exp_msg, exp_col):
        view = frame.copy()
        self.cov.draw(view, self.target, PER_BIN)
        if cc is not None:
            cv2.aruco.drawDetectedCornersCharuco(view, cc, ci, (0, 255, 255))
        if quad is not None:
            cv2.polylines(view, [quad.astype(np.int32)], True, (255, 0, 255), 2)

        cells, tot, sok, stot, tok, ttot = self.cov.status(self.target, PER_BIN)
        cv2.putText(view, f'saved {self.saved}   corners {n}   '
                    f'grid {cells}/{tot}  scale {sok}/{stot}  tilt {tok}/{ttot}'
                    f'{"   AUTO" if self.auto else ""}', (14, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if enough else (0, 0, 255), 2)
        cv2.putText(view, f'fill {fill:.2f} ({SCALE_BINS[_bin(SCALE_BINS, fill)][0]})   '
                    f'tilt {tilt:.2f} ({TILT_BINS[_bin(TILT_BINS, tilt)][0]})   '
                    f'{"STILL" if still else "MOVING"}', (14, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if still else (60, 160, 255), 2)
        bc = '' if board_clip is None else f'  board {board_clip:.1f}%'
        cv2.putText(view, f'mean {mean:.0f}  clipped {clip:.1f}%{bc}  '
                    f'dark {dark:.1f}%  {exp_msg}',
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, exp_col, 2)
        tip = self.cov.advice(self.target, PER_BIN)
        cv2.putText(view, tip or 'coverage complete -- run solve',
                    (14, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255) if tip else (0, 255, 0), 2)
        return view


def cmd_capture(args):
    """Thin cv2 driver over CaptureSession. All the logic lives in the class."""
    show = not args.no_display
    try:
        s = CaptureSession(args.device, args.out, args.target)
    except RuntimeError as e:
        print(f'REFUSING TO START: {e}')
        raise SystemExit(1)
    print(f'frames -> {s.out}')
    print(f'focus locked at {s.locked.get("focus_absolute")}, '
          f'zoom {s.locked.get("zoom_absolute")}')
    if s.saved:
        print(f'{s.saved} frames already in {s.out}; coverage meters start empty '
              f'(they are not reconstructed from disk)')

    win = 'scene cam calibration capture'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('SPACE keep a frame   a auto   u undo   q quit')
    else:
        # waitKey is the only input channel, so headless has to drive itself:
        # auto-keep from the start, and stop on the coverage meters rather than
        # on a keypress that can never arrive.
        s.auto = True
        print(f'headless: auto-keeping frames, stops when coverage is complete '
              f'or {args.target} frames are saved. Ctrl-C to stop early.')

    try:
        while True:
            r = s.step()
            if r is None:
                continue
            _, view, state = r
            if state['just_kept']:
                print(f'saved {os.path.join(s.out, "%04d.png" % (s.saved - 1))}  '
                      f'{state["corners"]} corners  '
                      f'fill {state["fill"]:.2f} tilt {state["tilt"]:.2f}')
            key = 255
            if show:
                disp = cv2.resize(view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                  interpolation=cv2.INTER_AREA)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                ok, why = s.keep()
                print(f'saved {why}  {state["corners"]} corners' if ok
                      else f'rejected: {why}')
            if key == ord('a'):
                s.auto = not s.auto
            if key == ord('u'):
                ok, why = s.undo()
                print('removed last frame (coverage meters not rewound)' if ok
                      else f'undo: {why}')
            if key in (ord('q'), 27):
                break
            if not show and (state['coverage']['done'] or s.saved >= args.target):
                break
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        s.close()
        if show:
            cv2.destroyAllWindows()

    print(f'\n{s.saved} frames in {s.out}')
    final = s.coverage_state()
    if not final['done']:
        print(f'coverage incomplete -- {final["advice"]}')
        print('solve will still run, but expect a less trustworthy fit')


# ------------------------------------------------------------------------ solve
# k1 k2 p1 p2 k3 [k4 k5 k6]. Ordered simplest first so ties go to fewer params.
MODELS = {
    'zero_tangent': (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, 5, 'k1,k2'),
    'fix_k3': (cv2.CALIB_FIX_K3, 5, 'k1,k2,p1,p2'),
    'plumb_bob': (0, 5, 'k1,k2,p1,p2,k3'),
    'rational': (cv2.CALIB_RATIONAL_MODEL, 8, 'k1..k6,p1,p2'),
}

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)


def collect(paths, adict, board, params, min_corners):
    """-> (objp, imgp, ids, used, size). Shapes are (N,3)/(N,1,2) float32.

    """
    all_obj = board_corners(board)
    objp, imgp, ids, used, size = [], [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f'skip {os.path.basename(p)}: unreadable')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f'skip {os.path.basename(p)}: {gray.shape[::-1]} != {size}')
            continue
        cc, ci = detect_charuco(gray, adict, board, params)
        if cc is None or len(cc) < min_corners:
            print(f'skip {os.path.basename(p)}: '
                  f'{0 if cc is None else len(cc)} corners')
            continue
        objp.append(all_obj[ci.flatten()].reshape(-1, 3).astype(np.float32))
        imgp.append(cc.reshape(-1, 1, 2).astype(np.float32))
        ids.append(ci.flatten())
        used.append(p)
    return objp, imgp, ids, used, size


def view_errors(objp, imgp, K, D):
    """Per-view RMS reprojection error, re-solving each view's own pose.

    Extrinsics are per-view nuisance parameters, so scoring a view the model was
    not fitted on requires solving its pose first -- otherwise the number is
    meaningless. This is what makes the held-out split honest.
    """
    errs = []
    for o, i in zip(objp, imgp):
        ok, rvec, tvec = cv2.solvePnP(o, i, K, D)
        if not ok:
            errs.append(float('inf'))
            continue
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, D)
        errs.append(float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return errs


def fit(objp, imgp, size, flags):
    """calibrateCameraExtended -> (rms, K, D, std_intrinsics, per_view)."""
    rms, K, D, _, _, sdi, _, pve = cv2.calibrateCameraExtended(
        objp, imgp, size, None, None, flags=flags, criteria=CRITERIA)
    return rms, K, D, np.asarray(sdi).ravel(), np.asarray(pve).ravel()


def data_radius(imgp, K):
    """How far from the principal point corner observations actually reached."""
    return max(float(np.hypot(p.reshape(-1, 2)[:, 0] - K[0, 2],
                              p.reshape(-1, 2)[:, 1] - K[1, 2]).max())
               for p in imgp)


def undistort_gap(size, K1, D1, K2, D2, n=60):
    """Per-point undistortion disagreement between two models, over the frame.

    Two models that fit the observed corners equally well can still diverge
    wildly where no corner was ever seen, because out there they are
    extrapolating a polynomial rather than fitting one. Reprojection error
    cannot show this -- it only ever looks at pixels that had data. This can,
    and it is the number that says whether the model choice actually mattered.
    """
    gx, gy = np.meshgrid(np.linspace(0, size[0] - 1, n),
                         np.linspace(0, size[1] - 1, n))
    p = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float64)
    a = cv2.undistortPoints(p, K1, D1, P=K1).reshape(-1, 2)
    b = cv2.undistortPoints(p, K2, D2, P=K2).reshape(-1, 2)
    return np.linalg.norm(a - b, axis=1)


def reject_loop(objp, imgp, ids, used, size, flags, sigma, iters, label):
    """Fit, drop views the model disagrees with, refit. -> fit results + kept views.

    A single mis-detected board -- wrong ids, motion blur, a corner snapped to
    the neighbouring saddle -- can reproject far out and drag the whole
    optimisation into a bad basin, with the distortion terms contorting to
    accommodate it. The threshold is a multiple of the MEDIAN per-view error,
    not the mean, because the mean is itself dragged up by the outliers it is
    meant to find.
    """
    rms, K, D, sdi, pve = fit(objp, imgp, size, flags)
    for it in range(iters):
        thr = max(sigma * float(np.median(pve)), 0.5)
        keep = pve < thr
        if keep.sum() < 8 or keep.all():
            break
        print(f'  {label} pass {it}: rms {rms:.4f}, dropping {(~keep).sum()} '
              f'view(s) over {thr:.2f} px')
        objp = [o for o, k in zip(objp, keep) if k]
        imgp = [i for i, k in zip(imgp, keep) if k]
        ids = [i for i, k in zip(ids, keep) if k]
        used = [u for u, k in zip(used, keep) if k]
        rms, K, D, sdi, pve = fit(objp, imgp, size, flags)
    return rms, K, D, sdi, pve, objp, imgp, ids, used


def split_holdout(n, frac):
    """Deterministic interleaved split, or None when there is too little data.

    Interleaved rather than random: captured views arrive as a sequence sweeping
    the frame, so every k-th view is a spread across the whole capture, whereas
    a random draw can leave a whole region entirely in one side of the split.
    """
    if frac <= 0 or n < 12:
        return None
    step = max(2, int(round(1.0 / frac)))
    held = set(range(0, n, step))
    keep = [i for i in range(n) if i not in held]
    if len(keep) < 8 or len(held) < 3:
        return None
    return keep, sorted(held)


def choose_model(objp, imgp, size, holdout, forced):
    """Pick a distortion model by held-out reprojection error.

    Error on the fitting set falls monotonically as coefficients are added, so
    it cannot choose between models -- rational will always "win" it while
    quietly contorting to fit noise. Held-out error is the one that turns back
    up when the model starts overfitting.
    """
    if forced:
        print(f'model forced to {forced}')
        return forced
    if holdout is None:
        print('too few views for a held-out split; defaulting to plumb_bob')
        return 'plumb_bob'

    keep, held = holdout
    fo, fi = [objp[i] for i in keep], [imgp[i] for i in keep]
    ho, hi = [objp[i] for i in held], [imgp[i] for i in held]
    print(f'\nmodel selection on {len(keep)} fit / {len(held)} held-out views')
    print(f'  {"model":<14}{"params":<14}{"fit rms":>9}{"held-out":>11}')

    scores = {}
    for name, (flags, _, plist) in MODELS.items():
        try:
            rms, K, D, _, _ = fit(fo, fi, size, flags)
        except cv2.error as e:
            print(f'  {name:<14}{plist:<14}   failed: '
                  f'{str(e).strip().splitlines()[-1][:40]}')
            continue
        h = float(np.mean(view_errors(ho, hi, K, D)))
        scores[name] = h
        print(f'  {name:<14}{plist:<14}{rms:>9.4f}{h:>11.4f}')

    if not scores:
        raise RuntimeError('every distortion model failed to fit')
    best = min(scores, key=scores.get)
    # Prefer the simplest model within 2% of the best -- differences that small
    # are noise in the split, and extra coefficients cost stability at the edges
    # where there is least data to constrain them.
    for name in MODELS:
        if name in scores and scores[name] <= scores[best] * 1.02:
            best = name
            break
    print(f'  -> {best}')
    return best


def straightness(imgp, ids, K, D, nx):
    """RMS bow of each board row/column, before and after undistortion, in px.

    A straight 3D line images as a straight line under an ideal pinhole camera,
    and lens distortion is exactly what bends it. Fitting a line to each row of
    corners and measuring the perpendicular residual therefore tests the
    distortion model on its own, without the focal length or the pose entering
    the number at all -- which reprojection error cannot do, since it folds
    every parameter together and a pose error can mask a distortion error.
    """
    def bow(points_per_line):
        r = []
        for pts in points_per_line:
            if len(pts) < 3:
                continue
            c = pts.mean(axis=0)
            # principal direction; residual is the spread across it
            u, s, _ = np.linalg.svd(pts - c)
            r.append(float(np.sqrt(np.mean((u[:, 1] * s[1]) ** 2))))
        return r

    raw, und = [], []
    for ii, pp in zip(ids, imgp):
        p_raw = pp.reshape(-1, 2).astype(np.float64)
        p_und = cv2.undistortPoints(pp, K, D, P=K).reshape(-1, 2)
        for key in (ii // nx, ii % nx):          # rows, then columns
            for v in np.unique(key):
                m = key == v
                if m.sum() >= 3:
                    raw.append(p_raw[m])
                    und.append(p_und[m])
    return bow(raw), bow(und)


def cmd_solve(args):
    if args.frames is None:
        args.frames = latest_calib_run()
        if args.frames is None:
            raise SystemExit(f'no calibration runs with frames under {CALIB_ROOT} -- '
                             f'run `capture` first, or pass a directory')
        # Announced, never silent: solving the wrong frame set produces a
        # perfectly plausible calibration that is wrong in every pose built on it.
        print(f'frames: {args.frames}  (newest run)')
    if args.preview is None:
        args.preview = os.path.join(args.frames, 'undistort_preview.png')

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    board = make_board(adict, (SQUARES_X, SQUARES_Y),
                       SQUARE_LEN, MARKER_LEN)
    params = make_params()
    nx, _ = inner_grid(SQUARES_X, SQUARES_Y)

    # Numbered names only. charuco_board.png and undistort_preview.png live in
    # this same directory, and the rendered board in particular detects corners
    # perfectly -- a bare *.png glob would fold both into the calibration.
    paths = sorted(glob.glob(os.path.join(args.frames, '[0-9]*.png')) +
                   glob.glob(os.path.join(args.frames, '[0-9]*.jpg')))
    if not paths:
        raise RuntimeError(f'no captured frames (0000.png, ...) in {args.frames}')

    objp, imgp, ids, used, size = collect(paths, adict, board, params,
                                          MIN_CORNERS)
    print(f'{len(used)}/{len(paths)} frames usable at {size[0]}x{size[1]}')
    if len(used) < 8:
        raise RuntimeError('need at least ~8 usable views (25-40 is better)')

    # Order matters: clean, THEN choose the model, THEN fit for real.
    #
    # Selecting a distortion model on uncleaned views compares candidates that
    # are all being wrecked by the same handful of bad detections, so the scores
    # collapse together and the "winner" is noise. Rejecting first is done with
    # plumb_bob because outlier detection only needs a model good enough to make
    # a bad view stand out, and the middle of the range is the safe choice --
    # too simple would flag well-fitted edge views, too complex would absorb the
    # outliers it is supposed to expose.
    n_start = len(used)
    rms, K, D, sdi, pve, objp, imgp, ids, used = reject_loop(
        objp, imgp, ids, used, size, MODELS['plumb_bob'][0],
        REJECT_SIGMA, REJECT_ITERS, 'clean')

    model = choose_model(objp, imgp, size, split_holdout(len(used), HOLDOUT),
                         None)
    flags, ncoef, _ = MODELS[model]

    rms, K, D, sdi, pve, objp, imgp, ids, used = reject_loop(
        objp, imgp, ids, used, size, flags,
        REJECT_SIGMA, REJECT_ITERS, model)

    dropped = n_start - len(used)
    if dropped:
        print(f'  dropped {dropped}/{n_start} views as outliers')
        if dropped > n_start * 0.25:
            print('  WARNING: over a quarter of views rejected. Usually motion '
                  'blur or rolling-shutter shear from grabbing frames while the '
                  'board moved. Re-shoot holding the board still.')

    d = np.asarray(D).ravel()[:ncoef]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    s_fx, s_fy, s_cx, s_cy = (float(x) for x in sdi[:4])

    print(f'\nmodel {model}   RMS {rms:.4f} px over {len(used)} views')
    for p, e in sorted(zip(used, pve), key=lambda t: -t[1])[:5]:
        print(f'  worst: {os.path.basename(p)}  {e:.3f} px')
    if rms > 1.0:
        print('  WARNING: rms > 1 px. Usually a few bad detections or a board '
              'that is not flat; re-run without --no-reject, or re-shoot.')

    print(f'\nfx {fx:.2f} +/- {s_fx:.2f}     fy {fy:.2f} +/- {s_fy:.2f}')
    print(f'cx {cx:.2f} +/- {s_cx:.2f}     cy {cy:.2f} +/- {s_cy:.2f}')
    print(f'D  {np.array2string(d, precision=6)}')

    # cx, cy should land near the image centre. A large offset is either a real
    # sensor/lens decentring or, far more often, too little coverage variety --
    # the principal point is the parameter that pose diversity constrains least.
    off_x, off_y = cx - size[0] / 2, cy - size[1] / 2
    print(f'principal point offset from centre: ({off_x:+.1f}, {off_y:+.1f}) px '
          f'= ({off_x / max(s_cx, 1e-6):+.1f}, {off_y / max(s_cy, 1e-6):+.1f}) sigma')
    if abs(off_x) > 4 * max(s_cx, 1e-6) or abs(off_y) > 4 * max(s_cy, 1e-6):
        print('  NOTE: significantly off centre. Plausible, but re-check that '
              'you shot tilted views in all four quadrants before trusting it.')

    aspect = fy / fx
    print(f'aspect fy/fx = {aspect:.5f}')
    if abs(aspect - 1) > 0.01:
        print('  WARNING: >1% from square pixels. Rare on a modern sensor; '
              'usually means the capture path is rescaling the image.')

    fovx = np.degrees(2 * np.arctan(size[0] / (2 * fx)))
    fovy = np.degrees(2 * np.arctan(size[1] / (2 * fy)))
    print(f'FOV {fovx:.1f} deg horizontal, {fovy:.1f} deg vertical')

    # How far out the data actually reached. Distortion grows with radius, so
    # everything past this is the polynomial extrapolating rather than fitting,
    # and reprojection error says nothing about it -- it only scores pixels that
    # had observations. This is the single easiest way to get a calibration with
    # a beautiful rms that is badly wrong in the frame corners.
    r_data = data_radius(imgp, K)
    r_corner = float(np.hypot(max(cx, size[0] - cx), max(cy, size[1] - cy)))
    frac = r_data / r_corner
    print(f'\ncorner data reaches r={r_data:.0f} px of the r={r_corner:.0f} px '
          f'frame corner ({100 * frac:.0f}%)')

    # Quantify what the extrapolation is worth: refit the most flexible model on
    # the same views and see where the two disagree. Small gap -> the data
    # pinned the model down. Large gap -> the corners are guesswork.
    gap = None
    if model != 'rational':
        try:
            _, Kr, Dr, _, _ = fit(objp, imgp, size, MODELS['rational'][0])
            g = undistort_gap(size, K, d, Kr, np.asarray(Dr).ravel()[:8])
            gap = (float(np.median(g)), float(np.percentile(g, 95)), float(g.max()))
            print(f'disagreement with the 8-term model across the frame: '
                  f'median {gap[0]:.2f} px, p95 {gap[1]:.2f} px, max {gap[2]:.2f} px')
        except cv2.error:
            pass
    if frac < 0.9:
        print('  WARNING: no observations near the frame corners, so distortion '
              'there is extrapolated and unconstrained. Re-shoot with the board '
              'pushed right into all four corners, including partial views.')
        if gap and gap[2] > 5:
            print(f'  This is not hypothetical here: two models that fit these '
                  f'views equally well disagree by {gap[2]:.0f} px at the edges.')

    raw_bow, und_bow = straightness(imgp, ids, K, d, nx)
    if raw_bow and und_bow:
        print(f'\nline straightness over {len(und_bow)} board rows/columns:')
        print(f'  before undistortion  mean {np.mean(raw_bow):.3f} px, '
              f'worst {np.max(raw_bow):.3f} px')
        print(f'  after  undistortion  mean {np.mean(und_bow):.3f} px, '
              f'worst {np.max(und_bow):.3f} px')
        if np.mean(und_bow) > 0.5:
            print('  WARNING: rows are still visibly bowed after undistortion -- '
                  'the distortion model is not capturing this lens. Try '
                  '--model rational.')

    holdout = split_holdout(len(used), HOLDOUT)
    ho_err = float('nan')
    if holdout is not None:
        keep, held = holdout
        r2, K2, D2, _, _ = fit([objp[i] for i in keep], [imgp[i] for i in keep],
                               size, flags)
        ho_err = float(np.mean(view_errors([objp[i] for i in held],
                                           [imgp[i] for i in held], K2, D2)))
        print(f'\nheld-out reprojection error {ho_err:.4f} px '
              f'(fit {r2:.4f} px on {len(keep)} views)')
        if ho_err > 2 * r2:
            print('  WARNING: held-out error is far above fit error, so the model '
                  'is memorising these views rather than describing the lens. '
                  'Shoot more variety, or force a simpler --model.')

    ctrl_path = os.path.join(args.frames, 'locked_controls.json')
    locked = {}
    if os.path.exists(ctrl_path):
        with open(ctrl_path) as f:
            locked = json.load(f)
        print(f'\nlocked controls at capture time: '
              + ', '.join(f'{k}={locked[k]}' for k in GEOMETRY_CTRLS if k in locked))
    else:
        print(f'\nNOTE: no {ctrl_path}; the output json will not record what '
              'focus/zoom this calibration is valid for.')

    out = {
        'intrinsic_type': 'PINHOLE',
        'distortion_model': model,
        # Which frames produced this. Nothing else in the file identifies the
        # run -- locked_controls and nr_calib_images are the closest and neither
        # names it -- so without this the UI cannot say which capture the active
        # calibration came from, and neither can you six months from now.
        'source_run': os.path.basename(os.path.normpath(args.frames)),
        'solved_at': datetime.now().isoformat(timespec='seconds'),
        # k and d are what video_processor_charuco.py's load_intrinsics() reads,
        # as a sensor_msgs/CameraInfo-shaped dict. Keeping this shape means that
        # file needs no changes at all.
        'k': [float(v) for v in K.ravel()],
        'd': [float(v) for v in d],
        'image_width': int(size[0]),
        'image_height': int(size[1]),
        'fps': float(FPS),
        'final_reproj_error': float(rms),
        'holdout_reproj_error': ho_err,
        'nr_calib_images': len(used),
        'std_dev': {'fx': s_fx, 'fy': s_fy, 'cx': s_cx, 'cy': s_cy},
        'straightness_px': {
            'raw_mean': float(np.mean(raw_bow)) if raw_bow else None,
            'undistorted_mean': float(np.mean(und_bow)) if und_bow else None,
        },
        'fov_deg': {'horizontal': float(fovx), 'vertical': float(fovy)},
        'locked_controls': locked,
        'board': {'dict': BOARD_DICT, 'squares': [SQUARES_X, SQUARES_Y],
                  'square_len_m': SQUARE_LEN, 'marker_len_m': MARKER_LEN},
        # Human-readable restatement of K. Nothing parses this block.
        'intrinsics': {
            'aspect_ratio': float(aspect),
            'focal_length': float(fx),
            'focal_length_y': float(fy),
            'principal_pt_x': float(cx),
            'principal_pt_y': float(cy),
            'skew': 0.0,
        },
        'stabelized': False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=4)
    print('\nwrote', args.out)

    # Archive a copy beside the frames it was solved from. A plain copy rather
    # than a symlink: symlinks do not survive a Docker bind mount, and the stable
    # path is what verify.py and build_zarr.py read.
    archive = os.path.join(args.frames, 'scene_intrinsics.json')
    if os.path.abspath(archive) != os.path.abspath(args.out):
        shutil.copyfile(args.out, archive)
        print('archived', archive)

    if args.preview:
        img = cv2.imread(used[0])
        newK, _ = cv2.getOptimalNewCameraMatrix(K, d, size, ALPHA, size)
        und = cv2.undistort(img, K, d, None, newK)
        cv2.imwrite(args.preview, np.hstack([img, und]))
        print('wrote', args.preview, '(raw | undistorted)')
        print('  check the frame edges: straight edges in the scene should be '
              'straight in the right half.')


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    lk = sub.add_parser('lock', help='freeze focus/zoom/exposure and verify')
    lk.add_argument('-d', '--device', default=DEVICE)
    lk.add_argument('--focus', type=int, default=0,
                    help='0-1023. 0 is focus-at-infinity on this UVC camera; '
                         'raise it if the board sits close and looks soft, then '
                         'keep the SAME value for recording (default: 0)')
    lk.add_argument('--zoom', type=int, default=100,
                    help='100-200 digital zoom; leave at 100 (default: 100)')
    lk.add_argument('--exposure', type=int, default=800, metavar='UNITS',
                    help='nominally UVC 100us ticks, but this camera does NOT '
                         'honour that linearly -- brightness is not proportional '
                         'to the setting and there is a step change around 200. '
                         'Treat it as a dial, not milliseconds (default: 800). '
                         'Far longer than the 5 ms capture.py records at, which '
                         'is correct: the board is held still for calibration so '
                         'there is no motion blur to buy down, and the frame rate '
                         'does NOT drop with exposure (120 fps even at the top of '
                         'the range). Only focus and zoom must match at recording '
                         'time -- exposure has no effect on geometry')
    lk.add_argument('--wb', type=int, default=4600, metavar='K')
    lk.add_argument('--gamma', type=int, default=128,
                    help='0-255, default 128 (the sensor default, a near-neutral '
                         'tone curve). Prefer buying brightness with --exposure: '
                         'gamma is applied after digitisation so it amplifies '
                         'noise, while real photons improve SNR. Measured at '
                         'matched brightness, exposure 800/gamma 128 gives half '
                         'the temporal noise of exposure 200/gamma 160')
    lk.add_argument('--gain', type=int, default=100,
                    help='0-1023, but weak on this camera above ~200 (mean 6.4 at '
                         'gain 200 vs 6.5 at gain 1023). Exposure is the lever '
                         'that actually works')
    lk.set_defaults(func=cmd_lock)

    b = sub.add_parser('board', help='render the printable board')
    b.add_argument('--out', default=None,
                   help='default: charuco_board.png in the newest run directory')
    b.set_defaults(func=cmd_board)

    c = sub.add_parser('capture', help='live capture with coverage meters')
    c.add_argument('-d', '--device', default=DEVICE)
    c.add_argument('--out', default=None,
                   help='frame directory (default: a fresh data/calibration/<datetime>/)')
    c.add_argument('--target', type=int, default=60,
                   help='corner hits per grid cell before it turns green')
    c.add_argument('--no-display', action='store_true',
                   help='headless: no window, frames are kept automatically and the '
                        'run ends once coverage is complete or --target is reached')
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser('solve', help='fit intrinsics over captured frames')
    s.add_argument('frames', nargs='?', default=None,
                   help='frame directory (default: the newest run under '
                        'data/calibration/, announced when chosen)')
    s.add_argument('--out', default=DEFAULT_INTRINSICS,
                   help=f'default: {DEFAULT_INTRINSICS}. A copy is always archived '
                        f'in the run directory alongside the frames.')
    s.add_argument('--preview', default=None,
                   help='default: undistort_preview.png in the run directory')
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
