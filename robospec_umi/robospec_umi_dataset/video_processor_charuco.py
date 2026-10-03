#!/usr/bin/env python3
"""Offline TCP-pose analysis for a recorded handheld-gripper video (ChArUco board).

Point it at a folder holding one .mp4 and it writes back into that folder:
  <stem>_tcp_pose.csv        per-frame board-centre and TCP pose
  <stem>_annotated.mp4       the video with both frames drawn on it
  <stem>_charuco_dets.npz    raw detections (cache, so re-runs are fast)

Pipeline:
  1. detect the board in every frame (cached): ArUco markers first, then the
     chessboard corners interpolated between them
  2. solve one board pose per frame from every visible chessboard corner at once
  3. move the origin from the board's corner to the board centre
  4. apply the fixed centre->TCP transform (translation, then body-fixed rotations)

Why chessboard corners and not the marker corners: a saddle point is localised
from the four intensity quadrants around it and refines to a fraction of a pixel,
whereas a marker corner is the intersection of two edges that printing and motion
blur both round off. Marker corners are the fallback only, for frames where too
few chessboard corners survive -- see solve_board().

The board is planar, so PnP has the usual two-fold depth-flip ambiguity when it is
seen small and near fronto-parallel. IPPE picks the lower-error branch, and the
previous frame's pose seeds the refinement to keep the track on one branch.

This file is deliberately standalone -- stdlib, cv2 and numpy only. episode_prep.py
imports it for the solver and the seed-poisoning guards in track(); it is also
runnable on its own for one-off analysis of a recorded folder.

Intrinsics default to the calibration written by
robospec_umi_calibration/calibrate_scene_cam.py (intrinsic_type PINHOLE). A plain
sensor_msgs/CameraInfo k/d dict and bare fx/fy/cx/cy json are both accepted, and
--intrinsics "" falls back to the built-in RealSense table.

Usage:
    python3 robospec_umi/robospec_umi_dataset/video_processor_charuco.py <folder>
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

CV_VER = tuple(int(v) for v in cv2.__version__.split('.')[:2])

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_dataset
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

# RealSense D435 colour intrinsics, keyed by (width, height). Both streams report
# plumb_bob with all-zero distortion. Kept only as a fallback for footage that
# arrived without a calibration file.
INTRINSICS = {
    (1920, 1080): (1363.149658203125, 1362.5545654296875,
                   970.6884155273438, 547.5191040039062),
    (960, 540): (681.5748291015625, 681.2772827148438,
                 485.3442077636719, 273.7595520019531),
}

# Written by robospec_umi_calibration/calibrate_scene_cam.py solve.
DEFAULT_INTRINSICS = os.path.join(DATA, 'calibration', 'scene_intrinsics.json')


# --------------------------------------------------------------------- camera
class Camera:
    """A rectilinear (pinhole) camera with rad-tan or rational distortion.

    Pose solving happens in undistorted pixel coordinates: image points are
    rectified first and solvePnP is then handed a zero D, which keeps the solve
    independent of how many distortion coefficients the calibration carries (5
    for plumb_bob, 8 for the rational model).
    """

    def __init__(self, K, D, label=''):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        self.label = label
        # what solvePnP is handed once the points are already rectified
        self.zero = np.zeros((1, 5))

    def undistort(self, pts):
        """Raw image points (N,2) -> pinhole pixel coordinates (N,2)."""
        p = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.D, P=self.K).reshape(-1, 2)

    def project(self, objp, rvec, tvec):
        """Object points (N,3) -> raw image points (N,2), plus a validity mask.

        Points behind or far outside the lens can come back as huge or
        non-finite coordinates; the mask lets callers drop them rather than
        hand int32 garbage to the drawing routines.
        """
        objp = np.ascontiguousarray(objp, dtype=np.float64).reshape(-1, 1, 3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        p, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.D)
        p = p.reshape(-1, 2)
        good = np.isfinite(p).all(axis=1) & (np.abs(p) < 1e5).all(axis=1)
        return p, good

    def polyline(self, objp, rvec, tvec, img, colour, thick):
        """Project a sampled 3D polyline and stroke it.

        The sampling is load-bearing, not a leftover. The instinct is that a
        rectilinear camera images straight lines as straight lines, making this
        unnecessary -- but that is only true of the ideal pinhole projection.
        Radial distortion still bends them, by up to ~105 px at the frame corner
        on this lens, so projecting two endpoints and connecting them puts the
        line visibly off the geometry it is meant to trace.
        """
        p, good = self.project(objp, rvec, tvec)
        if good.sum() < 2:
            return None
        pts = p[good].astype(np.int32)
        cv2.polylines(img, [pts], False, colour, thick, cv2.LINE_AA)
        return pts


# ---------------------------------------------------------------- SE(3) utils
def rot_axis(axis, deg):
    return cv2.Rodrigues(np.deg2rad(deg) * np.eye(3)['xyz'.index(axis)])[0]


def parse_rotation(spec):
    """['x', '180', 'z', '90'] -> one rotation matrix.

    Each turn is body-fixed (intrinsic) and they compose left to right, so
    `x 180 z 90` means "flip about the centre frame's X, then spin 90 deg about
    the X-flipped frame's own Z" -- the same convention the cube script used for
    its face mountings.
    """
    if len(spec) % 2:
        sys.exit(f'--tcp-rotation takes AXIS DEG pairs, got {len(spec)} values: {spec}')
    R = np.eye(3)
    for axis, deg in zip(spec[::2], spec[1::2]):
        if axis.lower() not in 'xyz':
            sys.exit(f'--tcp-rotation axis must be x, y or z, got {axis!r}')
        R = R @ rot_axis(axis.lower(), float(deg))
    return R


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T


def T_from_rt(rvec, tvec):
    return make_T(cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0], tvec)


def rt_from_T(T):
    return cv2.Rodrigues(T[:3, :3])[0].flatten(), T[:3, 3].copy()


def quat_from_R(R):
    """(x, y, z, w) from a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k]))
    q = np.zeros(4)
    q[i], q[j], q[k] = s / 2, (R[j, i] + R[i, j]) / (2 * s), (R[k, i] + R[i, k]) / (2 * s)
    q[3] = (R[k, j] - R[j, k]) / (2 * s)
    return q


# ------------------------------------------------------------------- geometry
def make_board(size, square, marker, dict_name, legacy=False):
    """Build the board on whichever aruco API this OpenCV has.

    Dispatch on the version, never try/except: on 4.6 the 4.7-style
    CharucoBoard(...) constructor does not raise. It accepts the arguments,
    hands back an object, and segfaults on the first attribute access -- so the
    except clause never runs and the process dies instead.
    """
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if CV_VER < (4, 7):
        board = cv2.aruco.CharucoBoard_create(size[0], size[1], square, marker, adict)
    else:
        board = cv2.aruco.CharucoBoard(tuple(size), square, marker, adict)
    if hasattr(board, 'setLegacyPattern'):
        board.setLegacyPattern(legacy)
    return board


def board_geometry(board):
    """(chessboard corners (N,3), {marker id: its 4 corners (4,3)}), both in board frame."""
    try:
        chess, ids, objp = (board.getChessboardCorners(), board.getIds(), board.getObjPoints())
    except AttributeError:  # cv2 < 4.7
        chess, ids, objp = board.chessboardCorners, board.ids, board.objPoints
    markers = {int(i): np.asarray(p, dtype=np.float64).reshape(4, 3)
               for i, p in zip(np.asarray(ids).flatten(), objp)}
    return np.asarray(chess, dtype=np.float64).reshape(-1, 3), markers


def board_centre(chess):
    """Board frame -> centre frame. The interior corners are symmetric about the
    board centre, so their mean is it exactly; orientation is left alone."""
    return make_T(np.eye(3), chess.mean(axis=0))


# ------------------------------------------------------------------ detection
def make_detector(board):
    """Return detect(gray) -> (markers {id: (4,2)}, charuco corners (N,2), ids (N,)).

    cv2 >= 4.7 has CharucoDetector; 4.6 only has the free functions, and the host
    python3 still ships 4.6, so both paths are kept alive.
    """
    # Same 4.6 trap as make_board: the DetectorParameters symbol exists on 4.6,
    # so hasattr is true, and calling it builds an object that segfaults on use.
    params = (cv2.aruco.DetectorParameters_create() if CV_VER < (4, 7)
              else cv2.aruco.DetectorParameters())
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Widened thresholding window: markers seen at a steep angle or far from the
    # optical axis shrink to a few sheared pixels.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 8
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.05

    def pack(mc, mi, cc, ci):
        markers = {}
        if mi is not None and len(mi):
            for c, m in zip(mc, np.asarray(mi).flatten()):
                markers[int(m)] = np.asarray(c, dtype=np.float64).reshape(4, 2)
        if ci is None or len(ci) == 0:
            return markers, None, None
        return (markers, np.asarray(cc, dtype=np.float64).reshape(-1, 2),
                np.asarray(ci).flatten().astype(np.int64))

    if hasattr(cv2.aruco, 'CharucoDetector'):
        detector = cv2.aruco.CharucoDetector(board)
        if hasattr(detector, 'setDetectorParameters'):
            detector.setDetectorParameters(params)

        def detect(gray):
            cc, ci, mc, mi = detector.detectBoard(gray)
            return pack(mc, mi, cc, ci)
        return detect

    adict = board.dictionary

    def detect(gray):
        mc, mi, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
        if mi is None or len(mi) == 0:
            return {}, None, None
        _, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
        return pack(mc, mi, cc, ci)
    return detect


def pick_legacy(path, size, square, marker, dict_name, n_probe=20):
    """Boards drawn by cv2 < 4.6 use the opposite black/white phase; guessing wrong
    costs most of the corners. Try both on a few frames and keep the better one."""
    if not hasattr(cv2.aruco.CharucoBoard, 'setLegacyPattern'):
        print(f'board pattern: cv2 {cv2.__version__} cannot switch it, using the built-in one')
        return False
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0, max(n - 1, 0), min(n_probe, max(n, 1)), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()

    score = {}
    for legacy in (False, True):
        detect = make_detector(make_board(size, square, marker, dict_name, legacy))
        score[legacy] = sum(0 if ci is None else len(ci) for _, _, ci in map(detect, frames))
    best = max(score, key=score.get)
    print(f'board pattern: legacy={best}  '
          f'(corners over {len(frames)} probe frames: '
          f'modern {score[False]}, legacy {score[True]})')
    return best


def detect_video(path, detect, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f'loaded detections from {cache}')
        return list(d['dets']), int(d['n_frames'])

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f'cannot open {path}')
    dets, idx = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        markers, cc, ci = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        dets.append({'markers': markers, 'corners': cc, 'ids': ci})
        idx += 1
        if idx % 100 == 0:
            print(f'  detected {idx} frames ...', end='\r', flush=True)
    cap.release()
    n_ch = sum(1 for d in dets if d['ids'] is not None)
    print(f'detected {idx} frames, {sum(1 for d in dets if d["markers"])} with >=1 marker, '
          f'{n_ch} with >=1 chessboard corner')
    if cache:
        np.savez_compressed(cache, dets=np.array(dets, dtype=object), n_frames=idx)
    return dets, idx


# ------------------------------------------------------------------- solvers
def well_spread(obj, ratio=0.1):
    """IPPE needs a 2D spread; a single visible row of corners is collinear and
    gives a pose whose tilt is pure noise."""
    p = obj[:, :2] - obj[:, :2].mean(axis=0)
    s = np.linalg.svd(p, compute_uv=False)
    return s[0] > 0 and s[-1] > ratio * s[0]


def observations(det, chess, markers, source):
    """(object points, image points, ids) for one frame, or None."""
    if source != 'markers' and det['ids'] is not None and len(det['ids']) >= 4:
        obj = chess[det['ids']]
        if well_spread(obj):
            return obj, det['corners'], det['ids'], 'charuco'
    vis = sorted(m for m in det['markers'] if m in markers)
    if len(vis) >= 2:
        obj = np.concatenate([markers[m] for m in vis])
        img = np.concatenate([det['markers'][m] for m in vis])
        if well_spread(obj):
            return obj, img, np.array(vis), 'markers'
    return None


def solve_board(det, chess, markers, cam, guess=None, source='auto'):
    """Joint PnP over one frame -> (T_cam_board, rms_px, n_points, ids, source).

    Every solver call runs on rectified points with zero distortion, which keeps
    it independent of how many coefficients the calibration carries. rms is
    reported back in the raw image, through the full distortion model, because
    that is the number the annotated video and the CSV are read against.
    """
    obs = observations(det, chess, markers, source)
    if obs is None:
        return None, np.nan, 0, [], ''
    obj, img, ids, src = obs
    obj = np.ascontiguousarray(obj, dtype=np.float64)
    img = np.ascontiguousarray(img, dtype=np.float64)
    und = cam.undistort(img)

    if guess is None:
        ok, rvec, tvec = cv2.solvePnP(obj, und, cam.K, cam.zero,
                                      flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None, np.nan, 0, [], ''
    else:
        rvec, tvec = rt_from_T(guess)
        rvec, tvec = rvec.reshape(3, 1), tvec.reshape(3, 1)
    ok, rvec, tvec = cv2.solvePnP(obj, und, cam.K, cam.zero,
                                  rvec.reshape(3, 1), tvec.reshape(3, 1),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, np.nan, 0, [], ''
    rvec, tvec = cv2.solvePnPRefineVVS(obj, und, cam.K, cam.zero, rvec, tvec)

    proj, good = cam.project(obj, rvec, tvec)
    if good.sum() == 0:
        return None, np.nan, 0, [], ''
    rms = float(np.sqrt(np.mean(
        np.sum((proj[good] - img[good]) ** 2, axis=1))))
    return T_from_rt(rvec, tvec), rms, len(obj), ids.tolist(), src


# ------------------------------------------------------------------- tracking
def track(dets, chess, markers, cam, T_centre, T_centre_tcp, source='auto',
          max_rms=5.0):
    """Per-frame (T_cam_centre, T_cam_tcp, n_points, ids, source, rms); None where lost.

    The previous pose seeds the next solve, which speeds up the refinement and
    keeps the planar flip ambiguity from toggling between frames.

    The seed is also the failure mode. A seeded solve that walks off to a wrong
    minimum still *returns* a pose, so the old `T is None` fallback never fired
    and the bad pose became the next frame's seed -- on this footage one bad
    frame took the whole remaining track with it, 97 frames of it, out to 1e13
    metres. So: score every solve, retry from scratch when the seeded one looks
    bad, keep whichever reprojects better, and refuse to seed from a pose that
    failed. Bad frames stay local instead of ending the track.
    """
    rows, prev, rejected = [], None, 0
    for d in dets:
        T, rms, n, ids, src = solve_board(d, chess, markers, cam, guess=prev,
                                          source=source)
        bad = T is None or not np.isfinite(rms) or rms > max_rms
        if bad and prev is not None:
            T2, rms2, n2, ids2, src2 = solve_board(d, chess, markers, cam,
                                                   source=source)
            if T2 is not None and np.isfinite(rms2) and (T is None or rms2 < rms):
                T, rms, n, ids, src = T2, rms2, n2, ids2, src2
                bad = rms > max_rms
        if bad:
            rejected += T is not None
            rows.append((None, None, 0, [], '', rms if T is not None else np.nan))
            prev = None
            continue
        prev = T
        T_cam_centre = T @ T_centre
        rows.append((T_cam_centre, T_cam_centre @ T_centre_tcp, n, ids, src, rms))
    track.rejected = rejected
    return rows


def jitter(positions):
    """Median second difference (mm): true hand motion is smooth, noise is not."""
    acc = []
    for a, b, c in zip(positions, positions[1:], positions[2:]):
        if a is not None and b is not None and c is not None:
            acc.append(np.linalg.norm(np.asarray(a) - 2 * np.asarray(b) + np.asarray(c)) * 1000)
    return float(np.median(acc)) if acc else np.nan


# --------------------------------------------------------------------- inputs
def find_video(folder):
    if not os.path.isdir(folder):
        sys.exit(f'{folder} is not a folder -- pass the folder holding the .mp4')
    vids = [v for v in sorted(glob.glob(os.path.join(folder, '*.mp4')))
            if not v.endswith('_annotated.mp4')]
    if not vids:
        sys.exit(f'no .mp4 found in {folder}')
    if len(vids) > 1:
        sys.exit(f'{len(vids)} .mp4 files in {folder}, expected one:\n  ' +
                 '\n  '.join(os.path.basename(v) for v in vids))
    return vids[0]


def video_props(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f'cannot open {path}')
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def scale_K(K, src, dst):
    """Rescale a camera matrix between resolutions. D is dimensionless."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def load_intrinsics(path, width, height):
    """-> Camera. Accepts a CameraInfo-shaped k/d dict or bare fx/fy/cx/cy."""
    if path:
        with open(path) as f:
            d = json.load(f)
        if d.get('intrinsic_type') == 'FISHEYE':
            sys.exit(f'{path} is a FISHEYE calibration. This pipeline is '
                     f'rectilinear only -- the two projection models are not '
                     f'interchangeable, and feeding Kannala-Brandt coefficients '
                     f'to a pinhole solver returns plausible, wrong poses with '
                     f'no error anywhere downstream.')
        if 'k' in d:  # a dumped sensor_msgs/CameraInfo
            K = np.array(d['k'], dtype=np.float64).reshape(3, 3)
            D = np.array(d.get('d', [0] * 5), dtype=np.float64)
        else:
            K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]])
            D = np.array(d.get('dist', [0] * 5), dtype=np.float64)
        # Rescale if the calibration was shot at another resolution. K scales with
        # resolution while D is dimensionless, so a 960x540 video solved against a
        # 1920x1080 K is wrong by a factor of two in depth -- and silently so,
        # since reprojection error cannot see a pure scale error.
        src = (d.get('image_width'), d.get('image_height'))
        if None not in src and tuple(src) != (width, height):
            print(f'  calibration is {src[0]}x{src[1]}, video is '
                  f'{width}x{height} -- rescaling K')
            K = scale_K(K, src, (width, height))
        print(f'intrinsics from {path}: pinhole fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
              f'cx={K[0,2]:.1f} cy={K[1,2]:.1f}')
        return Camera(K, D, label=f'pinhole {os.path.basename(path)}')

    if (width, height) not in INTRINSICS:
        sys.exit(f'no built-in intrinsics for {width}x{height} '
                 f'(have {", ".join(f"{w}x{h}" for w, h in INTRINSICS)}); '
                 f'pass --intrinsics with a calibration json')
    fx, fy, cx, cy = INTRINSICS[(width, height)]
    print(f'intrinsics for {width}x{height}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}'
          f'  [RealSense D435, borrowed -- not this camera]')
    return Camera(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                  np.zeros(5), label='RealSense D435 (borrowed)')


# -------------------------------------------------------------------- outputs
AXIS_COLOURS = ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0)))


def draw_axes(img, cam, T, length, thick, n=16):
    """Frame axes as projected polylines, not cv2.drawFrameAxes.

    drawFrameAxes projects the two endpoints and joins them with a straight
    segment, ignoring distortion between them -- off by up to ~105 px at the
    frame corner on this lens. Sampling the axis and projecting every sample
    keeps it on the right pixels.
    """
    rvec, tvec = rt_from_T(T)
    for axis, colour in AXIS_COLOURS:
        pts = np.zeros((n, 3))
        pts[:, axis] = np.linspace(0, length, n)
        cam.polyline(pts, rvec, tvec, img, colour, thick)


def draw_frame(img, det, row, cam):
    """Board-centre axes, TCP axes, and the arm between them."""
    if det['markers']:
        ids = sorted(det['markers'])
        cv2.aruco.drawDetectedMarkers(
            img, [det['markers'][m].reshape(1, 4, 2).astype(np.float32) for m in ids],
            np.array(ids).reshape(-1, 1))
    if det['ids'] is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            img, det['corners'].reshape(-1, 1, 2).astype(np.float32),
            det['ids'].reshape(-1, 1).astype(np.int32), (0, 255, 0))

    T_centre, T_tcp, n, ids, src, rms = row
    if T_centre is None:
        lines, colour = ['no board visible'], (0, 0, 255)
    else:
        for T, length, thick in ((T_centre, 0.030, 3), (T_tcp, 0.060, 5)):
            draw_axes(img, cam, T, length, thick)
        # The arm is a straight 3D segment, so it curves too -- sample it.
        arm = np.linspace(T_centre[:3, 3], T_tcp[:3, 3], 16)
        pts = cam.polyline(arm, np.zeros(3), np.zeros(3), img, (255, 0, 255), 2)
        if pts is not None:
            a, b = pts[0], pts[-1]
            cv2.circle(img, tuple(a), 5, (255, 0, 255), -1)
            cv2.putText(img, 'TCP', tuple(b + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        c = T_centre[:3, 3]
        lines = [
            f'n={n} {src} [{" ".join(str(i) for i in ids)}] | rms={rms:.2f} px',
            f'board [{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}] m  d={np.linalg.norm(c) * 1000:.0f} mm',
            f'tcp   [{T_tcp[0, 3]:+.3f} {T_tcp[1, 3]:+.3f} {T_tcp[2, 3]:+.3f}] m',
        ]
        colour = (0, 255, 255)

    for i, line in enumerate(lines):
        for thick, col in ((5, (0, 0, 0)), (2, colour)):
            cv2.putText(img, line, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, col, thick, cv2.LINE_AA)
    return img


def write_video(src, dst, dets, rows, cam, fps):
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # mp4v keeps the timebase as 1/fps with a 16-bit denominator and cv2 hands it
    # round(fps * 1000), so a raw 119.323 fps overflows it and the writer silently
    # never opens. Rounding to 2 dp keeps any sane frame rate inside the limit --
    # same trim record_video.py makes before opening its writer.
    fps = round(fps, 2)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not out.isOpened():
        sys.exit(f'cannot open a VideoWriter for {dst} at {fps} fps, {w}x{h}')
    idx = 0
    while idx < len(rows):
        ok, img = cap.read()
        if not ok:
            break
        out.write(draw_frame(img, dets[idx], rows[idx], cam))
        idx += 1
        if idx % 100 == 0:
            print(f'  rendered {idx}/{len(rows)} frames ...', end='\r', flush=True)
    cap.release()
    out.release()
    print(f'wrote {dst}                      ')


def write_csv(path, rows, fps):
    with open(path, 'w') as f:
        f.write('frame,time_s,n_points,source,ids,'
                'board_tx,board_ty,board_tz,board_qx,board_qy,board_qz,board_qw,'
                'tcp_tx,tcp_ty,tcp_tz,tcp_qx,tcp_qy,tcp_qz,tcp_qw,reproj_rms_px\n')
        for i, (T_centre, T_tcp, n, ids, src, rms) in enumerate(rows):
            vals = ([np.nan] * 14 if T_centre is None else
                    [*T_centre[:3, 3], *quat_from_R(T_centre[:3, :3]),
                     *T_tcp[:3, 3], *quat_from_R(T_tcp[:3, :3])])
            f.write(f'{i},{i / fps:.6f},{n},{src},{"|".join(str(m) for m in ids)},'
                    + ','.join(f'{v:.6f}' for v in vals) + f',{rms:.6f}\n')
    print(f'wrote {path}')


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='folder containing exactly one .mp4')
    ap.add_argument('--board-size', type=int, nargs=2, default=[4, 4],
                    metavar=('NX', 'NY'), help='board size in squares')
    ap.add_argument('--square-size', type=float, default=0.020, help='checker side (m)')
    ap.add_argument('--marker-size', type=float, default=0.015, help='marker side (m)')
    ap.add_argument('--dict', default='DICT_4X4_50', help='cv2.aruco dictionary name')
    ap.add_argument('--legacy-pattern', choices=['auto', 'yes', 'no'], default='auto',
                    help='black/white phase of the printed board (default: probe the video)')
    ap.add_argument('--tcp-offset', type=float, nargs=3, default=[0.0, -0.155, 0.0725],
                    metavar=('X', 'Y', 'Z'), help='board centre -> TCP translation (m)')
    ap.add_argument('--tcp-rotation', nargs='*', default=['x', '180', 'z', '90'],
                    metavar='AXIS DEG', help='body-fixed turns applied after the translation')
    ap.add_argument('--intrinsics', default=DEFAULT_INTRINSICS,
                    help='calibration json from calibrate_scene_cam.py solve: '
                         'CameraInfo k/d, or fx/fy/cx/cy. Pass "" for the '
                         'built-in RealSense fallback.')
    ap.add_argument('--max-rms', type=float, default=5.0,
                    help='drop poses reprojecting worse than this (px)')
    ap.add_argument('--no-video', action='store_true', help='skip the annotated mp4')
    args = ap.parse_args()

    video = find_video(args.folder)
    out_dir = os.path.abspath(args.folder)
    stem = os.path.splitext(os.path.basename(video))[0]
    w, h, fps, n_declared = video_props(video)
    print(f'{video}\n  {w}x{h}  {fps:.2f} fps  {n_declared} frames  '
          f'{n_declared / fps:.1f} s')

    cam = load_intrinsics(args.intrinsics, w, h)

    legacy = (pick_legacy(video, args.board_size, args.square_size, args.marker_size, args.dict)
              if args.legacy_pattern == 'auto' else args.legacy_pattern == 'yes')
    board = make_board(args.board_size, args.square_size, args.marker_size, args.dict, legacy)
    chess, markers = board_geometry(board)
    T_centre = board_centre(chess)
    # Trans(t) @ Rot(R) is just [R | t], so the translation stays in the centre
    # frame and the turns only reorient the TCP axes about that point
    T_centre_tcp = make_T(parse_rotation(args.tcp_rotation), args.tcp_offset)
    print(f'board: {args.board_size[0]}x{args.board_size[1]} squares of '
          f'{args.square_size * 1000:.1f} mm, markers {args.marker_size * 1000:.1f} mm, '
          f'{args.dict} ids {sorted(markers)}')
    print(f'  {len(chess)} interior corners spanning '
          f'{(chess[:, 0].max() - chess[:, 0].min()) * 1000:.0f} x '
          f'{(chess[:, 1].max() - chess[:, 1].min()) * 1000:.0f} mm; '
          f'centre at [{T_centre[0, 3] * 1000:+.1f} {T_centre[1, 3] * 1000:+.1f} '
          f'{T_centre[2, 3] * 1000:+.1f}] mm in the board frame')
    print(f'  centre -> TCP: [{args.tcp_offset[0] * 1000:+.1f} '
          f'{args.tcp_offset[1] * 1000:+.1f} {args.tcp_offset[2] * 1000:+.1f}] mm then '
          f'{" ".join(args.tcp_rotation) if args.tcp_rotation else "no rotation"}')
    R_tcp = T_centre_tcp[:3, :3]
    print('    TCP axes in the centre frame: ' + '  '.join(
        f'{ax}->[{R_tcp[0, i]:+.0f} {R_tcp[1, i]:+.0f} {R_tcp[2, i]:+.0f}]'
        for i, ax in enumerate('XYZ')))

    detect = make_detector(board)
    dets, n_frames = detect_video(video, detect,
                                  os.path.join(out_dir, f'{stem}_charuco_dets.npz'))

    seen = Counter(m for d in dets for m in d['markers'])
    print(f'\nmarker ids seen: {dict(sorted(seen.items()))}   '
          f'(not on the board: {sorted(m for m in seen if m not in markers)})')

    rows = track(dets, chess, markers, cam, T_centre, T_centre_tcp,
                 max_rms=args.max_rms)
    n_rejected = track.rejected
    mrows = track(dets, chess, markers, cam, T_centre, T_centre_tcp,
                  source='markers', max_rms=args.max_rms)

    # ---- per-run metrics -------------------------------------------------
    dt = 1.0 / fps
    tracked = [r for r in rows if r[0] is not None]
    hist = Counter(r[4] for r in rows if r[0] is not None)
    rms = np.array([r[5] for r in rows if np.isfinite(r[5])])

    print(f'\n=== run metrics: {os.path.basename(args.folder.rstrip("/"))} ===')
    print(f'  {w}x{h} @ {fps:.2f} fps, {n_frames} frames, {n_frames / fps:.1f} s')
    print(f'  tracked {len(tracked)}/{n_frames} frames '
          f'({100 * len(tracked) / max(n_frames, 1):.0f}%), '
          f'pose source: {dict(hist)}')
    if n_rejected:
        print(f'  {n_rejected} pose(s) dropped for rms > {args.max_rms} px')
    print(f'  points per frame: median {np.median([r[2] for r in tracked]):.0f}, '
          f'min {min((r[2] for r in tracked), default=0)}')
    if len(rms):
        print(f'  reprojection rms: median {np.median(rms):.3f} px, '
              f'90th pct {np.percentile(rms, 90):.3f} px')

    print('\n  jitter = median second difference, which mixes two regimes. Smooth true')
    print('  motion contributes a*dt^2, so mm/s^2 compares real acceleration across frame')
    print('  rates; white measurement noise contributes ~sigma*sqrt(6) at ANY dt, so for')
    print('  noise mm/s^2 penalises the faster rate by 1/dt^2. To compare noise between')
    print('  two runs, resample both to the same dt and read the mm column.')
    print(f'  {"":<26}{"mm":>10}{"mm/s^2":>12}')
    for label, seq in (
            ('board, charuco corners', [r[0][:3, 3] if r[0] is not None else None for r in rows]),
            ('board, marker corners', [r[0][:3, 3] if r[0] is not None else None for r in mrows]),
            ('TCP,   charuco corners', [r[1][:3, 3] if r[1] is not None else None for r in rows]),
            ('TCP,   marker corners', [r[1][:3, 3] if r[1] is not None else None for r in mrows])):
        j = jitter(seq)
        print(f'  {label:<26}{j:10.2f}{j / dt ** 2:12.1f}')

    write_csv(os.path.join(out_dir, f'{stem}_tcp_pose.csv'), rows, fps)
    if not args.no_video:
        write_video(video, os.path.join(out_dir, f'{stem}_annotated.mp4'),
                    dets, rows, cam, fps)


if __name__ == '__main__':
    main()
