"""Live ChArUco detection and pose, to validate the scene-camera intrinsics.

Reads the calibration written by calibrate_scene_cam.py (intrinsic_type PINHOLE,
top-level "k"/"d") and shows, live, whether it actually describes the camera.

    python3 robospec_umi/robospec_umi_calibration/scene_cam_charuco_detector.py

Keys:
    q / ESC  quit                 m  toggle marker outlines
    s        save annotated frame u  toggle undistorted view
    d        take a distance sample (scale check)
    r        reset accumulated stats

What this is actually testing
-----------------------------
Two failures matter, and they need different instruments.

1. FOCUS DRIFT. The intrinsics are only valid at the focus and zoom they were shot
   at. A replug resets this camera to autofocus, which changes the focal length,
   and nothing downstream notices -- the pose solve just returns plausible wrong
   depths. The JSON records locked_controls, so this is checked at startup.

2. A FOCAL-LENGTH SCALE ERROR, which reprojection error structurally CANNOT see.
   If fx were 10% wrong, solvePnP simply places the board 10% further away and
   reprojection stays perfect. Every metric translation downstream would be 10%
   off with no symptom anywhere. Only a measured physical distance catches it --
   that is what the `d` key is for. See ScaleCheck.

Everything else here (reprojection error, radial bins, jitter, ambiguity) is
diagnosing HOW WELL the model fits, which is a different question from whether it
is scaled correctly.
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np

# Same directory, so Python has already put it on sys.path when this is run as a
# script -- no path juggling needed. The board geometry and the v4l2 helpers both
# come from here, so the two can never disagree about what board is in front of
# the camera.
import calibrate_scene_cam as CS

WIDTH, HEIGHT = CS.WIDTH, CS.HEIGHT
FPS = CS.FPS

DISPLAY_SCALE = 0.5     # preview downscale; detection always runs at full res
AXIS_LEN = -1           # metres; -1 means two board squares
MAX_REPROJ = 1.5        # px -- reject the pose above this
WARN_REPROJ = 1.0       # px -- radial bars turn orange above this
AMBIGUITY = 2.0         # flag when the alternate IPPE branch is within this factor
ALPHA = 0.0             # undistorted view: 0 crops to valid pixels, 1 keeps all
JITTER_WINDOW = 30      # frames
SAMPLES = 30            # frames averaged per distance sample


# ----------------------------------------------------------------- intrinsics
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
    """-> (K, D, meta) from the PINHOLE json calibrate_scene_cam.py writes.

    Rescales K if the capture resolution differs from the calibration's: K scales
    with resolution while D is dimensionless, so a 960x540 stream read against a
    1920x1080 K is wrong by a factor of two in depth, silently.
    """
    with open(path) as f:
        d = json.load(f)
    if d.get('intrinsic_type') == 'FISHEYE':
        raise SystemExit(
            f'{path} is a FISHEYE calibration. This pipeline is rectilinear only '
            f'-- re-run calibrate_scene_cam.py solve to produce a PINHOLE one.')
    if 'k' not in d:
        raise SystemExit(f'{path} has no "k" -- not a calibrate_scene_cam.py output')

    K = np.array(d['k'], dtype=np.float64).reshape(3, 3)
    D = np.array(d.get('d', [0.0] * 5), dtype=np.float64)
    src = (int(d.get('image_width', width)), int(d.get('image_height', height)))
    if src != (width, height):
        print(f'  calibration is {src[0]}x{src[1]}, capturing at {width}x{height} '
              f'-- rescaling K')
        K = scale_K(K, src, (width, height))
    return K, D, d


def check_geometry_lock(device, meta):
    """Compare the camera's live focus/zoom against what was calibrated.

    This is the check that matters most and costs least. Focus changes the focal
    length, so a session shot at a different focus is described by different
    intrinsics -- and the only visible symptom is that distances come out wrong,
    which looks like nothing at all.
    """
    locked = meta.get('locked_controls') or {}
    if not locked:
        print('  NOTE: no locked_controls in the calibration; cannot verify focus')
        return True
    got = CS.v4l2_get(device, CS.GEOMETRY_CTRLS)
    bad = []
    for key in CS.GEOMETRY_CTRLS:
        want, have = locked.get(key), got.get(key)
        if want is None or have is None:
            continue
        if int(want) != int(have):
            bad.append(f'{key}: now {have}, calibrated at {want}')
    if bad:
        print('\n' + '=' * 70)
        print('  CAMERA GEOMETRY DOES NOT MATCH THE CALIBRATION')
        for b in bad:
            print(f'    {b}')
        print('  Focus and zoom change the focal length, so scene_intrinsics.json')
        print('  does not describe this camera right now. Distances will be wrong')
        print('  and nothing else will look unusual.')
        print(f'    fix:  python3 {os.path.join(CS.HERE, "calibrate_scene_cam.py")} lock')
        print('=' * 70 + '\n')
        return False
    print(f'  geometry matches calibration: '
          + ', '.join(f'{k}={got[k]}' for k in CS.GEOMETRY_CTRLS if k in got))
    return True


# ------------------------------------------------------------------ pose solve
def solve_board_pose(cc, ci, all_obj, K, D):
    """-> (rvec, tvec, err_best, err_alt).

    Corners go straight into solvePnP with K and D -- no pre-undistortion step,
    because solvePnP reads a rad-tan or rational D natively and rectifying first
    would only add an approximation.

    IPPE is used through solvePnPGeneric to get BOTH branches of the planar
    two-fold ambiguity. A plane seen near fronto-parallel has a second pose that
    reprojects almost as well, and picking the wrong one is what produces the
    depth spikes build_zarr.py's drop_spikes() has to filter out. err_alt/err_best
    is therefore a live confidence measure, not a curiosity.
    """
    obj = all_obj[ci.flatten()].reshape(-1, 1, 3).astype(np.float64)
    img = cc.reshape(-1, 1, 2).astype(np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        n = 0
    if not n:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D)
        return (rvec, tvec, float('nan'), float('nan')) if ok else (None, None, 0, 0)

    e = np.asarray(errs, dtype=np.float64).ravel()
    order = np.argsort(e)
    best = int(order[0])
    alt = float(e[order[1]]) if n > 1 else float('inf')
    return rvecs[best], tvecs[best], float(e[best]), alt


def detect_all(gray, adict, board, params):
    """Markers AND ChArUco corners from ONE detectMarkers pass.

    -> (marker_corners, marker_ids, charuco_corners, charuco_ids), any of which
    may be None.

    CS.detect_charuco() calls detectMarkers internally, so drawing marker
    outlines by calling detectMarkers separately first does the expensive work
    twice. Measured on this 1080p stream that is 23.6 ms each and 66% of the
    loop, which halves the frame rate for no benefit whatsoever.
    """
    mc, mi, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
    if mi is None or len(mi) == 0:
        return None, None, None, None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
    if n is None or n < 4:
        return mc, mi, None, None
    return mc, mi, cc, ci


def board_centre_cam(rvec, tvec, centre_obj):
    """Board CENTRE in camera coordinates, in metres.

    Not tvec. The ChArUco pose origin sits on the first interior corner, which for
    a 7x5 30 mm board is 129 mm from the centre of the printed pattern -- and as
    the board rotates that offset swings around, so |tvec| varies by up to +/-129 mm
    for a board whose centre never moved. That would wreck the scale check, since
    the offset is not constant and so cannot be absorbed by the fit's intercept.
    The centre is also the point you can actually put a tape measure on.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return (R @ np.asarray(centre_obj, dtype=np.float64).reshape(3, 1)
            + np.asarray(tvec, dtype=np.float64).reshape(3, 1)).ravel()


def project(points, rvec, tvec, K, D):
    """Project object points, flagging any that land somewhere absurd."""
    proj, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 3), rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    good = np.isfinite(proj).all(axis=1) & (np.abs(proj) < 1e4).all(axis=1)
    return proj, good


def reproj_error(cc, ci, all_obj, rvec, tvec, K, D):
    proj, good = project(all_obj[ci.flatten()], rvec, tvec, K, D)
    if good.sum() == 0:
        return float('nan')
    obs = cc.reshape(-1, 2)[good]
    return float(np.sqrt(np.mean(np.sum((proj[good] - obs) ** 2, axis=1))))


def draw_axes(img, rvec, tvec, K, D, length, n=24):
    """Axes as projected polylines, NOT cv2.drawFrameAxes.

    The instinct is that a rectilinear camera images straight lines as straight
    lines, making the sampling unnecessary. That is wrong for this lens: it is ~110 deg, and its fitted model displaces pixels by about
    105 px at the frame corner. drawFrameAxes projects only the two endpoints and
    joins them with a straight segment, which visibly misses near the edges --
    exactly where the calibration most needs scrutiny.
    """
    for axis, colour in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
        pts = np.zeros((n, 3))
        pts[:, axis] = np.linspace(0, length, n)
        proj, good = project(pts, rvec, tvec, K, D)
        if good.sum() < 2:
            continue
        cv2.polylines(img, [proj[good].astype(np.int32)], False,
                      colour, 3, cv2.LINE_AA)
        cv2.putText(img, 'XYZ'[axis], tuple(proj[good][-1].astype(np.int32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)


# ------------------------------------------------------------------ diagnostics
class RadialError:
    """Mean reprojection error binned by how far the board sat from the centre.

    Distortion grows with radius, and the calibration set reached only ~88% of the
    way to the frame corner, so the outer ring is the least constrained part of
    the model. If error is flat across these bins the calibration holds everywhere
    it will be used; if the outer bin climbs, the model is extrapolating and poses
    near the frame edge cannot be trusted. A single global average hides this
    completely, which is the whole reason for binning.
    """

    def __init__(self, K, size, n=4):
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.r_max = float(np.hypot(max(self.cx, size[0] - self.cx),
                                    max(self.cy, size[1] - self.cy)))
        self.n = n
        self.reset()

    def reset(self):
        self.sum = np.zeros(self.n)
        self.cnt = np.zeros(self.n, dtype=np.int64)

    def add(self, centre, err):
        if not np.isfinite(err):
            return
        r = float(np.hypot(centre[0] - self.cx, centre[1] - self.cy)) / self.r_max
        i = min(int(r * self.n), self.n - 1)
        self.sum[i] += err
        self.cnt[i] += 1

    def means(self):
        return np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), np.nan)

    def draw(self, img, x=14, y=300, w=220, warn=1.0):
        cv2.putText(img, 'reproj px by radius', (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        m = self.means()
        for i in range(self.n):
            yy = y + i * 26
            lo, hi = 100 * i / self.n, 100 * (i + 1) / self.n
            if self.cnt[i] == 0:
                cv2.putText(img, f'{lo:3.0f}-{hi:3.0f}%   --', (x, yy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
                continue
            frac = min(m[i] / max(warn, 1e-6), 1.0)
            col = (0, 255, 0) if m[i] <= warn else (60, 160, 255)
            cv2.rectangle(img, (x + 92, yy), (x + 92 + int(w * frac), yy + 16),
                          col, -1)
            cv2.rectangle(img, (x + 92, yy), (x + 92 + w, yy + 16), (110, 110, 110), 1)
            cv2.putText(img, f'{lo:3.0f}-{hi:3.0f}%', (x, yy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(img, f'{m[i]:.2f} ({self.cnt[i]})', (x + 92 + w + 8, yy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)


class Jitter:
    """Spread of the recovered position over a sliding window.

    On a stationary board this is measurement noise propagating through the solve,
    and it is exactly what becomes velocity noise once these poses are
    differentiated downstream. Reported in mm because that is the unit the error
    budget is argued in.
    """

    def __init__(self, n=30):
        self.buf = deque(maxlen=n)

    def add(self, t):
        self.buf.append(np.asarray(t, dtype=np.float64).ravel())

    def reset(self):
        self.buf.clear()

    def mm(self):
        if len(self.buf) < 5:
            return None
        return np.std(np.array(self.buf), axis=0) * 1000.0


class ScaleCheck:
    """Recovered distance vs tape-measured distance, fitted over >=2 samples.

    The one test reprojection error cannot do. A wrong focal length rescales every
    recovered translation by the same factor while leaving reprojection perfect,
    so the model looks flawless and the metres are wrong.

    Deliberately DIFFERENTIAL rather than single-point: the optical centre sits
    somewhere inside the lens barrel, not at the housing face, so any single
    absolute measurement carries an unknown offset of a centimetre or two. Fitting
    recovered = slope*measured + intercept lets that offset land in the intercept
    instead of contaminating the slope.

        slope      the focal-length scale error. 1.000 means fx is right;
                   1.08 means fx is 8% too large and every distance is too.
        intercept  where the optical centre is relative to your tape datum.
    """

    def __init__(self):
        self.samples = []          # (measured_mm, recovered_mm)

    def add(self, measured_mm, recovered_mm):
        self.samples.append((float(measured_mm), float(recovered_mm)))

    def reset(self):
        self.samples = []

    def fit(self):
        if len(self.samples) < 2:
            return None
        a = np.array(self.samples)
        # need genuinely different distances or the fit is meaningless
        if a[:, 0].max() - a[:, 0].min() < 50:
            return None
        slope, intercept = np.polyfit(a[:, 0], a[:, 1], 1)
        pred = slope * a[:, 0] + intercept
        resid = float(np.sqrt(np.mean((a[:, 1] - pred) ** 2)))
        return float(slope), float(intercept), resid

    def report(self, fx):
        print('\n' + '-' * 62)
        print(f'{"measured (mm)":>15}{"recovered (mm)":>17}{"ratio":>10}')
        for m, r in self.samples:
            print(f'{m:>15.1f}{r:>17.1f}{r / max(m, 1e-9):>10.4f}')
        f = self.fit()
        if f is None:
            print('\n  need >=2 samples at distances at least 50 mm apart')
            print('-' * 62)
            return
        slope, intercept, resid = f
        print(f'\n  slope      {slope:.4f}   <- focal-length scale error '
              f'({100 * (slope - 1):+.1f}%)')
        print(f'  intercept  {intercept:+.1f} mm  <- optical centre vs your tape datum')
        print(f'  residual   {resid:.1f} mm')
        if abs(slope - 1) <= 0.01:
            print(f'\n  fx = {fx:.1f} is consistent with the tape to within 1%.')
        else:
            print(f'\n  fx = {fx:.1f} looks WRONG by {100 * (slope - 1):+.1f}%. '
                  f'Implied fx = {fx / slope:.1f}.')
            print('  Every translation from this calibration carries that error.')
            print('  Re-check the printed square size (--square-len) first: a board')
            print('  printed at 96% scale produces exactly this signature.')
        print('-' * 62)


# --------------------------------------------------------------- headless use
class DetectorSession:
    """Per-frame detection and pose for a server-driven validation view.

    Deliberately separate from main()'s loop rather than shared, unlike
    CaptureSession. main() is built around cv2.waitKey and a blocking input()
    prompt for the tape-measure scale check, neither of which can serve a
    browser. The duplication is bounded and safe because nothing here decides
    what reaches disk -- it only decides what is drawn and reported. The capture
    loop was the opposite case, which is why that one is shared.
    """

    def __init__(self, device, intrinsics_path):
        self.K, self.D, self.meta = load_intrinsics(intrinsics_path, WIDTH, HEIGHT)
        self.intrinsics_path = intrinsics_path
        self.geometry_ok = check_geometry_lock(device, self.meta)

        self.adict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, CS.BOARD_DICT))
        self.board = CS.make_board(self.adict, (CS.SQUARES_X, CS.SQUARES_Y),
                                   CS.SQUARE_LEN, CS.MARKER_LEN)
        self.params = CS.make_params()
        self.all_obj = CS.board_corners(self.board)
        self.centre_obj = self.all_obj.mean(axis=0)
        self.nx, self.ny = CS.inner_grid(CS.SQUARES_X, CS.SQUARES_Y)
        self.axis_len = AXIS_LEN if AXIS_LEN > 0 else CS.SQUARE_LEN * 2

        self.cap = CS.open_camera(device, WIDTH, HEIGHT, FPS)
        self.radial = RadialError(self.K, (WIDTH, HEIGHT))
        self.jitter = Jitter(JITTER_WINDOW)

    def grab(self):
        """Dequeue a frame without decoding it -- ~0.1 ms against ~9 ms for the
        decode, so a caller wanting fewer frames than the camera produces can
        drop here and never pay for the JPEG."""
        return self.cap.grab()

    def step(self, grabbed=False):
        """-> (raw_frame, overlay_frame, state) or None on a bad read.

        grabbed=True decodes the frame the caller already grab()ed instead of
        dequeuing another.
        """
        if not grabbed and not self.cap.grab():
            return None
        ok, frame = self.cap.retrieve()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        mc, mi, cc, ci = detect_all(gray, self.adict, self.board, self.params)
        if mi is not None:
            cv2.aruco.drawDetectedMarkers(vis, mc, mi)

        st = {'corners': 0 if cc is None else int(len(cc)), 'pose': False,
              'reproj_px': None, 'distance_mm': None, 'ambiguity': None,
              'ambiguous': False, 'status': 'no board',
              'geometry_ok': bool(self.geometry_ok)}

        if cc is not None and len(cc) >= CS.MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 255))
            quad = CS.board_quad(cc, ci, self.nx, self.ny)
            if quad is not None:
                cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 0, 255), 2)
            rvec, tvec, e_best, e_alt = solve_board_pose(cc, ci, self.all_obj,
                                                         self.K, self.D)
            if rvec is not None:
                err = reproj_error(cc, ci, self.all_obj, rvec, tvec, self.K, self.D)
                t = board_centre_cam(rvec, tvec, self.centre_obj)
                ratio = e_alt / max(e_best, 1e-9)
                st['reproj_px'] = round(float(err), 3)
                st['ambiguity'] = None if not np.isfinite(ratio) else round(float(ratio), 2)
                st['ambiguous'] = bool(np.isfinite(ratio) and ratio < AMBIGUITY)
                if err > MAX_REPROJ:
                    # Motion blur makes interpolateCornersCharuco place a few
                    # corners on the wrong saddle. The pose still solves and is
                    # wrong; drawing it would put a confident frame in the wrong
                    # place.
                    st['status'] = f'POSE REJECTED  reproj={err:.2f} > {MAX_REPROJ:.1f} px'
                    self.jitter.reset()
                else:
                    draw_axes(vis, rvec, tvec, self.K, self.D, self.axis_len)
                    self.radial.add(cc.reshape(-1, 2).mean(axis=0), err)
                    self.jitter.add(t)
                    st['pose'] = True
                    st['distance_mm'] = round(float(np.linalg.norm(t)) * 1000.0, 1)
                    st['status'] = 'tracking'
        elif cc is not None:
            st['status'] = f'only {len(cc)} corners (need {CS.MIN_CORNERS})'
            self.jitter.reset()
        else:
            self.jitter.reset()

        self.radial.draw(vis, warn=WARN_REPROJ)
        m = self.radial.means()
        st['radial'] = [None if not np.isfinite(v) else round(float(v), 3) for v in m]
        st['radial_counts'] = [int(c) for c in self.radial.cnt]
        j = self.jitter.mm()
        st['jitter_mm'] = None if j is None else [round(float(v), 2) for v in j]
        return frame, vis, st

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--device', default=CS.DEVICE)
    ap.add_argument('-i', '--intrinsics', default=CS.DEFAULT_INTRINSICS)
    ap.add_argument('--expect-mm', type=float, default=-1,
                    help='if set, each sample is also compared against this '
                         'distance directly, without needing a second point')
    ap.add_argument('--no-display', action='store_true',
                    help='headless: no window, no undistorted view. Runs the '
                         'detection and jitter checks and prints the summary, '
                         'which is what makes this usable as an automated '
                         'post-calibration gate.')
    args = ap.parse_args()

    K, D, meta = load_intrinsics(args.intrinsics, WIDTH, HEIGHT)
    print(f'intrinsics {args.intrinsics}')
    print(f'  {meta.get("distortion_model", "?")}  fx={K[0,0]:.2f} fy={K[1,1]:.2f} '
          f'cx={K[0,2]:.2f} cy={K[1,2]:.2f}')
    print(f'  D={np.array2string(D, precision=5)}')
    print(f'  calibration reproj {meta.get("final_reproj_error", float("nan")):.4f} px '
          f'over {meta.get("nr_calib_images", "?")} views')

    if not check_geometry_lock(args.device, meta):
        raise SystemExit('refusing to run; pass --ignore-lock to override')

    jb = meta.get('board') or {}
    if jb and (jb.get('squares') != [CS.SQUARES_X, CS.SQUARES_Y]
               or abs(jb.get('square_len_m', 0) - CS.SQUARE_LEN) > 1e-9):
        print(f'  note: testing with a {CS.SQUARES_X}x{CS.SQUARES_Y} '
              f'{CS.SQUARE_LEN * 1000:.0f}mm board; the calibration used '
              f'{jb.get("squares")} {jb.get("square_len_m", 0) * 1000:.0f}mm. '
              f'That is fine -- intrinsics do not depend on the board.')

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CS.BOARD_DICT))
    board = CS.make_board(adict, (CS.SQUARES_X, CS.SQUARES_Y),
                          CS.SQUARE_LEN, CS.MARKER_LEN)
    params = CS.make_params()
    # The wide adaptive-threshold sweep (3..23 step 5) is kept deliberately. A
    # narrower 3..13 step 10 is 27% faster for identical output on a well-lit
    # static board, but the wide sweep earns its cost on uneven lighting and
    # steep board angles -- exactly the views a calibration check needs to see.
    all_obj = CS.board_corners(board)
    centre_obj = all_obj.mean(axis=0)
    nx, ny = CS.inner_grid(CS.SQUARES_X, CS.SQUARES_Y)
    print(f'  board centre is {np.linalg.norm(centre_obj) * 1000:.0f} mm from the '
          f'pose origin; distances below are to the CENTRE -- measure your tape '
          f'to the middle of the printed pattern')
    axis_len = AXIS_LEN if AXIS_LEN > 0 else CS.SQUARE_LEN * 2

    show = not args.no_display
    cap = CS.open_camera(args.device, WIDTH, HEIGHT, FPS)
    win = f'scene charuco ({args.device})'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('\nq quit   s save   m markers   u undistort   d distance sample   r reset')
        print('Walk the board into all four frame corners and watch the radial bars: '
              'that is the test the calibration set could not do for itself.\n')
    else:
        print('\nheadless: accumulating detection and jitter stats. Ctrl-C for the '
              'summary.\n')

    radial = RadialError(K, (WIDTH, HEIGHT))
    jitter = Jitter(JITTER_WINDOW)
    scale = ScaleCheck()
    show_markers, show_undist = True, False
    maps = None
    sampling, sample_buf = False, []
    n, t0, meas = 0, time.monotonic(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('frame grab failed')
                break
            n += 1
            dt = time.monotonic() - t0
            if dt >= 1.0:
                meas, n, t0 = n / dt, 0, time.monotonic()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            mc, mi, cc, ci = detect_all(gray, adict, board, params)
            if show_markers and mi is not None:
                cv2.aruco.drawDetectedMarkers(vis, mc, mi)
            status, colour, extra = 'no board', (0, 0, 255), ''

            if cc is not None and len(cc) >= CS.MIN_CORNERS:
                cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 255))
                quad = CS.board_quad(cc, ci, nx, ny)
                if quad is not None:
                    cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 0, 255), 2)

                rvec, tvec, e_best, e_alt = solve_board_pose(cc, ci, all_obj, K, D)
                if rvec is not None:
                    err = reproj_error(cc, ci, all_obj, rvec, tvec, K, D)
                    # board CENTRE, not tvec -- see board_centre_cam()
                    t = board_centre_cam(rvec, tvec, centre_obj)
                    dist_mm = float(np.linalg.norm(t)) * 1000.0
                    ratio = e_alt / max(e_best, 1e-9)

                    if err > MAX_REPROJ:
                        # Motion blur makes interpolateCornersCharuco place a few
                        # corners on the wrong saddle. The pose still solves, and
                        # it is wrong -- drawing it would put a confident-looking
                        # frame in the wrong place.
                        status = (f'{len(cc)} corners  POSE REJECTED  '
                                  f'reproj={err:.2f} > {MAX_REPROJ:.1f} px')
                        colour = (0, 165, 255)
                        jitter.reset()
                    else:
                        draw_axes(vis, rvec, tvec, K, D, axis_len)
                        centre = cc.reshape(-1, 2).mean(axis=0)
                        radial.add(centre, err)
                        jitter.add(t)
                        status = (f'{len(cc)} corners   centre '
                                  f'xyz=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m   '
                                  f'dist={dist_mm:.1f} mm')
                        colour = (0, 255, 0)
                        amb = ('AMBIGUOUS' if ratio < AMBIGUITY
                               else f'{ratio:.1f}x')
                        extra = f'reproj {err:.3f} px    2nd-branch margin {amb}'
                        if sampling:
                            sample_buf.append(dist_mm)
                            if len(sample_buf) >= SAMPLES:
                                sampling = False
                                rec = float(np.mean(sample_buf))
                                sd = float(np.std(sample_buf))
                                print(f'\nsampled {len(sample_buf)} frames: '
                                      f'{rec:.1f} mm (sd {sd:.1f})')
                                if args.expect_mm > 0:
                                    r = rec / args.expect_mm
                                    print(f'  vs --expect-mm {args.expect_mm:.1f}: '
                                          f'ratio {r:.4f} ({100 * (r - 1):+.1f}%)')
                                try:
                                    s = input('  tape-measured distance to the '
                                              'BOARD CENTRE in mm '
                                              '(blank to discard): ').strip()
                                except EOFError:
                                    s = ''
                                if s:
                                    try:
                                        scale.add(float(s), rec)
                                        scale.report(K[0, 0])
                                    except ValueError:
                                        print('  not a number; discarded')
                                # the camera kept streaming into the buffer while
                                # input() blocked; drop the backlog
                                for _ in range(10):
                                    cap.read()
                                t0, n = time.monotonic(), 0
            elif cc is not None:
                status = f'only {len(cc)} corners (need {CS.MIN_CORNERS})'
                colour = (0, 165, 255)
                jitter.reset()
            else:
                jitter.reset()

            # Drawing and the undistort remap are display-only work.
            if show:
                if show_undist:
                    if maps is None:
                        newK, _ = cv2.getOptimalNewCameraMatrix(
                            K, D, (WIDTH, HEIGHT), ALPHA,
                            (WIDTH, HEIGHT))
                        maps = cv2.initUndistortRectifyMap(
                            K, D, None, newK, (WIDTH, HEIGHT), cv2.CV_16SC2)
                    vis = cv2.remap(vis, maps[0], maps[1], cv2.INTER_LINEAR)

                cv2.putText(vis, f'{meas:.1f} fps', (14, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if show_undist:
                    cv2.putText(vis, 'UNDISTORTED - straight edges should be straight',
                                (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 0), 2)
                cv2.putText(vis, status, (14, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                if extra:
                    cv2.putText(vis, extra, (14, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                j = jitter.mm()
                if j is not None:
                    cv2.putText(vis, f'jitter over {len(jitter.buf)} frames: '
                                f'x{j[0]:.2f} y{j[1]:.2f} z{j[2]:.2f} mm', (14, 142),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                if sampling:
                    cv2.putText(vis, f'SAMPLING {len(sample_buf)}/{SAMPLES} '
                                f'- hold the board still', (14, 176),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                radial.draw(vis, warn=WARN_REPROJ)
                if scale.samples:
                    f = scale.fit()
                    msg = (f'scale: {len(scale.samples)} sample(s)' if f is None else
                           f'scale slope {f[0]:.4f} ({100 * (f[0] - 1):+.1f}%)  '
                           f'intercept {f[1]:+.0f} mm')
                    col = ((255, 255, 255) if f is None else
                           (0, 255, 0) if abs(f[0] - 1) <= 0.01 else (60, 160, 255))
                    cv2.putText(vis, msg, (14, vis.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

                disp = vis if DISPLAY_SCALE == 1.0 else cv2.resize(
                    vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA)
                cv2.imshow(win, disp)

            key = (cv2.waitKey(1) & 0xFF) if show else 255
            if key in (ord('q'), 27):
                break
            if key == ord('m'):
                show_markers = not show_markers
            if key == ord('u'):
                show_undist = not show_undist
            if key == ord('d') and not sampling:
                sampling, sample_buf = True, []
            if key == ord('r'):
                radial.reset()
                jitter.reset()
                scale.reset()
                print('stats reset')
            if key == ord('s'):
                name = os.path.join(
                    CS.CALIB_ROOT,
                    time.strftime('scene_charuco_%Y%m%d_%H%M%S.png'))
                os.makedirs(CS.CALIB_ROOT, exist_ok=True)
                cv2.imwrite(name, vis)
                print('saved', name)
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    m = radial.means()
    if np.isfinite(m).any():
        print('\nreprojection error by radius (0% = centre, 100% = frame corner):')
        for i in range(radial.n):
            if radial.cnt[i]:
                print(f'  {100*i/radial.n:3.0f}-{100*(i+1)/radial.n:3.0f}%  '
                      f'{m[i]:.3f} px over {radial.cnt[i]} frames')
        seen = m[np.isfinite(m)]
        if len(seen) > 1 and seen.max() > 2 * seen.min():
            print('  NOTE: error is not flat across the frame. The bins that are '
                  'worse are where the calibration had least data; poses there are '
                  'less trustworthy than the global rms suggests.')
    if scale.samples:
        scale.report(K[0, 0])


if __name__ == '__main__':
    main()
