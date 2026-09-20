#!/usr/bin/env python3
"""Live photometric preview for the scene camera -- exposure, gain, gamma, WB.

    python3 robospec_umi/robospec_umi_calibration/preview.py
    python3 robospec_umi/robospec_umi_calibration/preview.py --mask

Separate from CaptureSession and DetectorSession because it is a different job:
no board, no intrinsics, no coverage meters. Just "does this look right at this
exposure", which is what you need BEFORE there is a calibration to speak of.

WHY THIS EXISTS AT ALL
The photometric controls used to be frozen at fixed values by `lock`. Measuring
the camera showed that was unnecessary: at 1080p, exposure 1..10000, gain
0..1023, gamma 0..255 and white balance 2800..6500 all hold 119.4-119.8 fps, and
so do auto exposure and auto white balance. None of them changes the projection
either. So they can be tuned freely for the room -- and tuning them freely wants
a preview rather than a documented magic number.

What must still be fixed is only the geometry: autofocus off, focus and zoom
pinned. See lock_geometry() in calibrate_scene_cam.py.

CLIPPING MASK
The overlay can paint saturated pixels red and crushed pixels blue. This is the
reason a preview beats a printed number: a blown white board destroys the
intensity gradients that subpixel corner refinement works from, costing accuracy
rather than detections -- so nothing looks wrong, the calibration is just
quietly worse. The mask makes it obvious.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_scene_cam as CS                                    # noqa: E402

# Measured ceilings. Brightness stops responding above these, so a slider that
# travels further is travelling through nothing.
USEFUL_EXPOSURE_MAX = 800
USEFUL_GAIN_MAX = 250

# Controls the preview offers, in the order a UI should show them.
TUNABLE = ('auto_exposure', 'exposure_time_absolute',
           'white_balance_automatic', 'white_balance_temperature',
           'gain', 'gamma')


# Ranges do not change for the life of a device, so read them once. Without this
# a single slider drag is three v4l2-ctl processes per pixel of travel -- one to
# clamp, one to write, one for the UI's readback -- on a machine that also has to
# service a camera every 11.1 ms.
_RANGE_CACHE = {}


def _ranges(device):
    if device not in _RANGE_CACHE:
        _RANGE_CACHE[device] = CS.v4l2_controls(device)
    return _RANGE_CACHE[device]


def scene_controls(device):
    """Current values, ranges, and which sliders are live, for a v4l2 device.

    Module level, not a method, because both the calibration preview and a
    recording session need it and neither should have to hold an open camera to
    ask -- v4l2-ctl reads the node regardless of who is streaming it.

    `adjustable` is derived from auto_exposure / white_balance_automatic, NOT
    from the driver's INACTIVE flag: measured, this camera never raises that
    flag, and happily accepts writes it then ignores.
    """
    c = CS.v4l2_controls(device)
    auto_exp = c.get('auto_exposure', {}).get('value') == 3
    auto_wb = c.get('white_balance_automatic', {}).get('value') == 1
    out = {}
    for name in TUNABLE:
        e = c.get(name)
        if not e:
            continue
        e = dict(e)
        if name == 'exposure_time_absolute':
            e['useful_max'] = USEFUL_EXPOSURE_MAX
            e['adjustable'] = not auto_exp
        elif name == 'white_balance_temperature':
            e['adjustable'] = not auto_wb
        elif name == 'gain':
            e['useful_max'] = USEFUL_GAIN_MAX
            # NOT adjustable under auto exposure. Auto is a *brightness* loop and
            # gain is one of its two levers, so raising gain just makes it shorten
            # exposure and land on the same target. Measured, mean luma sweeping
            # gain 0..1023:
            #   manual (exp 800)  17.0  125.1  143.5  146.4  145.3
            #   auto exposure     81.7   81.5   81.1   82.2   81.3
            # The write is accepted and read back; the effect is overridden.
            e['adjustable'] = not auto_exp
        else:
            e['adjustable'] = True
        out[name] = e
    out['_auto_exposure'] = auto_exp
    out['_auto_wb'] = auto_wb
    return out


def scene_set(device, ctrl, value):
    """-> (ok, detail). Clamps against the cached range, then writes."""
    if ctrl not in TUNABLE:
        return False, f'{ctrl} is not tunable here'
    rng = _ranges(device).get(ctrl)
    if rng:
        value = int(np.clip(int(value), rng['min'], rng['max']))
    ok, err = CS.v4l2_set(device, ctrl, value)
    return ok, err or str(value)


class PreviewSession:
    """Camera + photometric readout. No board, no pose, no disk."""

    def __init__(self, device=None, mask=False, auto=True):
        self.device = device or CS.DEVICE
        self.mask = mask
        if auto:
            # Start from auto every time rather than inheriting whatever the last
            # session left behind. Measured: no frame-rate cost (119.6 fps under
            # auto exposure, 119.5 under auto WB), and it is the setting that
            # works in an unfamiliar room without touching anything.
            CS.v4l2_set(self.device, 'auto_exposure', 3)
            CS.v4l2_set(self.device, 'white_balance_automatic', 1)
        self.cap = CS.open_camera(self.device, CS.WIDTH, CS.HEIGHT, CS.FPS)

    # ------------------------------------------------------------- controls
    def controls(self):
        return scene_controls(self.device)

    def set(self, ctrl, value):
        return scene_set(self.device, ctrl, value)

    # ----------------------------------------------------------------- loop
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
        mean, clip, dark, _, verdict, colour = CS.exposure_stats(gray)

        view = frame.copy()
        if self.mask:
            view[gray >= 250] = (0, 0, 255)      # blown
            view[gray <= 8] = (255, 60, 0)       # crushed
        cv2.putText(view, f'mean {mean:.0f}   clipped {clip:.1f}%   '
                    f'crushed {dark:.1f}%   {verdict}',
                    (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        if self.mask:
            cv2.putText(view, 'MASK: red = blown, blue = crushed', (14, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        state = {
            'mean': round(mean, 1), 'clipped': round(clip, 2),
            'crushed': round(dark, 2), 'verdict': verdict,
            'histogram': (hist / max(hist.max(), 1)).round(3).tolist(),
            'mask': self.mask,
        }
        return frame, view, state

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--device', default=CS.DEVICE)
    ap.add_argument('--mask', action='store_true',
                    help='paint blown pixels red and crushed pixels blue')
    args = ap.parse_args()

    s = PreviewSession(args.device, args.mask)
    win = 'scene camera preview'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print('m toggle clipping mask   q quit')
    try:
        while True:
            r = s.step()
            if r is None:
                continue
            _, view, _ = r
            cv2.imshow(win, cv2.resize(view, None, fx=0.5, fy=0.5,
                                       interpolation=cv2.INTER_AREA))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                s.mask = not s.mask
            if k in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        s.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
