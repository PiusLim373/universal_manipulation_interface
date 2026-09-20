#!/usr/bin/env python3
"""Check a capture session before you trust it for training.

The failures this catches are the ones that leave no visible trace in the data. A
sidecar one entry longer than its video shifts every subsequent frame by ~10 ms,
which is ~1 cm of TCP label error applied silently for the rest of the recording;
the video still plays and the poses still look reasonable. Likewise, arrival-based
wrist timestamps would look perfectly plausible while carrying 12 ms of jitter.

Usage:
    python3 robospec_umi/robospec_umi_capture/verify.py data/capture/20260919_150000
"""

import json
import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_capture
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

OK, BAD, WARN = '  [ok]  ', '  [FAIL]', '  [warn]'


def check_invariant(label, n_video, n_ts, fails):
    """One timestamp per written frame, or every later frame is shifted.

    A mismatch is normally an interrupted recording rather than corruption: the
    sidecar is checkpointed about once a second so it can trail the video, and the
    encoder holds a few frames that are lost if the flush never runs. Both streams
    are written in capture order, so the leading min(n_video, n_ts) frames are still
    correctly paired and the tail can simply be dropped.
    """
    if n_video == n_ts:
        print(OK + f' index invariant: {n_video} frames == {n_ts} timestamps')
        return
    print(BAD + f' index invariant BROKEN: {n_video} frames vs {n_ts} timestamps')
    n = min(n_video, n_ts)
    print(f'         looks like an interrupted recording '
          f'({"video ahead" if n_video > n_ts else "sidecar ahead"}). '
          f'Both are in capture order, so use the first {n} frames of each and '
          f'discard the tail.')
    fails.append(f'{label} index mismatch ({n_video} video vs {n_ts} timestamps)')


def count_packets(path):
    c = av.open(path)
    s = c.streams.video[0]
    n = sum(1 for p in c.demux(s) if p.pts is not None and p.size > 0)
    c.close()
    return n


def intervals(t_ns, label, nominal_fps, fails):
    t = t_ns.astype(np.float64) / 1e9
    d = np.diff(t) * 1000
    period = 1000.0 / nominal_fps
    print(f'    {len(t)} frames over {t[-1]-t[0]:.2f} s = '
          f'{len(t)/(t[-1]-t[0]):.2f} fps  (nominal {nominal_fps})')
    print(f'    interval ms: median {np.median(d):7.3f}  p1 {np.percentile(d,1):7.3f}  '
          f'p99 {np.percentile(d,99):7.3f}  max {d.max():8.3f}')
    if not np.all(np.diff(t) > 0):
        print(BAD + f' {label}: timestamps are not strictly increasing')
        fails.append(f'{label} non-monotonic timestamps')
    gaps = int(np.sum(d > 1.5 * period))
    print(f'    {gaps} gaps > 1.5x nominal period'
          + (f'  ({100*gaps/len(d):.1f}% of intervals)' if gaps else ''))
    return d


def check_scene(d, meta, fails):
    print('\nSCENE')
    vid, side = os.path.join(d, 'scene', 'scene.mkv'), os.path.join(d, 'scene', 'scene_ts.npz')
    if not (os.path.exists(vid) and os.path.exists(side)):
        print(BAD + ' missing scene.mkv or scene_ts.npz')
        fails.append('scene files missing')
        return
    t_ns = np.load(side)['t_ns']
    n_pkt = count_packets(vid)
    if t_ns.dtype != np.int64:
        print(BAD + f' t_ns dtype is {t_ns.dtype}, must be int64')
        fails.append('scene t_ns dtype')
    check_invariant('scene', n_pkt, len(t_ns), fails)
    intervals(t_ns, 'scene', meta['scene']['fps_requested'], fails)
    codec = av.open(vid).streams.video[0].codec_context.name
    print(f'    codec {codec}' + (OK.strip() + ' passthrough preserved'
                                  if codec == 'mjpeg' else WARN.strip() + ' re-encoded?'))
    print(f'    size {os.path.getsize(vid)/1e6:.1f} MB '
          f'({os.path.getsize(vid)/1e6/max((t_ns[-1]-t_ns[0])/1e9, 1e-9):.1f} MB/s)')


def check_wrist(d, meta, fails):
    print('\nWRIST')
    vid, side = os.path.join(d, 'wrist', 'wrist.mkv'), os.path.join(d, 'wrist', 'wrist_ts.npz')
    if not (os.path.exists(vid) and os.path.exists(side)):
        print(BAD + ' missing wrist.mkv or wrist_ts.npz')
        fails.append('wrist files missing')
        return
    z = np.load(side)
    t_ns, dev, arr, seq = z['t_ns'], z['t_device_ns'], z['t_arrival_ns'], z['seq']
    n_frm = count_packets(vid)
    check_invariant('wrist', n_frm, len(t_ns), fails)

    doms = meta.get('wrist', {}).get('timestamp_domains', [])
    if doms == ['timestamp_domain.global_time']:
        print(OK + f' timestamp domain {doms}')
    else:
        print(BAD + f' timestamp domain {doms} -- expected global_time only. '
                    'Without it the stamps are host-arrival and carry ~12 ms jitter.')
        fails.append(f'wrist timestamp domain {doms}')

    intervals(t_ns, 'wrist', meta['wrist']['fps'], fails)

    # device-vs-arrival is the evidence that device stamps are actually being used
    lag = (arr - dev + int(meta['mono_to_epoch_offset_s'] * 1e9)) / 1e6
    print(f'    arrival - device: median {np.median(lag):6.2f} ms   '
          f'spread(p1-p99) {np.percentile(lag,99)-np.percentile(lag,1):6.2f} ms')
    jit_dev = np.diff(t_ns.astype(np.float64)/1e6)
    jit_arr = np.diff(arr.astype(np.float64)/1e6)
    iqr = lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    print(f'    interval jitter (IQR): device {iqr(jit_dev):.3f} ms  vs  '
          f'arrival {iqr(jit_arr):.3f} ms   <- why global time is used')

    gaps = np.diff(seq) - 1
    nd = int(gaps[gaps > 0].sum())
    if nd == 0:
        print(OK + ' no device-side frame drops (seq is contiguous)')
    else:
        print(WARN + f' device dropped {nd} frames in {int((gaps>0).sum())} bursts. '
                     'Harmless -- resampling is by time, not index -- but worth knowing.')
    print(f'    size {os.path.getsize(vid)/1e6:.1f} MB')


def check_overlap(d, fails):
    """Both streams must actually cover the same wall-clock window, or there is
    nothing to interpolate onto at the ends."""
    sp = os.path.join(d, 'scene', 'scene_ts.npz')
    wp = os.path.join(d, 'wrist', 'wrist_ts.npz')
    if not (os.path.exists(sp) and os.path.exists(wp)):
        return
    s, w = np.load(sp)['t_ns'], np.load(wp)['t_ns']
    lo, hi = max(s[0], w[0]), min(s[-1], w[-1])
    print('\nOVERLAP (both on CLOCK_MONOTONIC)')
    print(f'    scene {s[0]/1e9:.3f} .. {s[-1]/1e9:.3f}')
    print(f'    wrist {w[0]/1e9:.3f} .. {w[-1]/1e9:.3f}')
    if hi <= lo:
        print(BAD + ' the two streams do not overlap at all')
        fails.append('no temporal overlap')
        return
    inside = int(((w >= lo) & (w <= hi)).sum())
    print(OK + f' overlap {(hi-lo)/1e9:.2f} s; {inside}/{len(w)} wrist frames '
               f'({100*inside/len(w):.1f}%) fall inside the scene track')
    if inside < len(w):
        print(WARN + f' {len(w)-inside} wrist frames sit outside and cannot be '
                     'labelled -- they must be dropped, never extrapolated')


CALIB_JSON = os.path.join(DATA, 'calibration', 'scene_intrinsics.json')


def check_scene_geometry(d, meta):
    """Did this session record at the focus and zoom the intrinsics were shot at?

    Nothing downstream can notice a mismatch. A different focus means a different
    focal length, so the ChArUco solve scales every translation by the ratio --
    the poses stay smooth, plausible and self-consistent, and are simply wrong.
    A session recorded at a different zoom is worse still, because that moves the
    principal point too.

    Silent (not a failure) when the calibration file does not exist yet: a
    session can legitimately be verified before the camera has been calibrated.
    """
    path = os.environ.get('SCENE_INTRINSICS', CALIB_JSON)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            locked = json.load(f).get('locked_controls') or {}
    except (ValueError, OSError):
        return []
    if not locked:
        return []

    scene = meta.get('scene', {})
    out = []
    for key in ('focus_absolute', 'zoom_absolute'):
        want, got = locked.get(key), scene.get(key)
        # negative means the flag was "leave this control alone"
        if want is None or got is None or got < 0:
            continue
        if int(want) != int(got):
            print(BAD + f' scene {key}={got} but {os.path.basename(path)} was '
                        f'calibrated at {want} -- the intrinsics do not apply to '
                        f'this session')
            out.append(f'scene {key} differs from calibration')
        else:
            print(OK + f' scene {key}={got} matches the calibration')
    return out


def check_episode(ep_dir, meta, fails):
    print(f'\n{"="*64}\nEPISODE {os.path.basename(ep_dir)}')
    if os.path.isdir(os.path.join(ep_dir, 'scene')):
        check_scene(ep_dir, meta, fails)
    if os.path.isdir(os.path.join(ep_dir, 'wrist')):
        check_wrist(ep_dir, meta, fails)
    check_overlap(ep_dir, fails)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = os.path.abspath(sys.argv[1])
    meta_path = os.path.join(d, 'session.json')
    if not os.path.exists(meta_path):
        sys.exit(f'no session.json in {d}')
    meta = json.load(open(meta_path))
    fails_top = []
    eps = sorted(x for x in os.listdir(d)
                 if x.startswith('ep') and os.path.isdir(os.path.join(d, x)))
    print(f'session {meta["session"]}   {len(eps)} episodes')

    dsp = meta.get('display', {})
    if dsp.get('enabled'):
        sk = dsp.get('frames_skipped', {})
        print(f'  preview was on at {dsp.get("fps")} fps, skipped '
              + ', '.join(f'{k} {v}' for k, v in sk.items())
              + ' frames (by design -- it samples the stream)')

    # locked settings are what make the camera-to-camera offset a constant
    w = meta.get('wrist', {})
    for label_, val, want in (('wrist auto exposure', w.get('auto_exposure'), 0),
                              ('wrist auto white balance', w.get('auto_white_balance'), 0)):
        if val is None:
            continue
        if float(val) != want:
            print(BAD + f' {label_} is ON -- exposure midpoint or colour will drift '
                        'between episodes')
            fails_top.append(f'{label_} not locked')
    sc = meta.get('scene', {}).get('controls', '')
    if sc and ('white_balance_automatic: 1' in sc or 'auto_exposure: 3' in sc):
        print(BAD + f' scene camera not fully locked: {sc}')
        fails_top.append('scene controls not locked')

    # Focus and zoom are geometry, not brightness: autofocus changes the focal
    # length, so footage shot with it on does not match the intrinsics that were
    # calibrated, and the ChArUco pose solve returns plausible wrong depths with
    # no error anywhere downstream. This is the only place that catches it.
    if sc and 'focus_automatic_continuous: 1' in sc:
        print(BAD + ' scene camera AUTOFOCUS WAS ON -- focal length drifts during '
                    'recording, so scene_intrinsics.json does not apply to this '
                    'session and every solved pose is suspect')
        fails_top.append('scene autofocus not locked')
    fails_top += check_scene_geometry(d, meta)

    fails = list(fails_top)
    if not eps:
        print(BAD + ' no episode directories found')
        sys.exit(1)
    for ep in eps:
        check_episode(os.path.join(d, ep), meta, fails)

    print('\n' + '=' * 64)
    if fails:
        print(f'FAILED ({len(fails)}):')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print(f'all checks passed across {len(eps)} episodes -- session is safe to use')


if __name__ == '__main__':
    main()
