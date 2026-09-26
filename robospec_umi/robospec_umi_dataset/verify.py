#!/usr/bin/env python3
"""Check a capture session before you trust it for training.

The failures this catches are the ones that leave no visible trace in the data. A
sidecar one entry longer than its video shifts every subsequent frame by ~10 ms,
which is ~1 cm of TCP label error applied silently for the rest of the recording;
the video still plays and the poses still look reasonable. Likewise, arrival-based
wrist timestamps would look perfectly plausible while carrying 12 ms of jitter.

verify_session() returns the result per episode; a session-level failure (focus
or zoom not matching the calibration) fails every episode.

Usage:
    python3 robospec_umi/robospec_umi_dataset/verify.py data/capture/20260919_150000
"""

import json
import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_dataset
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, 'data')
CALIB_JSON = os.path.join(DATA, 'calibration', 'scene_intrinsics.json')

OK, BAD, WARN = '  [ok]  ', '  [FAIL]', '  [warn]'


class Log:
    """Collects the report lines plus the fails / warnings they announce."""

    def __init__(self):
        self.lines, self.fails, self.warns = [], [], []

    def __call__(self, s=''):
        self.lines.append(s)

    def fail(self, line, why):
        self(BAD + line)
        self.fails.append(why)

    def warn(self, line, why):
        self(WARN + line)
        self.warns.append(why)


def check_invariant(label, n_video, n_ts, log):
    """One timestamp per written frame, or every later frame is shifted.

    A mismatch is normally an interrupted recording: the sidecar is checkpointed
    about once a second and the encoder holds a few frames.
    """
    if n_video == n_ts:
        log(OK + f' index invariant: {n_video} frames == {n_ts} timestamps')
        return
    n = min(n_video, n_ts)
    log.fail(f' index invariant BROKEN: {n_video} frames vs {n_ts} timestamps',
             f'{label} index mismatch ({n_video} video vs {n_ts} timestamps)')
    log(f'         looks like an interrupted recording '
        f'({"video ahead" if n_video > n_ts else "sidecar ahead"}). '
        f'Both are in capture order, so use the first {n} frames of each and '
        f'discard the tail.')


def count_packets(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return sum(1 for p in c.demux(s) if p.pts is not None and p.size > 0)


def intervals(t_ns, label, nominal_fps, log):
    t = t_ns.astype(np.float64) / 1e9
    d = np.diff(t) * 1000
    period = 1000.0 / nominal_fps
    log(f'    {len(t)} frames over {t[-1]-t[0]:.2f} s = '
        f'{len(t)/(t[-1]-t[0]):.2f} fps  (nominal {nominal_fps})')
    log(f'    interval ms: median {np.median(d):7.3f}  p1 {np.percentile(d,1):7.3f}  '
        f'p99 {np.percentile(d,99):7.3f}  max {d.max():8.3f}')
    if not np.all(np.diff(t) > 0):
        log.fail(f' {label}: timestamps are not strictly increasing',
                 f'{label} non-monotonic timestamps')
    gaps = int(np.sum(d > 1.5 * period))
    log(f'    {gaps} gaps > 1.5x nominal period'
        + (f'  ({100*gaps/len(d):.1f}% of intervals)' if gaps else ''))


def check_scene(d, meta, log):
    log('\nSCENE')
    vid, side = os.path.join(d, 'scene', 'scene.mkv'), os.path.join(d, 'scene', 'scene_ts.npz')
    if not (os.path.exists(vid) and os.path.exists(side)):
        log.fail(' missing scene.mkv or scene_ts.npz', 'scene files missing')
        return
    t_ns = np.load(side)['t_ns']
    if t_ns.dtype != np.int64:
        log.fail(f' t_ns dtype is {t_ns.dtype}, must be int64', 'scene t_ns dtype')
    check_invariant('scene', count_packets(vid), len(t_ns), log)
    intervals(t_ns, 'scene', meta['scene']['fps_requested'], log)
    with av.open(vid) as c:
        codec = c.streams.video[0].codec_context.name
    log(f'    codec {codec}' + (OK.strip() + ' passthrough preserved'
                                if codec == 'mjpeg' else WARN.strip() + ' re-encoded?'))
    log(f'    size {os.path.getsize(vid)/1e6:.1f} MB '
        f'({os.path.getsize(vid)/1e6/max((t_ns[-1]-t_ns[0])/1e9, 1e-9):.1f} MB/s)')


def check_wrist(d, meta, log):
    log('\nWRIST')
    vid, side = os.path.join(d, 'wrist', 'wrist.mkv'), os.path.join(d, 'wrist', 'wrist_ts.npz')
    if not (os.path.exists(vid) and os.path.exists(side)):
        log.fail(' missing wrist.mkv or wrist_ts.npz', 'wrist files missing')
        return
    z = np.load(side)
    t_ns, dev, arr, seq = z['t_ns'], z['t_device_ns'], z['t_arrival_ns'], z['seq']
    check_invariant('wrist', count_packets(vid), len(t_ns), log)

    doms = meta.get('wrist', {}).get('timestamp_domains', [])
    if doms == ['timestamp_domain.global_time']:
        log(OK + f' timestamp domain {doms}')
    else:
        log.fail(f' timestamp domain {doms} -- expected global_time only. '
                 'Without it the stamps are host-arrival and carry ~12 ms jitter.',
                 f'wrist timestamp domain {doms}')

    intervals(t_ns, 'wrist', meta['wrist']['fps'], log)

    # device-vs-arrival is the evidence that device stamps are actually being used
    lag = (arr - dev + int(meta['mono_to_epoch_offset_s'] * 1e9)) / 1e6
    log(f'    arrival - device: median {np.median(lag):6.2f} ms   '
        f'spread(p1-p99) {np.percentile(lag,99)-np.percentile(lag,1):6.2f} ms')
    jit_dev = np.diff(t_ns.astype(np.float64)/1e6)
    jit_arr = np.diff(arr.astype(np.float64)/1e6)
    iqr = lambda x: np.percentile(x, 75) - np.percentile(x, 25)  # noqa: E731
    log(f'    interval jitter (IQR): device {iqr(jit_dev):.3f} ms  vs  '
        f'arrival {iqr(jit_arr):.3f} ms   <- why global time is used')

    gaps = np.diff(seq) - 1
    nd = int(gaps[gaps > 0].sum())
    if nd == 0:
        log(OK + ' no device-side frame drops (seq is contiguous)')
    else:
        log.warn(f' device dropped {nd} frames in {int((gaps>0).sum())} bursts. '
                 'Harmless -- resampling is by time, not index -- but worth knowing.',
                 f'wrist dropped {nd} frames')
    log(f'    size {os.path.getsize(vid)/1e6:.1f} MB')


def check_overlap(d, log):
    """Both streams must cover the same wall-clock window."""
    sp = os.path.join(d, 'scene', 'scene_ts.npz')
    wp = os.path.join(d, 'wrist', 'wrist_ts.npz')
    if not (os.path.exists(sp) and os.path.exists(wp)):
        return
    s, w = np.load(sp)['t_ns'], np.load(wp)['t_ns']
    lo, hi = max(s[0], w[0]), min(s[-1], w[-1])
    log('\nOVERLAP (both on CLOCK_MONOTONIC)')
    log(f'    scene {s[0]/1e9:.3f} .. {s[-1]/1e9:.3f}')
    log(f'    wrist {w[0]/1e9:.3f} .. {w[-1]/1e9:.3f}')
    if hi <= lo:
        log.fail(' the two streams do not overlap at all', 'no temporal overlap')
        return
    inside = int(((w >= lo) & (w <= hi)).sum())
    log(OK + f' overlap {(hi-lo)/1e9:.2f} s; {inside}/{len(w)} wrist frames '
             f'({100*inside/len(w):.1f}%) fall inside the scene track')
    if inside < len(w):
        log(WARN + f' {len(w)-inside} wrist frames sit outside and cannot be '
                   'labelled -- they must be dropped, never extrapolated')


def check_scene_geometry(meta, log, path):
    """Did this session record at the focus and zoom the intrinsics were shot at?

    A different focus means a different focal length, so every solved pose is
    scaled and nothing downstream can notice. Silent when there is no calibration.
    """
    try:
        with open(path) as f:
            locked = json.load(f).get('locked_controls') or {}
    except (ValueError, OSError):
        return
    scene = meta.get('scene', {})
    for key in ('focus_absolute', 'zoom_absolute'):
        want, got = locked.get(key), scene.get(key)
        # negative means the flag was "leave this control alone"
        if want is None or got is None or got < 0:
            continue
        if int(want) != int(got):
            log.fail(f' scene {key}={got} but {os.path.basename(path)} was '
                     f'calibrated at {want} -- the intrinsics do not apply to '
                     f'this session', f'scene {key} differs from calibration')
        else:
            log(OK + f' scene {key}={got} matches the calibration')


def check_session(meta, log, intrinsics):
    """Session-wide checks. Auto exposure / white balance are allowed: they
    change brightness and colour, not geometry or timing."""
    dsp = meta.get('display', {})
    if dsp.get('enabled'):
        sk = dsp.get('frames_skipped', {})
        log(f'  preview was on at {dsp.get("fps")} fps, skipped '
            + ', '.join(f'{k} {v}' for k, v in sk.items())
            + ' frames (by design -- it samples the stream)')

    sc = meta.get('scene', {}).get('controls', '')
    # Focus and zoom are geometry: autofocus changes the focal length, so the
    # calibrated intrinsics no longer apply and every solved pose is suspect.
    if 'focus_automatic_continuous: 1' in sc:
        log.fail(' scene camera AUTOFOCUS WAS ON -- focal length drifts during '
                 'recording, so scene_intrinsics.json does not apply to this '
                 'session and every solved pose is suspect', 'scene autofocus on')
    check_scene_geometry(meta, log, intrinsics)


def verify_session(d, intrinsics=None):
    """-> {session, ok, session_fails, warnings, episodes: {ep: {ok, fails, warnings}}, lines}"""
    d = os.path.abspath(d)
    intrinsics = intrinsics or os.environ.get('SCENE_INTRINSICS', CALIB_JSON)
    with open(os.path.join(d, 'session.json')) as f:
        meta = json.load(f)
    eps = sorted(x for x in os.listdir(d)
                 if x.startswith('ep') and os.path.isdir(os.path.join(d, x)))
    top = Log()
    top(f'session {meta["session"]}   {len(eps)} episodes')
    check_session(meta, top, intrinsics)
    if not eps:
        top.fail(' no episode directories found', 'no episodes')

    lines, episodes = list(top.lines), {}
    for ep in eps:
        log = Log()
        log(f'\n{"="*64}\nEPISODE {ep}')
        p = os.path.join(d, ep)
        if os.path.isdir(os.path.join(p, 'scene')):
            check_scene(p, meta, log)
        if os.path.isdir(os.path.join(p, 'wrist')):
            check_wrist(p, meta, log)
        check_overlap(p, log)
        fails = top.fails + log.fails
        episodes[ep] = {'ok': not fails, 'fails': fails, 'warnings': log.warns}
        lines += log.lines
    return {'session': meta['session'], 'ok': all(e['ok'] for e in episodes.values()) and not top.fails,
            'session_fails': top.fails, 'warnings': top.warns, 'episodes': episodes,
            'lines': lines}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = os.path.abspath(sys.argv[1])
    if not os.path.exists(os.path.join(d, 'session.json')):
        sys.exit(f'no session.json in {d}')
    r = verify_session(d)
    print('\n'.join(r['lines']))
    fails = r['session_fails'] + [f'{ep}: {f}' for ep, e in r['episodes'].items()
                                  for f in e['fails'] if f not in r['session_fails']]
    print('\n' + '=' * 64)
    if fails:
        print(f'FAILED ({len(fails)}):')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print(f'all checks passed across {len(r["episodes"])} episodes -- session is safe to use')


if __name__ == '__main__':
    main()
