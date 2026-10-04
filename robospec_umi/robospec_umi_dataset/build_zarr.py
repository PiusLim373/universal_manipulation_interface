#!/usr/bin/env python3
"""Turn capture sessions into the dataset.zarr.zip that UmiDataset trains from.

The SLAM pipeline cannot do this: scripts_slam_pipeline/00-06 are GoPro/SLAM/IMU
specific, and 07 assumes frames are evenly spaced in time. Ours are not -- the
wrist drops frames in bursts -- so this emits 07's schema directly.

Poses are written ABSOLUTE, never pre-subtracted: UmiDataset converts to relative
at training time, so subtracting here would convert twice.

Each episode is prepared first (episode_prep: ChArUco detection + pose track,
cached in <ep>/derived/; missing ones in parallel, without the preview videos),
then resampled onto the 60 Hz grid (timeline.py).

Usage:
    python3 robospec_umi/robospec_umi_dataset/build_zarr.py \\
        data/capture/20260919_150000 -o data/dataset/dataset.zarr.zip
    python3 robospec_umi/robospec_umi_dataset/build_zarr.py \\
        --project data/dataset/20260925_101500_dataset.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import cv2
import numpy as np
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_dataset
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, 'data')
CAPTURE_ROOT = os.path.join(DATA, 'capture')
DATASET_ROOT = os.path.join(DATA, 'dataset')

# The repo root carries diffusion_policy/ and umi/, which are not installed.
sys.path.append(REPO)

from diffusion_policy.common.replay_buffer import ReplayBuffer          # noqa: E402
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl  # noqa: E402
from umi.common.cv_util import get_image_transform                      # noqa: E402

import episode_prep as P                                                # noqa: E402
import timeline as TL                                                   # noqa: E402

register_codecs()

CALIB_JSON = P.CALIB_JSON
OUT_RES = 224           # UmiDataset's camera0_rgb edge
COMPRESSION_LEVEL = 99  # JpegXl quality for camera0_rgb
PREP_WORKERS = 3        # parallel episode preps for anything not yet cached


def check_intrinsics(session_dir, path):
    """Refuse a calibration that does not describe this session.

    A mismatch produces plausible, wrong poses with no error downstream, so this
    is fatal rather than a warning. verify.py makes the same focus/zoom check.
    """
    if not os.path.exists(path):
        sys.exit(f'{path} not found -- run calibrate_scene_cam.py solve, or pass '
                 f'--intrinsics explicitly')
    with open(path) as f:
        intr = json.load(f)
    kind = intr.get('intrinsic_type', '?')
    if kind != 'PINHOLE':
        sys.exit(f'{path} is {kind}, expected PINHOLE -- projection models are '
                 f'not interchangeable')
    scene = {}
    p = os.path.join(session_dir, 'session.json')
    if os.path.exists(p):
        with open(p) as f:
            scene = json.load(f).get('scene', {})
    locked = intr.get('locked_controls') or {}
    for key in ('focus_absolute', 'zoom_absolute'):
        wanted, got = locked.get(key), scene.get(key)
        if wanted is None or got is None or int(got) < 0:
            continue
        if int(wanted) != int(got):
            sys.exit(f'{os.path.basename(session_dir)}: scene {key}={got} but the '
                     f'calibration was shot at {wanted} -- re-calibrate, or pass the '
                     f'matching --intrinsics')


def episode_lowdim(ep):
    """The six low-dim arrays for one segment. Only demo_start is read by
    UmiDataset; demo_end is carried for schema parity with 07."""
    p = ep['pose']
    n = len(p)
    start = np.broadcast_to(ep['demo_start'], (n, 6)).astype(np.float32)
    end = np.broadcast_to(ep['demo_end'], (n, 6)).astype(np.float32)
    return {
        'robot0_eef_pos': p[:, :3].copy(),
        'robot0_eef_rot_axis_angle': p[:, 3:].copy(),
        'robot0_gripper_width': TL.gripper_on_grid(ep['dir'], ep['grid']),
        'robot0_demo_start_pose': start.copy(),
        'robot0_demo_end_pose': end.copy(),
    }


def fill_images(ep, img_array, buffer_start, tf):
    """Decode the wrist video once, writing the selected frames to their slots.
    Sequential decode, not seeking: h264 random access is unreliable."""
    want = {}
    for i, fi in enumerate(ep['wrist_idx']):
        want.setdefault(int(fi), []).append(buffer_start + i)
    c = av.open(os.path.join(ep['dir'], 'wrist', 'wrist.mkv'))
    st = c.streams.video[0]
    st.thread_count = 4
    written = 0
    for idx, frame in enumerate(c.decode(st)):
        slots = want.pop(idx, None)
        if slots is None:
            continue
        img = tf(frame.to_ndarray(format='rgb24'))
        for s in slots:
            img_array[s] = img
        written += len(slots)
        if not want:
            break
    c.close()
    return written


def episode_dirs(session_dir):
    return [os.path.join(session_dir, d) for d in sorted(os.listdir(session_dir))
            if d.startswith('ep') and os.path.isdir(os.path.join(session_dir, d))]


def load_project(path):
    """Edit-project json -> ([(ep_dir, trim)], output path). Excluded and
    verify-failed episodes are left out."""
    with open(path) as f:
        proj = json.load(f)
    items = []
    for sess in proj['sessions']:
        ver = (proj.get('verify') or {}).get(sess, {}).get('episodes', {})
        for d in episode_dirs(os.path.join(CAPTURE_ROOT, sess)):
            ep = os.path.basename(d)
            e = proj.get('episodes', {}).get(f'{sess}/{ep}', {})
            if e.get('excluded') or not ver.get(ep, {}).get('ok', False):
                continue
            items.append((d, e.get('trim')))
    return items, os.path.join(DATASET_ROOT, f'{proj["id"]}_dataset.zarr.zip')


def ep_name(d):
    return f'{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}'


def prepare_missing(dirs, intr_path, workers):
    """Detection + pose track for episodes the trimmer never opened, in parallel.
    No preview videos: nobody is going to watch them."""
    key = P.cache_key(intr_path)
    todo = [d for d in dirs if not P.is_prepped(d, key, videos=False)]
    if not todo:
        return
    print(f'preparing {len(todo)} episode(s), {workers} at a time '
          f'(detection + pose track, no videos) ...', flush=True)
    t0 = time.time()
    # spawn, not fork: a forked child inherits whatever cv2/av threads exist
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        futs = {ex.submit(P.prepare_tracks, d, intr_path): d for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            m = f.result()
            print(f'  [{k}/{len(todo)}] {ep_name(futs[f])}: {P.summary(m)}  '
                  f'({m["seconds"]} s)', flush=True)
    print(f'prepared {len(todo)} in {time.time() - t0:.0f} s\n', flush=True)


def build(items, output, intr_path, workers=PREP_WORKERS):
    """[(ep_dir, trim)] -> dataset.zarr.zip. Returns the report dict."""
    t_start = time.time()
    for s in sorted({os.path.dirname(d) for d, _ in items}):
        check_intrinsics(s, intr_path)
    with open(intr_path) as f:
        intr = {'run': json.load(f).get('source_run'), 'sha1': P.sha1(intr_path)}
    print(f'{len(items)} episodes, intrinsics {intr_path}\n', flush=True)
    prepare_missing([d for d, _ in items], intr_path, workers)

    kept, dropped = [], []
    for k, (d, trim) in enumerate(items):
        name = ep_name(d)
        print(f'[{k+1}/{len(items)}] {name}'
              + (f'  trim {trim[0]}..{trim[1]} s' if trim else '') + ' ...', flush=True)
        P.prepare(d, intr_path, videos=False)      # no-op: prepare_missing did it
        plan = TL.plan_episode(d, trim)
        print(f'      {plan["note"]}', flush=True)
        if plan['reason']:
            print(f'      dropped: {plan["reason"]}', flush=True)
            dropped.append({'episode': name, 'reason': plan['reason']})
            continue
        segs = plan['segments']
        fused = bool(((P.read_prep(d) or {}).get('imu') or {}).get('fused'))
        for s in segs:
            s['name'] = f'{name}_s{s["seg"]}' if len(segs) > 1 else name
            s['imu_fused'] = fused
            print(f'      -> {s["name"]}: {len(s["grid"])} steps '
                  f'({len(s["grid"])/TL.GRID_HZ:.1f}s)', flush=True)
            kept.append(s)

    if not kept:
        sys.exit('\nno segments passed -- nothing to write')

    # low-dim first, then one image array sized to the total, filled by offset
    out_res = (OUT_RES, OUT_RES)
    rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    starts, total = [], 0
    for ep in kept:
        starts.append(total)
        rb.add_episode(data=episode_lowdim(ep), compressors=None)
        total += len(ep['grid'])

    img_array = rb.data.require_dataset(
        name='camera0_rgb', shape=(total,) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=JpegXl(level=COMPRESSION_LEVEL, numthreads=1),
        dtype=np.uint8)

    sizes = set()
    for ep in kept:
        with av.open(os.path.join(ep['dir'], 'wrist', 'wrist.mkv')) as c:
            st = c.streams.video[0]
            sizes.add((st.width, st.height))
    if len(sizes) > 1:      # one crop per dataset, and the model learns one view
        sys.exit(f'wrist frame sizes differ across episodes ({sorted(sizes)}); '
                 'build D405 and D455 recordings into separate datasets')
    (w, h), = sizes
    x0, side = TL.wrist_crop(w, h)
    tf = get_image_transform((w, h), out_res, crop_x=x0)
    print(f'\nimage transform: {w}x{h} -> crop x {x0}-{x0 + side} -> '
          f'{out_res[0]}x{out_res[1]}', flush=True)
    for ep, buf in zip(kept, starts):
        n = fill_images(ep, img_array, buf, tf)
        print(f'  {ep["name"]}: {n}/{len(ep["grid"])} frames written', flush=True)

    print(f'\nsaving {total} steps across {len(kept)} episodes -> {output}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
    tmp = output + '.tmp'
    with zarr.ZipStore(tmp, mode='w') as zs:
        rb.save_to_store(store=zs)
        # what a robot-side deploy must reproduce; save_to_store drops root attrs
        zarr.group(zs).attrs['wrist_crop'] = [int(x0), 0, int(side), int(side)]
    os.replace(tmp, output)

    size = os.path.getsize(output) / 1e6
    print(f'\n{"="*60}')
    print(f'included {len(kept)} episodes, {total} steps at {TL.GRID_HZ:.0f} Hz '
          f'({total/TL.GRID_HZ:.1f} s)')
    print(f'imu fused: {sum(s["imu_fused"] for s in kept)}/{len(kept)} episodes')
    for e in dropped:
        print(f'  excluded {e["episode"]}: {e["reason"]}')
    print(f'wrote {output}  ({size:.1f} MB)', flush=True)
    return {'path': output, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'seconds': round(time.time() - t_start, 1), 'episodes': len(kept),
            'recordings': len(items) - len(dropped), 'steps': int(total),
            'duration_s': round(total / TL.GRID_HZ, 2), 'size_mb': round(size, 1),
            'intrinsics': intr,
            'included': [{'episode': s['name'], 'steps': len(s['grid']),
                          'imu_fused': s['imu_fused']} for s in kept],
            'dropped': dropped}


def write_export(project, report):
    """Merge the report into the project json (re-read: the UI may have saved)."""
    with open(project) as f:
        proj = json.load(f)
    proj['export'] = report
    tmp = project + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(proj, f, indent=2)
    os.replace(tmp, project)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sessions', nargs='*', help='capture session directories')
    ap.add_argument('-o', '--output', help='dataset.zarr.zip path')
    ap.add_argument('--project', help='edit-project json (sessions, exclusions, trims)')
    ap.add_argument('--intrinsics', default=CALIB_JSON,
                    help=f'scene-camera calibration (default: {CALIB_JSON})')
    ap.add_argument('--workers', type=int, default=PREP_WORKERS,
                    help='parallel preps for episodes not yet cached')
    args = ap.parse_args()
    av.logging.set_level(av.logging.PANIC)   # MJPEG APP-marker and pix-fmt noise
    cv2.setNumThreads(4)

    if args.project:
        items, output = load_project(args.project)
        output = args.output or output
    else:
        if not args.sessions or not args.output:
            ap.error('pass session directories and -o, or --project')
        items = [(d, None) for s in args.sessions
                 for d in episode_dirs(os.path.abspath(os.path.expanduser(s)))]
        output = args.output
    if not items:
        sys.exit('no episodes to build')

    report = build(items, output, args.intrinsics, args.workers)
    if args.project:
        write_export(args.project, report)


if __name__ == '__main__':
    main()
