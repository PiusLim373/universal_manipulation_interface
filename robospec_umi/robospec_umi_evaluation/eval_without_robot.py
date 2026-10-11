"""Run a trained UMI policy on a recording, with no robot attached.

The model is fed what it would get on the robot: the raw wrist video, cropped and
resized exactly as for training, plus the TCP pose and gripper width at each
step, built into its input by the robot-side helpers in
umi/real_world/real_inference_util.py. At every observation step it forecasts
the next 0.75 s. Each forecast is scored against what the hand actually did, and
against standing still, and is saved for the Eval Without Robot page.

    python3 robospec_umi/robospec_umi_evaluation/eval_without_robot.py \\
        -c data/outputs/<date>/<run>/checkpoints/<epoch>.ckpt -e data/capture/<session>/ep003

The checkpoint names the dataset it trained on, so the episode is labelled held
out (never trained on), trained on, or not in that dataset; a trimmed episode is
evaluated over exactly the span that went into the dataset. Results go to
data/evaluation/<checkpoint>/<session>_<ep>.{json,npz}.

Open loop: every forecast starts from the recorded pose, so errors never compound
the way they will on a robot. Read the numbers as a lower bound.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_evaluation
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')
OUTPUTS = os.path.join(DATA, 'outputs')
EVAL_ROOT = os.path.join(DATA, 'evaluation')
CAPTURE_ROOT = os.path.join(DATA, 'capture')
DATASET_ROOT = os.path.join(DATA, 'dataset')
CALIB_JSON = os.path.join(DATA, 'calibration', 'scene_intrinsics.json')

# The repo root carries diffusion_policy/ and umi/, which are not installed.
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(PKG, 'robospec_umi_dataset'))

import episode_prep as EP              # noqa: E402
import timeline as TL                  # noqa: E402
import video_processor_charuco as V    # noqa: E402

BATCH = 16          # forecasts per forward pass
POSE_TOL = 1e-5     # m; a recomputed pose must match the dataset's this closely
N_CHECK = 6         # images per segment compared with the dataset's


# ------------------------------------------------------------- checkpoint info
def read_cfg(ckpt):
    """-> (cfg, {epoch, global_step}) without loading the weights: only data.pkl
    is unpickled, with every tensor left out (0.06 s instead of ~5 s)."""
    import dill

    class Head(dill.Unpickler):
        def persistent_load(self, pid):
            return None

        def find_class(self, mod, name):
            if mod.startswith('torch'):
                return lambda *a, **k: None
            return super().find_class(mod, name)

    with zipfile.ZipFile(ckpt) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))
    p = Head(io.BytesIO(raw)).load()
    meta = {k: dill.loads(v) for k, v in p.get('pickles', {}).items()
            if k in ('epoch', 'global_step')}
    return p['cfg'], meta


def describe(ckpt):
    """What the checkpoint was trained on, and how that dataset was split.

    recordings: 'sess/epNNN' -> its segments in the dataset, each with the zarr
    episode index and held_out. Episode i of the zarr is export.included[i] of
    its project file, and the split is UmiDataset's own get_val_mask.
    """
    from diffusion_policy.common.sampler import get_val_mask
    cfg, meta = read_cfg(ckpt)
    name = os.path.basename(str(cfg.task.dataset_path))
    zarr_path = os.path.join(DATASET_ROOT, name)
    m = re.match(r'^(\d{8}_\d{6})_dataset\.zarr\.zip$', name)
    proj = None
    if m and os.path.exists(os.path.join(DATASET_ROOT, f'{m.group(1)}_dataset.json')):
        with open(os.path.join(DATASET_ROOT, f'{m.group(1)}_dataset.json')) as f:
            proj = json.load(f)
    included = ((proj or {}).get('export') or {}).get('included') or []
    ds = cfg.task.dataset
    val = get_val_mask(len(included), ds.val_ratio, ds.seed) if included else []
    recordings = {}
    for i, e in enumerate(included):
        rec, _, seg = e['episode'].partition('_s')
        recordings.setdefault(rec, []).append(
            {'seg': int(seg or 0), 'steps': e['steps'], 'zarr_index': i, 'held_out': bool(val[i])})
    a = cfg.task.shape_meta.action
    return {'cfg': cfg, 'epoch': meta.get('epoch'), 'global_step': meta.get('global_step'),
            'dataset': name, 'project': m and m.group(1), 'project_file': proj,
            'zarr': zarr_path if os.path.exists(zarr_path) else None,
            'recordings': recordings, 'horizon': int(a.horizon), 'down_sample': int(a.down_sample_steps),
            'n_action_steps': cfg.get('n_action_steps'), 'run_dir': cfg.multi_run.run_dir}


def result_dir(ckpt):
    """data/evaluation/<the name the Training page downloads the checkpoint as>."""
    rel = os.path.relpath(os.path.realpath(ckpt), os.path.realpath(OUTPUTS))
    if rel.startswith('..'):
        rel = os.path.basename(ckpt)
    name = rel.replace(os.sep + 'checkpoints' + os.sep, os.sep).replace(os.sep, '_')
    return os.path.join(EVAL_ROOT, re.sub(r'\.ckpt$', '', name))


def episode_key(ep_dir):
    return os.path.relpath(os.path.abspath(ep_dir), CAPTURE_ROOT)


def result_paths(ckpt, ep_dir):
    stem = os.path.join(result_dir(ckpt), episode_key(ep_dir).replace(os.sep, '_'))
    return stem + '.json', stem + '.npz'


def read_result(ckpt, ep_dir):
    try:
        with open(result_paths(ckpt, ep_dir)[0]) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------- inputs
def wrist_frames(ep_dir, need, out_res):
    """{frame index: image} for the frames needed, cropped and resized as
    build_zarr does it. Sequential decode: h264 random access is unreliable."""
    from umi.common.cv_util import get_image_transform
    out, last = {}, max(need)
    with av.open(os.path.join(ep_dir, 'wrist', 'wrist.mkv')) as c:
        st = c.streams.video[0]
        st.thread_count = 4
        tf = get_image_transform((st.width, st.height), out_res,
                                 crop_x=TL.wrist_crop(st.width, st.height)[0])
        for i, f in enumerate(c.decode(st)):
            if i in need:
                out[i] = tf(f.to_ndarray(format='rgb24'))
            if i >= last:
                break
    return out


def obs_index(i, horizon, down, pad):
    """UmiDataset's history for grid step i: images repeat the first valid frame
    (pad), low-dim clips at the segment start."""
    if not pad:
        return [max(i - k * down, 0) for k in reversed(range(horizon))]
    n = min(horizon, i // down + 1)
    idx = [i - k * down for k in reversed(range(n))]
    return [idx[0]] * (horizon - n) + idx


def check_rows(n):
    """The steps of a segment whose images are compared with the dataset's."""
    return np.linspace(0, n - 1, N_CHECK).astype(int)


def check_dataset(info, key, segs, frames):
    """Do the recomputed segments reproduce the dataset? -> verify dict.

    Steps and poses must match the zarr exactly, or a trim changed since the
    export. The raw-path images differ only by the zarr's lossy compression;
    input_diff reports it in /255.
    """
    mine = info['recordings'].get(key)
    if mine is None:
        return {'status': 'not in dataset'}
    if len(mine) != len(segs) or any(m['steps'] != len(s['grid']) for m, s in zip(mine, segs)):
        return {'status': 'mismatch', 'detail': 'segment steps differ from the export: '
                'the dataset was exported with another trim'}
    if info['zarr'] is None:
        return {'status': 'steps only', 'detail': f'{info["dataset"]} is not here'}
    import zarr
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
    z = zarr.open(info['zarr'], 'r')
    ends = z['meta/episode_ends'][:]
    pos, rot, grip = z['data/robot0_eef_pos'], z['data/robot0_eef_rot_axis_angle'], z['data/robot0_gripper_width']
    diffs = []
    for m, s in zip(mine, segs):
        a, b = (ends[m['zarr_index'] - 1] if m['zarr_index'] else 0), ends[m['zarr_index']]
        dp = max(np.abs(s['pose'][:, :3] - pos[a:b]).max(), np.abs(s['pose'][:, 3:] - rot[a:b]).max())
        dg = np.abs(TL.gripper_on_grid(s['dir'], s['grid']) - grip[a:b]).max()
        if dp > POSE_TOL or dg > POSE_TOL:
            return {'status': 'mismatch', 'detail': f'poses differ from the export by {dp * 1000:.2f} mm: '
                    're-prepped with another intrinsic or pipeline since'}
        for k in check_rows(b - a):
            diffs.append(np.abs(z['data/camera0_rgb'][a + k].astype(np.int16)
                                - frames[int(s['wrist_idx'][k])].astype(np.int16)).mean())
    return {'status': 'verified', 'input_diff': round(float(np.mean(diffs)), 2)}


# -------------------------------------------------------------------- model
def load_policy(ckpt, device):
    """-> (policy, cfg). Every weight comes from the checkpoint, so a ViT encoder
    is built without its pretrained download (no BatchNorm, same architecture)."""
    import dill
    import hydra
    import torch
    from omegaconf import open_dict
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    enc = cfg.policy.obs_encoder
    if str(enc.get('model_name', '')).startswith('vit'):
        with open_dict(cfg):
            enc.pretrained = False
    ws = hydra.utils.get_class(cfg._target_)(cfg)
    ws.load_payload(payload)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    return policy.eval().to(device), cfg


def split_of(mine, si):
    """held_out / trained_on for a segment of a dataset recording."""
    if not mine:
        return 'not_in_dataset'
    return 'held_out' if mine[min(si, len(mine) - 1)]['held_out'] else 'trained_on'


def angle_deg(Ra, Rb):
    """Geodesic angle between rotation stacks, degrees."""
    c = (np.einsum('nij,nij->n', Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stamp(ckpt, ep_dir, intr_path, seed, stride):
    """What a result was made from; a cached one is reused only while it all matches."""
    st = os.stat(ckpt)
    with open(intr_path, 'rb') as f:
        intr = hashlib.sha1(f.read()).hexdigest()
    return {'checkpoint_size': st.st_size, 'checkpoint_mtime': int(st.st_mtime),
            'prepared_at': (EP.read_prep(ep_dir) or {}).get('prepared_at'),
            'intrinsics_sha1': intr, 'seed': seed, 'stride': stride}


# ----------------------------------------------------------------- evaluate
def evaluate(ckpt, ep_dir, intr_path=CALIB_JSON, seed=0, stride=None, force=False, progress=None):
    """Forecast along one recording and score it. -> the result json dict."""
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
    from umi.common.pose_util import pose_to_mat

    progress = progress or (lambda *a: None)
    ep_dir, t_start = os.path.abspath(ep_dir), time.time()
    key = episode_key(ep_dir)
    info = describe(ckpt)
    down, horizon = info['down_sample'], info['horizon']
    stride = stride or down
    EP.prepare(ep_dir, intr_path, progress, videos=False)    # a no-op once prepped
    st = stamp(ckpt, ep_dir, intr_path, seed, stride)
    old = read_result(ckpt, ep_dir)
    if old and not force and old.get('stamp') == st:
        return old

    mine = info['recordings'].get(key)
    proj = info['project_file'] or {}
    trim = ((proj.get('episodes') or {}).get(key) or {}).get('trim') if mine else None
    plan = TL.plan_episode(ep_dir, trim)
    segs = plan['segments']
    if not segs:
        raise SystemExit(f'{key}: nothing to evaluate ({plan["reason"]})')
    t0 = TL.load_track(ep_dir)['t_ns'][0] / 1e9

    meta = info['cfg'].task.shape_meta
    obs_meta = meta.obs
    rgb = [k for k, v in obs_meta.items() if v.get('type') == 'rgb']
    hr, dr = obs_meta[rgb[0]].horizon, obs_meta[rgb[0]].down_sample_steps
    hl, dl = obs_meta['robot0_eef_pos'].horizon, obs_meta['robot0_eef_pos'].down_sample_steps
    out_res = tuple(obs_meta[rgb[0]].shape[1:][::-1])
    starts = [list(range(0, len(s['grid']), stride)) for s in segs]

    progress('decode', 0, 1)
    need = {int(s['wrist_idx'][j]) for s, st_ in zip(segs, starts)
            for i in [*st_, *check_rows(len(s['grid']))] for j in obs_index(i, hr, dr, True)}
    frames = wrist_frames(ep_dir, need, out_res)
    verify = check_dataset(info, key, segs, frames)

    progress('load', 0, 1)
    t_load = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, cfg = load_policy(ckpt, device)
    t_load = time.time() - t_load
    repr_ = cfg.task.pose_repr

    with av.open(os.path.join(ep_dir, 'scene', 'scene.mkv')) as c:
        size = (c.streams.video[0].width, c.streams.video[0].height)
    cam = V.load_intrinsics(intr_path, *size)

    def px(p):
        q, good = cam.project(p, np.zeros(3), np.zeros(3))
        q[~good] = np.nan
        return q

    torch.manual_seed(seed)
    total, done = sum(map(len, starts)), 0
    F = {k: [] for k in ('seg', 'i', 't', 'xyz', 'rpy', 'grip', 'px')}
    grid = {k: [] for k in ('t', 'seg', 'px')}
    scores = {}
    t_inf = time.time()
    for si, (s, st_) in enumerate(zip(segs, starts)):
        pose, n = s['pose'].astype(np.float64), len(s['grid'])
        grip = TL.gripper_on_grid(s['dir'], s['grid'])[:, 0].astype(np.float64)
        R = pose_to_mat(pose)[:, :3, :3]
        grid['t'].append(s['grid'] - t0)
        grid['seg'].append(np.full(n, si))
        grid['px'].append(px(pose[:, :3]))
        split = split_of(mine, si)
        sc = scores.setdefault(split, {'n': 0, 'err': {}, 'still': {}})
        for b0 in range(0, len(st_), BATCH):
            batch, envs = [], []
            for i in st_[b0:b0 + BATCH]:
                li = obs_index(i, hl, dl, False)
                env = {rgb[0]: np.stack([frames[int(s['wrist_idx'][j])] for j in obs_index(i, hr, dr, True)]),
                       'robot0_eef_pos': pose[li, :3], 'robot0_eef_rot_axis_angle': pose[li, 3:],
                       'robot0_gripper_width': grip[li, None]}
                envs.append(env)
                batch.append(get_real_umi_obs_dict(env, meta, repr_.obs_pose_repr,
                                                   episode_start_pose=[s['demo_start'].astype(np.float64)]))
            obs = {k: np.stack([b[k] for b in batch]) for k in batch[0]}
            with torch.no_grad():
                raw = policy.predict_action(dict_apply(
                    obs, lambda x: torch.as_tensor(x, dtype=torch.float32).to(device)))['action_pred']
            raw = raw.cpu().numpy()
            for i, env, r in zip(st_[b0:b0 + BATCH], envs, raw):
                a = get_real_umi_action(r, env, repr_.action_pose_repr)     # (H, 7) absolute
                Ra = pose_to_mat(a[:, :6])[:, :3, :3]
                F['seg'].append(si)
                F['i'].append(i)
                F['t'].append(s['grid'][i] - t0)
                F['xyz'].append(a[:, :3] * 1000)
                F['rpy'].append(np.degrees(np.unwrap(np.radians(TL.rpy_deg(Ra)), axis=0)))
                F['grip'].append(a[:, 6] * 1000)
                F['px'].append(px(a[:, :3]))
                sc['n'] += 1
                for k in range(1, horizon):
                    j = i + k * down
                    if j >= n:
                        break
                    e = sc['err'].setdefault(k, [[], [], []])
                    w = sc['still'].setdefault(k, [[], [], []])
                    e[0].append(np.linalg.norm(a[k, :3] - pose[j, :3]) * 1000)
                    w[0].append(np.linalg.norm(pose[i, :3] - pose[j, :3]) * 1000)
                    e[1].append(angle_deg(Ra[k:k + 1], R[j:j + 1])[0])
                    w[1].append(angle_deg(R[i:i + 1], R[j:j + 1])[0])
                    e[2].append(abs(a[k, 6] - grip[j]) * 1000)
                    w[2].append(abs(grip[i] - grip[j]) * 1000)
            done += len(batch)
            progress('forecast', done, total)
    t_inf = time.time() - t_inf

    hz = TL.GRID_HZ
    metrics = {}
    for split, sc in scores.items():
        ahead = []
        for k in sorted(sc['err']):
            e, w = (np.mean(x, axis=1) for x in (sc['err'][k], sc['still'][k]))
            ahead.append({'step': k, 's': round(k * down / hz, 3), 'n': len(sc['err'][k][0]),
                          'pos_mm': round(e[0], 2), 'pos_still_mm': round(w[0], 2),
                          'rot_deg': round(e[1], 2), 'rot_still_deg': round(w[1], 2),
                          'grip_mm': round(e[2], 2), 'grip_still_mm': round(w[2], 2)})
        metrics[split] = {'forecasts': sc['n'], 'ahead': ahead}

    progress('save', 0, 1)
    jp, zp = result_paths(ckpt, ep_dir)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    f32 = lambda x: np.asarray(x, np.float32)    # noqa: E731
    np.savez_compressed(zp[:-4] + '.tmp.npz', f_seg=np.array(F['seg'], np.int16),
                        f_i=np.array(F['i'], np.int32), f_t=f32(F['t']), f_xyz=f32(F['xyz']),
                        f_rpy=f32(F['rpy']), f_grip=f32(F['grip']), f_px=f32(F['px']),
                        g_t=f32(np.concatenate(grid['t'])), g_seg=np.concatenate(grid['seg']).astype(np.int16),
                        g_px=f32(np.concatenate(grid['px'])))
    os.replace(zp[:-4] + '.tmp.npz', zp)
    res = {
        'checkpoint': os.path.relpath(os.path.abspath(ckpt), REPO), 'episode': key,
        'epoch': info['epoch'], 'dataset': info['dataset'], 'stamp': st,
        'segments': [{'seg': si, 'span': [round(float(s['grid'][0] - t0), 3), round(float(s['grid'][-1] - t0), 3)],
                      'steps': len(s['grid']),
                      'split': split_of(mine, si)}
                     for si, s in enumerate(segs)],
        'trim': trim, 'verify': verify, 'metrics': metrics,
        'horizon': horizon, 'down_sample': down, 'step_s': round(down / hz, 4),
        'n_action_steps': info['n_action_steps'], 'scene_size': list(size),
        'settings': {'device': str(device), 'ema': bool(cfg.training.use_ema),
                     'inference_steps': int(policy.num_inference_steps)},
        'seconds': {'load': round(t_load, 1), 'forecast': round(t_inf, 1),
                    'total': round(time.time() - t_start, 1)},
        'made_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(jp + '.tmp', 'w') as f:
        json.dump(res, f, indent=1)
    os.replace(jp + '.tmp', jp)
    return res


def summary(r):
    """The result in a few lines: how far off the forecasts are, against standing still."""
    out = [f'{r["episode"]}  epoch {r["epoch"]}  {r["verify"]["status"]}'
           + (f'  (input vs training {r["verify"]["input_diff"]}/255)' if 'input_diff' in r['verify'] else '')]
    for split, m in r['metrics'].items():
        out.append(f'  {split.replace("_", " ")}: {m["forecasts"]} forecasts')
        for a in m['ahead']:
            if a['step'] in (r['n_action_steps'], r['horizon'] - 1):
                out.append(f'    {a["s"]:.2f} s ahead   position {a["pos_mm"]:6.1f} mm (still {a["pos_still_mm"]:6.1f})'
                           f'   rotation {a["rot_deg"]:5.1f} deg (still {a["rot_still_deg"]:5.1f})'
                           f'   gripper {a["grip_mm"]:5.1f} mm (still {a["grip_still_mm"]:5.1f})')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('-e', '--episode', required=True, help='data/capture/<session>/epNNN')
    ap.add_argument('--intrinsics', default=CALIB_JSON)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stride', type=int, default=None,
                    help='grid steps between forecasts (default: one observation step)')
    ap.add_argument('--force', action='store_true', help='re-run even if a result is cached')
    args = ap.parse_args()
    av.logging.set_level(av.logging.PANIC)

    def progress(stage, i, n):
        print(f'PROGRESS {stage} {i} {n}', flush=True)
    r = evaluate(args.checkpoint, args.episode, args.intrinsics, args.seed, args.stride,
                 args.force, progress)
    print(summary(r), flush=True)
    print(f'wrote {result_paths(args.checkpoint, args.episode)[0]}', flush=True)


if __name__ == '__main__':
    main()
