"""Run a trained UMI policy on recorded footage, with no robot attached.

Feeds the policy real observations and compares the action chunk it predicts
against what the hand actually did, so you can tell whether a checkpoint learned
anything before committing hardware to finding out.

    python3 robospec_umi/robospec_umi_evaluation/eval_without_robot.py
    ... --no-video                     numbers only, skips the ~2 GB wrist decode
    ... --log-samples 0 200            per-step delta tables for those samples

With no arguments it picks the newest checkpoint under data/outputs/, the newest
dataset under data/dataset/, and ep000 of the newest session under data/capture/,
announcing each choice. Results go to data/evaluation/<datetime>/.


HOW TO READ THE NUMBERS
-----------------------
The policy emits a (16, 10) chunk: 16 future steps of
[3 position | 6D rotation | 1 gripper], expressed RELATIVE to the current pose --
literally "move this far from where you are". Ground truth from the zarr has the
same shape, so they subtract directly. Error is 3D Euclidean distance in mm.

Three things decide whether that error means anything.

1. THE NO-MOTION BASELINE. Because actions are relative, predicting all zeros --
   "stay exactly where you are" -- already scores well when the hand moves
   slowly. Its error equals the distance the hand actually moved, so it is
   printed beside every result. A model that learned nothing still posts
   flattering millimetre errors; only the ratio to this baseline is meaningful.

2. TRAIN vs VAL. Reproducing training data proves the plumbing works and nothing
   about generalisation. Results are split, and the gap between them is the
   actual finding.

3. PER HORIZON STEP, never a bare mean. Step 0 is degenerate -- the first action
   is the current pose relative to itself, so true motion is 0.0 mm and the
   baseline ratio is undefined. What matters is that the ratio RISES with
   horizon: that is what separates predicting motion from predicting small
   numbers.

This is open loop. Ground truth is fed at every step, so errors never compound
the way they will on a robot closing the loop on its own predictions. Read it as
a lower bound on real error, never an estimate of it.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import av
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robospec_umi_evaluation
PKG = os.path.dirname(HERE)                         # .../robospec_umi
REPO = os.path.dirname(PKG)                         # repo root
DATA = os.path.join(REPO, 'data')

# The repo root carries diffusion_policy/ and umi/, which are not installed.
sys.path.insert(0, REPO)

import dill                                                    # noqa: E402
import hydra                                                   # noqa: E402
from omegaconf import open_dict                                # noqa: E402
from umi.common.cv_util import get_image_transform             # noqa: E402
from umi.common.pose_util import pose10d_to_mat, mat_to_pose   # noqa: E402

# The 10-dim action for "do not move": zero translation, and the 6D rotation of
# the identity matrix (its first two rows).
IDENTITY10 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)

# ------------------------------------------------------------------- fixed run
SPLIT = 'both'          # reporting train and val separately is the whole point
MAX_SAMPLES = 200       # per split; inference is ~0.1 s each
STRIDE = 1              # take consecutive samples
INFERENCE_STEPS = 16    # DDIM steps
VIDEO_FRAMES = 200      # frames in the annotated video
VIDEO_FPS = 20.0


# --------------------------------------------------------------- input finding
def _newest(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def find_checkpoint():
    """Newest *.ckpt anywhere under data/outputs/.

    Two layouts live there -- a bare file, and hydra's
    <run>/checkpoints/<epoch>.ckpt -- so this globs recursively rather than
    assuming either. Newest wins; pass -c for a specific epoch.
    """
    return _newest(glob.glob(os.path.join(DATA, 'outputs', '**', '*.ckpt'),
                             recursive=True))


def find_zarr():
    """Newest *.zarr.zip, preferring data/dataset/ over data/ itself."""
    return _newest(glob.glob(os.path.join(DATA, 'dataset', '*.zarr.zip'))
                   or glob.glob(os.path.join(DATA, '*.zarr.zip')))


def find_episode():
    """ep000 of the newest capture session.

    A guess, but a guarded one: the zarr records no link back to the recording it
    was built from, and align_episode() below catches a mismatched pair by image
    correlation rather than letting it through.
    """
    sessions = sorted(glob.glob(os.path.join(DATA, 'capture', '*')))
    for s in reversed([d for d in sessions if os.path.isdir(d)]):
        ep = os.path.join(s, 'ep000')
        if os.path.isdir(ep):
            return ep
    return None


# --------------------------------------------------------------------- loading
def load_policy(ckpt, device, inference_steps):
    """-> (policy, cfg, meta). The checkpoint is self-contained.

    No normalizer file is needed: LinearNormalizer is an nn.Module held by the
    policy, so its scale/offset tensors are already inside model.state_dict().
    The normalizer.pkl that appears in a training run directory is a handshake
    between ranks at train time, not an inference artefact.
    """
    from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)
    # EMA weights are the ones validated during training and are NOT the same
    # tensors as `model`. eval_real.py makes the same choice.
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = inference_steps
    policy.eval().to(device)
    meta = {k: dill.loads(v) for k, v in payload['pickles'].items()
            if k in ('epoch', 'global_step')}
    return policy, cfg, meta


def load_dataset(cfg, zarr_path):
    """Instantiate UmiDataset against the zarr we were asked to evaluate.

    cfg.task.dataset_path is baked into the checkpoint as whatever path training
    ran against, which may not exist here and is almost certainly not the dataset
    being evaluated. Left alone it would quietly read the wrong file, so it is
    overridden before instantiation.
    """
    with open_dict(cfg):
        cfg.task.dataset_path = zarr_path
        if 'dataset_path' in cfg.task.dataset:
            cfg.task.dataset.dataset_path = zarr_path
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset, dataset.get_validation_dataset()


# ------------------------------------------------------------------ pose maths
def action_pos_mm(a):
    """(...,10) action -> (...,3) translation in mm.

    Dims 3-8 are a 6D rotation (the first two rows of R); reading them directly
    is meaningless, they have to go through pose10d_to_mat, which Gram-Schmidts
    them back into a rotation matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    return mat_to_pose(pose10d_to_mat(a[..., :9]))[..., :3] * 1000.0


def action_rot_deg(a):
    """(...,10) action -> (...,3) rotation vector in degrees."""
    a = np.asarray(a, dtype=np.float64)
    return np.degrees(mat_to_pose(pose10d_to_mat(a[..., :9]))[..., 3:])


def rot_mag(rotvec):
    return np.linalg.norm(rotvec, axis=-1)


# -------------------------------------------------------------- video alignment
def decode_wrist(mkv, out_res):
    """Decode the whole wrist video, transformed exactly as build_zarr.py did.

    Sequential decode, no seeking: h264 random access is unreliable, and the
    whole clip is wanted anyway.
    """
    frames = []
    with av.open(mkv) as c:
        st = c.streams.video[0]
        st.thread_count = 4
        tf = get_image_transform((st.width, st.height), (out_res, out_res))
        src = f'{st.width}x{st.height}'
        for f in c.decode(st):
            frames.append(tf(f.to_ndarray(format='rgb24')))
    return np.array(frames), src


def signatures(imgs):
    """16x16 mean-removed grayscale signatures, for correlation matching."""
    s = np.stack([cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA).mean(axis=2)
                  for x in imgs]).reshape(len(imgs), -1)
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def align_episode(video, replay_buffer, min_corr=0.99, max_pixel_diff=6.0):
    """Match wrist video frames to zarr rows, per zarr episode. -> {ep: info}

    The zarr records no link back to the recording it came from, so the mapping
    has to be recovered. Re-deriving it through build_zarr.plan_episode would
    mean importing that module AND the ChArUco processor and re-solving poses
    that are already sitting in the zarr -- so instead the IMAGES are matched.
    build_zarr wrote exactly these transformed frames into the zarr, so
    correlation recovers the mapping directly.

    Two independent checks guard it. Correlation must be near 1, and the match
    must be MONOTONIC: both sequences are time-ordered, so a mapping that jumps
    backwards is matching noise. Measured on the right episode this gives
    corr 0.9999 and a 1.4/255 pixel residual (the lossy JpegXl round trip);
    on a different recording, corr 0.54 and non-monotonic.
    """
    sv = signatures(video)
    ends = replay_buffer.episode_ends[:]
    out = {}
    for i, end in enumerate(ends):
        start = 0 if i == 0 else ends[i - 1]
        z = replay_buffer['camera0_rgb'][start:end]
        corr = signatures(z) @ sv.T / sv.shape[1]
        idx, score = corr.argmax(1), corr.max(1)
        diff = float(np.abs(z.astype(np.int16)
                            - video[idx].astype(np.int16)).mean())
        mono = bool(np.all(np.diff(idx) >= 0))
        ok = score.min() >= min_corr and mono and diff <= max_pixel_diff
        if ok:
            out[i] = {'start': start, 'end': end, 'wrist_idx': idx,
                      'corr': float(score.min()), 'diff': diff}
    return out


# ---------------------------------------------------------------------- metrics
class Metrics:
    """Per-horizon-step error, against ground truth and against doing nothing."""

    def __init__(self, horizon):
        self.h = horizon
        self.pos, self.rot, self.base_pos, self.base_rot = [], [], [], []

    def add(self, pred, gt):
        pp, gp = action_pos_mm(pred), action_pos_mm(gt)
        pr, gr = action_rot_deg(pred), action_rot_deg(gt)
        zp = action_pos_mm(np.tile(IDENTITY10, (self.h, 1)))
        zr = action_rot_deg(np.tile(IDENTITY10, (self.h, 1)))
        self.pos.append(np.linalg.norm(pp - gp, axis=-1))
        self.rot.append(rot_mag(pr - gr))
        # the no-motion baseline's error IS the distance the hand actually moved
        self.base_pos.append(np.linalg.norm(zp - gp, axis=-1))
        self.base_rot.append(rot_mag(zr - gr))

    def empty(self):
        return not self.pos

    def report(self, label, rate=60.0, down_sample=3):
        p, b = np.array(self.pos), np.array(self.base_pos)
        r, br = np.array(self.rot), np.array(self.base_rot)
        print(f'\n=== {label}   n={len(p)} samples x {p.shape[1]} horizon steps ===')
        print(f'{"step":>5}{"t ahead":>9}{"true move":>11}{"model err":>11}'
              f'{"gain":>7}   {"true rot":>10}{"rot err":>10}{"gain":>7}')
        for s in list(range(0, p.shape[1], max(p.shape[1] // 4, 1))) + [p.shape[1] - 1]:
            if s >= p.shape[1]:
                continue
            gp = b[:, s].mean() / max(p[:, s].mean(), 1e-9)
            gr = br[:, s].mean() / max(r[:, s].mean(), 1e-9)
            print(f'{s:>5}{(s + 1) * down_sample / rate:>8.2f}s'
                  f'{b[:, s].mean():>11.1f}{p[:, s].mean():>11.1f}{gp:>6.2f}x   '
                  f'{br[:, s].mean():>10.2f}{r[:, s].mean():>10.2f}{gr:>6.2f}x')
        # step 0 is excluded from the headline: true motion there is 0.0 mm, so
        # the ratio is undefined and averaging it in just dilutes the result
        gain = b[:, 1:].mean() / max(p[:, 1:].mean(), 1e-9)
        rgain = br[:, 1:].mean() / max(r[:, 1:].mean(), 1e-9)
        print(f'{"MEAN":>5}{"":>9}{b[:, 1:].mean():>11.1f}{p[:, 1:].mean():>11.1f}'
              f'{gain:>6.2f}x   {br[:, 1:].mean():>10.2f}{r[:, 1:].mean():>10.2f}'
              f'{rgain:>6.2f}x   (step 0 excluded)')
        return float(p[:, 1:].mean()), float(gain), float(rgain)


def log_sample(i, pred, gt, split):
    """The literal 'what would the robot do next' readout."""
    pp, gp = action_pos_mm(pred), action_pos_mm(gt)
    pr, gr = action_rot_deg(pred), action_rot_deg(gt)
    print(f'\n--- sample {i} [{split}] --- motion relative to the current pose')
    print(f'{"step":>5}{"PRED dx":>9}{"dy":>8}{"dz":>8}{"|d|":>8}{"rot":>7}'
          f'  |{"GT dx":>9}{"dy":>8}{"dz":>8}{"|d|":>8}{"rot":>7}{"err":>8}')
    for s in range(len(pp)):
        print(f'{s:>5}{pp[s][0]:>9.1f}{pp[s][1]:>8.1f}{pp[s][2]:>8.1f}'
              f'{np.linalg.norm(pp[s]):>8.1f}{rot_mag(pr[s]):>7.1f}'
              f'  |{gp[s][0]:>9.1f}{gp[s][1]:>8.1f}{gp[s][2]:>8.1f}'
              f'{np.linalg.norm(gp[s]):>8.1f}{rot_mag(gr[s]):>7.1f}'
              f'{np.linalg.norm(pp[s] - gp[s]):>8.1f}')


# ------------------------------------------------------------------------ video
def draw_panel(img_rgb, pred, gt, row, split, view_px=560, panel_px=430):
    """Wrist frame beside the predicted-vs-actual motion for that frame.

    The path is drawn as a top-down plot in the gripper's own XY plane rather
    than projected into the image. Projecting would need the wrist camera's
    intrinsics and its transform to the TCP, neither of which this rig has
    measured -- inventing them would put a confident-looking overlay in the
    wrong place, which is worse than an honest 2D plot.
    """
    view = cv2.cvtColor(cv2.resize(img_rgb, (view_px, view_px),
                                   interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_RGB2BGR)
    pp, gp = action_pos_mm(pred), action_pos_mm(gt)
    pr, gr = action_rot_deg(pred), action_rot_deg(gt)
    panel = np.full((view_px, panel_px, 3), 26, np.uint8)
    # cv2 is BGR: (120,200,255) reads as orange, not blue. Ground truth is drawn
    # in a real blue so the legend does not lie about which line is which.
    GREEN, BLUE, GREY = (120, 240, 120), (255, 190, 90), (150, 150, 150)

    def put(t, y, col=(235, 235, 235), sc=0.5, th=1):
        cv2.putText(panel, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                    cv2.LINE_AA)

    badge = (0, 200, 80) if split == 'TRAIN' else (40, 190, 245)
    cv2.rectangle(panel, (12, 12), (108, 40), badge, -1)
    cv2.putText(panel, split, (22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, f'zarr row {row}', (120, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, GREY, 1, cv2.LINE_AA)
    put('final step of the 16-step chunk (0.80 s ahead)', 62, GREY, 0.44)
    put(f'PRED {pp[-1][0]:+7.1f}{pp[-1][1]:+7.1f}{pp[-1][2]:+7.1f} mm', 90, GREEN, 0.52)
    put(f'     |d| {np.linalg.norm(pp[-1]):6.1f} mm  rot {rot_mag(pr[-1]):5.1f} deg',
        112, GREEN, 0.48)
    put(f'GT   {gp[-1][0]:+7.1f}{gp[-1][1]:+7.1f}{gp[-1][2]:+7.1f} mm', 142, BLUE, 0.52)
    put(f'     |d| {np.linalg.norm(gp[-1]):6.1f} mm  rot {rot_mag(gr[-1]):5.1f} deg',
        164, BLUE, 0.48)
    err = float(np.linalg.norm(pp[-1] - gp[-1]))
    put(f'error {err:6.1f} mm', 196,
        GREEN if err < 20 else (60, 160, 255), 0.6, 2)

    cx, cy, r = panel_px // 2, 400, 150
    span = max(float(np.abs(np.concatenate([pp[:, :2], gp[:, :2]])).max()), 1.0)
    cv2.line(panel, (cx - r, cy), (cx + r, cy), (70, 70, 70), 1)
    cv2.line(panel, (cx, cy - r), (cx, cy + r), (70, 70, 70), 1)
    put(f'top-down XY, +/-{span:.0f} mm   green=pred  blue=actual',
        cy + r + 26, GREY, 0.42)
    for pts, col in ((gp, BLUE), (pp, GREEN)):
        xy = np.stack([cx + pts[:, 0] / span * r, cy - pts[:, 1] / span * r], 1)
        cv2.polylines(panel, [xy.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(xy[-1].astype(int)), 4, col, -1)
    cv2.circle(panel, (cx, cy), 3, (255, 255, 255), -1)
    return np.hstack([view, panel])


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', default=None,
                    help='default: newest *.ckpt under data/outputs/')
    ap.add_argument('-z', '--zarr', default=None,
                    help='default: newest *.zarr.zip under data/dataset/')
    ap.add_argument('-e', '--episode', default=None,
                    help='default: ep000 of the newest data/capture/ session')
    ap.add_argument('--no-video', action='store_true',
                    help='skip decoding wrist.mkv entirely (pure zarr replay). '
                         'Much faster when only the numbers are wanted.')
    ap.add_argument('--log-samples', type=int, nargs='*', default=[],
                    help='sample indices to print a full per-step delta table for')
    args = ap.parse_args()

    # Each default is ANNOUNCED, never silently assumed: evaluating the wrong
    # checkpoint or the wrong recording produces numbers that look entirely
    # ordinary, and the mistake surfaces much later if at all.
    for attr, finder, what in (('checkpoint', find_checkpoint, 'data/outputs/'),
                               ('zarr', find_zarr, 'data/dataset/'),
                               ('episode', find_episode, 'data/capture/')):
        if getattr(args, attr) is None:
            found = finder()
            if found is None and attr != 'episode':
                raise SystemExit(f'no {attr} found under {what} -- pass '
                                 f'--{attr} explicitly')
            setattr(args, attr, found)
            if found:
                print(f'{attr:<12}{found}  (newest in {what})')

    for p in (args.checkpoint, args.zarr):
        if not os.path.exists(p):
            raise SystemExit(f'missing: {p}')

    out_dir = os.path.join(DATA, 'evaluation',
                           datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device      {device}')
    print(f'results     {out_dir}')
    print(f'checkpoint  {args.checkpoint}')
    policy, cfg, meta = load_policy(args.checkpoint, device, INFERENCE_STEPS)
    print(f'            epoch {meta.get("epoch")}  step {meta.get("global_step")}  '
          f'{"EMA" if cfg.training.use_ema else "raw"} weights  '
          f'{INFERENCE_STEPS} DDIM steps')
    print(f'zarr        {args.zarr}')
    dataset, val = load_dataset(cfg, args.zarr)
    horizon = cfg.task.shape_meta['action']['horizon']
    down = cfg.task.shape_meta['action']['down_sample_steps']
    print(f'            {len(dataset)} train / {len(val)} val samples, '
          f'action horizon {horizon} at every {down}th 60 Hz step')

    # ---- align the recorded video with the zarr rows ------------------------
    aligned, video = {}, None
    if not args.no_video and args.episode is None:
        print('video       no session found under data/capture/; continuing on '
              'zarr images only (pass -e for one)')
    elif not args.no_video:
        mkv = os.path.join(args.episode, 'wrist', 'wrist.mkv')
        if not os.path.exists(mkv):
            print(f'video       {mkv} not found; continuing on zarr images only')
        else:
            res = cfg.task.shape_meta['obs']['camera0_rgb']['shape'][1]
            video, src = decode_wrist(mkv, res)
            print(f'video       {mkv}')
            print(f'            {len(video)} frames, {src} -> centre crop -> '
                  f'{res}x{res}')
            aligned = align_episode(video, dataset.replay_buffer)
            if not aligned:
                print('            NO zarr episode matches this video. Either it is '
                      'not the recording this zarr was built from, or build_zarr '
                      'settings changed since. Falling back to zarr images.')
            for i, a in sorted(aligned.items()):
                print(f'            zarr episode {i} rows {a["start"]}:{a["end"]} '
                      f'<- wrist frames {a["wrist_idx"][0]}..{a["wrist_idx"][-1]}  '
                      f'corr {a["corr"]:.4f}  |video-zarr| {a["diff"]:.2f}/255')
            if aligned and max(a['diff'] for a in aligned.values()) > 3.0:
                print('            WARNING: the video path does not reproduce the '
                      'zarr images closely. A deployed policy would see different '
                      'pixels than this model trained on.')

    # row -> decoded frame, for --video-images and for the annotated video
    row_img = {}
    for a in aligned.values():
        for k, w in enumerate(a['wrist_idx']):
            row_img[a['start'] + k] = video[int(w)]

    # ---- run the policy ----------------------------------------------------
    results = {}
    for name, ds in (('train', dataset), ('val', val)):
        if SPLIT not in (name, 'both') or len(ds) == 0:
            continue
        idxs = list(range(0, len(ds), max(STRIDE, 1)))[:MAX_SAMPLES]
        m, logged = Metrics(horizon), []
        with torch.no_grad():
            for i in idxs:
                batch = ds[i]
                obs = {k: v.unsqueeze(0).to(device) for k, v in batch['obs'].items()}
                pred = policy.predict_action(obs)['action_pred'][0].cpu().numpy()
                gt = batch['action'].numpy()
                m.add(pred, gt)
                if i in args.log_samples:
                    logged.append((i, pred, gt))
        for i, pred, gt in logged:
            log_sample(i, pred, gt, name)
        if not m.empty():
            results[name] = m.report(f'{name} split', down_sample=down)

    # ---- verdict -----------------------------------------------------------
    print('\n' + '=' * 74)
    if 'train' in results and 'val' in results:
        tp, tg, _ = results['train']
        vp, vg, _ = results['val']
        print(f'position:  train {tp:.1f} mm ({tg:.2f}x baseline)    '
              f'val {vp:.1f} mm ({vg:.2f}x baseline)')
        if vg < 1.2:
            print('The model barely beats predicting no motion on held-out data. '
                  'It has not learned a useful policy yet.')
        elif vp > tp * 1.6:
            print('Held-out error is well above training error: the model is '
                  'fitting these specific demonstrations rather than the task. '
                  'Expected at this dataset size -- the fix is more data, not '
                  'more epochs.')
        else:
            print('Held-out error is close to training error and clearly beats the '
                  'no-motion baseline, which is what generalisation looks like.')

    # Rotation is scored separately because it can fail while position succeeds,
    # and the position headline would hide it completely. A gain below 1.0 means
    # the predicted rotation is further from the truth than predicting NO
    # rotation would have been -- the model is actively making it worse.
    for name, (_, _, rgain) in results.items():
        if rgain < 1.0:
            print(f'ROTATION [{name}]: {rgain:.2f}x baseline -- WORSE than '
                  f'predicting no rotation at all. The position head is learning '
                  f'and the rotation head is not.')
        elif rgain < 1.2:
            print(f'rotation [{name}]: {rgain:.2f}x baseline -- barely better than '
                  f'no rotation.')
    for name, (p, g, _) in results.items():
        if g < 1.2:
            print(f'  {name}: position {g:.2f}x baseline -- at or below '
                  f'"stay still".')
    print('\nCaveats this measurement cannot escape:')
    print('  * Open loop. Ground truth is fed at every step, so errors never '
          'compound\n    as they will on a robot. This is a lower bound on real '
          'error.')
    print('  * robot0_gripper_width is constant 0.0 in this dataset, so that '
          'input\n    channel carries no information and the gripper action is '
          'not meaningful.')
    if len(val):
        print(f'  * val is {len(val)} samples from a single held-out segment. '
              f'It is the only\n    held-out data that exists here; one segment '
              f'is not a distribution.')
    print('=' * 74)

    # ---- summary.json ------------------------------------------------------
    # Written so two checkpoints can be compared without re-reading scrollback.
    # The gains matter more than the millimetres: because actions are relative,
    # raw error is flattering whenever the hand moved slowly.
    summary = {
        'checkpoint': args.checkpoint,
        'epoch': meta.get('epoch'),
        'global_step': meta.get('global_step'),
        'zarr': args.zarr,
        'episode': args.episode,
        'inference_steps': INFERENCE_STEPS,
        'n_train_samples': len(dataset),
        'n_val_samples': len(val),
        'splits': {name: {'pos_err_mm': p, 'pos_gain': g, 'rot_gain': rg}
                   for name, (p, g, rg) in results.items()},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {os.path.join(out_dir, "summary.json")}')

    # ---- annotated video ---------------------------------------------------
    video_path = os.path.join(out_dir, 'eval_annotated.mp4')
    if row_img:
        val_rows = set()
        for i in range(len(val)):
            val_rows.add(int(val.sampler.indices[i][0]))
        rows = {int(dataset.sampler.indices[i][0]): (i, dataset)
                for i in range(len(dataset))}
        rows.update({int(val.sampler.indices[i][0]): (i, val)
                     for i in range(len(val))})
        # Spread evenly over every matched row rather than taking the first N: a
        # prefix biases the video toward whichever split happens to come first in
        # the episode, and the train/val comparison is the point of it.
        #
        # Whether any VAL frames appear at all depends on the episode passed to
        # -e. UmiDataset holds out whole zarr episodes, so a video of ep000 shows
        # val frames only when ep000 is itself the held-out one. That is reported
        # below rather than left to be inferred from a "0 val" count.
        allrows = sorted(r for r in row_img if r in rows)
        if len(allrows) > VIDEO_FRAMES:
            pick = np.linspace(0, len(allrows) - 1, VIDEO_FRAMES)
            allrows = [allrows[int(round(i))] for i in pick]
        avail = allrows
        if not avail:
            print('\nno overlap between decoded frames and dataset samples; '
                  'skipping video')
        else:
            writer = None
            with torch.no_grad():
                for r in avail:
                    si, ds = rows[r]
                    batch = ds[si]
                    obs = {k: v.unsqueeze(0).to(device)
                           for k, v in batch['obs'].items()}
                    pred = policy.predict_action(obs)['action_pred'][0].cpu().numpy()
                    frame = draw_panel(row_img[r], pred, batch['action'].numpy(), r,
                                       'VAL' if r in val_rows else 'TRAIN')
                    if writer is None:
                        writer = cv2.VideoWriter(
                            video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                            VIDEO_FPS, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
            if writer is not None:
                writer.release()
                n_val = sum(1 for r in avail if r in val_rows)
                print(f'wrote {video_path}  {len(avail)} frames '
                      f'({len(avail) - n_val} train, {n_val} val)')
                if n_val == 0:
                    print(f'            all TRAIN: the held-out split is in a '
                          f'different zarr episode than\n            '
                          f'{os.path.basename(args.episode)}. Point -e at the '
                          f'held-out episode to see VAL frames.')


if __name__ == '__main__':
    main()
