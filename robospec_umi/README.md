# robospec_umi

A ChArUco-tracking UMI pipeline: calibrate the scene camera, record handheld
gripper demonstrations with a wrist camera, build a training dataset, train a
diffusion policy, evaluate it.

There is no SLAM and no GoPro. Scene-camera ChArUco tracking supplies the TCP
pose, which means the whole pipeline runs on two USB cameras and a printed board.

```
robospec_umi/
├── robospec_umi_calibration/   calibrate_scene_cam.py, scene_cam_charuco_detector.py
├── robospec_umi_capture/       capture.py, verify.py, build_zarr.py, video_processor_charuco.py
├── robospec_umi_evaluation/    eval_without_robot.py
├── robospec_umi_ui/            (not yet built)
├── robospec_umi_server.py      serves the UI; the container's entrypoint
├── robospec_umi_conda.yaml
├── robospec_umi.Dockerfile
├── robospec_umi_compose.yaml
└── 99-decxin-cam.rules         pins the scene camera to /dev/scene_cam
```

Everything reads and writes under `data/` at the repo root:

```
data/
├── calibration/
│   ├── scene_intrinsics.json     <- the ACTIVE calibration; capture and build_zarr check against it
│   └── <datetime>/               <- one directory per calibration run
├── capture/
│   └── <datetime>/               <- one directory per recording session
├── dataset/
│   └── *.zarr.zip                <- what training and evaluation read
├── outputs/
│   └── <date>/                   <- training runs and checkpoints
└── evaluation/
    └── <datetime>/               <- one directory per evaluation run
```

---

## The one thing that will silently ruin your data

Focus and zoom are **geometry**. Focusing physically moves lens elements, which
changes the focal length; digital zoom crops and rescales, which moves both the
focal length and the principal point.

This camera ships with `focus_automatic_continuous=1` and resets to it on every
replug. If you record at a different focus than you calibrated at, the ChArUco
pose solve returns **plausible, smooth, self-consistent, wrong depths**. Nothing
looks broken. The video plays fine, the poses look reasonable, reprojection error
stays low — and every label in your dataset is scaled wrong.

So: run `lock` before every calibration and before every recording session, and
let `verify.py` confirm it afterwards. Three separate scripts cross-check focus
and zoom against `scene_intrinsics.json` for this reason alone.

---

## 1. Setup

### 1a. The udev rule — do this first, on the host

Every command below opens `/dev/scene_cam`. That symlink does not exist until
this rule is installed, and it is not a convenience:

```
/dev/video4   8086:0b5b   Intel RealSense D405
/dev/video10  1bcf:2d50   DECXIN Camera        <- the scene camera
/dev/video11  1bcf:2d50   DECXIN Camera        <- second node, delivers no frames
```

USB enumeration order is not stable. This camera has been `/dev/video4` and
`/dev/video10` on the same machine depending on what was plugged in first, and
`/dev/video4` is currently the RealSense. Opening the wrong node does not error —
it hands you a different camera. The rule also filters on `index==0`, because the
DECXIN's second node opens fine and returns nothing.

```bash
sudo cp robospec_umi/99-decxin-cam.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/scene_cam
v4l2-ctl -d /dev/scene_cam --all | head    # must say DECXIN, not RealSense
```

Check the wrist camera too:

```bash
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])"
```

### 1b. Then either: Docker (recommended)

One image with the full environment, GPU access, the cameras, and the UI server.

```bash
export ROBOSPEC_UMI_DIRECTORY=/path/to/universal_manipulation_interface
xhost +local:docker            # only needed for cv2.imshow windows
# only if `id -u` is not 1000:
#   export ROBOSPEC_UID=$(id -u) ROBOSPEC_GID=$(id -g)

docker compose -f robospec_umi/robospec_umi_compose.yaml build
docker compose -f robospec_umi/robospec_umi_compose.yaml up -d
docker compose -f robospec_umi/robospec_umi_compose.yaml exec robospec bash
```

The UI is then at <http://localhost:8080>. Everything else is run from the shell
that `exec` gives you, exactly as documented below.

### 1b-i. The web UI

Build it once and the server picks it up — the directory is bind-mounted, so no
container rebuild:

```bash
cd robospec_umi/robospec_umi_ui
npm install && npm run build      # -> dist/, served at :8080
```

To work on the UI, run Vite on the host instead; it proxies `/api` to the
server, so edits are live:

```bash
npm run dev                       # http://localhost:5173
```

Calibration is fully wired. Capture, Edit, Training and Evaluation are
placeholder pages that name the CLI command to use meanwhile.

**Only one thing can stream a camera at a time.** V4L2 gives exclusive access,
so starting calibration capture while the test view holds the camera returns a
clear "in use by …" message with a link, rather than a failure. The header shows
which device is held and by what, and a **Take over** button recovers a camera
stranded by a crash. Note that `v4l2-ctl` can still *change* controls while
another process streams — the kernel allows it — which is why the server refuses
a `lock` while a recording is in progress.

Four things worth knowing:

- **The first build takes about 10 minutes and the image is ~20 GB.** Measured,
  not estimated. Most of it is unavoidable: the CUDA runtime libraries bundled
  with torch cu128 are 3.6 GB, torch itself 2.0 GB, triton 544 MB. The rest is
  the upstream UMI dependency set — `robosuite`, `ray`, `free-mujoco-py` and
  `ur-rtde` together add over a gigabyte and are imported by nothing in this
  pipeline, so there is room to trim if the size ever matters.
- **`export ROBOSPEC_UID`, not `export UID`.** Bash marks `UID` readonly, so
  `export UID=$(id -u)` errors outright. The build args default to 1000, so on a
  normal single-user machine you can skip this and files still land owned by you.
- **`xhost +local:docker` grants any local container access to your X server.**
  Fine on a personal rig; think before running it on a shared machine. Skip it
  entirely if you do not need `cv2.imshow`.
- **The container is privileged with `/dev` bound.** That is what makes the
  cameras work, and it means the container is not a security boundary.

### 1c. Or: conda on the host

```bash
conda env create -f robospec_umi/robospec_umi_conda.yaml
conda activate umi2
sudo apt install v4l-utils     # the scripts shell out to v4l2-ctl
```

PyTorch comes from the yaml's `pip:` block via the cu128 index, pinned to exactly
`2.7.0` — the last release with a cp39 cu128 wheel. Verify the GPU arch is
present, or training fails at the first kernel launch:

```bash
python -c "import torch; print(torch.__version__, 'sm_120' in torch.cuda.get_arch_list())"
# expect: 2.7.0+cu128 True
```

Then confirm both cameras' libraries are present. `pyrealsense2` in particular
fails *quietly* if it is missing — `capture.py` guards the import, so you get a
session that records scene video and no wrist stream at all, with no error:

```bash
python -c "import pyrealsense2, cv2, av; print('cameras ok')"
```

---

## 2. Calibration

Produces `data/calibration/scene_intrinsics.json`, which everything downstream
reads. You need a printed 7×5 ChArUco board, 30 mm squares, 22 mm markers.

### 2a. Print the board

```bash
python robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py board
```

Writes `charuco_board.png` (210 × 150 mm, fits A4). **Print at 100% / "actual
size"** — any fit-to-page scaling changes the square size and silently rescales
your whole calibration. Measure a printed square with calipers; if it is not
30.0 mm, correct `SQUARE_LEN` at the top of the script.

### 2b. Lock the camera

```bash
python robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py lock
```

Turns autofocus off and pins focus 0, zoom 100, exposure 800, gamma 128,
gain 100, white balance 4600 K. It reads every value back and reports what
actually stuck — UVC silently clamps out-of-range writes and reports success, so
the readback is the only proof.

Note the focus and zoom it prints. You will pass the same values to `capture.py`.

### 2c. Capture views

```bash
python robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py capture
```

Creates a fresh `data/calibration/<datetime>/` and opens a preview with three
coverage meters. It **refuses to start** if autofocus is on or zoom ≠ 100.

| key | |
|---|---|
| `SPACE` | keep this frame |
| `a` | toggle auto-keep (one frame/second while the board is still and detected) |
| `u` | undo the last frame |
| `q` | done |

Aim for 40–60 views. What matters is not the count but the spread, and the
meters track three things independently:

- **grid** — walk the board into all four frame *corners*. Distortion grows with
  radius, so a calibration fitted only on central views is extrapolating at the
  edges. This is the single easiest way to get a beautiful RMS that is badly
  wrong where it counts.
- **scale** — mix near and far views, or focal length and board distance stay
  correlated and `f` is poorly determined.
- **tilt** — tilt in all directions. Fronto-parallel views alone cannot separate
  the principal point from the board position.

Hold the board **still** for each keep; a moving board is sheared by the rolling
shutter, which is a systematic corner bias, not noise. Frames taken while moving
are rejected.

`--no-display` runs it headless: auto-keep from the start, stopping when coverage
completes or `--target` frames are saved.

### 2d. Solve

```bash
python robospec_umi/robospec_umi_calibration/calibrate_scene_cam.py solve
```

Defaults to the newest run directory and says which one it picked. Fits four
distortion models, picks by held-out error, then refits with outlier rejection.
Writes `data/calibration/scene_intrinsics.json` and archives a copy beside the
frames.

Read the output rather than just the RMS:

- **`final_reproj_error`** under ~0.5 px is healthy. On its own it means little.
- **corner data reach** — how far out your views actually went. Everything past
  that radius is the polynomial extrapolating, and reprojection error says
  nothing about it because there were no observations there to score.
- **`std_dev`** on fx/fy/cx/cy — large values mean the views did not pin the
  parameter down, regardless of RMS.
- **aspect `fy/fx`** should be within 1% of 1.0 on a modern sensor. Further off
  usually means something in the capture path is rescaling the image.

### 2e. Validate

```bash
python robospec_umi/robospec_umi_calibration/scene_cam_charuco_detector.py
```

Live detection against the calibration you just wrote. Verifies focus and zoom
still match at startup, then shows per-radius reprojection error, pose jitter,
and the planar-ambiguity margin.

**Do the scale check.** Reprojection error structurally cannot see a focal-length
scale error: if `fx` were 10% wrong, solvePnP simply places the board 10% further
away and reprojects perfectly. Only a measured physical distance catches it.

Press `d`, hold the board still, and type the tape-measured distance **to the
centre of the printed pattern** (not to the pose origin — the board centre sits
129 mm from it on this board, and it swings with rotation). Take 3–4 samples at
different distances. The fitted slope should be within ±1% of 1.000.

| key | |
|---|---|
| `d` | take a distance sample | 
| `u` | toggle undistorted view — straight scene edges should be straight |
| `m` | toggle marker outlines |
| `r` | reset stats |
| `s` | save an annotated frame |

---

## 3. Capture

### 3a. Record

In the web UI, **Capture** is two stages — *Preview & Lock*, then *Capture
Dataset* — reached from the home page. One session owns both cameras across both
stages: they open once when you enter Preview & Lock and stay open until you
press Finish, so the 3 s warm-up is paid while you are tuning rather than again
on the way into recording, and the D405's pipeline is started exactly once.

Preview & Lock shows both cameras. The scene panel is the same one calibration
uses (auto or manual exposure, white balance, gamma, gain) plus the geometry
lock. The wrist panel is manual exposure and gain with an auto white balance
toggle — the D405 has no usable auto exposure, for the reason printed on the
panel. Controls are frozen for the duration of each episode.

Or from the CLI:

```bash
python robospec_umi/robospec_umi_capture/capture.py
```

Focus and zoom are pinned to **0 / 100** (`SCENE_FOCUS` / `SCENE_ZOOM` in
`capture.py`), which is what the calibration is shot at; `verify.py` checks the
recording against `locked_controls` in `scene_intrinsics.json`. Both cameras stream
continuously from launch and are gated per episode — nothing opens or closes
between takes, so the startup transient is paid once during warm-up rather than
at the start of every episode.

| key | |
|---|---|
| `SPACE` / `ENTER` | start / stop an episode |
| `Q` / `ESC` | finish the session |

Keys work in the preview window *and* in the terminal, so a foot pedal that types
ENTER drives it. `--no-display` runs headless on the terminal alone.

Writes `data/capture/<datetime>/` with one `epNNN/` per episode, each holding
`scene/scene.mkv` + `scene_ts.npz` and `wrist/wrist.mkv` + `wrist_ts.npz`.

**The preview is deliberately lossy.** It samples the stream and skips frames on
purpose so recording never waits on the display. A large on-screen `skipped`
count is correct, not a problem.

### 3b. Verify

```bash
python robospec_umi/robospec_umi_capture/verify.py data/capture/<datetime>
```

Run this before trusting a session. It catches the failures that leave no visible
trace: a sidecar one entry longer than its video shifts every subsequent frame by
~10 ms — about 1 cm of TCP label error, applied silently for the rest of the
recording, while the video still plays and the poses still look reasonable.

It also re-checks focus and zoom against the calibration. A **match** line is the
proof the intrinsics apply to this footage; silence means it could not find the
calibration file.

### 3c. Build the dataset

```bash
python robospec_umi/robospec_umi_capture/build_zarr.py \
    data/capture/<datetime> -o data/dataset/dataset.zarr.zip
```

Tracks the small 4×4 / 20 mm board through the scene video, solves a TCP pose per
frame, resamples onto a 60 Hz time grid and pairs each grid point with the
nearest wrist frame. Accepts several session directories at once.

Grid points are chosen **by time, never by frame index** — the wrist camera drops
~3.4% of frames in bursts up to 167 ms, so striding every Nth frame stops being
uniform the moment one is dropped.

Read the per-episode summary. Each line reports what was thrown away and why:
spikes (planar depth-ambiguity flips), pose-track gaps, grid points with no wrist
frame close enough, and contiguous runs too short to yield a training sample.

To see *where* those losses happen:

```bash
python robospec_umi/robospec_umi_capture/build_zarr.py \
    data/capture/<datetime> -o data/dataset/dataset.zarr.zip \
    --annotate-video --annotate-episode ep003
```

Writes `<ep>/scene/scene_annotated.mp4` — the tracking overlay plus a colour-coded
banner per frame saying what the dataset did with it. Off by default; roughly
30–60 s and 30–60 MB per episode. Scrub the red bands to see whether a run of
dropped frames is the board sitting near fronto-parallel (a recording-geometry
problem, fixed by tilting the board or working closer) or something else.

---

## 4. Training

*Not yet migrated.* Training currently runs from the repo root against
`diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml`, logging
to wandb, writing checkpoints under `data/outputs/<date>/`.

---

## 5. Evaluation

Runs a trained policy on recorded footage with no robot attached, so you can tell
whether a checkpoint learned anything before committing hardware to finding out.

```bash
python robospec_umi/robospec_umi_evaluation/eval_without_robot.py
```

With no arguments it picks the newest checkpoint under `data/outputs/`, the
newest dataset under `data/dataset/`, and `ep000` of the newest session under
`data/capture/` — **announcing each choice**, because evaluating the wrong
checkpoint produces numbers that look entirely ordinary. Override any of them:

```bash
... -c data/outputs/2026.09.09/epoch=0110-train_loss=0.013.ckpt
... -z data/dataset/dataset.zarr.zip
... -e data/capture/20260919_150000/ep003
... --no-video           numbers only; skips the ~2 GB wrist decode, much faster
... --log-samples 0 200  full per-step predicted-vs-actual tables for those samples
```

Results land in `data/evaluation/<datetime>/`:

- **`summary.json`** — checkpoint, epoch, inputs, and per-split error and gains.
  Written so two checkpoints can be compared without re-reading scrollback.
- **`eval_annotated.mp4`** — wrist frame beside the predicted and actual motion
  for that frame, badged TRAIN or VAL.

### How to read the numbers

The policy emits 16 future steps of `[3 position | 6D rotation | 1 gripper]`,
expressed **relative to the current pose**. Three things decide whether the error
means anything, and none of them is the raw millimetre figure.

**The no-motion baseline is the whole story.** Because actions are relative,
predicting all zeros — "stay exactly where you are" — already scores well
whenever the hand moves slowly. Its error *is* the distance the hand actually
moved, so it is printed beside every result. A model that learned nothing still
posts flattering millimetre errors. Only the ratio to this baseline means
anything.

**Train vs val.** Reproducing training data proves the plumbing works and nothing
about generalisation. The gap between the two splits is the actual finding.

**Per horizon step, never a bare mean.** Step 0 is degenerate — the first action
is the current pose relative to itself, so true motion is 0.0 mm and the ratio is
undefined. What matters is that the ratio **rises** with horizon: that is what
separates predicting motion from predicting small numbers.

A healthy result looks like this — gain climbing from 4× to 10× across the chunk:

```
 step  t ahead  true move  model err   gain     true rot   rot err   gain
    4    0.25s       21.3        5.3  4.02x         1.84      1.66  1.11x
    8    0.45s       41.8        7.0  5.98x         2.52      1.62  1.56x
   15    0.80s       75.6        7.6  9.94x         3.60      1.77  2.03x
```

Rotation is scored separately because it can fail while position succeeds, and a
position headline would hide that completely. A rotation gain **below 1.0** means
the predicted rotation is further from the truth than predicting *no* rotation —
the model is actively making it worse.

### Two things this measurement cannot escape

**It is open loop.** Ground truth is fed at every step, so errors never compound
the way they will on a robot closing the loop on its own predictions. Read it as
a lower bound on real error, never an estimate of it.

**The annotated video may contain no VAL frames.** `UmiDataset` holds out whole
zarr episodes, so a video of `ep000` shows val frames only when `ep000` is itself
the held-out episode. The script says so when it happens; point `-e` at the
held-out episode to see the comparison.
