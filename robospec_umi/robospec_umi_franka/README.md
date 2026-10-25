# robospec_umi_franka

Running a trained robospec_umi policy on the FR3, as a skill that the Robospec
main program calls.

**Status: design notes only.** None of this is implemented or tested on the robot
yet. The Robospec details come from an older copy of Robospec-demo, so check them
against the current branch. This is a proposal: change anything that doesn't fit.

---

## 1. Use case

UMI is one skill the main program calls, like grasp or lift:

1. The main program moves the arm to an observation pose with its usual PTP
   motion. That means gripper open, object in the wrist camera's view, roughly
   where the demonstrations started.
2. It calls the `/umi/run` action.
3. UMI drives the arm and gripper itself until its stop rule ends the run.
4. It returns a result, and the main program takes back control.

Not needed at eval time:
- **SpaceMouse:** upstream UMI's eval uses one to move the arm between runs.
- **DECXIN scene camera:** it only produces training labels.

**Assumed hardware:**
- FR3 with the Franka Hand, fitted with the same fingers as the handheld gripper.
- D455 wrist camera on the same mount as in data collection. D405 support comes later.
- The TCP at the same place relative to the fingers as in data collection.

---

## 2. Architecture

```
Robospec DAG, or you via `ros2 action send_goal`
        │  /umi/run  (RunUmi action)
        ▼
umi_node            system Python 3.10 + rclpy, no torch
  │
  ├─ UmiClient ── JSON lines over AF_UNIX ──▶ umi_server     robospec_umi conda env, Python 3.9, no ROS
  │                                             owns the D455 (pyrealsense2) and the policy on the GPU
  │
  ├─ target pose @ 200 Hz ──▶ Cartesian impedance controller ──▶ FR3
  │                           (C++, 1 kHz, inside controller_manager)  ← MISSING, see 7.1
  ├─ grasp / move goals ────▶ franka_gripper
  └─ switch_controller ─────▶ controller_manager
```

There are three pieces, named the way Robospec names its cuRobo pieces
(`curobo_server`, `CuroboClient`, `grasp_node`):

- **`umi_server`** (this folder). Runs in the `robospec_umi` conda env.
  - At startup it opens the D455, loads the checkpoint and runs one warmup
    inference.
  - After that it answers requests.
  - The launch file starts it as its own process with respawn, the same way it
    starts `curobo_server` and `sam3_server`. Clients reach it over an AF_UNIX
    socket.
  - It never imports ROS.
- **`UmiClient`**. A plain-Python socket helper that `umi_node` imports. It
  connects with retries, sends a request and waits for the reply.
- **`umi_node`**. A Robospec primitive in the system Python, with rclpy. It never
  imports torch. It:
  - hosts the action server;
  - buffers robot state;
  - switches controllers;
  - runs the 200 Hz interpolator;
  - sends gripper goals;
  - enforces the safety limits;
  - handles cancel.

**Why two processes:**
- Humble's rclpy is a compiled extension for Python 3.10, so it can't be imported
  from 3.9.
- The env is pinned to 3.9 because torch 2.7.0 is the last release with a
  cp39 cu128 wheel. cu128 is the minimum for the RTX 50-series (sm_120).
- With the socket between them, neither interpreter has to change.

**Why not the Docker image:** robot runs don't need the web UI, and the container
would add device, network and lifecycle setup. Use the conda env on the host.

**How upstream UMI maps onto this:**

| Layer | Upstream UMI | Here |
|---|---|---|
| Policy loop | `eval_real.py` | `umi_server` |
| Waypoint interpolation, 200 Hz | `FrankaInterpolationController` | `umi_node` |
| Transport | zerorpc to a NUC | socket (server ↔ node), ROS topic (node → controller) |
| Cartesian impedance, 1 kHz | Polymetis | **missing** (7.1) |

---

## 3. The skill: `/umi/run`

Proposed `RunUmi.action` in `robospec_msgs`:

```
# goal
string  task            # reserved; one checkpoint per umi_server for now
float32 timeout_s
---
# result
bool    success         # ended cleanly (stop rule / timeout), NOT "task succeeded"
string  reason          # done | timeout | cancelled | robot_error | server_lost
geometry_msgs/Pose final_tcp
float32 gripper_width
---
# feedback
int32   step
float32 elapsed_s
```

What `umi_node` does with a goal:

```
check: robot state fresh, no robot errors, umi_server ready
switch_controller(STRICT): deactivate current controllers, activate impedance
    (impedance target starts at the measured pose, so nothing jumps)
umi_server ← reset(start_pose)
every ~0.4 s (8 action steps × 50 ms):
    umi_server ← step(robot state of the last ~0.3 s)
    umi_server → ~8 waypoints [t, pose, width]
    give them to the interpolator
    gripper width crosses threshold → async grasp / move
    send feedback; stop on cancel, timeout or stop rule
hold the measured pose, switch_controller back
return result
```

**The smoothing happens in the interpolator, so don't skip it.** Sending raw
20 Hz waypoints straight to the controller produces steps, and steps trigger
Franka reflexes. Upstream works the same way:
- **By timestamp.** Waypoints carry absolute timestamps and are played back by
  time. Uneven inference or transport time doesn't matter, as long as a chunk
  arrives before its first waypoint is due.
- **Joined at the commanded pose.** Each new chunk is joined onto the pose
  currently being commanded, not the measured pose. Waypoints already in the
  past are dropped.
- **Speed limits.** A waypoint that would need more than the max linear or
  angular speed is pushed later instead of jumped to.
- **Controller damping.** The impedance controller acts as a spring-damper
  towards its target, which rounds off whatever corners remain.

Jitter can still come from two places:
- **Consecutive chunks that disagree.** Position stays continuous but velocity
  changes. If it shows on the robot, blend the overlapping waypoints or add a
  target filter in the controller.
- **A chunk that arrives late.** The arm holds the last waypoint and hesitates.
  DDIM 16 on the 5090 should take tens of ms against a ~400 ms budget.

Port the interpolator logic from upstream: `PoseTrajectoryInterpolator` and its
`schedule_waypoint`, in `umi/common/pose_trajectory_interpolator.py`. It needs
only numpy and scipy, so it runs in the system Python.

---

## 4. `umi_server` protocol

One JSON object per line, one reply per request:

```
{"cmd": "ping"}                                     → {"status": "ok", "message": "ready"}
{"cmd": "reset", "start_pose": [x, y, z, rx, ry, rz]} → {"status": "ok"}
{"cmd": "step",  "robot_state": [[t, x, y, z, rx, ry, rz, width], ...]}
    → {"status": "ok", "waypoints": [[t, x, y, z, rx, ry, rz, width], ...]}
```

- **Poses:** the TCP in the robot base frame, rotation as axis-angle.
- **Times:** seconds, on the clock both sides share (7.9).
- **Readiness:** `ping` only reports ready after the warmup inference.

On each `step` the server:
1. Takes the 2 newest frames that are 50 ms apart.
2. Interpolates the robot-state history at those frame timestamps.
3. Builds the observation with `get_real_umi_obs_dict`, runs the policy, and
   converts the output to absolute TCP poses with `get_real_umi_action`.
4. Stamps each action with its time (frame time + k × 50 ms), drops the ones
   already past, and returns the rest.

When the client disconnects, the server drops its episode state, i.e. the start
pose. A crashed `umi_node` can't leave a stale one behind.

`robospec_umi_evaluation/eval_without_robot.py` already does step 3, and the
cropping, on recorded episodes. Start from it.

---

## 5. What must match training

**Observations** (`diffusion_policy/config/task/umi.yaml`), 2 steps each:
- `camera0_rgb`: 3×224×224;
- `robot0_eef_pos` and `robot0_eef_rot_axis_angle` (6D rotation), relative to the
  current TCP;
- `robot0_gripper_width`;
- `robot0_eef_rot_axis_angle_wrt_start`, relative to the pose at `reset`.

**Actions:** 16 steps of [position, 6D rotation, gripper width], relative to the
current TCP. 8 are executed per inference.

**Timing:** the dataset grid is 60 Hz and the policy downsamples by 3, so
observations and actions are 50 ms apart (20 Hz). Inference uses 16 DDIM steps.

**Relative poses:** the robot's base frame doesn't matter. The TCP frame does (7.2).

**Image pipeline:**
- 640×360 → 360×360 crop starting at x = 106 → 224×224.
- Use `get_image_transform(..., crop_x=...)` from `umi/common/cv_util.py`, with
  the crop from `wrist_crop` in `robospec_umi_dataset/timeline.py`.
- The similar function in `diffusion_policy/common/cv2_util.py` centre-crops
  (x = 140), which is wrong for this data.

**Camera settings:** the same as capture. See `WRIST_MODELS['D455']` and
`WristRecorder._configure` in `robospec_umi_capture/capture.py`:
- 640×360 @ 90 fps;
- auto exposure on, with `auto_exposure_priority` 0, which caps exposure near
  11 ms so 90 fps holds;
- global time on.

Configure the options through the started pipeline's sensor; the comment in
`_configure` explains why.

**Code to reuse:**
- `load_policy` in `eval_without_robot.py`.
- `get_real_umi_obs_dict` / `get_real_umi_action` in `umi/real_world/real_inference_util.py`.
- The timestamp alignment in `BimanualUmiEnv.get_obs`, in `umi/real_world/bimanual_umi_env.py`.

**Don't use `eval_real.py` as-is.**
- It requires a SpaceMouse.
- It hardcodes GoPro/Elgato capture, the WSG gripper and UMI's own flange→tip
  transform.
- It uses a 0.17 s camera latency measured for GoPro.
- It defaults to 10 Hz, but this policy runs at 20 Hz.

Its loop is still the right reference for how to drop late actions and schedule
waypoints.

---

## 6. Setup on the FR3 PC

```bash
# build deps for the yaml's pip block (same list the Dockerfile installs)
sudo apt install build-essential cmake patchelf libusb-1.0-0 \
    libspnav-dev spacenavd libosmesa6-dev libglfw3
conda env create -f robospec_umi/robospec_umi_conda.yaml
conda activate robospec_umi
python -c "import torch; print(torch.__version__, 'sm_120' in torch.cuda.get_arch_list())"
# expect: 2.7.0+cu128 True
python -c "import pyrealsense2; print('realsense ok')"
```

- **Repo:** clone this repo onto the FR3 PC. `umi_server` imports `umi/` and
  `diffusion_policy/` from the repo root.
- **Checkpoint:** copy the trained `.ckpt` over.
- **udev rules:** the DECXIN and gripper-sensor rules in `robospec_umi/README.md`
  are for data collection only. librealsense's own rules are needed, and are
  probably already installed for Robospec's RealSense cameras.
- **Robospec config:** add the env's interpreter to Robospec's scene config, next
  to the other model envs, e.g.
  `"umi_python": "/home/<user>/miniconda3/envs/robospec_umi/bin/python"`.
- **Server environment:** start `umi_server` with `PYTHONPATH` set to this repo's
  root only, plus `PYTHONNOUSERSITE=1`. The launch file runs in a ROS-sourced
  shell, and the 3.9 interpreter must not see ROS's Python 3.10 site-packages.
- **`umi_node`:** needs `numpy` and `scipy` in the system Python, for the
  interpolator.

---

## 7. Missing parts

### 7.1 Streaming Cartesian impedance controller (blocker)

Nothing on the FR3 side can follow a stream of target poses today:
- **Polymetis** provided this for upstream UMI. It is unmaintained, pinned to an
  old libfranka, and supports only the Panda, so it's not an option for the FR3.
- **franka_ros2 v2.1.0's example controllers** all generate their own motion,
  and none takes a target pose from a topic. That covers Cartesian
  pose/velocity/orientation/elbow, joint impedance/position/velocity, gravity
  compensation and move-to-start. The ROS 1 Cartesian impedance example was
  never ported.
- **Robospec's PTP action** takes one goal per call and blocks, so it can't be
  streamed.

Options:
- **crisp_controllers** (recommended). A torque-based Cartesian impedance
  controller for ros2_control, built for learned policies and demoed on the FR3.
- **A port of ROS 1 `franka_ros`'s `cartesian_impedance_example_controller`** to
  ros2_control. That controller subscribes to an equilibrium pose.
- **Franka's own Cartesian pose interface.** Avoid it: it is stiff, and any jerk
  in the commands ends in a reflex.

Running it next to the existing controller:
- **Load it inactive** (`--inactive`), from its own controller config file. Leave
  the existing `controllers.yaml` alone.
- **One controller at a time.** Only one controller can hold the joint command
  interfaces; activating a second is refused, not fought over. Swap them in one
  `switch_controller` call with STRICT.
- **Re-reading the pose.** Check that the PTP side's controller re-reads the
  current pose when it's reactivated. If it resumes its old target, the arm snaps
  back to the pre-UMI pose.
- **Interface mode.** If the existing controller uses a different command
  interface than crisp (position, velocity or Cartesian instead of effort), the
  hardware interface changes libfranka's control mode during the switch. This is
  supported, but test it once.
- **No PTP goals** may be sent while the skill runs.
- **Build location.** Build it in an overlay workspace on top of franka_ws, so
  franka_ws itself stays unchanged.

Starting gains, from upstream's Polymetis setup:
- stiffness `[750, 750, 750, 15, 15, 15]` (N/m ×3, Nm/rad ×3);
- damping `[37, 37, 37, 2, 2, 2]`.

### 7.2 TCP frame on the robot (blocker)

In training, the TCP is defined from the ChArUco board on the handheld gripper
(`timeline.py`):
- `TCP_OFFSET = (0, -0.155, 0.0725)` from the board centre;
- then body-fixed turns of x 180° and z 90°.

The robot has no board, so the same frame has to be measured from the flange:
- **Origin:** the point between the fingertips that the training TCP lands on,
  with these fingers on the Franka Hand.
- **Axes:** pointing the same way relative to the fingers and camera as in
  training. The annotated charuco videos draw the TCP axes, so read off from
  them which axis is which.

Actions are relative to the current TCP. A wrong axis sends the arm the wrong
way, and a wrong origin turns every rotation into a sideways translation.

Get the flange pose from the robot state (`o_t_ee` combined with the inverse of
`f_t_ee`), then apply the fixed flange→TCP transform. That way the result
doesn't depend on Desk's end-effector setting. Check the field names in the
franka_msgs version on the robot.

### 7.3 Gripper

- **One move at a time.** The Franka Hand runs one blocking move or grasp at a
  time. It can't follow a continuous width stream the way upstream's WSG does.
- **Range.** Max width is 80 mm. Training widths reach 96.7 mm (1–2 % of samples
  are above 80 mm), and episodes started at 64–81 mm.
- **Width mapping.** Training widths are the fingertip gap, measured by the
  handheld's encoder. The width the Hand reports isn't necessarily the fingertip
  gap with these fingers. Measure the gap with the Hand fully closed and fully
  open, and fit a mapping. Both the observation and the command use it.
- **Commands from the loop.** Robospec's gripper service is binary and blocking:
  it waits for the grasp to finish and settle, 0.6 s or more. It can't be called
  from the control loop. Instead, send `franka_gripper` grasp (with force) and
  move goals asynchronously, on a width threshold with hysteresis. Stop a running
  goal before sending the opposite one.

### 7.4 Wrist camera sharing (unknown)

It isn't settled yet whether anything other than UMI needs the D455.

What the policy saw in training: 640×360 @ 90 fps, auto exposure capped near
11 ms, and sensor timestamps (global time).
- **Robospec's current wrist-camera path can't feed the policy.** It is an
  OpenCV/V4L2 USB camera at 1280×720, publishing a 5 Hz JPEG preview stamped
  with arrival times.
- **1280×720 on the D455 tops out at 30 fps.** Auto exposure can then go up to
  ~33 ms, which means more blur while moving, and frames are coarser in time.
  Resizing to 640×360 probably gives the same field of view, but compare the two
  modes' intrinsics before relying on it.

Options:
- **UMI-only camera.** `umi_server` owns it from startup. This is the simplest
  and matches training exactly.
- **One shared driver.**
  - A `realsense2_camera` node at the training settings publishes raw frames.
  - Other primitives subscribe too.
  - `umi_node` crops and resizes the frames and hands them to `umi_server`,
    e.g. via `/dev/shm`.
  - Verify once that its frames match what capture recorded.
- **Handover.**
  - Run the driver as a composable node and unload it through the container's
    unload/load services, not with `pkill`. `umi_server` then opens the camera,
    and the steps reverse afterwards.
  - Each switch costs seconds: opening the camera, auto exposure settling, and a
    hardware reset if `initial_reset` is set.
  - Repeated reopening is where RealSense devices wedge. During capture, a D405
    hung until it was physically replugged.

**D405 (later):** 848×480. It needs its own crop entry, and almost certainly its
own data.

### 7.5 Feedback and success detection

The policy has no done signal and no success estimate. `success` in the result
only means the run ended cleanly. Options:
- check the gripper width after a grasp (fully closed means empty);
- let the main program verify with its own perception;
- later, train a success classifier on the wrist view.

### 7.6 Stop rule

Today the run only ends on timeout or cancel. Demos lasted 3.6–6.0 s, so use a
timeout of ~8–10 s.

Candidates for a real stop rule:
- N steps after a grasp completes;
- TCP motion has settled;
- predicted actions near zero for N chunks.

The right one depends on the task, so decide it with the task.

### 7.7 Payload

Enter the wrist camera and its mount as end-effector load in Desk. Under
impedance control, a load error becomes a steady sag: 0.1 kg is about 1.3 mm at
750 N/m. Actions are relative to the measured pose, so the sag adds up every
cycle, at roughly 3–4 mm/s.

### 7.8 Safety

- **Speed:** clamp linear and angular speed in the interpolator.
- **Workspace:** a box around the start pose. Start small.
- **Stiffness:** moderate for the first runs.
- **Abort on:** any robot error or reflex, a `umi_server` disconnect, or stale
  robot state.
- **On cancel or abort:** hold the measured pose and switch the controllers back.

### 7.9 Timestamps and latency

- **One clock.** Camera frames carry RealSense global-time stamps (host clock).
  Robot state carries ROS header stamps (wall clock). Put both on the same clock
  before aligning them.
- **No latency constant to start with.** Training aligned the streams by sensor
  timestamps, with no extra latency shift (`dataset_frequeny: 0`, `DELTA = 0`).
  Start eval the same way. Upstream's 0.17 s is a GoPro number.
- **Measure later:**
  - any remaining camera latency;
  - how far the impedance controller lags behind its target. Upstream subtracts
    an action-execution latency when dropping late waypoints.

### 7.10 Real-time

franka_ros2 shares the PC with GPU workloads: cuRobo, SAM3, and now UMI. PTP
motion runs fine on it. If streaming impedance control shows
`communication_constraints_violation` reflexes, check the kernel (PREEMPT_RT or
not) and the CPU load first.

---

## 8. To confirm on the robot side

- [ ] Which controller serves the PTP action? Its name, its command interface,
      and whether it re-reads the pose when it's reactivated.
- [ ] Is there already a controller that follows a streamed Cartesian target?
- [ ] Desk's end-effector settings and the configured load.
- [ ] The robot-state topic rate: `ros2 topic hz /franka_robot_state_broadcaster/robot_state`.
- [ ] Does any other primitive need the D455?
- [ ] How is the arm hand-guided while FCI is active?

---

## 9. Manual test (no main program)

Before starting, you need:
- a streaming controller (7.1);
- the flange→TCP transform (7.2) and the gripper mapping (7.3), both measured;
- the checkpoint on the FR3 PC.

1. **Dry-run `umi_server` without the robot.**
   - Plug in the D455 and start the server.
   - Drive it with a small script that replays one recorded episode's robot state
     through `reset` and `step`.
   - Check that the cropped image matches training, how long inference takes, and
     that the waypoints are sane: small, smooth, and stamped in the future.
2. **Bring up only what the test needs.** Use a small launch file, not Robospec's
   full FR3 launch, which also starts cuRobo, SAM3, rosbridge, HITL, the DAG and
   the scene cameras. The small launch file starts:
   - franka_bringup `franka.launch.py` with
     `robot_type:=fr3 robot_ip:=<ip> load_gripper:=true use_fake_hardware:=false`;
   - the impedance controller, loaded inactive;
   - `umi_server` (with respawn) and `umi_node`.
3. **Move the arm to a demo-like start pose:** gripper open, object in the wrist
   view. One way is to activate `gravity_compensation_example_controller` and
   guide the arm by hand. `umi_node` deactivates whatever is listed in its
   `deactivate_controllers` parameter, in the same switch call.
4. **Run:**
   ```bash
   ros2 action send_goal /umi/run robospec_msgs/action/RunUmi "{timeout_s: 10.0}" --feedback
   ```
   Ctrl+C cancels the goal. `umi_node` then holds the pose and switches back.
5. **First runs:**
   - low stiffness, a small workspace box and low speed limits;
   - a hand on the user stop;
   - check the result and the logs after each run.
