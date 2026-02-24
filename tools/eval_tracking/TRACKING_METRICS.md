# SONIC Tracking Metrics (Offline, No Core Changes)

This document defines the metrics used by `eval_sonic_tracking.py`.

## Scope

The evaluator compares:

- **Reference motion**: `gear_sonic_deploy/reference/textop_converted/<motion>/...`
- **Tracked simulation logs**: StateLogger CSV logs (`q.csv`, `dq.csv`, etc.)

It does **not** require editing `gear_sonic_deploy` source code.

## Signals Used

From reference motion folder:

- `joint_pos.csv` (29 joints, IsaacLab order)
- `joint_vel.csv` (29 joints, IsaacLab order)
- `body_quat.csv` (root/body_0 quaternion, wxyz)
- `body_ang_vel.csv` (root/body_0 angular velocity, xyz)

From StateLogger logs:

- `q.csv`
- `dq.csv`
- `base_quat.csv`
- `base_ang_vel.csv`
- `motion_name.csv`
- `motion_playing.csv`

## Joint-Order Conversion

`q.csv` and `dq.csv` are logged in MuJoCo/hardware order.

- `q.csv` includes `default_angles` offset.
- The evaluator converts to IsaacLab order using `MUJOCO_TO_ISAACLAB` mapping
  and subtracts default angles before comparison.

## Segmentation and Alignment

1. Build segments where `motion_playing == 1` and `motion_name` is constant.
2. For each segment, use the matching reference folder by `motion_name`.
3. Drop the first `warmup_frames` (default: 20) to avoid switch transient bias.
4. Align from frame 0 and evaluate up to `min(T_log, T_ref)`.

## Metric Definitions

Let `q_t, dq_t` be tracked joint vectors (29D), `q*_t, dq*_t` reference vectors.

- **MPJPE (joint-angle space)**
  - Per-frame: `||q_t - q*_t||_2`
  - Report: temporal mean over aligned frames
  - Unit: `rad`

- **MPJVE (joint-velocity space)**
  - Per-frame: `||dq_t - dq*_t||_2`
  - Report: temporal mean
  - Unit: `rad/s`

- **Root orientation error**
  - `theta_t = 2 * acos(|dot(q_root_t, q_root*_t)|)`
  - Quaternions are normalized before computation
  - Report: temporal mean
  - Unit: `rad`

- **Root angular velocity error**
  - Per-frame: `||w_t - w*_t||_2`
  - Report: temporal mean
  - Unit: `rad/s`

## About the "first-frame jump"

When switching to a new motion, the controller may rapidly move from the previous pose to the new motion's first frame. This transient can dominate metrics if included.

Default handling:

- Drop first `20` frames (0.4s @ 50Hz) per segment.
- Keep the dropped-frame count in reports for transparency.

## Batch autoplay startup reliability

For `auto_run_motion_eval.py`, startup success is detected from generated logs
(`q.csv` or `motion_playing.csv`) instead of a single one-shot key press.

Recommended controls:

- `--start-timeout-sec`: max wait time for control startup.
- `--inject-interval-sec`: repeated `]` key injection interval.
- `--min-log-rows`: minimum data rows to treat startup as successful.

`--single-session` is also supported. In this mode, one deploy process plays
many motions (via `T`/`N`) and writes one continuous log run. The evaluator
still works because it segments by `motion_playing` + `motion_name`.

## Not Included in This Version

- Base position error
- Base linear velocity error

Reason: StateLogger CSV does not include base position / linear velocity directly.
