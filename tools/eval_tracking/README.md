# eval_tracking tools

External tools for automatic motion playback and offline tracking evaluation.

- `auto_run_motion_eval.py`: generate/run per-motion deploy commands (optional sim launch)
- `eval_sonic_tracking.py`: compute MPJPE/MPJVE/root metrics from StateLogger logs
- `TRACKING_METRICS.md`: formal metric definitions and alignment rules
- `EVAL_RESULTS_GUIDE.md`: how to read and validate evaluation outputs

## 1) Dry-run (recommended first)

```bash
python tools/eval_tracking/auto_run_motion_eval.py \
  --repo-root /path/to/GR00T-WholeBodyControl \
  --reference-root gear_sonic_deploy/reference/textop_converted
```

This generates a run plan and commands but does not execute them.

## 2) Execute batch playback

```bash
python tools/eval_tracking/auto_run_motion_eval.py \
  --repo-root /path/to/GR00T-WholeBodyControl \
  --reference-root gear_sonic_deploy/reference/textop_converted \
  --launch-sim \
  --execute
```

Notes:

- Each deploy run uses an isolated single-motion dataset root (no motion-index ambiguity).
- Logs are written under the auto-run workspace (`.../logs/...`).
- Startup is handshake-based: the runner repeatedly injects `]` until `q.csv` or
  `motion_playing.csv` has enough rows.
- Useful stability knobs:
  - `--start-timeout-sec` (default 25)
  - `--inject-interval-sec` (default 1.0)
  - `--min-log-rows` (default 2)
  - `--no-inject-start-key` (disable auto `]` injection)

Optional filter list:

```bash
python tools/eval_tracking/auto_run_motion_eval.py \
  --repo-root /path/to/GR00T-WholeBodyControl \
  --reference-root gear_sonic_deploy/reference/textop_converted \
  --motion-list-txt tools/eval_tracking/my_motion_subset.txt
```

Single-session mode (load models once, then auto `T/N` in one process):

```bash
python tools/eval_tracking/auto_run_motion_eval.py \
  --repo-root /path/to/GR00T-WholeBodyControl \
  --reference-root gear_sonic_deploy/reference/textop_converted \
  --single-session \
  --single-session-max-motions 0 \
  --execute
```

Useful single-session knobs:

- `--single-session-max-motions` (0 = all)
- `--single-session-post-complete-sec`
- `--single-session-key-gap-sec`

## 3) Offline evaluation only

```bash
python tools/eval_tracking/eval_sonic_tracking.py \
  --reference-root /path/to/GR00T-WholeBodyControl/gear_sonic_deploy/reference/textop_converted \
  --logs-dir /path/to/logs_root_or_single_run \
  --out-dir /path/to/output \
  --warmup-frames 20
```

Outputs:

- `metrics_human_readable.md`
- `metrics_summary.csv`
- `metrics_per_segment.csv`
- `metrics_per_motion.csv`
- `metrics_summary.json`
- `alignment_report.json`

Recommended first read: `metrics_human_readable.md`, then detailed drill-down in
`metrics_per_motion.csv` / `metrics_per_segment.csv`.

## 4) First-frame jump impact

If the robot jumps at the beginning of each motion, metrics will be biased high.
Use `--warmup-frames` (default 20) to remove this transient from scoring.
