# SONIC Tracking Evaluation: How to Read Results

This guide explains how to read outputs from `tools/eval_tracking/eval_sonic_tracking.py`.

## 1) Run command

```bash
python tools/eval_tracking/eval_sonic_tracking.py \
  --reference-root gear_sonic_deploy/reference/textop_converted \
  --logs-dir tools/eval_tracking/outputs/<run>/logs \
  --out-dir tools/eval_tracking/outputs/<run>/eval_results \
  --warmup-frames 20
```

## 2) Output files and what each means

- `metrics_human_readable.md`
  - Best entry point. Contains confidence, global metrics, best/worst motions, and data-quality warnings.
- `metrics_summary.csv`
  - One-line summary for spreadsheet / experiment tracking.
- `metrics_summary.json`
  - Full structured summary with quality checks and confidence reasons.
- `metrics_per_motion.csv`
  - Per-motion aggregated metrics. Use this to rank motions and find weak cases.
- `metrics_per_segment.csv`
  - Per-segment detailed metrics. Use this when a motion looks wrong and you need drill-down.
- `alignment_report.json`
  - Alignment diagnostics (missing reference, empty segment after warmup, eval failures).

## 3) Recommended reading order (fast)

1. Open `metrics_human_readable.md`.
2. Check `Confidence` and `Data Quality` section first.
3. Check global metrics table (frame-weighted first, segment mean second).
4. Open `metrics_per_motion.csv` and inspect worst MPJPE/MPJVE entries.
5. If needed, open `metrics_per_segment.csv` + `alignment_report.json` for root cause.

## 4) Metric definitions and units

- `mpjpe_joint_rad` / `mpjpe_joint_deg`
  - Mean L2 error of 29 joint angles per frame.
- `mpjve_joint_rads` / `mpjve_joint_degps`
  - Mean L2 error of 29 joint angular velocities per frame.
- `root_orientation_error_rad` / `root_orientation_error_deg`
  - Geodesic angle difference between tracked root quaternion and reference root quaternion.
- `root_angular_velocity_error_rads` / `root_angular_velocity_error_degps`
  - Mean L2 error of root angular velocity.

The script reports:

- `segment_unweighted` mean: each segment counts equally.
- `frame_weighted` mean: long segments weigh more. This is usually the main metric for comparing runs.

## 5) How to judge whether numbers are trustworthy

Check in `metrics_summary.json`:

- `quality_checks.status_counts`
  - Should mostly be `ok`.
- `quality_checks.non_ok_alignment_rows`
  - Should be near 0.
- `quality_checks.short_segments`
  - High value means warmup/drop leaves too little data.
- `quality_checks.non_finite_segments`
  - Must be 0.
- `quality_checks.root_quat_norm_error_max`
  - Should be very small.

Then check:

- `confidence` and `confidence_reasons`
  - `low` means sample size or quality is insufficient; do not use for model ranking.

## 6) Typical failure patterns

- High MPJPE + high MPJVE on almost all motions
  - Often means tracking quality is poor, or reference/log joint mapping is wrong.
- Root orientation error very high but MPJPE moderate
  - Check heading/base alignment behavior around motion switch.
- Many `empty_after_warmup` in `alignment_report.json`
  - Warmup too large or motion segments too short.
- Only 1-2 segments evaluated
  - Statistics are unstable. Run more motions before comparing policies.

## 7) Practical checklist before comparing two runs

- Same `--warmup-frames`
- Similar segment count and frames used total
- `confidence` not `low`
- Low non-ok alignment rows
- Compare frame-weighted means in `metrics_summary.csv`

## 8) Useful quick commands

```bash
# quick summary
cat tools/eval_tracking/outputs/<run>/eval_results/metrics_summary.csv

# top 10 worst motions by MPJPE
python - <<'PY'
import pandas as pd
p = 'tools/eval_tracking/outputs/<run>/eval_results/metrics_per_motion.csv'
df = pd.read_csv(p)
print(df.sort_values('mpjpe_joint_rad_frame_weighted_mean', ascending=False).head(10)[
    ['motion_name', 'segments', 'frames_used_total', 'mpjpe_joint_rad_frame_weighted_mean']
])
PY
```
