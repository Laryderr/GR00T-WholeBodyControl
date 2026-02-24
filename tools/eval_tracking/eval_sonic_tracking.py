#!/usr/bin/env python3
"""Offline tracking-metric evaluator for SONIC reference-motion tracking.

This script compares GR00T reference motion CSVs against StateLogger CSV logs and
computes per-segment / per-motion metrics:
  - MPJPE (joint-angle space)
  - MPJVE (joint-velocity space)
  - root orientation error
  - root angular-velocity error

It emits both machine-readable CSV/JSON outputs and a human-readable Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Mappings copied from gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

REQUIRED_LOG_FILES = {
    "q.csv",
    "dq.csv",
    "base_quat.csv",
    "base_ang_vel.csv",
    "motion_name.csv",
    "motion_playing.csv",
}

DEFAULT_CONF_THRESHOLDS = {
    "low": {
        "min_segments": 3,
        "min_frames": 500,
        "max_failure_ratio": 0.20,
    },
    "high": {
        "min_segments": 10,
        "min_frames": 3000,
        "max_failure_ratio": 0.0,
    },
}


@dataclass
class Segment:
    run_name: str
    motion_name: str
    start_idx: int
    end_idx: int


@dataclass
class SegmentMetrics:
    run_name: str
    motion_name: str
    segment_id: int
    frames_raw: int
    frames_drop: int
    frames_used: int
    truncated_to_ref: bool
    mpjpe_joint_rad: float
    mpjve_joint_rads: float
    root_orientation_error_rad: float
    root_angular_velocity_error_rads: float
    root_quat_norm_error_mean: float
    root_quat_norm_error_max: float
    quality_flag: str


def _rad_to_deg(x: float) -> float:
    return float(x * 180.0 / math.pi)


def _contains_non_finite(*arrs: np.ndarray) -> bool:
    return not all(np.all(np.isfinite(a)) for a in arrs)


def _load_numeric_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        return header, np.zeros((0, len(header)), dtype=np.float64)
    return header, np.asarray(rows, dtype=np.float64)


def _load_motion_name_csv(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "motion_name" not in reader.fieldnames:
            raise ValueError(f"motion_name column not found in {path}")
        for row in reader:
            names.append((row.get("motion_name") or "").strip())
    return names


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(q, axis=1, keepdims=True)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return q / denom


def _quat_geodesic_rad(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    dots = np.sum(q_a * q_b, axis=1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def _contains_required_log_files(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_LOG_FILES)


def discover_log_runs(logs_root: Path) -> List[Path]:
    if _contains_required_log_files(logs_root):
        return [logs_root]

    runs: List[Path] = []
    seen = set()
    for q_path in logs_root.rglob("q.csv"):
        parent = q_path.parent
        if parent in seen:
            continue
        if _contains_required_log_files(parent):
            seen.add(parent)
            runs.append(parent)
    return sorted(runs)


def _truncate_all_to_min_len(arrays: Sequence[np.ndarray], names: Sequence[str]) -> Tuple[List[np.ndarray], int]:
    if len(arrays) != len(names):
        raise ValueError("arrays/names length mismatch")
    min_len = min((a.shape[0] for a in arrays), default=0)
    return [a[:min_len] for a in arrays], min_len


def load_reference_motion(motion_dir: Path) -> Dict[str, np.ndarray]:
    _, joint_pos = _load_numeric_csv(motion_dir / "joint_pos.csv")
    _, joint_vel = _load_numeric_csv(motion_dir / "joint_vel.csv")
    _, body_quat = _load_numeric_csv(motion_dir / "body_quat.csv")
    _, body_ang_vel = _load_numeric_csv(motion_dir / "body_ang_vel.csv")

    if joint_pos.shape[1] < 29 or joint_vel.shape[1] < 29:
        raise ValueError(f"{motion_dir} joint_pos/joint_vel columns < 29")
    if body_quat.shape[1] < 9:
        raise ValueError(f"{motion_dir} body_quat.csv columns invalid")
    if body_ang_vel.shape[1] < 8:
        raise ValueError(f"{motion_dir} body_ang_vel.csv columns invalid")

    return {
        "joint_pos": joint_pos[:, -29:],
        "joint_vel": joint_vel[:, -29:],
        "root_quat": body_quat[:, 5:9],
        "root_ang_vel": body_ang_vel[:, 5:8],
    }


def load_log_run(run_dir: Path) -> Dict[str, np.ndarray | List[str]]:
    _, q = _load_numeric_csv(run_dir / "q.csv")
    _, dq = _load_numeric_csv(run_dir / "dq.csv")
    _, base_quat = _load_numeric_csv(run_dir / "base_quat.csv")
    _, base_ang_vel = _load_numeric_csv(run_dir / "base_ang_vel.csv")
    _, motion_playing = _load_numeric_csv(run_dir / "motion_playing.csv")
    motion_names = _load_motion_name_csv(run_dir / "motion_name.csv")

    if q.shape[1] < 34 or dq.shape[1] < 34:
        raise ValueError(f"{run_dir}: q/dq csv columns invalid")
    if base_quat.shape[1] < 9 or base_ang_vel.shape[1] < 8:
        raise ValueError(f"{run_dir}: base_quat/base_ang_vel csv columns invalid")

    q_mj = q[:, -29:]
    dq_mj = dq[:, -29:]
    q_isaac = (q_mj - DEFAULT_ANGLES_MUJOCO)[..., MUJOCO_TO_ISAACLAB]
    dq_isaac = dq_mj[..., MUJOCO_TO_ISAACLAB]

    arrays, min_len = _truncate_all_to_min_len(
        [q_isaac, dq_isaac, base_quat[:, 5:9], base_ang_vel[:, 5:8], motion_playing[:, -1:]],
        ["q", "dq", "base_quat", "base_ang_vel", "motion_playing"],
    )
    q_isaac, dq_isaac, base_quat_vec, base_ang_vel_vec, motion_playing_vec = arrays

    names_arr = motion_names[:min_len]
    if len(names_arr) < min_len:
        names_arr += [""] * (min_len - len(names_arr))

    return {
        "q": q_isaac,
        "dq": dq_isaac,
        "base_quat": base_quat_vec,
        "base_ang_vel": base_ang_vel_vec,
        "motion_playing": motion_playing_vec[:, 0] > 0.5,
        "motion_name": names_arr,
    }


def segment_by_motion_name_and_play(log_data: Dict[str, np.ndarray | List[str]], run_name: str) -> List[Segment]:
    playing = np.asarray(log_data["motion_playing"], dtype=bool)
    names = list(log_data["motion_name"])
    n = len(names)

    out: List[Segment] = []
    i = 0
    while i < n:
        if not playing[i]:
            i += 1
            continue
        name = names[i].strip()
        j = i + 1
        while j < n and playing[j] and names[j].strip() == name:
            j += 1
        if name:
            out.append(Segment(run_name=run_name, motion_name=name, start_idx=i, end_idx=j))
        i = j
    return out


def compute_segment_metrics(
    seg: Segment,
    seg_idx: int,
    log_data: Dict[str, np.ndarray | List[str]],
    ref_data: Dict[str, np.ndarray],
    warmup_frames: int,
    min_frames_per_segment: int,
) -> SegmentMetrics | None:
    s = seg.start_idx
    e = seg.end_idx

    q_log = np.asarray(log_data["q"])[s:e]
    dq_log = np.asarray(log_data["dq"])[s:e]
    root_q_log = np.asarray(log_data["base_quat"])[s:e]
    root_w_log = np.asarray(log_data["base_ang_vel"])[s:e]

    drop = min(max(warmup_frames, 0), q_log.shape[0])
    q_log = q_log[drop:]
    dq_log = dq_log[drop:]
    root_q_log = root_q_log[drop:]
    root_w_log = root_w_log[drop:]

    if q_log.shape[0] == 0:
        return None

    q_ref = ref_data["joint_pos"]
    dq_ref = ref_data["joint_vel"]
    root_q_ref = ref_data["root_quat"]
    root_w_ref = ref_data["root_ang_vel"]

    used = min(q_log.shape[0], q_ref.shape[0], dq_ref.shape[0], root_q_ref.shape[0], root_w_ref.shape[0])
    if used <= 0:
        return None

    q_log = q_log[:used]
    dq_log = dq_log[:used]
    root_q_log = root_q_log[:used]
    root_w_log = root_w_log[:used]

    q_ref = q_ref[:used]
    dq_ref = dq_ref[:used]
    root_q_ref = root_q_ref[:used]
    root_w_ref = root_w_ref[:used]

    err_joint = np.linalg.norm(q_log - q_ref, axis=1)
    err_joint_vel = np.linalg.norm(dq_log - dq_ref, axis=1)

    q_norm_err_log = np.abs(np.linalg.norm(root_q_log, axis=1) - 1.0)
    q_norm_err_ref = np.abs(np.linalg.norm(root_q_ref, axis=1) - 1.0)
    root_q_log_n = _normalize_quat_wxyz(root_q_log)
    root_q_ref_n = _normalize_quat_wxyz(root_q_ref)
    err_root_ori = _quat_geodesic_rad(root_q_log_n, root_q_ref_n)
    err_root_w = np.linalg.norm(root_w_log - root_w_ref, axis=1)

    flags: List[str] = []
    truncated = used < (e - s - drop)
    if truncated:
        flags.append("truncated_to_ref")
    if used < max(min_frames_per_segment, 1):
        flags.append("short_segment")
    if _contains_non_finite(err_joint, err_joint_vel, err_root_ori, err_root_w):
        flags.append("non_finite")

    quality_flag = "ok" if not flags else "|".join(flags)

    return SegmentMetrics(
        run_name=seg.run_name,
        motion_name=seg.motion_name,
        segment_id=seg_idx,
        frames_raw=e - s,
        frames_drop=drop,
        frames_used=used,
        truncated_to_ref=truncated,
        mpjpe_joint_rad=float(np.mean(err_joint)),
        mpjve_joint_rads=float(np.mean(err_joint_vel)),
        root_orientation_error_rad=float(np.mean(err_root_ori)),
        root_angular_velocity_error_rads=float(np.mean(err_root_w)),
        root_quat_norm_error_mean=float(np.mean(np.concatenate([q_norm_err_log, q_norm_err_ref]))),
        root_quat_norm_error_max=float(np.max(np.concatenate([q_norm_err_log, q_norm_err_ref]))),
        quality_flag=quality_flag,
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    if np.sum(w) <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / np.sum(w))


def aggregate_per_motion(metrics: Sequence[SegmentMetrics]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[SegmentMetrics]] = {}
    for m in metrics:
        buckets.setdefault(m.motion_name, []).append(m)

    out: List[Dict[str, object]] = []
    for motion in sorted(buckets):
        vals = buckets[motion]
        weights = [v.frames_used for v in vals]
        frames_used_total = int(sum(weights))
        segments = len(vals)

        mpjpe_fw = _weighted_mean([v.mpjpe_joint_rad for v in vals], weights)
        mpjve_fw = _weighted_mean([v.mpjve_joint_rads for v in vals], weights)
        root_ori_fw = _weighted_mean([v.root_orientation_error_rad for v in vals], weights)
        root_w_fw = _weighted_mean([v.root_angular_velocity_error_rads for v in vals], weights)

        mpjpe_seg = float(np.mean([v.mpjpe_joint_rad for v in vals]))
        mpjve_seg = float(np.mean([v.mpjve_joint_rads for v in vals]))
        root_ori_seg = float(np.mean([v.root_orientation_error_rad for v in vals]))
        root_w_seg = float(np.mean([v.root_angular_velocity_error_rads for v in vals]))

        if segments < 2 or frames_used_total < 200:
            confidence_hint = "low"
        elif segments < 5 or frames_used_total < 1000:
            confidence_hint = "medium"
        else:
            confidence_hint = "high"

        out.append(
            {
                "motion_name": motion,
                "segments": segments,
                "frames_used_total": frames_used_total,
                "confidence_hint": confidence_hint,
                "mpjpe_joint_rad_frame_weighted_mean": mpjpe_fw,
                "mpjpe_joint_deg_frame_weighted_mean": _rad_to_deg(mpjpe_fw),
                "mpjpe_joint_rad_segment_mean": mpjpe_seg,
                "mpjpe_joint_deg_segment_mean": _rad_to_deg(mpjpe_seg),
                "mpjve_joint_rads_frame_weighted_mean": mpjve_fw,
                "mpjve_joint_degps_frame_weighted_mean": _rad_to_deg(mpjve_fw),
                "mpjve_joint_rads_segment_mean": mpjve_seg,
                "mpjve_joint_degps_segment_mean": _rad_to_deg(mpjve_seg),
                "root_orientation_error_rad_frame_weighted_mean": root_ori_fw,
                "root_orientation_error_deg_frame_weighted_mean": _rad_to_deg(root_ori_fw),
                "root_orientation_error_rad_segment_mean": root_ori_seg,
                "root_orientation_error_deg_segment_mean": _rad_to_deg(root_ori_seg),
                "root_angular_velocity_error_rads_frame_weighted_mean": root_w_fw,
                "root_angular_velocity_error_degps_frame_weighted_mean": _rad_to_deg(root_w_fw),
                "root_angular_velocity_error_rads_segment_mean": root_w_seg,
                "root_angular_velocity_error_degps_segment_mean": _rad_to_deg(root_w_seg),
            }
        )
    return out


def _stat(arr: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray(list(arr), dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _build_metric_summary(metrics: Sequence[SegmentMetrics], attr: str) -> Dict[str, object]:
    vals = [float(getattr(m, attr)) for m in metrics]
    weights = [m.frames_used for m in metrics]
    segment_stats = _stat(vals)
    frame_weighted_mean = _weighted_mean(vals, weights)

    return {
        "segment_unweighted": segment_stats,
        "frame_weighted_mean": frame_weighted_mean,
        "segment_unweighted_deg": {k: _rad_to_deg(v) for k, v in segment_stats.items()},
        "frame_weighted_mean_deg": _rad_to_deg(frame_weighted_mean),
    }


def _failure_ratio(alignment_rows: Sequence[Dict[str, object]]) -> float:
    if not alignment_rows:
        return 1.0
    fail = sum(1 for r in alignment_rows if str(r.get("status", "")) != "ok")
    return float(fail / max(len(alignment_rows), 1))


def _build_quality_checks(
    metrics: Sequence[SegmentMetrics],
    alignment_rows: Sequence[Dict[str, object]],
    min_frames_per_segment: int,
) -> Dict[str, object]:
    status_counts: Dict[str, int] = {}
    for row in alignment_rows:
        st = str(row.get("status", "unknown"))
        status_counts[st] = status_counts.get(st, 0) + 1

    short_segments = sum(1 for m in metrics if "short_segment" in m.quality_flag)
    truncated_segments = sum(1 for m in metrics if m.truncated_to_ref)
    non_finite_segments = sum(1 for m in metrics if "non_finite" in m.quality_flag)

    frames_used = [m.frames_used for m in metrics]
    quat_norm_means = [m.root_quat_norm_error_mean for m in metrics]
    quat_norm_maxes = [m.root_quat_norm_error_max for m in metrics]

    return {
        "status_counts": status_counts,
        "non_ok_alignment_rows": int(sum(v for k, v in status_counts.items() if k != "ok")),
        "short_segments": short_segments,
        "truncated_segments": truncated_segments,
        "non_finite_segments": non_finite_segments,
        "min_frames_per_segment_threshold": int(min_frames_per_segment),
        "frames_used_min": int(min(frames_used)) if frames_used else 0,
        "frames_used_max": int(max(frames_used)) if frames_used else 0,
        "root_quat_norm_error_mean": float(np.mean(quat_norm_means)) if quat_norm_means else float("nan"),
        "root_quat_norm_error_max": float(np.max(quat_norm_maxes)) if quat_norm_maxes else float("nan"),
        "alignment_failure_ratio": _failure_ratio(alignment_rows),
    }


def _resolve_conf_thresholds(path: str) -> Dict[str, Dict[str, float]]:
    if not path:
        return DEFAULT_CONF_THRESHOLDS

    p = Path(path)
    if p.is_file():
        data = json.loads(p.read_text())
    else:
        data = json.loads(path)

    out = {
        "low": {
            "min_segments": float(data.get("low", {}).get("min_segments", DEFAULT_CONF_THRESHOLDS["low"]["min_segments"])),
            "min_frames": float(data.get("low", {}).get("min_frames", DEFAULT_CONF_THRESHOLDS["low"]["min_frames"])),
            "max_failure_ratio": float(
                data.get("low", {}).get("max_failure_ratio", DEFAULT_CONF_THRESHOLDS["low"]["max_failure_ratio"])
            ),
        },
        "high": {
            "min_segments": float(data.get("high", {}).get("min_segments", DEFAULT_CONF_THRESHOLDS["high"]["min_segments"])),
            "min_frames": float(data.get("high", {}).get("min_frames", DEFAULT_CONF_THRESHOLDS["high"]["min_frames"])),
            "max_failure_ratio": float(
                data.get("high", {}).get("max_failure_ratio", DEFAULT_CONF_THRESHOLDS["high"]["max_failure_ratio"])
            ),
        },
    }
    return out


def _infer_confidence(
    segment_count: int,
    frames_used_total: int,
    quality_checks: Dict[str, object],
    conf_thresholds: Dict[str, Dict[str, float]],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    failure_ratio = float(quality_checks.get("alignment_failure_ratio", 1.0))
    non_finite_segments = int(quality_checks.get("non_finite_segments", 0))

    low_cfg = conf_thresholds["low"]
    high_cfg = conf_thresholds["high"]

    if segment_count < int(low_cfg["min_segments"]):
        reasons.append(f"segment_count<{int(low_cfg['min_segments'])}")
    if frames_used_total < int(low_cfg["min_frames"]):
        reasons.append(f"frames_used_total<{int(low_cfg['min_frames'])}")
    if failure_ratio > float(low_cfg["max_failure_ratio"]):
        reasons.append(f"failure_ratio>{float(low_cfg['max_failure_ratio']):.2f}")
    if non_finite_segments > 0:
        reasons.append("non_finite_segments>0")

    if reasons:
        return "low", reasons

    if (
        segment_count >= int(high_cfg["min_segments"])
        and frames_used_total >= int(high_cfg["min_frames"])
        and failure_ratio <= float(high_cfg["max_failure_ratio"])
        and non_finite_segments == 0
    ):
        return "high", ["enough_samples", "clean_alignment", "no_non_finite"]

    return "medium", ["sample_count_or_frames_not_high_enough"]


def build_summary(
    metrics: Sequence[SegmentMetrics],
    alignment_rows: Sequence[Dict[str, object]],
    warmup_frames: int,
    min_frames_per_segment: int,
    conf_thresholds: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "segment_count": len(metrics),
        "frames_used_total": int(sum(m.frames_used for m in metrics)),
        "warmup_frames": int(warmup_frames),
        "min_frames_per_segment": int(min_frames_per_segment),
    }

    quality_checks = _build_quality_checks(metrics, alignment_rows, min_frames_per_segment)
    summary["quality_checks"] = quality_checks

    if not metrics:
        summary["metrics"] = {}
        summary["confidence"] = "low"
        summary["confidence_reasons"] = ["no_valid_segments"]
        summary["confidence_thresholds"] = conf_thresholds
        return summary

    summary["metrics"] = {
        "mpjpe_joint_rad": _build_metric_summary(metrics, "mpjpe_joint_rad"),
        "mpjve_joint_rads": _build_metric_summary(metrics, "mpjve_joint_rads"),
        "root_orientation_error_rad": _build_metric_summary(metrics, "root_orientation_error_rad"),
        "root_angular_velocity_error_rads": _build_metric_summary(metrics, "root_angular_velocity_error_rads"),
    }

    confidence, reasons = _infer_confidence(
        segment_count=len(metrics),
        frames_used_total=int(summary["frames_used_total"]),
        quality_checks=quality_checks,
        conf_thresholds=conf_thresholds,
    )
    summary["confidence"] = confidence
    summary["confidence_reasons"] = reasons
    summary["confidence_thresholds"] = conf_thresholds
    return summary


def build_summary_csv_row(summary: Dict[str, object]) -> Dict[str, object]:
    metrics = summary.get("metrics", {})

    def _metric(metric_key: str, field: str, default: float = float("nan")) -> float:
        return float(metrics.get(metric_key, {}).get(field, default))

    def _metric_nested(metric_key: str, nested_key: str, field: str, default: float = float("nan")) -> float:
        return float(metrics.get(metric_key, {}).get(nested_key, {}).get(field, default))

    q = summary.get("quality_checks", {})
    status_counts = q.get("status_counts", {})
    return {
        "segment_count": int(summary.get("segment_count", 0)),
        "frames_used_total": int(summary.get("frames_used_total", 0)),
        "confidence": summary.get("confidence", "low"),
        "ok_rows": int(status_counts.get("ok", 0)),
        "non_ok_rows": int(q.get("non_ok_alignment_rows", 0)),
        "short_segments": int(q.get("short_segments", 0)),
        "truncated_segments": int(q.get("truncated_segments", 0)),
        "mpjpe_joint_rad_frame_weighted_mean": _metric("mpjpe_joint_rad", "frame_weighted_mean"),
        "mpjpe_joint_deg_frame_weighted_mean": _metric("mpjpe_joint_rad", "frame_weighted_mean_deg"),
        "mpjpe_joint_rad_segment_mean": _metric_nested("mpjpe_joint_rad", "segment_unweighted", "mean"),
        "mpjpe_joint_deg_segment_mean": _metric_nested("mpjpe_joint_rad", "segment_unweighted_deg", "mean"),
        "mpjve_joint_rads_frame_weighted_mean": _metric("mpjve_joint_rads", "frame_weighted_mean"),
        "mpjve_joint_degps_frame_weighted_mean": _metric("mpjve_joint_rads", "frame_weighted_mean_deg"),
        "root_orientation_error_rad_frame_weighted_mean": _metric("root_orientation_error_rad", "frame_weighted_mean"),
        "root_orientation_error_deg_frame_weighted_mean": _metric("root_orientation_error_rad", "frame_weighted_mean_deg"),
        "root_angular_velocity_error_rads_frame_weighted_mean": _metric(
            "root_angular_velocity_error_rads", "frame_weighted_mean"
        ),
        "root_angular_velocity_error_degps_frame_weighted_mean": _metric(
            "root_angular_velocity_error_rads", "frame_weighted_mean_deg"
        ),
    }


def _fmt(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_human_readable_report(
    out_path: Path,
    summary: Dict[str, object],
    per_motion_rows: Sequence[Dict[str, object]],
    alignment_rows: Sequence[Dict[str, object]],
    top_k: int,
) -> None:
    metrics = summary.get("metrics", {})
    quality = summary.get("quality_checks", {})

    def m(metric_key: str) -> Dict[str, object]:
        return metrics.get(metric_key, {})

    def frame_weight(metric_key: str) -> float:
        return float(m(metric_key).get("frame_weighted_mean", float("nan")))

    def frame_weight_deg(metric_key: str) -> float:
        return float(m(metric_key).get("frame_weighted_mean_deg", float("nan")))

    def seg_mean(metric_key: str) -> float:
        return float(m(metric_key).get("segment_unweighted", {}).get("mean", float("nan")))

    def seg_mean_deg(metric_key: str) -> float:
        return float(m(metric_key).get("segment_unweighted_deg", {}).get("mean", float("nan")))

    by_mpjpe = sorted(per_motion_rows, key=lambda r: float(r.get("mpjpe_joint_rad_frame_weighted_mean", float("inf"))))
    top_k = max(int(top_k), 1)
    best = by_mpjpe[:top_k]
    worst = by_mpjpe[-top_k:][::-1]

    lines: List[str] = []
    lines.append("# Tracking Evaluation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Confidence: **{summary.get('confidence', 'low')}**")
    lines.append(f"- Segment count: {summary.get('segment_count', 0)}")
    lines.append(f"- Frames used total: {summary.get('frames_used_total', 0)}")
    lines.append(f"- Warmup frames: {summary.get('warmup_frames', 0)}")
    lines.append("")
    reasons = summary.get("confidence_reasons", [])
    if reasons:
        lines.append("- Confidence reasons: " + ", ".join(str(x) for x in reasons))
        lines.append("")

    lines.append("## Global Metrics")
    lines.append("")
    lines.append("| Metric | Frame-weighted | Segment mean |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| MPJPE (joint) | {_fmt(frame_weight('mpjpe_joint_rad'))} rad ({_fmt(frame_weight_deg('mpjpe_joint_rad'))} deg) | {_fmt(seg_mean('mpjpe_joint_rad'))} rad ({_fmt(seg_mean_deg('mpjpe_joint_rad'))} deg) |"
    )
    lines.append(
        f"| MPJVE (joint vel) | {_fmt(frame_weight('mpjve_joint_rads'))} rad/s ({_fmt(frame_weight_deg('mpjve_joint_rads'))} deg/s) | {_fmt(seg_mean('mpjve_joint_rads'))} rad/s ({_fmt(seg_mean_deg('mpjve_joint_rads'))} deg/s) |"
    )
    lines.append(
        f"| Root orientation | {_fmt(frame_weight('root_orientation_error_rad'))} rad ({_fmt(frame_weight_deg('root_orientation_error_rad'))} deg) | {_fmt(seg_mean('root_orientation_error_rad'))} rad ({_fmt(seg_mean_deg('root_orientation_error_rad'))} deg) |"
    )
    lines.append(
        f"| Root angular velocity | {_fmt(frame_weight('root_angular_velocity_error_rads'))} rad/s ({_fmt(frame_weight_deg('root_angular_velocity_error_rads'))} deg/s) | {_fmt(seg_mean('root_angular_velocity_error_rads'))} rad/s ({_fmt(seg_mean_deg('root_angular_velocity_error_rads'))} deg/s) |"
    )
    lines.append("")

    def _table_motion(rows: Sequence[Dict[str, object]], title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| motion_name | segments | frames | MPJPE(rad) | MPJVE(rad/s) | root_ori(rad) | root_w(rad/s) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                "| {motion} | {segments} | {frames} | {mpjpe} | {mpjve} | {root_ori} | {root_w} |".format(
                    motion=r.get("motion_name", ""),
                    segments=r.get("segments", 0),
                    frames=r.get("frames_used_total", 0),
                    mpjpe=_fmt(float(r.get("mpjpe_joint_rad_frame_weighted_mean", float("nan")))),
                    mpjve=_fmt(float(r.get("mpjve_joint_rads_frame_weighted_mean", float("nan")))),
                    root_ori=_fmt(float(r.get("root_orientation_error_rad_frame_weighted_mean", float("nan")))),
                    root_w=_fmt(float(r.get("root_angular_velocity_error_rads_frame_weighted_mean", float("nan")))),
                )
            )
        lines.append("")

    _table_motion(best, f"Best {min(top_k, len(best))} Motions (by MPJPE)")
    _table_motion(worst, f"Worst {min(top_k, len(worst))} Motions (by MPJPE)")

    lines.append("## Data Quality")
    lines.append("")
    lines.append(f"- Alignment status counts: {quality.get('status_counts', {})}")
    lines.append(f"- Non-ok alignment rows: {quality.get('non_ok_alignment_rows', 0)}")
    lines.append(f"- Short segments: {quality.get('short_segments', 0)}")
    lines.append(f"- Truncated segments: {quality.get('truncated_segments', 0)}")
    lines.append(f"- Non-finite segments: {quality.get('non_finite_segments', 0)}")
    lines.append(
        f"- Root quaternion norm error: mean={_fmt(float(quality.get('root_quat_norm_error_mean', float('nan'))), 6)}, max={_fmt(float(quality.get('root_quat_norm_error_max', float('nan'))), 6)}"
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Use `metrics_per_motion.csv` for quick ranking and comparison across runs.")
    lines.append("- Use `metrics_per_segment.csv` + `alignment_report.json` to debug outliers.")
    lines.append("- If confidence is low, increase number of motions or frames before comparing models.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SONIC tracking accuracy from StateLogger CSV logs.")
    parser.add_argument(
        "--reference-root",
        default="gear_sonic_deploy/reference/textop_converted",
        help="Root directory containing per-motion reference folders.",
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Either one log run directory (contains q.csv, dq.csv, ...) or a parent containing many runs.",
    )
    parser.add_argument(
        "--out-dir",
        default="tools/eval_tracking/outputs/eval_result",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Drop first N frames of each playing segment to avoid switch transient bias.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing reference motions or malformed runs. Default: skip bad entries and continue.",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Reserved for future frame-level dump (not used in this version).",
    )
    parser.add_argument(
        "--min-frames-per-segment",
        type=int,
        default=30,
        help="Threshold for marking a segment as short (warning only).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K best/worst motions to include in markdown report.",
    )
    parser.add_argument(
        "--confidence-thresholds-json",
        default="",
        help="Optional path to JSON file (or inline JSON) overriding confidence thresholds.",
    )

    parser.add_argument("--emit-markdown-report", dest="emit_markdown_report", action="store_true")
    parser.add_argument("--no-emit-markdown-report", dest="emit_markdown_report", action="store_false")
    parser.set_defaults(emit_markdown_report=True)

    parser.add_argument("--summary-csv", dest="summary_csv", action="store_true")
    parser.add_argument("--no-summary-csv", dest="summary_csv", action="store_false")
    parser.set_defaults(summary_csv=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root).resolve()
    logs_root = Path(args.logs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_thresholds = _resolve_conf_thresholds(args.confidence_thresholds_json)

    runs = discover_log_runs(logs_root)
    if not runs:
        raise RuntimeError(f"No valid log runs found under: {logs_root}")

    all_metrics: List[SegmentMetrics] = []
    alignment_rows: List[Dict[str, object]] = []

    for run_dir in runs:
        run_name = run_dir.name
        try:
            log_data = load_log_run(run_dir)
        except Exception as exc:
            if args.strict:
                raise
            alignment_rows.append(
                {
                    "run_name": run_name,
                    "motion_name": "",
                    "status": "run_load_failed",
                    "message": str(exc),
                }
            )
            continue

        segments = segment_by_motion_name_and_play(log_data, run_name)
        if not segments:
            alignment_rows.append(
                {
                    "run_name": run_name,
                    "motion_name": "",
                    "status": "no_playing_segments",
                    "message": "No motion_playing==1 segments found",
                }
            )
            continue

        for seg_idx, seg in enumerate(segments):
            motion_dir = reference_root / seg.motion_name
            if not motion_dir.is_dir():
                msg = f"reference motion not found: {motion_dir}"
                if args.strict:
                    raise FileNotFoundError(msg)
                alignment_rows.append(
                    {
                        "run_name": run_name,
                        "motion_name": seg.motion_name,
                        "status": "missing_reference",
                        "segment_id": seg_idx,
                        "start_idx": seg.start_idx,
                        "end_idx": seg.end_idx,
                        "message": msg,
                    }
                )
                continue

            try:
                ref_data = load_reference_motion(motion_dir)
                metrics = compute_segment_metrics(
                    seg,
                    seg_idx,
                    log_data,
                    ref_data,
                    args.warmup_frames,
                    args.min_frames_per_segment,
                )
            except Exception as exc:
                if args.strict:
                    raise
                alignment_rows.append(
                    {
                        "run_name": run_name,
                        "motion_name": seg.motion_name,
                        "status": "segment_eval_failed",
                        "segment_id": seg_idx,
                        "start_idx": seg.start_idx,
                        "end_idx": seg.end_idx,
                        "message": str(exc),
                    }
                )
                continue

            if metrics is None:
                alignment_rows.append(
                    {
                        "run_name": run_name,
                        "motion_name": seg.motion_name,
                        "status": "empty_after_warmup",
                        "segment_id": seg_idx,
                        "start_idx": seg.start_idx,
                        "end_idx": seg.end_idx,
                        "message": "Segment dropped by warmup or has no overlap with reference",
                    }
                )
                continue

            all_metrics.append(metrics)
            alignment_rows.append(
                {
                    "run_name": run_name,
                    "motion_name": seg.motion_name,
                    "status": "ok",
                    "segment_id": seg_idx,
                    "start_idx": seg.start_idx,
                    "end_idx": seg.end_idx,
                    "frames_raw": metrics.frames_raw,
                    "frames_drop": metrics.frames_drop,
                    "frames_used": metrics.frames_used,
                    "truncated_to_ref": metrics.truncated_to_ref,
                    "quality_flag": metrics.quality_flag,
                }
            )

    per_segment_rows = [
        {
            "run_name": m.run_name,
            "motion_name": m.motion_name,
            "segment_id": m.segment_id,
            "frames_raw": m.frames_raw,
            "frames_drop": m.frames_drop,
            "frames_used": m.frames_used,
            "truncated_to_ref": m.truncated_to_ref,
            "quality_flag": m.quality_flag,
            "mpjpe_joint_rad": m.mpjpe_joint_rad,
            "mpjpe_joint_deg": _rad_to_deg(m.mpjpe_joint_rad),
            "mpjve_joint_rads": m.mpjve_joint_rads,
            "mpjve_joint_degps": _rad_to_deg(m.mpjve_joint_rads),
            "root_orientation_error_rad": m.root_orientation_error_rad,
            "root_orientation_error_deg": _rad_to_deg(m.root_orientation_error_rad),
            "root_angular_velocity_error_rads": m.root_angular_velocity_error_rads,
            "root_angular_velocity_error_degps": _rad_to_deg(m.root_angular_velocity_error_rads),
            "root_quat_norm_error_mean": m.root_quat_norm_error_mean,
            "root_quat_norm_error_max": m.root_quat_norm_error_max,
        }
        for m in all_metrics
    ]

    per_motion_rows = aggregate_per_motion(all_metrics)
    summary = build_summary(
        all_metrics,
        alignment_rows,
        warmup_frames=args.warmup_frames,
        min_frames_per_segment=args.min_frames_per_segment,
        conf_thresholds=conf_thresholds,
    )

    write_csv(out_dir / "metrics_per_segment.csv", per_segment_rows)
    write_csv(out_dir / "metrics_per_motion.csv", per_motion_rows)

    if args.summary_csv:
        write_csv(out_dir / "metrics_summary.csv", [build_summary_csv_row(summary)])

    with (out_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "alignment_report.json").open("w") as f:
        json.dump(alignment_rows, f, indent=2)

    if args.emit_markdown_report:
        write_human_readable_report(
            out_path=out_dir / "metrics_human_readable.md",
            summary=summary,
            per_motion_rows=per_motion_rows,
            alignment_rows=alignment_rows,
            top_k=args.top_k,
        )

    print(f"[DONE] evaluated segments: {len(all_metrics)}")
    print(f"[DONE] outputs: {out_dir}")


if __name__ == "__main__":
    main()
