#!/usr/bin/env python3
"""Paper-aligned SONIC tracking evaluation (separate from eval_sonic_tracking.py).

This evaluator targets the metric style described in the SONIC paper:
  - Success rate (Succ)
  - Root-relative 3D MPJPE (mm)
  - Velocity distance Evel (mm/frame)
  - Acceleration distance Eacc (mm/frame^2)

Important:
  1) It does not modify deploy/runtime code.
  2) It requires MuJoCo Python (`pip install mujoco`) for FK.
  3) Full paper success criterion additionally needs base position logs (`base_pos.csv`).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - handled by runtime check
    mujoco = None


MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAACLAB)

REQUIRED_LOG_FILES = {"q.csv", "base_quat.csv", "motion_name.csv", "motion_playing.csv"}


@dataclass
class Segment:
    run_name: str
    motion_name: str
    start_idx: int
    end_idx: int


@dataclass
class SegmentResult:
    run_name: str
    motion_name: str
    segment_id: int
    status: str
    reason: str
    success_flag: Optional[bool]
    frames_total: int
    frames_used: int
    mpjpe_mm: float
    evel_mm_per_frame: float
    eacc_mm_per_frame2: float
    max_root_ori_err_rad: float
    max_root_z_dev_m: float


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(q, axis=-1, keepdims=True)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return q / denom


def _quat_geodesic_wxyz(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    a = _quat_normalize_wxyz(q_a)
    b = _quat_normalize_wxyz(q_b)
    dots = np.sum(a * b, axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_rotate_vec_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q: (..., 4), v: (..., 3)
    q = _quat_normalize_wxyz(q)
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def world_to_root_frame(points_w: np.ndarray, root_pos_w: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    rel_w = points_w - root_pos_w[:, None, :]
    q_inv = _quat_conjugate_wxyz(_quat_normalize_wxyz(root_quat_wxyz))
    q_inv_expanded = np.repeat(q_inv[:, None, :], rel_w.shape[1], axis=1)
    return _quat_rotate_vec_wxyz(q_inv_expanded, rel_w)


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


def _find_playing_col(header: Sequence[str]) -> int:
    for i, name in enumerate(header):
        lower = name.lower()
        if lower == "playing" or lower.startswith("playing_"):
            return i
    return len(header) - 1


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


def parse_body_part_indexes(metadata_txt: Path) -> Optional[List[int]]:
    if not metadata_txt.is_file():
        return None
    text = metadata_txt.read_text()
    marker = "Body part indexes:"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1].strip().splitlines()
    if not tail:
        return None
    line = tail[0].strip()
    if not line:
        return None
    # Support formats like "[0, 1, 2]" or "[ 0 1 2 ]".
    if "," in line:
        try:
            vals = ast.literal_eval(line)
            return [int(x) for x in vals]
        except Exception:
            return None
    line = line.strip("[]").strip()
    if not line:
        return None
    try:
        return [int(x) for x in line.split()]
    except Exception:
        return None


def load_reference_motion(motion_dir: Path) -> Dict[str, np.ndarray]:
    _, joint_pos_isaac = _load_numeric_csv(motion_dir / "joint_pos.csv")
    _, body_quat = _load_numeric_csv(motion_dir / "body_quat.csv")
    _, body_pos = _load_numeric_csv(motion_dir / "body_pos.csv")
    if joint_pos_isaac.shape[1] < 29:
        raise ValueError(f"{motion_dir}: joint_pos.csv columns < 29")
    if body_quat.shape[1] < 4:
        raise ValueError(f"{motion_dir}: body_quat.csv columns < 4")
    if body_pos.shape[1] < 3:
        raise ValueError(f"{motion_dir}: body_pos.csv columns < 3")

    joint_pos_isaac = joint_pos_isaac[:, -29:]
    joint_pos_mj = joint_pos_isaac[:, ISAACLAB_TO_MUJOCO]
    root_quat_wxyz = body_quat[:, :4]
    root_pos_w = body_pos[:, :3]

    body_part_indexes = parse_body_part_indexes(motion_dir / "metadata.txt")
    return {
        "joint_pos_mj": joint_pos_mj,
        "root_quat_wxyz": root_quat_wxyz,
        "root_pos_w": root_pos_w,
        "body_part_indexes": np.asarray(body_part_indexes if body_part_indexes else [], dtype=np.int64),
    }


def load_log_run(run_dir: Path) -> Dict[str, np.ndarray | List[str] | bool]:
    _, q = _load_numeric_csv(run_dir / "q.csv")
    quat_header, base_quat = _load_numeric_csv(run_dir / "base_quat.csv")
    playing_header, motion_playing = _load_numeric_csv(run_dir / "motion_playing.csv")
    motion_names = _load_motion_name_csv(run_dir / "motion_name.csv")

    if q.shape[1] < 29:
        raise ValueError(f"{run_dir}: q.csv columns < 29")
    if base_quat.shape[1] < 4:
        raise ValueError(f"{run_dir}: base_quat.csv columns < 4")
    q_mj = q[:, -29:]
    root_quat_wxyz = base_quat[:, -4:]

    playing_col = _find_playing_col(playing_header)
    playing_val = motion_playing[:, playing_col] > 0.5 if motion_playing.size else np.zeros((0,), dtype=bool)

    base_pos_w: Optional[np.ndarray] = None
    has_base_pos = False
    base_pos_csv = run_dir / "base_pos.csv"
    if base_pos_csv.is_file():
        _, base_pos = _load_numeric_csv(base_pos_csv)
        if base_pos.shape[1] >= 3:
            base_pos_w = base_pos[:, -3:]
            has_base_pos = True

    min_len = min(q_mj.shape[0], root_quat_wxyz.shape[0], playing_val.shape[0], len(motion_names))
    if has_base_pos and base_pos_w is not None:
        min_len = min(min_len, base_pos_w.shape[0])

    out: Dict[str, np.ndarray | List[str] | bool] = {
        "q_mj": q_mj[:min_len],
        "root_quat_wxyz": root_quat_wxyz[:min_len],
        "motion_playing": playing_val[:min_len],
        "motion_name": motion_names[:min_len],
        "has_base_pos": has_base_pos,
    }
    if has_base_pos and base_pos_w is not None:
        out["root_pos_w"] = base_pos_w[:min_len]
    return out


def segment_by_motion_name_and_play(log_data: Dict[str, np.ndarray | List[str] | bool], run_name: str) -> List[Segment]:
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


class MujocoFK:
    def __init__(self, model_xml: Path, dof_count: int = 29):
        if mujoco is None:
            raise RuntimeError("mujoco Python package is required. Please `pip install mujoco`.")
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)

        free_joint_ids = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free_joint_ids:
            raise RuntimeError("No free root joint found in MuJoCo model.")
        self.root_joint_id = free_joint_ids[0]
        self.root_qpos_adr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_body_id = int(self.model.jnt_bodyid[self.root_joint_id])

        non_free_joint_ids = [j for j in range(self.model.njnt) if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        if len(non_free_joint_ids) < dof_count:
            raise RuntimeError(
                f"MuJoCo model has only {len(non_free_joint_ids)} non-free joints, but need {dof_count}."
            )
        self.act_joint_ids = non_free_joint_ids[:dof_count]
        self.act_qpos_adr = np.array([int(self.model.jnt_qposadr[j]) for j in self.act_joint_ids], dtype=np.int64)
        self.act_body_ids = np.array([int(self.model.jnt_bodyid[j]) for j in self.act_joint_ids], dtype=np.int64)

    def fk_body_positions(
        self,
        q_mj: np.ndarray,
        root_quat_wxyz: np.ndarray,
        body_ids: np.ndarray,
        root_pos_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        t = q_mj.shape[0]
        out = np.zeros((t, len(body_ids), 3), dtype=np.float64)
        for i in range(t):
            self.data.qpos[:] = 0.0
            if root_pos_w is not None:
                self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = root_pos_w[i]
            self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat_wxyz[i]
            self.data.qpos[self.act_qpos_adr] = q_mj[i]
            mujoco.mj_forward(self.model, self.data)
            out[i] = self.data.xpos[body_ids]
        return out


def compute_segment_metrics(
    fk: MujocoFK,
    seg: Segment,
    log_data: Dict[str, np.ndarray | List[str] | bool],
    ref_data: Dict[str, np.ndarray],
    warmup_frames: int,
    success_criterion: str,
    require_base_pos_for_success: bool,
) -> SegmentResult:
    q_log = np.asarray(log_data["q_mj"], dtype=np.float64)[seg.start_idx : seg.end_idx]
    quat_log = np.asarray(log_data["root_quat_wxyz"], dtype=np.float64)[seg.start_idx : seg.end_idx]
    root_pos_log = None
    if bool(log_data.get("has_base_pos", False)):
        root_pos_log = np.asarray(log_data["root_pos_w"], dtype=np.float64)[seg.start_idx : seg.end_idx]

    q_ref = np.asarray(ref_data["joint_pos_mj"], dtype=np.float64)
    quat_ref = np.asarray(ref_data["root_quat_wxyz"], dtype=np.float64)
    root_pos_ref = np.asarray(ref_data["root_pos_w"], dtype=np.float64)

    t = min(q_log.shape[0], q_ref.shape[0], quat_log.shape[0], quat_ref.shape[0], root_pos_ref.shape[0])
    if t <= 0:
        return SegmentResult(
            run_name=seg.run_name,
            motion_name=seg.motion_name,
            segment_id=-1,
            status="failed",
            reason="empty_alignment",
            success_flag=None,
            frames_total=0,
            frames_used=0,
            mpjpe_mm=float("nan"),
            evel_mm_per_frame=float("nan"),
            eacc_mm_per_frame2=float("nan"),
            max_root_ori_err_rad=float("nan"),
            max_root_z_dev_m=float("nan"),
        )

    q_log = q_log[:t]
    quat_log = quat_log[:t]
    q_ref = q_ref[:t]
    quat_ref = quat_ref[:t]
    root_pos_ref = root_pos_ref[:t]
    if root_pos_log is not None:
        root_pos_log = root_pos_log[:t]

    body_ids = fk.act_body_ids
    pred_root_pos = root_pos_log if root_pos_log is not None else np.zeros((t, 3), dtype=np.float64)
    pred_pos_w = fk.fk_body_positions(q_log, quat_log, body_ids=body_ids, root_pos_w=pred_root_pos)
    ref_pos_w = fk.fk_body_positions(q_ref, quat_ref, body_ids=body_ids, root_pos_w=root_pos_ref)

    pred_rr = world_to_root_frame(pred_pos_w, pred_root_pos, quat_log)
    ref_rr = world_to_root_frame(ref_pos_w, root_pos_ref, quat_ref)

    start_eval = min(max(warmup_frames, 0), t)
    pred_rr_eval = pred_rr[start_eval:]
    ref_rr_eval = ref_rr[start_eval:]
    frames_used = pred_rr_eval.shape[0]
    if frames_used <= 0:
        return SegmentResult(
            run_name=seg.run_name,
            motion_name=seg.motion_name,
            segment_id=-1,
            status="failed",
            reason="empty_after_warmup",
            success_flag=None,
            frames_total=t,
            frames_used=0,
            mpjpe_mm=float("nan"),
            evel_mm_per_frame=float("nan"),
            eacc_mm_per_frame2=float("nan"),
            max_root_ori_err_rad=float("nan"),
            max_root_z_dev_m=float("nan"),
        )

    pos_err_mm = np.linalg.norm((pred_rr_eval - ref_rr_eval) * 1000.0, axis=-1)
    mpjpe_mm = _safe_mean(pos_err_mm)

    pred_vel = np.diff(pred_rr_eval, axis=0) * 1000.0
    ref_vel = np.diff(ref_rr_eval, axis=0) * 1000.0
    evel_mm_per_frame = _safe_mean(np.linalg.norm(pred_vel - ref_vel, axis=-1))

    pred_acc = np.diff(pred_vel, axis=0)
    ref_acc = np.diff(ref_vel, axis=0)
    eacc_mm_per_frame2 = _safe_mean(np.linalg.norm(pred_acc - ref_acc, axis=-1))

    ori_err = _quat_geodesic_wxyz(quat_log, quat_ref)
    max_root_ori_err = float(np.max(ori_err)) if ori_err.size else float("nan")

    max_root_z_dev = float("nan")
    if root_pos_log is not None:
        max_root_z_dev = float(np.max(np.abs(root_pos_log[:, 2] - root_pos_ref[:, 2])))

    success_flag: Optional[bool]
    status = "ok"
    reason = ""
    if require_base_pos_for_success and root_pos_log is None:
        success_flag = None
        status = "unknown"
        reason = "missing_base_pos_for_success"
    else:
        fail_ori = bool(np.any(ori_err > 1.0)) if success_criterion == "paper" else False
        fail_height = False
        if root_pos_log is not None:
            fail_height = bool(np.any(np.abs(root_pos_log[:, 2] - root_pos_ref[:, 2]) > 0.25))
        elif success_criterion == "fall_only":
            success_flag = None
            status = "unknown"
            reason = "missing_base_pos_for_fall_only"
            return SegmentResult(
                run_name=seg.run_name,
                motion_name=seg.motion_name,
                segment_id=-1,
                status=status,
                reason=reason,
                success_flag=success_flag,
                frames_total=t,
                frames_used=frames_used,
                mpjpe_mm=mpjpe_mm,
                evel_mm_per_frame=evel_mm_per_frame,
                eacc_mm_per_frame2=eacc_mm_per_frame2,
                max_root_ori_err_rad=max_root_ori_err,
                max_root_z_dev_m=max_root_z_dev,
            )
        if success_criterion == "fall_only":
            success_flag = not fail_height
        else:
            success_flag = not (fail_height or fail_ori)
        if not success_flag:
            status = "failed"
            if fail_height and fail_ori:
                reason = "height_dev>0.25_and_root_ori>1.0"
            elif fail_height:
                reason = "height_dev>0.25"
            else:
                reason = "root_ori>1.0"

    return SegmentResult(
        run_name=seg.run_name,
        motion_name=seg.motion_name,
        segment_id=-1,
        status=status,
        reason=reason,
        success_flag=success_flag,
        frames_total=t,
        frames_used=frames_used,
        mpjpe_mm=mpjpe_mm,
        evel_mm_per_frame=evel_mm_per_frame,
        eacc_mm_per_frame2=eacc_mm_per_frame2,
        max_root_ori_err_rad=max_root_ori_err,
        max_root_z_dev_m=max_root_z_dev,
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_per_motion(results: List[SegmentResult], successful_only: bool) -> List[Dict[str, object]]:
    bucket: Dict[str, List[SegmentResult]] = {}
    for r in results:
        if successful_only and r.success_flag is not True:
            continue
        bucket.setdefault(r.motion_name, []).append(r)

    out: List[Dict[str, object]] = []
    for motion_name, items in sorted(bucket.items()):
        frames = np.array([x.frames_used for x in items], dtype=np.float64)
        if np.sum(frames) <= 0:
            continue
        def _fw(attr: str) -> float:
            vals = np.array([getattr(x, attr) for x in items], dtype=np.float64)
            mask = np.isfinite(vals) & (frames > 0)
            if not np.any(mask):
                return float("nan")
            return float(np.sum(vals[mask] * frames[mask]) / np.sum(frames[mask]))

        out.append(
            {
                "motion_name": motion_name,
                "segments": len(items),
                "frames_used_total": int(np.sum(frames)),
                "mpjpe_mm_frame_weighted": _fw("mpjpe_mm"),
                "evel_mm_per_frame_weighted": _fw("evel_mm_per_frame"),
                "eacc_mm_per_frame2_weighted": _fw("eacc_mm_per_frame2"),
            }
        )
    return out


def summarize(results: List[SegmentResult], successful_only_for_errors: bool) -> Dict[str, object]:
    success = [r for r in results if r.success_flag is True]
    failed = [r for r in results if r.success_flag is False]
    unknown = [r for r in results if r.success_flag is None]
    measured = success if successful_only_for_errors else [r for r in results if r.frames_used > 0]

    def _frame_weighted(attr: str) -> float:
        vals = np.array([getattr(r, attr) for r in measured], dtype=np.float64)
        frames = np.array([r.frames_used for r in measured], dtype=np.float64)
        mask = np.isfinite(vals) & (frames > 0)
        if not np.any(mask):
            return float("nan")
        return float(np.sum(vals[mask] * frames[mask]) / np.sum(frames[mask]))

    out = {
        "segments_total": len(results),
        "segments_success": len(success),
        "segments_failed": len(failed),
        "segments_unknown": len(unknown),
        "success_rate": float(len(success) / (len(success) + len(failed))) if (len(success) + len(failed)) > 0 else float("nan"),
        "errors_on": "successful_segments_only" if successful_only_for_errors else "all_segments",
        "mpjpe_mm_frame_weighted": _frame_weighted("mpjpe_mm"),
        "evel_mm_per_frame_weighted": _frame_weighted("evel_mm_per_frame"),
        "eacc_mm_per_frame2_weighted": _frame_weighted("eacc_mm_per_frame2"),
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned SONIC tracking evaluator (separate script).")
    parser.add_argument(
        "--reference-root",
        required=True,
        help="Root directory containing converted reference motions (each motion in a subfolder).",
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="One run directory or a parent directory containing many runs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for paper-aligned evaluation results.",
    )
    parser.add_argument(
        "--model-xml",
        default="gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
        help="MuJoCo XML used for FK (must match runtime kinematic convention).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Frames dropped from each segment before error metrics.",
    )
    parser.add_argument(
        "--success-criterion",
        choices=["paper", "fall_only"],
        default="paper",
        help="paper: height_dev>0.25m OR root_ori_err>1rad; fall_only: height_dev>0.25m only.",
    )
    parser.add_argument(
        "--allow-missing-base-pos",
        action="store_true",
        help="If set, keep running even when base_pos.csv is missing (Succ may become unknown).",
    )
    parser.add_argument(
        "--errors-on-all-segments",
        action="store_true",
        help="By default paper style reports errors on successful segments only. Set this to use all aligned segments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root).resolve()
    logs_root = Path(args.logs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if mujoco is None:
        raise RuntimeError("Missing dependency: mujoco Python package. Install with `pip install mujoco`.")

    model_xml = Path(args.model_xml)
    if not model_xml.is_absolute():
        model_xml = (Path(__file__).resolve().parents[2] / model_xml).resolve()
    if not model_xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_xml}")

    fk = MujocoFK(model_xml=model_xml, dof_count=29)
    runs = discover_log_runs(logs_root)
    if not runs:
        raise RuntimeError(f"No valid log runs found under: {logs_root}")

    all_results: List[SegmentResult] = []
    per_segment_rows: List[Dict[str, object]] = []

    for run_dir in runs:
        run_name = run_dir.name
        log_data = load_log_run(run_dir)
        segments = segment_by_motion_name_and_play(log_data, run_name=run_name)
        if not segments:
            continue
        for seg_idx, seg in enumerate(segments):
            motion_dir = reference_root / seg.motion_name
            if not motion_dir.is_dir():
                all_results.append(
                    SegmentResult(
                        run_name=run_name,
                        motion_name=seg.motion_name,
                        segment_id=seg_idx,
                        status="failed",
                        reason="missing_reference_motion",
                        success_flag=None,
                        frames_total=0,
                        frames_used=0,
                        mpjpe_mm=float("nan"),
                        evel_mm_per_frame=float("nan"),
                        eacc_mm_per_frame2=float("nan"),
                        max_root_ori_err_rad=float("nan"),
                        max_root_z_dev_m=float("nan"),
                    )
                )
                continue

            ref_data = load_reference_motion(motion_dir)
            result = compute_segment_metrics(
                fk=fk,
                seg=seg,
                log_data=log_data,
                ref_data=ref_data,
                warmup_frames=args.warmup_frames,
                success_criterion=args.success_criterion,
                require_base_pos_for_success=not args.allow_missing_base_pos,
            )
            result.segment_id = seg_idx
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No evaluable segments found.")

    for r in all_results:
        per_segment_rows.append(
            {
                "run_name": r.run_name,
                "motion_name": r.motion_name,
                "segment_id": r.segment_id,
                "status": r.status,
                "reason": r.reason,
                "success_flag": "" if r.success_flag is None else bool(r.success_flag),
                "frames_total": r.frames_total,
                "frames_used": r.frames_used,
                "mpjpe_mm": r.mpjpe_mm,
                "evel_mm_per_frame": r.evel_mm_per_frame,
                "eacc_mm_per_frame2": r.eacc_mm_per_frame2,
                "max_root_ori_err_rad": r.max_root_ori_err_rad,
                "max_root_z_dev_m": r.max_root_z_dev_m,
            }
        )

    per_motion_rows = aggregate_per_motion(all_results, successful_only=not args.errors_on_all_segments)
    summary = summarize(all_results, successful_only_for_errors=not args.errors_on_all_segments)
    summary.update(
        {
            "reference_root": str(reference_root),
            "logs_root": str(logs_root),
            "model_xml": str(model_xml),
            "warmup_frames": int(args.warmup_frames),
            "success_criterion": args.success_criterion,
            "allow_missing_base_pos": bool(args.allow_missing_base_pos),
        }
    )

    write_csv(out_dir / "paper_metrics_per_segment.csv", per_segment_rows)
    write_csv(out_dir / "paper_metrics_per_motion.csv", per_motion_rows)
    write_csv(out_dir / "paper_metrics_summary.csv", [summary])
    with (out_dir / "paper_metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    lines: List[str] = []
    lines.append("# Paper-Aligned Tracking Evaluation")
    lines.append("")
    lines.append("This report follows paper-style metrics as closely as possible with offline logs.")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- success_criterion: `{args.success_criterion}`")
    lines.append(f"- warmup_frames (for errors): `{args.warmup_frames}`")
    lines.append(f"- errors_on: `{summary['errors_on']}`")
    lines.append(f"- allow_missing_base_pos: `{args.allow_missing_base_pos}`")
    lines.append("")
    lines.append("## Global")
    lines.append("")
    lines.append(f"- Segments total: {summary['segments_total']}")
    lines.append(f"- Success / Failed / Unknown: {summary['segments_success']} / {summary['segments_failed']} / {summary['segments_unknown']}")
    lines.append(f"- Success rate (Succ): {summary['success_rate']:.6f}")
    lines.append(f"- MPJPE (mm): {summary['mpjpe_mm_frame_weighted']:.6f}")
    lines.append(f"- Evel (mm/frame): {summary['evel_mm_per_frame_weighted']:.6f}")
    lines.append(f"- Eacc (mm/frame^2): {summary['eacc_mm_per_frame2_weighted']:.6f}")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `paper_metrics_summary.csv`")
    lines.append("- `paper_metrics_summary.json`")
    lines.append("- `paper_metrics_per_motion.csv`")
    lines.append("- `paper_metrics_per_segment.csv`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Success criterion is strict paper logic only when base position logs are available.")
    lines.append("- If base position is missing and fallback is enabled, success may be `unknown`.")
    lines.append("- Metrics use root-relative 3D joint positions computed via MuJoCo FK.")
    (out_dir / "paper_metrics_report.md").write_text("\n".join(lines))

    print(f"[DONE] evaluated segments: {len(all_results)}")
    print(f"[DONE] outputs: {out_dir}")


if __name__ == "__main__":
    main()
