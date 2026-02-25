#!/usr/bin/env python3
"""
Convert EVVC GMR-style PKL motions to SONIC reference-motion CSV folders.

Input (per pkl):
  {
    "fps": scalar or shape-(1,),
    "root_pos": (T, 3),
    "root_rot": (T, 4),
    "dof_pos": (T, 29),
    "local_body_pos": (T, N, 3),    # optional for conversion
    "link_body_list": list[str],    # optional for conversion
  }

Output (per motion folder):
  - joint_pos.csv   (T', 29)   IsaacLab order
  - joint_vel.csv   (T', 29)   computed at target fps
  - body_pos.csv    (T', 3)    root-only
  - body_quat.csv   (T', 4)    root-only, wxyz
  - metadata.txt
  - info.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
except Exception as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "This script requires scipy (Rotation/Slerp). Install scipy before running."
    ) from exc


# SONIC mapping from gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


@dataclass
class MotionResult:
    source: str
    motion_name: str
    status: str
    message: str
    input_fps: float | None = None
    output_fps: float | None = None
    input_frames: int | None = None
    output_frames: int | None = None
    input_joint_order: str | None = None
    output_joint_order: str | None = "isaaclab"
    quat_order_mode: str | None = None
    quat_order_detected: str | None = None
    quat_confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EVVC GMR PKL motion files to SONIC reference CSV format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("gear_sonic_deploy/reference/ECCV/evvc_pose_pkl"),
        help="Directory containing input *.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gear_sonic_deploy/reference/ECCV/evvc_pose_csv"),
        help="Directory to write SONIC reference motion folders.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.pkl",
        help="Glob pattern for input motion files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output motion folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and plan conversion without writing files.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=50.0,
        help="Output target FPS (SONIC expected: 50).",
    )
    parser.add_argument(
        "--input-joint-order",
        choices=["mujoco", "isaaclab"],
        default="mujoco",
        help="Input dof_pos order. GMR unitree_g1 defaults to mujoco order.",
    )
    parser.add_argument(
        "--quat-order",
        choices=["wxyz", "xyzw", "auto"],
        default="wxyz",
        help="Input root_rot quaternion order.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on ambiguous quaternion auto-detection or invalid values.",
    )
    parser.add_argument(
        "--export-info",
        action="store_true",
        default=True,
        help="Export info.txt and conversion_report.json.",
    )
    return parser.parse_args()


def _to_float_scalar(x: Any, name: str) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return float(arr[0])


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf values.")


def load_motion_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a dict.")
    return data


def validate_motion_dict(data: dict[str, Any], path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    required_keys = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    fps = _to_float_scalar(data["fps"], "fps")
    if not (fps > 0.0):
        raise ValueError(f"Invalid fps={fps} in {path}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T,3), got {root_pos.shape} in {path}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T,4), got {root_rot.shape} in {path}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"dof_pos must have shape (T,29), got {dof_pos.shape} in {path}")

    t = root_pos.shape[0]
    if root_rot.shape[0] != t or dof_pos.shape[0] != t:
        raise ValueError(
            f"Frame count mismatch in {path}: root_pos={root_pos.shape[0]}, "
            f"root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )
    if t < 2:
        raise ValueError(f"Need at least 2 frames, got {t} in {path}")

    _assert_finite("root_pos", root_pos)
    _assert_finite("root_rot", root_rot)
    _assert_finite("dof_pos", dof_pos)

    return fps, root_pos, root_rot, dof_pos


def reorder_joints_to_isaaclab(dof_pos: np.ndarray, input_joint_order: str) -> np.ndarray:
    if input_joint_order == "isaaclab":
        return dof_pos.copy()
    # input is mujoco order -> output isaaclab order
    return dof_pos[:, MUJOCO_TO_ISAACLAB]


def normalize_quat_wxyz(quat_wxyz: np.ndarray) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    min_norm = float(np.min(norms))
    if min_norm < 1e-10:
        raise ValueError("Quaternion norm too small; cannot normalize safely.")
    normalized = quat_wxyz / norms
    max_dev = float(np.max(np.abs(norms - 1.0)))
    return normalized, max_dev


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)


def quat_continuity_score_wxyz(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    rot = R.from_quat(quat_xyzw)
    rel = rot[:-1].inv() * rot[1:]
    angles = rel.magnitude()
    if angles.size == 0:
        return 0.0
    return float(np.median(angles))


def detect_quat_order_auto(root_rot_raw: np.ndarray, strict: bool) -> tuple[str, float]:
    cand_wxyz, _ = normalize_quat_wxyz(root_rot_raw.copy())
    cand_xyzw, _ = normalize_quat_wxyz(xyzw_to_wxyz(root_rot_raw.copy()))

    s_wxyz = quat_continuity_score_wxyz(cand_wxyz)
    s_xyzw = quat_continuity_score_wxyz(cand_xyzw)

    # Lower continuity score is better.
    best_order = "wxyz" if s_wxyz <= s_xyzw else "xyzw"
    best = min(s_wxyz, s_xyzw)
    other = max(s_wxyz, s_xyzw)
    confidence = float((other - best) / (other + 1e-12))

    # Ambiguous case: both look similarly smooth.
    if strict and confidence < 0.05:
        raise ValueError(
            f"Quaternion auto-detection ambiguous (score_wxyz={s_wxyz:.6f}, "
            f"score_xyzw={s_xyzw:.6f}, confidence={confidence:.6f}). "
            "Use --quat-order wxyz or --quat-order xyzw explicitly."
        )
    return best_order, confidence


def resolve_quat_wxyz(root_rot_raw: np.ndarray, quat_order: str, strict: bool) -> tuple[np.ndarray, str, float]:
    if quat_order == "wxyz":
        quat_wxyz = root_rot_raw.copy()
        detected = "wxyz"
        confidence = 1.0
    elif quat_order == "xyzw":
        quat_wxyz = xyzw_to_wxyz(root_rot_raw.copy())
        detected = "xyzw"
        confidence = 1.0
    else:
        detected, confidence = detect_quat_order_auto(root_rot_raw, strict=strict)
        if detected == "wxyz":
            quat_wxyz = root_rot_raw.copy()
        else:
            quat_wxyz = xyzw_to_wxyz(root_rot_raw.copy())

    quat_wxyz, _ = normalize_quat_wxyz(quat_wxyz)
    return quat_wxyz, detected, confidence


def resample_linear(times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    out = np.empty((times_dst.shape[0], values.shape[1]), dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(times_dst, times_src, values[:, i])
    return out


def resample_quat_slerp_wxyz(times_src: np.ndarray, quat_wxyz_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    quat_xyzw_src = wxyz_to_xyzw(quat_wxyz_src)
    rotations = R.from_quat(quat_xyzw_src)
    slerp = Slerp(times_src, rotations)
    quat_xyzw_dst = slerp(times_dst).as_quat()
    quat_wxyz_dst = xyzw_to_wxyz(quat_xyzw_dst)
    quat_wxyz_dst, _ = normalize_quat_wxyz(quat_wxyz_dst)
    return quat_wxyz_dst


def build_resample_times(num_frames: int, input_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    times_src = np.arange(num_frames, dtype=np.float64) / input_fps
    duration = times_src[-1]
    out_frames = int(math.floor(duration * target_fps + 1e-12)) + 1
    times_dst = np.arange(out_frames, dtype=np.float64) / target_fps
    times_dst = np.clip(times_dst, times_src[0], times_src[-1])
    return times_src, times_dst


def compute_joint_velocity(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(joint_pos)
    n = joint_pos.shape[0]
    if n < 2:
        return vel

    vel[0] = (joint_pos[1] - joint_pos[0]) * fps
    vel[-1] = (joint_pos[-1] - joint_pos[-2]) * fps
    if n > 2:
        vel[1:-1] = (joint_pos[2:] - joint_pos[:-2]) * (0.5 * fps)
    return vel


def write_csv(path: Path, headers: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in data:
            writer.writerow([f"{float(v):.8f}" for v in row])


def write_metadata(path: Path, motion_name: str, total_timesteps: int) -> None:
    text = (
        f"Metadata for: {motion_name}\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        f"Total timesteps: {total_timesteps}\n"
    )
    path.write_text(text, encoding="utf-8")


def write_info(path: Path, info: dict[str, Any]) -> None:
    lines = ["EVVC PKL -> SONIC reference conversion info", "========================================", ""]
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_one_file(
    pkl_path: Path,
    out_root: Path,
    overwrite: bool,
    dry_run: bool,
    target_fps: float,
    input_joint_order: str,
    quat_order: str,
    strict: bool,
    export_info: bool,
) -> MotionResult:
    motion_name = pkl_path.stem
    result = MotionResult(
        source=str(pkl_path),
        motion_name=motion_name,
        status="error",
        message="unknown",
        input_joint_order=input_joint_order,
        quat_order_mode=quat_order,
    )

    try:
        data = load_motion_pkl(pkl_path)
        in_fps, root_pos, root_rot_raw, dof_pos = validate_motion_dict(data, pkl_path)

        result.input_fps = in_fps
        result.input_frames = int(root_pos.shape[0])

        dof_pos_isaac = reorder_joints_to_isaaclab(dof_pos, input_joint_order=input_joint_order)
        quat_wxyz_src, detected_order, confidence = resolve_quat_wxyz(
            root_rot_raw=root_rot_raw, quat_order=quat_order, strict=strict
        )
        result.quat_order_detected = detected_order
        result.quat_confidence = confidence

        times_src, times_dst = build_resample_times(root_pos.shape[0], in_fps, target_fps)

        root_pos_out = resample_linear(times_src, root_pos, times_dst)
        dof_pos_out = resample_linear(times_src, dof_pos_isaac, times_dst)
        quat_out = resample_quat_slerp_wxyz(times_src, quat_wxyz_src, times_dst)
        joint_vel_out = compute_joint_velocity(dof_pos_out, fps=target_fps)

        out_motion_dir = out_root / motion_name
        if out_motion_dir.exists():
            if overwrite:
                if not dry_run:
                    shutil.rmtree(out_motion_dir)
            else:
                result.status = "skipped"
                result.message = f"Output exists: {out_motion_dir}"
                result.output_fps = target_fps
                result.output_frames = int(times_dst.shape[0])
                return result

        if not dry_run:
            out_motion_dir.mkdir(parents=True, exist_ok=True)

            write_csv(
                out_motion_dir / "joint_pos.csv",
                [f"joint_{i}" for i in range(29)],
                dof_pos_out,
            )
            write_csv(
                out_motion_dir / "joint_vel.csv",
                [f"joint_vel_{i}" for i in range(29)],
                joint_vel_out,
            )
            write_csv(
                out_motion_dir / "body_pos.csv",
                ["body_0_x", "body_0_y", "body_0_z"],
                root_pos_out,
            )
            write_csv(
                out_motion_dir / "body_quat.csv",
                ["body_0_w", "body_0_x", "body_0_y", "body_0_z"],
                quat_out,
            )
            write_metadata(
                out_motion_dir / "metadata.txt",
                motion_name=motion_name,
                total_timesteps=int(times_dst.shape[0]),
            )
            if export_info:
                write_info(
                    out_motion_dir / "info.txt",
                    {
                        "source_file": str(pkl_path),
                        "input_fps": in_fps,
                        "target_fps": target_fps,
                        "input_frames": int(root_pos.shape[0]),
                        "output_frames": int(times_dst.shape[0]),
                        "input_joint_order": input_joint_order,
                        "output_joint_order": "isaaclab",
                        "quat_order_mode": quat_order,
                        "quat_order_detected": detected_order,
                        "quat_confidence": confidence,
                        "local_body_pos_shape": (
                            tuple(np.asarray(data["local_body_pos"]).shape)
                            if "local_body_pos" in data
                            else None
                        ),
                        "link_body_list_len": (
                            len(data["link_body_list"]) if "link_body_list" in data else None
                        ),
                    },
                )

        result.status = "success"
        result.message = "converted"
        result.output_fps = target_fps
        result.output_frames = int(times_dst.shape[0])
        return result
    except Exception as exc:
        result.status = "error"
        result.message = str(exc)
        return result


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.target_fps <= 0:
        raise ValueError(f"target-fps must be positive, got {args.target_fps}")

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_dir}/{args.glob}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Files matched: {len(files)}")
    print(f"[INFO] target_fps={args.target_fps}, input_joint_order={args.input_joint_order}, quat_order={args.quat_order}")
    print(f"[INFO] strict={args.strict}, dry_run={args.dry_run}, overwrite={args.overwrite}")

    results: list[MotionResult] = []
    for idx, pkl_path in enumerate(files, start=1):
        r = convert_one_file(
            pkl_path=pkl_path,
            out_root=output_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            target_fps=float(args.target_fps),
            input_joint_order=args.input_joint_order,
            quat_order=args.quat_order,
            strict=bool(args.strict),
            export_info=bool(args.export_info),
        )
        results.append(r)
        print(f"[{idx}/{len(files)}] {r.status.upper()}: {pkl_path.name} :: {r.message}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "glob": args.glob,
        "target_fps": args.target_fps,
        "input_joint_order": args.input_joint_order,
        "quat_order": args.quat_order,
        "strict": args.strict,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "total": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "error": sum(1 for r in results if r.status == "error"),
        "results": [asdict(r) for r in results],
    }

    if args.export_info and (not args.dry_run):
        report_file = output_dir / "conversion_report.json"
        report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote report: {report_file}")

    print(
        "[INFO] Done. "
        f"success={summary['success']} skipped={summary['skipped']} "
        f"error={summary['error']} total={summary['total']}"
    )

    if summary["error"] > 0 and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
