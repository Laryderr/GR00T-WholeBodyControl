#!/usr/bin/env python3
"""Self-check validator for eval_sonic_tracking_paper.py.

This script validates paper-style evaluation logic with three deterministic tests:
1) Identity test (prediction == reference)
2) Known perturbation test (single-joint constant offset)
3) Threshold trigger test (root orientation error > 1 rad)

It writes machine-readable and human-readable reports and returns non-zero when
validation fails.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_INFRA_ERROR = 2


@dataclass
class CheckResult:
    case_name: str
    passed: bool
    expected: str
    actual: str
    message: str
    status: str
    reason: str
    success_flag: str
    mpjpe_mm: float
    evel_mm_per_frame: float
    eacc_mm_per_frame2: float
    max_root_ori_err_rad: float
    max_root_z_dev_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper-style SONIC evaluator with deterministic self-tests.")
    parser.add_argument(
        "--paper-eval-script",
        default="tools/eval_tracking/eval_sonic_tracking_paper.py",
        help="Path to eval_sonic_tracking_paper.py.",
    )
    parser.add_argument(
        "--model-xml",
        default="gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
        help="MuJoCo XML path for FK checks.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: tools/eval_tracking/outputs/eval_paper_validation_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reserved for future randomized extensions).")
    parser.add_argument("--num-frames", type=int, default=200, help="Frame count for synthetic cases.")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Warmup frames passed to evaluator.")
    parser.add_argument("--tol-mpjpe-mm", type=float, default=1e-4, help="Identity tolerance for MPJPE (mm).")
    parser.add_argument("--tol-evel-mm-per-frame", type=float, default=1e-4, help="Identity tolerance for Evel.")
    parser.add_argument("--tol-eacc-mm-per-frame2", type=float, default=1e-4, help="Identity tolerance for Eacc.")
    parser.add_argument(
        "--threshold-root-ori-rad",
        type=float,
        default=1.0,
        help="Orientation threshold used by threshold-trigger check.",
    )
    parser.add_argument("--strict", action="store_true", default=True, help="Return non-zero if any case fails.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep generated temporary case data.")
    return parser.parse_args()


def _resolve_repo_root() -> Path:
    # tools/eval_tracking/validate_paper_eval.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("eval_paper_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_paper_module"] = module
    spec.loader.exec_module(module)
    return module


def _wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    half = 0.5 * yaw_rad
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _write_csv(path: Path, header: List[str], rows: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row.tolist())


def _create_reference_motion(
    motion_dir: Path,
    q_mj: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_pos_w: np.ndarray,
    mj_to_isaac: np.ndarray,
) -> None:
    motion_dir.mkdir(parents=True, exist_ok=True)
    q_isaac = q_mj[:, mj_to_isaac]

    joint_header = [f"joint_{i}" for i in range(29)]
    quat_header = [f"body_0_{k}" for k in ("w", "x", "y", "z")]
    pos_header = [f"body_0_{k}" for k in ("x", "y", "z")]

    _write_csv(motion_dir / "joint_pos.csv", joint_header, q_isaac)
    _write_csv(motion_dir / "body_quat.csv", quat_header, root_quat_wxyz)
    _write_csv(motion_dir / "body_pos.csv", pos_header, root_pos_w)

    # Metadata is optional in the evaluator, but writing one avoids ambiguity.
    (motion_dir / "metadata.txt").write_text(
        "Metadata for synthetic validation motion\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        f"Total timesteps: {q_mj.shape[0]}\n"
    )


def _create_log_run(
    run_dir: Path,
    motion_name: str,
    q_mj: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_pos_w: np.ndarray,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    t = q_mj.shape[0]
    idx = np.arange(t, dtype=np.float64).reshape(-1, 1)
    t_ms = (idx[:, 0] * 20.0).reshape(-1, 1)
    zeros = np.zeros((t, 3), dtype=np.float64)

    q_rows = np.hstack([idx, t_ms, zeros, q_mj])
    q_header = ["index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp"] + [
        f"q_{i}" for i in range(29)
    ]
    _write_csv(run_dir / "q.csv", q_header, q_rows)

    quat_rows = np.hstack([idx, t_ms, zeros, root_quat_wxyz])
    quat_header = ["index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp"] + [
        "base_qw",
        "base_qx",
        "base_qy",
        "base_qz",
    ]
    _write_csv(run_dir / "base_quat.csv", quat_header, quat_rows)

    pos_rows = np.hstack([idx, t_ms, zeros, root_pos_w])
    pos_header = ["index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp"] + [
        "base_x",
        "base_y",
        "base_z",
    ]
    _write_csv(run_dir / "base_pos.csv", pos_header, pos_rows)

    playing = np.ones((t, 1), dtype=np.float64)
    playing_rows = np.hstack([idx, t_ms, zeros, playing])
    playing_header = ["index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp", "playing_0"]
    _write_csv(run_dir / "motion_playing.csv", playing_header, playing_rows)

    with (run_dir / "motion_name.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "time_ms", "motion_name"])
        writer.writeheader()
        for i in range(t):
            writer.writerow({"index": i, "time_ms": int(i * 20), "motion_name": motion_name})


def _build_case_dirs(base_dir: Path, case_name: str) -> Tuple[Path, Path, str]:
    motion_name = f"validation_{case_name}"
    ref_dir = base_dir / "reference" / motion_name
    run_dir = base_dir / "logs" / f"run_{case_name}"
    return ref_dir, run_dir, motion_name


def _eval_one_case(
    ep: Any,
    fk: Any,
    ref_motion_dir: Path,
    run_dir: Path,
    motion_name: str,
    warmup_frames: int,
) -> Any:
    ref_data = ep.load_reference_motion(ref_motion_dir)
    log_data = ep.load_log_run(run_dir)
    segments = ep.segment_by_motion_name_and_play(log_data, run_name=run_dir.name)
    if len(segments) != 1:
        raise RuntimeError(f"Expected exactly one segment for {motion_name}, got {len(segments)}")
    seg = segments[0]
    return ep.compute_segment_metrics(
        fk=fk,
        seg=seg,
        log_data=log_data,
        ref_data=ref_data,
        warmup_frames=warmup_frames,
        success_criterion="paper",
        require_base_pos_for_success=True,
    )


def _run_checks(args: argparse.Namespace, ep: Any, fk: Any, tmp_dir: Path) -> List[CheckResult]:
    np.random.seed(args.seed)
    t = int(args.num_frames)
    if t <= args.warmup_frames + 2:
        raise RuntimeError("--num-frames must be > warmup_frames + 2")

    mj_to_isaac = np.asarray(ep.MUJOCO_TO_ISAACLAB, dtype=np.int64)
    q_base = np.zeros((t, 29), dtype=np.float64)
    root_pos = np.tile(np.array([[0.0, 0.0, 0.85]], dtype=np.float64), (t, 1))
    quat_ref = np.tile(_wxyz_from_yaw(0.0), (t, 1))

    checks: List[CheckResult] = []

    # Case 1: identity.
    case = "identity"
    ref_dir, run_dir, motion_name = _build_case_dirs(tmp_dir, case)
    _create_reference_motion(ref_dir, q_base, quat_ref, root_pos, mj_to_isaac)
    _create_log_run(run_dir, motion_name, q_base, quat_ref, root_pos)
    r_identity = _eval_one_case(ep, fk, ref_dir, run_dir, motion_name, args.warmup_frames)

    identity_pass = (
        r_identity.success_flag is True
        and r_identity.status == "ok"
        and abs(r_identity.mpjpe_mm) <= args.tol_mpjpe_mm
        and abs(r_identity.evel_mm_per_frame) <= args.tol_evel_mm_per_frame
        and abs(r_identity.eacc_mm_per_frame2) <= args.tol_eacc_mm_per_frame2
    )
    checks.append(
        CheckResult(
            case_name=case,
            passed=bool(identity_pass),
            expected=(
                f"success=True, status=ok, mpjpe<={args.tol_mpjpe_mm}, "
                f"evel<={args.tol_evel_mm_per_frame}, eacc<={args.tol_eacc_mm_per_frame2}"
            ),
            actual=(
                f"success={r_identity.success_flag}, status={r_identity.status}, "
                f"mpjpe={r_identity.mpjpe_mm:.8f}, evel={r_identity.evel_mm_per_frame:.8f}, "
                f"eacc={r_identity.eacc_mm_per_frame2:.8f}"
            ),
            message="Identity case should be numerically near zero.",
            status=r_identity.status,
            reason=r_identity.reason,
            success_flag=str(r_identity.success_flag),
            mpjpe_mm=float(r_identity.mpjpe_mm),
            evel_mm_per_frame=float(r_identity.evel_mm_per_frame),
            eacc_mm_per_frame2=float(r_identity.eacc_mm_per_frame2),
            max_root_ori_err_rad=float(r_identity.max_root_ori_err_rad),
            max_root_z_dev_m=float(r_identity.max_root_z_dev_m),
        )
    )

    # Case 2: single-joint perturbation.
    case = "perturbation"
    q_perturb = q_base.copy()
    q_perturb[:, 0] += 0.2  # deterministic known offset
    ref_dir, run_dir, motion_name = _build_case_dirs(tmp_dir, case)
    _create_reference_motion(ref_dir, q_base, quat_ref, root_pos, mj_to_isaac)
    _create_log_run(run_dir, motion_name, q_perturb, quat_ref, root_pos)
    r_perturb = _eval_one_case(ep, fk, ref_dir, run_dir, motion_name, args.warmup_frames)

    perturb_pass = (
        r_perturb.success_flag is True
        and r_perturb.status == "ok"
        and r_perturb.mpjpe_mm > (r_identity.mpjpe_mm + max(1e-6, args.tol_mpjpe_mm))
        and abs(r_perturb.evel_mm_per_frame) <= args.tol_evel_mm_per_frame
        and abs(r_perturb.eacc_mm_per_frame2) <= args.tol_eacc_mm_per_frame2
    )
    checks.append(
        CheckResult(
            case_name=case,
            passed=bool(perturb_pass),
            expected=(
                "success=True, status=ok, mpjpe(identity)<mpjpe(perturb), "
                f"evel<={args.tol_evel_mm_per_frame}, eacc<={args.tol_eacc_mm_per_frame2}"
            ),
            actual=(
                f"success={r_perturb.success_flag}, status={r_perturb.status}, "
                f"mpjpe={r_perturb.mpjpe_mm:.8f}, identity_mpjpe={r_identity.mpjpe_mm:.8f}, "
                f"evel={r_perturb.evel_mm_per_frame:.8f}, eacc={r_perturb.eacc_mm_per_frame2:.8f}"
            ),
            message="Known joint offset should increase MPJPE without creating velocity/acceleration error for static poses.",
            status=r_perturb.status,
            reason=r_perturb.reason,
            success_flag=str(r_perturb.success_flag),
            mpjpe_mm=float(r_perturb.mpjpe_mm),
            evel_mm_per_frame=float(r_perturb.evel_mm_per_frame),
            eacc_mm_per_frame2=float(r_perturb.eacc_mm_per_frame2),
            max_root_ori_err_rad=float(r_perturb.max_root_ori_err_rad),
            max_root_z_dev_m=float(r_perturb.max_root_z_dev_m),
        )
    )

    # Case 3: threshold trigger (root orientation > 1 rad).
    case = "threshold_trigger"
    quat_bad = np.tile(_wxyz_from_yaw(1.2), (t, 1))
    ref_dir, run_dir, motion_name = _build_case_dirs(tmp_dir, case)
    _create_reference_motion(ref_dir, q_base, quat_ref, root_pos, mj_to_isaac)
    _create_log_run(run_dir, motion_name, q_base, quat_bad, root_pos)
    r_thresh = _eval_one_case(ep, fk, ref_dir, run_dir, motion_name, args.warmup_frames)

    threshold_pass = (
        r_thresh.success_flag is False
        and r_thresh.status == "failed"
        and ("root_ori>1.0" in (r_thresh.reason or ""))
        and (r_thresh.max_root_ori_err_rad > args.threshold_root_ori_rad)
    )
    checks.append(
        CheckResult(
            case_name=case,
            passed=bool(threshold_pass),
            expected=f"success=False due to root_ori>{args.threshold_root_ori_rad}",
            actual=(
                f"success={r_thresh.success_flag}, status={r_thresh.status}, "
                f"reason={r_thresh.reason}, max_root_ori={r_thresh.max_root_ori_err_rad:.8f}"
            ),
            message="Root orientation threshold must trigger failure.",
            status=r_thresh.status,
            reason=r_thresh.reason,
            success_flag=str(r_thresh.success_flag),
            mpjpe_mm=float(r_thresh.mpjpe_mm),
            evel_mm_per_frame=float(r_thresh.evel_mm_per_frame),
            eacc_mm_per_frame2=float(r_thresh.eacc_mm_per_frame2),
            max_root_ori_err_rad=float(r_thresh.max_root_ori_err_rad),
            max_root_z_dev_m=float(r_thresh.max_root_z_dev_m),
        )
    )

    return checks


def _write_outputs(out_dir: Path, checks: List[CheckResult], args: argparse.Namespace, exit_code: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for c in checks:
        rows.append(
            {
                "case_name": c.case_name,
                "passed": c.passed,
                "expected": c.expected,
                "actual": c.actual,
                "message": c.message,
                "status": c.status,
                "reason": c.reason,
                "success_flag": c.success_flag,
                "mpjpe_mm": c.mpjpe_mm,
                "evel_mm_per_frame": c.evel_mm_per_frame,
                "eacc_mm_per_frame2": c.eacc_mm_per_frame2,
                "max_root_ori_err_rad": c.max_root_ori_err_rad,
                "max_root_z_dev_m": c.max_root_z_dev_m,
            }
        )

    with (out_dir / "validation_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for c in checks if c.passed)
    failed = len(checks) - passed
    summary = {
        "overall_pass": failed == 0,
        "cases_total": len(checks),
        "cases_passed": passed,
        "cases_failed": failed,
        "exit_code": exit_code,
        "args": vars(args),
    }
    with (out_dir / "validation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    md: List[str] = []
    md.append("# Paper Evaluator Validation Report")
    md.append("")
    md.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- Overall: {'PASS' if summary['overall_pass'] else 'FAIL'}")
    md.append(f"- Cases: {passed}/{len(checks)} passed")
    md.append(f"- Exit code: {exit_code}")
    md.append("")
    md.append("| Case | Result | Status | Reason | MPJPE (mm) | Evel | Eacc |")
    md.append("|---|---|---|---|---:|---:|---:|")
    for c in checks:
        md.append(
            f"| {c.case_name} | {'PASS' if c.passed else 'FAIL'} | {c.status} | {c.reason or '-'} | "
            f"{c.mpjpe_mm:.6f} | {c.evel_mm_per_frame:.6f} | {c.eacc_mm_per_frame2:.6f} |"
        )
    md.append("")
    for c in checks:
        md.append(f"## {c.case_name}")
        md.append("")
        md.append(f"- Expected: {c.expected}")
        md.append(f"- Actual: {c.actual}")
        md.append(f"- Check: {c.message}")
        md.append("")

    (out_dir / "validation_report.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    repo_root = _resolve_repo_root()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        repo_root / "tools" / "eval_tracking" / "outputs" / f"eval_paper_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_cases"

    try:
        eval_script = Path(args.paper_eval_script)
        if not eval_script.is_absolute():
            eval_script = (repo_root / eval_script).resolve()
        if not eval_script.is_file():
            raise FileNotFoundError(f"paper eval script not found: {eval_script}")

        model_xml = Path(args.model_xml)
        if not model_xml.is_absolute():
            model_xml = (repo_root / model_xml).resolve()
        if not model_xml.is_file():
            raise FileNotFoundError(f"model xml not found: {model_xml}")

        ep = _load_module(eval_script)
        if getattr(ep, "mujoco", None) is None:
            raise RuntimeError("mujoco Python package is required. Install with `pip install mujoco`.")

        fk = ep.MujocoFK(model_xml=model_xml, dof_count=29)
        checks = _run_checks(args, ep, fk, tmp_dir)
        exit_code = EXIT_OK if all(c.passed for c in checks) else EXIT_VALIDATION_FAILED
        _write_outputs(out_dir, checks, args, exit_code)

        print(f"[DONE] validation cases: {len(checks)}")
        print(f"[DONE] outputs: {out_dir}")
        print(f"[DONE] overall: {'PASS' if exit_code == EXIT_OK else 'FAIL'}")

        if (not args.keep_temp) and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return exit_code if args.strict else EXIT_OK

    except Exception as exc:
        err = {
            "overall_pass": False,
            "cases_total": 0,
            "cases_passed": 0,
            "cases_failed": 0,
            "exit_code": EXIT_INFRA_ERROR,
            "error": str(exc),
            "args": vars(args),
        }
        with (out_dir / "validation_summary.json").open("w") as f:
            json.dump(err, f, indent=2)
        (out_dir / "validation_report.md").write_text(
            "# Paper Evaluator Validation Report\n\n"
            f"- Overall: FAIL (infra)\n"
            f"- Error: {exc}\n"
        )
        print(f"[ERROR] {exc}")
        print(f"[DONE] outputs: {out_dir}")
        return EXIT_INFRA_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
