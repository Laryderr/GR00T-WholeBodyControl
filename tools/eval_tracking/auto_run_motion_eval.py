#!/usr/bin/env python3
"""External runner for automatic batch playback + evaluation.

Design goals:
- Do NOT modify gear_sonic_deploy core source.
- Reproducible batch playback order.
- Optional end-to-end execution (sim + deploy + evaluation), default is dry-run plan output.

Execution modes:
- Multi-process (default): one deploy process per motion.
- Single-session: one deploy process, auto key sequence (T/N) across motions.
"""

from __future__ import annotations

import argparse
import csv
import os
import pty
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

POLL_SEC = 0.2
VALID_TERMINATE_EXIT_CODES = (0, -2, 130)


@dataclass
class MotionItem:
    name: str
    path: Path
    frames: int


@dataclass
class StartHandshakeResult:
    success: bool
    status: str
    elapsed_sec: float
    inject_attempts: int
    q_rows: int
    motion_playing_rows: int
    message: str


@dataclass
class CsvTailState:
    path: Path
    header: list[str] | None = None
    offset: int = 0
    rows_read: int = 0
    last_row: list[str] | None = None


def count_frames_in_joint_csv(joint_pos_csv: Path) -> int:
    with joint_pos_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        return sum(1 for _ in reader)


def list_motion_items(reference_root: Path, motion_list: Sequence[str] | None) -> List[MotionItem]:
    candidates = sorted([p for p in reference_root.iterdir() if p.is_dir()])
    if motion_list:
        keep = set(motion_list)
        candidates = [p for p in candidates if p.name in keep]

    items: List[MotionItem] = []
    for d in candidates:
        joint_csv = d / "joint_pos.csv"
        if not joint_csv.is_file():
            continue
        frames = count_frames_in_joint_csv(joint_csv)
        items.append(MotionItem(name=d.name, path=d, frames=frames))
    return items


def read_motion_list(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            names.append(raw)
    return names


def make_single_motion_set(root: Path, item: MotionItem, copy_mode: str) -> Path:
    dataset_root = root / f"single_{item.name}"
    motion_dst = dataset_root / item.name
    dataset_root.mkdir(parents=True, exist_ok=True)

    if motion_dst.exists() or motion_dst.is_symlink():
        if motion_dst.is_dir() and not motion_dst.is_symlink():
            import shutil

            shutil.rmtree(motion_dst)
        else:
            motion_dst.unlink()

    if copy_mode == "symlink":
        motion_dst.symlink_to(item.path, target_is_directory=True)
    else:
        import shutil

        shutil.copytree(item.path, motion_dst)

    return dataset_root


def make_session_motion_set(root: Path, items: Sequence[MotionItem], copy_mode: str) -> Path:
    dataset_root = root / "session_motion_set"
    if dataset_root.exists() or dataset_root.is_symlink():
        if dataset_root.is_dir() and not dataset_root.is_symlink():
            import shutil

            shutil.rmtree(dataset_root)
        else:
            dataset_root.unlink()
    dataset_root.mkdir(parents=True, exist_ok=True)

    for item in items:
        dst = dataset_root / item.name
        if copy_mode == "symlink":
            dst.symlink_to(item.path, target_is_directory=True)
        else:
            import shutil

            shutil.copytree(item.path, dst)

    (dataset_root / "session_order.txt").write_text("\n".join(item.name for item in items) + "\n")
    return dataset_root


def write_single_line_playback(playback_path: Path) -> None:
    playback_path.parent.mkdir(parents=True, exist_ok=True)
    with playback_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0, 0, 1, 1, 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def build_deploy_cmd(
    deploy_bin: Path,
    network_interface: str,
    decoder_model: Path,
    encoder_model: Path,
    planner_model: Path,
    obs_config: Path,
    motion_data_root: Path,
    logs_dir: Path,
    output_type: str,
    input_type: str,
    playback_file: Path | None,
) -> List[str]:
    cmd = [
        str(deploy_bin),
        network_interface,
        str(decoder_model),
        str(motion_data_root),
        "--obs-config",
        str(obs_config),
        "--encoder-file",
        str(encoder_model),
        "--planner-file",
        str(planner_model),
        "--input-type",
        input_type,
        "--output-type",
        output_type,
        "--disable-crc-check",
        "--enable-csv-logs",
        "--logs-dir",
        str(logs_dir),
    ]
    if playback_file is not None:
        cmd.extend(["--playback-input-file", str(playback_file)])
    return cmd


def launch_process(cmd: List[str], cwd: Path | None = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)


def launch_process_with_pty_stdin(cmd: List[str], cwd: Path | None = None) -> tuple[subprocess.Popen, int]:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdin=slave_fd)
    os.close(slave_fd)
    return proc, master_fd


def terminate_process(proc: subprocess.Popen, timeout_sec: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def count_csv_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return 0


def log_row_counts(run_logs_dir: Path) -> tuple[int, int]:
    q_rows = count_csv_data_rows(run_logs_dir / "q.csv")
    motion_playing_rows = count_csv_data_rows(run_logs_dir / "motion_playing.csv")
    return q_rows, motion_playing_rows


def wait_for_control_start(
    proc: subprocess.Popen,
    stdin_master_fd: int,
    run_logs_dir: Path,
    inject_start_key: bool,
    start_timeout_sec: float,
    inject_interval_sec: float,
    min_log_rows: int,
    initial_inject_delay_sec: float,
) -> StartHandshakeResult:
    start_time = time.time()
    deadline = start_time + max(start_timeout_sec, 0.0)
    next_inject_time = start_time + max(initial_inject_delay_sec, 0.0)
    inject_attempts = 0
    min_rows = max(min_log_rows, 1)

    while time.time() < deadline:
        if proc.poll() is not None:
            q_rows, motion_playing_rows = log_row_counts(run_logs_dir)
            return StartHandshakeResult(
                success=False,
                status="process_exit_early",
                elapsed_sec=time.time() - start_time,
                inject_attempts=inject_attempts,
                q_rows=q_rows,
                motion_playing_rows=motion_playing_rows,
                message="deploy process exited before control start",
            )

        q_rows, motion_playing_rows = log_row_counts(run_logs_dir)
        if q_rows >= min_rows or motion_playing_rows >= min_rows:
            return StartHandshakeResult(
                success=True,
                status="ok",
                elapsed_sec=time.time() - start_time,
                inject_attempts=inject_attempts,
                q_rows=q_rows,
                motion_playing_rows=motion_playing_rows,
                message="",
            )

        now = time.time()
        if inject_start_key and now >= next_inject_time:
            try:
                os.write(stdin_master_fd, b"]")
                inject_attempts += 1
            except OSError:
                pass
            next_inject_time = now + max(inject_interval_sec, 0.1)

        time.sleep(POLL_SEC)

    q_rows, motion_playing_rows = log_row_counts(run_logs_dir)
    if q_rows >= min_rows or motion_playing_rows >= min_rows:
        return StartHandshakeResult(
            success=True,
            status="ok",
            elapsed_sec=time.time() - start_time,
            inject_attempts=inject_attempts,
            q_rows=q_rows,
            motion_playing_rows=motion_playing_rows,
            message="",
        )

    return StartHandshakeResult(
        success=False,
        status="timeout",
        elapsed_sec=time.time() - start_time,
        inject_attempts=inject_attempts,
        q_rows=q_rows,
        motion_playing_rows=motion_playing_rows,
        message="control start timeout: logs not generated",
    )


def sleep_while_running(proc: subprocess.Popen, duration_sec: float) -> bool:
    deadline = time.time() + max(duration_sec, 0.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        time.sleep(min(POLL_SEC, max(deadline - time.time(), 0.0)))
    return proc.poll() is None


def write_status_csv(status_file: Path, rows: list[list[str]]) -> None:
    with status_file.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line]))


def update_csv_tail(state: CsvTailState) -> None:
    if not state.path.is_file():
        return
    with state.path.open("r", newline="") as f:
        f.seek(state.offset)
        if state.header is None:
            header_line = f.readline()
            if not header_line:
                return
            state.header = _parse_csv_line(header_line)

        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = _parse_csv_line(line)
            state.rows_read += 1
            state.last_row = row

        state.offset = f.tell()


def tail_get_value(state: CsvTailState, col_name: str) -> Optional[str]:
    if state.header is None or state.last_row is None:
        return None
    try:
        idx = state.header.index(col_name)
    except ValueError:
        return None
    if idx >= len(state.last_row):
        return None
    return state.last_row[idx]


def tail_get_motion_name(state: CsvTailState) -> str:
    value = tail_get_value(state, "motion_name")
    return (value or "").strip()


def tail_get_playing_bool(state: CsvTailState) -> bool:
    value = tail_get_value(state, "playing")
    if value is None:
        return False
    try:
        return float(value) > 0.5
    except ValueError:
        return False


def send_key(stdin_master_fd: int, key: str) -> None:
    os.write(stdin_master_fd, key.encode("ascii", errors="ignore"))


def wait_for_motion_name_available(
    proc: subprocess.Popen,
    motion_name_tail: CsvTailState,
    timeout_sec: float,
) -> tuple[bool, str]:
    deadline = time.time() + max(timeout_sec, 0.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, ""
        update_csv_tail(motion_name_tail)
        name = tail_get_motion_name(motion_name_tail)
        if name:
            return True, name
        time.sleep(POLL_SEC)
    return False, ""


def wait_for_motion_name_change(
    proc: subprocess.Popen,
    motion_name_tail: CsvTailState,
    prev_name: str,
    timeout_sec: float,
) -> tuple[bool, str, str]:
    deadline = time.time() + max(timeout_sec, 0.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, prev_name, "deploy exited while waiting for motion switch"
        update_csv_tail(motion_name_tail)
        current = tail_get_motion_name(motion_name_tail)
        if current and current != prev_name:
            return True, current, ""
        time.sleep(POLL_SEC)
    return False, prev_name, "motion name did not change after 'N'"


def wait_for_play_cycle(
    proc: subprocess.Popen,
    motion_name_tail: CsvTailState,
    motion_playing_tail: CsvTailState,
    timeout_sec: float,
) -> tuple[bool, str, int, int, str]:
    started = False
    observed_name = ""
    start_row = -1
    end_row = -1
    deadline = time.time() + max(timeout_sec, 0.0)

    while time.time() < deadline:
        if proc.poll() is not None:
            return False, observed_name, start_row, end_row, "deploy exited during play cycle"

        update_csv_tail(motion_name_tail)
        update_csv_tail(motion_playing_tail)

        current_name = tail_get_motion_name(motion_name_tail)
        if current_name:
            observed_name = current_name

        playing = tail_get_playing_bool(motion_playing_tail)
        row = motion_playing_tail.rows_read

        if not started:
            if playing:
                started = True
                start_row = row
        else:
            if not playing:
                end_row = row
                return True, observed_name, start_row, end_row, ""

        time.sleep(POLL_SEC)

    if not started:
        return False, observed_name, start_row, end_row, "play did not start (motion_playing stayed 0)"
    return False, observed_name, start_row, end_row, "play did not complete before timeout"


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Automatic batch runner for SONIC motion tracking evaluation.")
    parser.add_argument("--repo-root", required=True, help="Path to GR00T-WholeBodyControl repo root.")
    parser.add_argument(
        "--reference-root",
        default="gear_sonic_deploy/reference/textop_converted",
        help="Reference motion root relative to repo-root or absolute path.",
    )
    parser.add_argument(
        "--motion-list-txt",
        default="",
        help="Optional txt file (one motion folder name per line). If empty, run all motions in lexicographic order.",
    )
    parser.add_argument(
        "--workspace-root",
        default=f"tools/eval_tracking/outputs/auto_run_{ts}",
        help="Output workspace for generated playback files and logs.",
    )

    parser.add_argument("--deploy-bin", default="gear_sonic_deploy/target/release/g1_deploy_onnx_ref")
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--decoder-model", default="gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--encoder-model", default="gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--planner-model", default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    parser.add_argument("--obs-config", default="gear_sonic_deploy/policy/release/observation_config.yaml")
    parser.add_argument("--input-type", default="keyboard")
    parser.add_argument("--output-type", default="all")

    parser.add_argument("--launch-sim", action="store_true", help="Launch MuJoCo sim loop from this script.")
    parser.add_argument(
        "--sim-cmd",
        default="python gear_sonic/scripts/run_sim_loop.py",
        help="Command used when --launch-sim is enabled.",
    )

    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--duration-factor",
        type=float,
        default=1.10,
        help="Run each deploy for duration_factor * (frames/fps) after control starts.",
    )
    parser.add_argument(
        "--startup-sec",
        type=float,
        default=6.0,
        help="Extra startup wait after control starts before playback timing window.",
    )
    parser.add_argument("--settle-sec", type=float, default=1.0, help="Extra settle time before stopping deploy process.")
    parser.add_argument("--execute", action="store_true", help="Actually execute commands. Default is dry-run print only.")

    parser.add_argument(
        "--inject-start-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically inject ']' during startup until logs appear.",
    )
    parser.add_argument(
        "--start-key-delay-sec",
        type=float,
        default=2.5,
        help="Initial delay before first injected start key.",
    )
    parser.add_argument(
        "--start-timeout-sec",
        type=float,
        default=25.0,
        help="Maximum time to wait for control start evidence in logs.",
    )
    parser.add_argument(
        "--inject-interval-sec",
        type=float,
        default=1.0,
        help="Interval between repeated start key injections.",
    )
    parser.add_argument(
        "--min-log-rows",
        type=int,
        default=2,
        help="Min data rows in q.csv or motion_playing.csv to mark startup success.",
    )
    parser.add_argument(
        "--require-start-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat a motion as failed if startup handshake times out.",
    )
    parser.add_argument(
        "--require-log-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated alias for requiring startup success via log files.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop batch on first failed motion run.")

    parser.add_argument("--single-session", action="store_true", help="Run one deploy session and cycle motions via T/N.")
    parser.add_argument(
        "--single-session-max-motions",
        type=int,
        default=0,
        help="Maximum motions to play in single-session mode (0 = all).",
    )
    parser.add_argument(
        "--single-session-post-complete-sec",
        type=float,
        default=0.2,
        help="Wait time after one motion completes before sending next key.",
    )
    parser.add_argument(
        "--single-session-key-gap-sec",
        type=float,
        default=0.15,
        help="Gap between injected keyboard keys.",
    )

    parser.add_argument("--run-eval", action="store_true", help="Call eval_sonic_tracking.py after runs (with --execute).")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation even if --run-eval is set.")
    parser.add_argument("--warmup-frames", type=int, default=20)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Optional[List[str]]]:
    repo_root = Path(args.repo_root).resolve()

    reference_root = Path(args.reference_root)
    if not reference_root.is_absolute():
        reference_root = (repo_root / reference_root).resolve()

    workspace_root = Path(args.workspace_root)
    if not workspace_root.is_absolute():
        workspace_root = (repo_root / workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    motion_list: List[str] | None = None
    if args.motion_list_txt:
        motion_list_txt = Path(args.motion_list_txt)
        if not motion_list_txt.is_absolute():
            motion_list_txt = (repo_root / motion_list_txt).resolve()
        motion_list = read_motion_list(motion_list_txt)

    deploy_bin = Path(args.deploy_bin)
    if not deploy_bin.is_absolute():
        deploy_bin = (repo_root / deploy_bin).resolve()

    decoder_model = Path(args.decoder_model)
    if not decoder_model.is_absolute():
        decoder_model = (repo_root / decoder_model).resolve()

    encoder_model = Path(args.encoder_model)
    if not encoder_model.is_absolute():
        encoder_model = (repo_root / encoder_model).resolve()

    planner_model = Path(args.planner_model)
    if not planner_model.is_absolute():
        planner_model = (repo_root / planner_model).resolve()

    obs_config = Path(args.obs_config)
    if not obs_config.is_absolute():
        obs_config = (repo_root / obs_config).resolve()

    return (
        repo_root,
        reference_root,
        workspace_root,
        deploy_bin,
        decoder_model,
        encoder_model,
        planner_model,
        obs_config,
        motion_list,
    )


def append_status_row(
    rows: list[list[str]],
    mode: str,
    index: int,
    session_index: str,
    motion_name: str,
    motion_name_observed: str,
    status: str,
    start_status: str,
    deploy_exit_code: str,
    started_control: str,
    has_required_logs: str,
    handshake_elapsed_sec: str,
    inject_attempts: str,
    q_rows: str,
    motion_playing_rows: str,
    play_window_sec: str,
    total_run_sec: str,
    play_start_row: str,
    play_end_row: str,
    message: str,
) -> None:
    rows.append(
        [
            mode,
            str(index),
            session_index,
            motion_name,
            motion_name_observed,
            status,
            start_status,
            deploy_exit_code,
            started_control,
            has_required_logs,
            handshake_elapsed_sec,
            inject_attempts,
            q_rows,
            motion_playing_rows,
            play_window_sec,
            total_run_sec,
            play_start_row,
            play_end_row,
            message,
        ]
    )


def run_multi_process_batch(
    args: argparse.Namespace,
    repo_root: Path,
    items: Sequence[MotionItem],
    deploy_bin: Path,
    decoder_model: Path,
    encoder_model: Path,
    planner_model: Path,
    obs_config: Path,
    workspace_root: Path,
    plan_lines: list[str],
    status_rows: list[list[str]],
    status_file: Path,
) -> None:
    motion_sets_root = workspace_root / "single_motion_sets"
    playbacks_root = workspace_root / "playback_files"
    logs_root = workspace_root / "logs"
    require_start_success = args.require_start_success and args.require_log_files

    for idx, item in enumerate(items):
        dataset_root = make_single_motion_set(motion_sets_root, item, args.copy_mode)
        playback_file = playbacks_root / f"{idx:04d}_{item.name}.csv"
        write_single_line_playback(playback_file)

        run_logs_dir = logs_root / f"{idx:04d}_{item.name}"
        run_logs_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_deploy_cmd(
            deploy_bin=deploy_bin,
            network_interface=args.network_interface,
            decoder_model=decoder_model,
            encoder_model=encoder_model,
            planner_model=planner_model,
            obs_config=obs_config,
            motion_data_root=dataset_root,
            logs_dir=run_logs_dir,
            output_type=args.output_type,
            input_type=args.input_type,
            playback_file=playback_file,
        )

        est_motion_sec = item.frames / max(args.fps, 1e-6)
        play_window_sec = args.duration_factor * est_motion_sec

        plan_lines.append(f"[multi motion {idx:04d}] {item.name}")
        plan_lines.append(f"frames={item.frames} est_sec={est_motion_sec:.3f} play_window_sec={play_window_sec:.3f}")
        plan_lines.append(f"dataset_root={dataset_root}")
        plan_lines.append(f"playback_file={playback_file}")
        plan_lines.append(f"logs_dir={run_logs_dir}")
        plan_lines.append("command=" + " ".join(shlex.quote(x) for x in cmd))
        plan_lines.append("")

        if not args.execute:
            continue

        status = "ok"
        start_status = "skipped"
        message = ""
        handshake_elapsed_sec = 0.0
        inject_attempts = 0
        started_control = False
        has_required_logs = False
        q_rows = 0
        motion_playing_rows = 0
        deploy_exit_code = -999
        process_exited_early = False
        motion_start_time = time.time()

        proc, stdin_master_fd = launch_process_with_pty_stdin(cmd, cwd=repo_root / "gear_sonic_deploy")
        try:
            handshake = wait_for_control_start(
                proc=proc,
                stdin_master_fd=stdin_master_fd,
                run_logs_dir=run_logs_dir,
                inject_start_key=args.inject_start_key,
                start_timeout_sec=args.start_timeout_sec,
                inject_interval_sec=args.inject_interval_sec,
                min_log_rows=args.min_log_rows,
                initial_inject_delay_sec=args.start_key_delay_sec,
            )
            started_control = handshake.success
            start_status = handshake.status
            handshake_elapsed_sec = handshake.elapsed_sec
            inject_attempts = handshake.inject_attempts
            q_rows = handshake.q_rows
            motion_playing_rows = handshake.motion_playing_rows
            if handshake.message:
                message = handshake.message

            if handshake.success:
                if args.startup_sec > 0 and not sleep_while_running(proc, args.startup_sec):
                    process_exited_early = True
                    message = "deploy process exited during post-startup wait"
                if not process_exited_early and play_window_sec > 0 and not sleep_while_running(proc, play_window_sec):
                    process_exited_early = True
                    message = "deploy process exited before playback window completed"
                if not process_exited_early and args.settle_sec > 0 and not sleep_while_running(proc, args.settle_sec):
                    process_exited_early = True
                    message = "deploy process exited during settle window"
            elif not require_start_success:
                fallback_run_sec = max(args.startup_sec, 0.0) + max(play_window_sec, 0.0) + max(args.settle_sec, 0.0)
                if fallback_run_sec > 0 and not sleep_while_running(proc, fallback_run_sec):
                    process_exited_early = True
                    if not message:
                        message = "deploy process exited during fallback run window"
        finally:
            terminate_process(proc)
            try:
                os.close(stdin_master_fd)
            except OSError:
                pass

        deploy_exit_code = proc.poll() if proc.poll() is not None else -999
        q_rows, motion_playing_rows = log_row_counts(run_logs_dir)
        has_required_logs = q_rows >= max(args.min_log_rows, 1) or motion_playing_rows >= max(args.min_log_rows, 1)

        if require_start_success and not started_control:
            status = "failed"
            if not message:
                message = "control startup handshake failed"
        elif process_exited_early:
            status = "failed"
        elif args.require_log_files and not has_required_logs:
            status = "failed"
            if not message:
                message = "required logs missing or too short"
        elif deploy_exit_code not in VALID_TERMINATE_EXIT_CODES and not has_required_logs:
            status = "failed"
            if not message:
                message = "deploy exited abnormally and logs are missing"

        total_run_sec = time.time() - motion_start_time
        append_status_row(
            status_rows,
            mode="multi_process",
            index=idx,
            session_index="",
            motion_name=item.name,
            motion_name_observed="",
            status=status,
            start_status=start_status,
            deploy_exit_code=str(deploy_exit_code),
            started_control=str(started_control),
            has_required_logs=str(has_required_logs),
            handshake_elapsed_sec=f"{handshake_elapsed_sec:.3f}",
            inject_attempts=str(inject_attempts),
            q_rows=str(q_rows),
            motion_playing_rows=str(motion_playing_rows),
            play_window_sec=f"{play_window_sec:.3f}",
            total_run_sec=f"{total_run_sec:.3f}",
            play_start_row="",
            play_end_row="",
            message=message,
        )
        write_status_csv(status_file, status_rows)

        if args.fail_fast and status != "ok":
            break


def run_single_session_batch(
    args: argparse.Namespace,
    repo_root: Path,
    items: Sequence[MotionItem],
    deploy_bin: Path,
    decoder_model: Path,
    encoder_model: Path,
    planner_model: Path,
    obs_config: Path,
    workspace_root: Path,
    plan_lines: list[str],
    status_rows: list[list[str]],
    status_file: Path,
) -> None:
    if args.input_type != "keyboard":
        raise RuntimeError("--single-session currently supports only --input-type keyboard")

    motion_sets_root = workspace_root / "single_motion_sets"
    logs_root = workspace_root / "logs"
    run_logs_dir = logs_root / "0000_single_session"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = make_session_motion_set(motion_sets_root, items, args.copy_mode)
    cmd = build_deploy_cmd(
        deploy_bin=deploy_bin,
        network_interface=args.network_interface,
        decoder_model=decoder_model,
        encoder_model=encoder_model,
        planner_model=planner_model,
        obs_config=obs_config,
        motion_data_root=dataset_root,
        logs_dir=run_logs_dir,
        output_type=args.output_type,
        input_type=args.input_type,
        playback_file=None,
    )

    total_candidates = len(items)
    total_target = total_candidates
    if args.single_session_max_motions > 0:
        total_target = min(total_candidates, args.single_session_max_motions)

    frame_map = {item.name: item.frames for item in items}

    plan_lines.append("[single_session]")
    plan_lines.append(f"motions_total={total_candidates}")
    plan_lines.append(f"motions_target={total_target}")
    plan_lines.append(f"dataset_root={dataset_root}")
    plan_lines.append(f"logs_dir={run_logs_dir}")
    plan_lines.append(f"session_order_file={dataset_root / 'session_order.txt'}")
    plan_lines.append("command=" + " ".join(shlex.quote(x) for x in cmd))
    plan_lines.append("")

    if not args.execute:
        return

    proc, stdin_master_fd = launch_process_with_pty_stdin(cmd, cwd=repo_root / "gear_sonic_deploy")
    try:
        handshake = wait_for_control_start(
            proc=proc,
            stdin_master_fd=stdin_master_fd,
            run_logs_dir=run_logs_dir,
            inject_start_key=args.inject_start_key,
            start_timeout_sec=args.start_timeout_sec,
            inject_interval_sec=args.inject_interval_sec,
            min_log_rows=args.min_log_rows,
            initial_inject_delay_sec=args.start_key_delay_sec,
        )

        q_rows, motion_playing_rows = log_row_counts(run_logs_dir)
        has_required_logs = q_rows >= max(args.min_log_rows, 1) or motion_playing_rows >= max(args.min_log_rows, 1)

        if not handshake.success and (args.require_start_success and args.require_log_files):
            append_status_row(
                status_rows,
                mode="single_session",
                index=0,
                session_index="0",
                motion_name="",
                motion_name_observed="",
                status="failed",
                start_status=handshake.status,
                deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                started_control=str(False),
                has_required_logs=str(has_required_logs),
                handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                inject_attempts=str(handshake.inject_attempts),
                q_rows=str(q_rows),
                motion_playing_rows=str(motion_playing_rows),
                play_window_sec="",
                total_run_sec=f"{handshake.elapsed_sec:.3f}",
                play_start_row="",
                play_end_row="",
                message=handshake.message or "control startup handshake failed",
            )
            write_status_csv(status_file, status_rows)
            return

        motion_name_tail = CsvTailState(path=run_logs_dir / "motion_name.csv")
        motion_playing_tail = CsvTailState(path=run_logs_dir / "motion_playing.csv")

        ok_name, current_motion_name = wait_for_motion_name_available(proc, motion_name_tail, timeout_sec=8.0)
        if not ok_name:
            append_status_row(
                status_rows,
                mode="single_session",
                index=0,
                session_index="0",
                motion_name="",
                motion_name_observed="",
                status="failed",
                start_status=handshake.status,
                deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                started_control=str(handshake.success),
                has_required_logs=str(has_required_logs),
                handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                inject_attempts=str(handshake.inject_attempts),
                q_rows=str(q_rows),
                motion_playing_rows=str(motion_playing_rows),
                play_window_sec="",
                total_run_sec=f"{handshake.elapsed_sec:.3f}",
                play_start_row="",
                play_end_row="",
                message="motion_name.csv did not produce a motion name",
            )
            write_status_csv(status_file, status_rows)
            return

        if args.startup_sec > 0:
            if not sleep_while_running(proc, args.startup_sec):
                append_status_row(
                    status_rows,
                    mode="single_session",
                    index=0,
                    session_index="0",
                    motion_name=current_motion_name,
                    motion_name_observed=current_motion_name,
                    status="failed",
                    start_status=handshake.status,
                    deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                    started_control=str(handshake.success),
                    has_required_logs=str(has_required_logs),
                    handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                    inject_attempts=str(handshake.inject_attempts),
                    q_rows=str(q_rows),
                    motion_playing_rows=str(motion_playing_rows),
                    play_window_sec="",
                    total_run_sec="0.000",
                    play_start_row="",
                    play_end_row="",
                    message="deploy exited during single-session startup wait",
                )
                write_status_csv(status_file, status_rows)
                return

        for session_idx in range(total_target):
            expected_name = current_motion_name
            est_frames = frame_map.get(expected_name, 0)
            if est_frames > 0:
                est_motion_sec = est_frames / max(args.fps, 1e-6)
            else:
                est_motion_sec = 5.0
            play_window_sec = args.duration_factor * est_motion_sec
            per_motion_timeout = max(play_window_sec + args.settle_sec + args.single_session_post_complete_sec + 5.0, 10.0)

            motion_start_time = time.time()
            try:
                send_key(stdin_master_fd, "T")
            except OSError:
                append_status_row(
                    status_rows,
                    mode="single_session",
                    index=session_idx,
                    session_index=str(session_idx),
                    motion_name=expected_name,
                    motion_name_observed=expected_name,
                    status="failed",
                    start_status=handshake.status,
                    deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                    started_control=str(handshake.success),
                    has_required_logs=str(has_required_logs),
                    handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                    inject_attempts=str(handshake.inject_attempts),
                    q_rows=str(q_rows),
                    motion_playing_rows=str(motion_playing_rows),
                    play_window_sec=f"{play_window_sec:.3f}",
                    total_run_sec="0.000",
                    play_start_row="",
                    play_end_row="",
                    message="failed to send key 'T'",
                )
                write_status_csv(status_file, status_rows)
                if args.fail_fast:
                    break
                continue

            time.sleep(max(args.single_session_key_gap_sec, 0.0))
            ok_play, observed_name, start_row, end_row, play_msg = wait_for_play_cycle(
                proc=proc,
                motion_name_tail=motion_name_tail,
                motion_playing_tail=motion_playing_tail,
                timeout_sec=per_motion_timeout,
            )

            total_run_sec = time.time() - motion_start_time
            current_motion_name = observed_name or expected_name
            row_status = "ok" if ok_play else "failed"

            append_status_row(
                status_rows,
                mode="single_session",
                index=session_idx,
                session_index=str(session_idx),
                motion_name=expected_name,
                motion_name_observed=current_motion_name,
                status=row_status,
                start_status=handshake.status,
                deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                started_control=str(handshake.success),
                has_required_logs=str(True),
                handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                inject_attempts=str(handshake.inject_attempts),
                q_rows=str(log_row_counts(run_logs_dir)[0]),
                motion_playing_rows=str(log_row_counts(run_logs_dir)[1]),
                play_window_sec=f"{play_window_sec:.3f}",
                total_run_sec=f"{total_run_sec:.3f}",
                play_start_row=str(start_row if start_row >= 0 else ""),
                play_end_row=str(end_row if end_row >= 0 else ""),
                message=play_msg,
            )
            write_status_csv(status_file, status_rows)

            if not ok_play and args.fail_fast:
                break

            if args.single_session_post_complete_sec > 0:
                if not sleep_while_running(proc, args.single_session_post_complete_sec):
                    break

            if session_idx + 1 >= total_target:
                continue

            prev_name = current_motion_name
            try:
                send_key(stdin_master_fd, "N")
            except OSError:
                if args.fail_fast:
                    break
                continue
            time.sleep(max(args.single_session_key_gap_sec, 0.0))

            switched, new_name, switch_msg = wait_for_motion_name_change(
                proc=proc,
                motion_name_tail=motion_name_tail,
                prev_name=prev_name,
                timeout_sec=max(args.startup_sec, 2.0) + 5.0,
            )
            if switched:
                current_motion_name = new_name
            elif args.fail_fast:
                append_status_row(
                    status_rows,
                    mode="single_session",
                    index=session_idx,
                    session_index=str(session_idx),
                    motion_name=prev_name,
                    motion_name_observed=prev_name,
                    status="failed",
                    start_status=handshake.status,
                    deploy_exit_code=str(proc.poll() if proc.poll() is not None else -999),
                    started_control=str(handshake.success),
                    has_required_logs=str(True),
                    handshake_elapsed_sec=f"{handshake.elapsed_sec:.3f}",
                    inject_attempts=str(handshake.inject_attempts),
                    q_rows=str(log_row_counts(run_logs_dir)[0]),
                    motion_playing_rows=str(log_row_counts(run_logs_dir)[1]),
                    play_window_sec="",
                    total_run_sec="0.000",
                    play_start_row="",
                    play_end_row="",
                    message=switch_msg,
                )
                write_status_csv(status_file, status_rows)
                break

    finally:
        terminate_process(proc)
        try:
            os.close(stdin_master_fd)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    (
        repo_root,
        reference_root,
        workspace_root,
        deploy_bin,
        decoder_model,
        encoder_model,
        planner_model,
        obs_config,
        motion_list,
    ) = resolve_paths(args)

    items = list_motion_items(reference_root, motion_list)
    if not items:
        raise RuntimeError(f"No motion folders found in {reference_root}")

    plans_file = workspace_root / "run_plan.txt"
    status_file = workspace_root / "run_status.csv"
    logs_root = workspace_root / "logs"

    plan_lines: List[str] = [
        f"reference_root={reference_root}",
        f"motions={len(items)}",
        f"mode={'single_session' if args.single_session else 'multi_process'}",
        f"inject_start_key={args.inject_start_key}",
        f"start_timeout_sec={args.start_timeout_sec}",
        f"inject_interval_sec={args.inject_interval_sec}",
        f"min_log_rows={args.min_log_rows}",
        "",
    ]

    status_rows: list[list[str]] = [
        [
            "mode",
            "index",
            "session_index",
            "motion_name",
            "motion_name_observed",
            "status",
            "start_status",
            "deploy_exit_code",
            "started_control",
            "has_required_logs",
            "handshake_elapsed_sec",
            "inject_attempts",
            "q_rows",
            "motion_playing_rows",
            "play_window_sec",
            "total_run_sec",
            "play_start_row",
            "play_end_row",
            "message",
        ]
    ]

    sim_proc: subprocess.Popen | None = None
    interrupted = False

    if args.launch_sim:
        sim_cmd = shlex.split(args.sim_cmd)
        plan_lines.append("[sim]")
        plan_lines.append(" ".join(shlex.quote(x) for x in sim_cmd))
        plan_lines.append("")
        if args.execute:
            sim_proc = launch_process(sim_cmd, cwd=repo_root)
            time.sleep(2.0)
            if sim_proc.poll() is not None:
                raise RuntimeError("Sim process exited immediately. Check sim python environment (e.g., missing tyro / wrong venv).")

    try:
        if args.single_session:
            run_single_session_batch(
                args=args,
                repo_root=repo_root,
                items=items,
                deploy_bin=deploy_bin,
                decoder_model=decoder_model,
                encoder_model=encoder_model,
                planner_model=planner_model,
                obs_config=obs_config,
                workspace_root=workspace_root,
                plan_lines=plan_lines,
                status_rows=status_rows,
                status_file=status_file,
            )
        else:
            run_multi_process_batch(
                args=args,
                repo_root=repo_root,
                items=items,
                deploy_bin=deploy_bin,
                decoder_model=decoder_model,
                encoder_model=encoder_model,
                planner_model=planner_model,
                obs_config=obs_config,
                workspace_root=workspace_root,
                plan_lines=plan_lines,
                status_rows=status_rows,
                status_file=status_file,
            )

    except KeyboardInterrupt:
        interrupted = True
        print("[WARN] Interrupted by user. Writing partial run plan and status.")
    finally:
        plans_file.write_text("\n".join(plan_lines))
        write_status_csv(status_file, status_rows)
        if sim_proc is not None:
            terminate_process(sim_proc)

    print(f"[DONE] run plan written: {plans_file}")
    print(f"[DONE] run status written: {status_file}")

    if interrupted:
        return

    if args.run_eval and not args.no_eval:
        eval_script = repo_root / "tools" / "eval_tracking" / "eval_sonic_tracking.py"
        eval_out = workspace_root / "eval_results"
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--reference-root",
            str(reference_root),
            "--logs-dir",
            str(logs_root),
            "--out-dir",
            str(eval_out),
            "--warmup-frames",
            str(args.warmup_frames),
        ]
        print("[INFO] eval command:")
        print(" ".join(shlex.quote(x) for x in eval_cmd))
        if args.execute:
            subprocess.check_call(eval_cmd, cwd=repo_root)

    if not args.execute:
        print("[DRY-RUN] Commands and run plan generated. Re-run with --execute to run them.")


if __name__ == "__main__":
    main()
