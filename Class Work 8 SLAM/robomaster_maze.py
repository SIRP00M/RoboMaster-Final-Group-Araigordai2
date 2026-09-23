#!/usr/bin/env python3
"""
RoboMaster EP / EP Core - 4x5 Maze Explorer v12 (FAST Frontier-Trémaux SLAM Exploration)
============================================

Hardware / assumptions
----------------------
- RoboMaster EP/EP Core
- RoboMaster ToF sensor mounted on the gimbal and pointing with gimbal yaw
- Arena: 4 x 5 cells
- Cell size: 0.60 m
- Robot starts near the center of one cell
- Connection mode: AP by default

Behavior
--------
1. Stop chassis.
2. Save chassis IMU yaw.
3. Smooth-scan gimbal from -180 to +180 deg.
4. Actively hold chassis IMU yaw while the turret scans (25 Hz correction).
5. Read ToF while scanning and compensate each scan angle with any residual chassis yaw drift.
6. Recenter gimbal.
7. Fine-restore chassis yaw to the value before scanning.
7. Classify Front / Right / Back / Left as WALL / OPEN / UNKNOWN.
8. Trémaux: choose OPEN passages with the fewest edge marks (0 -> 1 -> 2).
9. Frontier exploration: prefer local OPEN passages never traversed before.
10. If the local cell has no new passage, plan through the already-built OPEN graph to the nearest known frontier.
11. If a selected passage is blocked, roll back, temporarily quarantine that edge, stop and scan 360 again instead of retrying forever.
12. Trusted retrace: previously traversed passages use a safer return profile near the destination cell.
13. Maintain logical localization as (row, col, heading) after each verified 0.60 m move.
14. Rotate chassis to the selected cardinal direction.
15. Drive exactly one 0.60 m cell with IMU heading hold + front ToF safety.
16. Repeat scan -> map update -> frontier selection until all reachable frontiers are exhausted.

Outputs
-------
- runs/<timestamp>/tof_log.csv
- runs/<timestamp>/imu_scan_log.csv
- runs/<timestamp>/maze_cell_trajectory.csv
- runs/<timestamp>/trajectory.csv
- runs/<timestamp>/maze_state.json
- runs/<timestamp>/summary.json

Install
-------
    pip install robomaster numpy

Run
---
    python3 robomaster_maze_4x5_tof_gimbal.py

Optional STA mode
-----------------
    python3 robomaster_maze_4x5_tof_gimbal.py --conn-type sta
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from robomaster import robot


# ============================================================
# USER CONFIG
# ============================================================

# FAST profile tuned from successful v7/v8 runs.
# Normal motion is faster, but obstacle approach still falls back to SLOW/CRAWL.
# If the floor is slippery or the maze walls are fragile, reduce MOVE_SPEED_MPS
# and TURN_MAX_DPS first.

# ToF port index from sensor.sub_distance(): 0..3
TOF_INDEX = 0

# Maze geometry
MAZE_COLS = 4
MAZE_ROWS = 5
CELL_SIZE_M = 0.60

# Logical start cell. Change to match where you place the robot.
# 0 = top row, MAZE_ROWS-1 = bottom row.
START_ROW = MAZE_ROWS - 1
START_COL = 0

# Heading index in the logical maze:
# 0 = initial forward
# 1 = +90 deg chassis turn from start
# 2 = 180 deg from start
# 3 = -90 deg chassis turn from start
START_HEADING = 0

# Gimbal scan
SCAN_START_DEG = -180.0
SCAN_END_DEG = +180.0
SCAN_SPEED_DPS = 24.0
SCAN_SAMPLE_STEP_DEG = 2.0
SCAN_TIMEOUT_S = 22.0
SCAN_LOOP_S = 0.006
GIMBAL_CENTER_TOL_DEG = 1.5
GIMBAL_SETTLE_S = 0.08

# A real EP may report -174..-180 deg after a completed -180 action.
# That is still sufficient because the +180 side covers the rear cardinal ray.
# Reject only a materially incomplete start position.
SCAN_START_VERIFY_TOL_DEG = 8.0
SCAN_MIN_RAW_SPAN_DEG = 345.0

# Chassis motion
TURN_MAX_DPS = 55.0
TURN_MIN_DPS = 9.0
TURN_KP = 1.8
TURN_TOL_DEG = 1.0
TURN_STABLE_COUNT = 3

MOVE_SPEED_MPS = 0.20
MOVE_SLOW_MPS = 0.10
MOVE_CRAWL_MPS = 0.06
MOVE_TOL_M = 0.015
MOVE_LOOP_S = 0.025
# v12 uses a progress watchdog instead of one short fixed move timeout.
# This allows a trusted retrace to crawl the final few centimeters without
# timing out, but still aborts if the chassis truly stops making progress.
MOVE_TOTAL_TIMEOUT_S = 18.0
MOVE_STALL_TIMEOUT_S = 2.2

# v12: if a confirmed retrace reaches within a few cm of the next logical
# cell center, do not throw away ~0.58 m of good motion just because the
# forward ToF sees the wall at the far side of that cell. Stop where we are
# and snap the logical pose to the destination cell if heading/lateral error
# are still small and there is still non-contact clearance.
NEAR_TARGET_ACCEPT_REMAINING_M = 0.035
NEAR_TARGET_ACCEPT_MIN_FRONT_M = 0.085
NEAR_TARGET_ACCEPT_MAX_LATERAL_M = 0.025
NEAR_TARGET_ACCEPT_MAX_YAW_ERR_DEG = 2.0
MOVE_PROGRESS_EPS_M = 0.004

# IMU heading hold while translating
DRIVE_YAW_KP = 2.1
DRIVE_MAX_Z_DPS = 25.0
DRIVE_YAW_DEADBAND_DEG = 0.25
DRIVE_YAW_ABORT_DEG = 7.0
DRIVE_LATERAL_ABORT_M = 0.07

# ToF classification
TOF_MIN_VALID_M = 0.05
TOF_MAX_VALID_M = 4.00

# From a cell center, a wall is nominally ~0.30 m away.
# An open adjacent cell usually gives substantially more than 0.60 m.
WALL_MAX_M = 0.43
OPEN_MIN_M = 0.68

# Navigation sector width around each cardinal direction
# Use a narrower central sector. The old +/-12 deg + low percentile could
# see a doorway/cell corner and call an otherwise-open axis a wall.
SECTOR_HALF_WIDTH_DEG = 8.0

# Safety while moving
FRONT_SLOWDOWN_M = 0.45

# IMPORTANT for 60-cm maze cells:
# A fixed 0.28-m emergency stop is too large. At the CENTER of a 60-cm cell,
# the front wall can naturally be only ~0.18-0.30 m from the gimbal ToF,
# depending on sensor offset. v6 therefore stopped ~0.49 m into a 0.60-m move.
#
# v7 uses projected clearance at the destination:
#   projected_clearance = current_front_range - remaining_move_distance
# We only continue if the predicted range after reaching the cell center stays
# above MIN_PROJECTED_FRONT_CLEARANCE_M.
ABSOLUTE_EMERGENCY_STOP_M = 0.08
MIN_PROJECTED_FRONT_CLEARANCE_M = 0.11
TOF_STALE_S = 0.30

# v10: When Trémaux is retracing a passage that the robot has already
# traversed successfully, the passage geometry is known to be physically open.
# Near the destination cell, projected-clearance can become pessimistic because
# the ToF origin is not exactly at chassis center and walls are seen at an angle.
# Therefore, for a confirmed OPEN retrace we keep normal projected safety over
# the first part of the move, then switch to slow odometry/IMU-guided motion for
# the final segment while retaining a hard ToF stop.
TRUSTED_RETRACE_RELAX_REMAINING_M = 0.22
TRUSTED_RETRACE_HARD_STOP_M = 0.10
TRUSTED_RETRACE_SLOW_MPS = 0.10
TRUSTED_RETRACE_CRAWL_MPS = 0.05

# Do not retry the same passage forever. A confirmed return passage may be
# retried once after a rollback; a second failure stops safely with a clear
# diagnostic instead of entering an infinite scan/turn/rollback loop.
# v12 no longer stops the whole exploration after N failures on one edge.
# A failing passage is quarantined for a few complete 360 scans while the
# explorer searches other local/frontier routes.
MAX_EDGE_MOVE_FAILURES = 99
TEMP_BLOCK_BASE_RESCAN_CYCLES = 2
TEMP_BLOCK_MAX_RESCAN_CYCLES = 5
FRONTIER_COMPLETE_CONFIRM_SCANS = 3

# If any one-cell move aborts after the robot has already translated, v7 rolls
# back along the just-traversed path so the physical robot and logical DFS cell
# stay synchronized.
ROLLBACK_MIN_DISTANCE_M = 0.03
ROLLBACK_SPEED_MPS = 0.12
ROLLBACK_TOL_M = 0.015
ROLLBACK_LATERAL_ABORT_M = 0.08
# v12 rollback is progress-watchdog based. Total time is generous, but if the
# robot is not actually reducing its displacement for several seconds we stop.
ROLLBACK_TOTAL_TIMEOUT_S = 24.0
ROLLBACK_STALL_TIMEOUT_S = 2.8
ROLLBACK_PROGRESS_EPS_M = 0.008

# Active IMU yaw hold while the gimbal performs the 360-degree scan.
# Real log showed ~10 deg of chassis yaw drift during one passive sweep,
# so v5 actively counters that reaction torque while the gimbal sweep itself uses drive_speed(), avoiding overlapping SDK Action objects.
SCAN_YAW_HOLD_ENABLED = True
SCAN_YAW_HOLD_KP = 1.45
SCAN_YAW_HOLD_DEADBAND_DEG = 0.30
SCAN_YAW_HOLD_MIN_DPS = 2.0
SCAN_YAW_HOLD_MAX_DPS = 12.0
SCAN_YAW_HOLD_COMMAND_PERIOD_S = 0.04   # 25 Hz chassis correction
SCAN_YAW_HOLD_ABORT_DEG = 6.0           # hold failure => stop safely

# Some EP gimbals report a few degrees of pitch offset even when a yaw action
# completes correctly. Maze direction mapping depends on yaw, so moderate
# pitch mismatch is only a warning.
GIMBAL_YAW_VERIFY_TOL_DEG = 3.0
STARTUP_SELFTEST_YAW_TOL_DEG = 6.0
GIMBAL_PITCH_WARN_DEG = 2.5
GIMBAL_PITCH_HARD_TOL_DEG = 6.0

# Closed-loop gimbal centering used after a speed-mode sweep.
# This deliberately avoids gimbal.recenter() after scanning because the SDK
# action can occasionally time out immediately after a drive_speed() sweep.
CENTER_GIMBAL_YAW_KP = 1.65
CENTER_GIMBAL_PITCH_KP = 1.30
CENTER_GIMBAL_MIN_DPS = 6.0
CENTER_GIMBAL_MAX_YAW_DPS = 55.0
CENTER_GIMBAL_MAX_PITCH_DPS = 28.0
CENTER_GIMBAL_YAW_TOL_DEG = 1.2
CENTER_GIMBAL_PITCH_TOL_DEG = 4.5
CENTER_GIMBAL_STABLE_COUNT = 3
CENTER_GIMBAL_TIMEOUT_S = 7.0
CENTER_GIMBAL_LOOP_S = 0.03
CENTER_GIMBAL_TELEMETRY_STALE_S = 0.6

# After scan, chassis must be restored this accurately before driving.
RESTORE_YAW_TOL_DEG = 1.0
RESTORE_YAW_HARD_FAIL_DEG = 2.0

# Trémaux may traverse each reachable passage up to twice, so keep a generous budget.
MAX_ACTIONS = 120

# Output
OUTPUT_ROOT = "runs"


# ============================================================
# SHARED STATE
# ============================================================


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gimbal_yaw: float = 0.0
    gimbal_pitch: float = 0.0
    tof_mm: Optional[Tuple[float, ...]] = None
    pose_time: float = 0.0
    gimbal_time: float = 0.0
    tof_time: float = 0.0


STATE = RobotState()
STATE_LOCK = threading.Lock()
STOP = threading.Event()
SIGINT_COUNT = 0

# +1 means positive chassis z increases IMU yaw.
# We calibrate this automatically at startup.
YAW_CMD_SIGN = 1.0

# (time, x, y, yaw)
TRAJECTORY: List[Tuple[float, float, float, float]] = []
TRAJECTORY_LOCK = threading.Lock()


def now() -> float:
    return time.time()


def angle_diff(target_deg: float, current_deg: float) -> float:
    """Shortest signed target-current angle in [-180, 180)."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def wrap_angle(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def snapshot() -> RobotState:
    with STATE_LOCK:
        return RobotState(**STATE.__dict__)


def stop_requested() -> None:
    if STOP.is_set():
        raise KeyboardInterrupt


def on_position(data) -> None:
    x, y, _z = data
    t = now()
    with STATE_LOCK:
        STATE.x = float(x)
        STATE.y = float(y)
        STATE.pose_time = t
        yaw = STATE.yaw

    with TRAJECTORY_LOCK:
        if not TRAJECTORY or t - TRAJECTORY[-1][0] >= 0.05:
            TRAJECTORY.append((t, float(x), float(y), float(yaw)))


def on_attitude(data) -> None:
    yaw, pitch, roll = data
    with STATE_LOCK:
        STATE.yaw = float(yaw)
        STATE.pitch = float(pitch)
        STATE.roll = float(roll)
        STATE.pose_time = now()


def on_gimbal_angle(data) -> None:
    # SDK callback: pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle
    pitch, yaw, _pitch_ground, _yaw_ground = data
    with STATE_LOCK:
        STATE.gimbal_pitch = float(pitch)
        STATE.gimbal_yaw = float(yaw)
        STATE.gimbal_time = now()


def on_tof(data) -> None:
    try:
        vals = tuple(float(v) for v in data)
    except Exception:
        return
    with STATE_LOCK:
        STATE.tof_mm = vals
        STATE.tof_time = now()


def signal_handler(_signum, _frame) -> None:
    global SIGINT_COUNT
    SIGINT_COUNT += 1
    STOP.set()
    print("\n[STOP] Ctrl+C -> stopping robot...", flush=True)
    if SIGINT_COUNT >= 2:
        print("[STOP] second Ctrl+C -> force exit", flush=True)
        os._exit(130)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception:
        pass


# ============================================================
# LOW-LEVEL ROBOT HELPERS
# ============================================================


def safe_stop_chassis(chassis) -> None:
    try:
        chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
    except Exception:
        pass


def safe_stop_gimbal(gimbal) -> None:
    try:
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    except Exception:
        pass


def stream_status_text() -> str:
    """Human-readable live status so startup failures are obvious."""
    s = snapshot()
    t = now()
    pose_ok = s.pose_time > 0 and (t - s.pose_time) < 1.0
    gimbal_ok = s.gimbal_time > 0 and (t - s.gimbal_time) < 1.0
    tof_ok = s.tof_time > 0 and s.tof_mm is not None and (t - s.tof_time) < 1.0
    return (
        f"CHASSIS={'OK' if pose_ok else 'WAIT'} "
        f"GIMBAL={'OK' if gimbal_ok else 'WAIT'} "
        f"TOF={'OK' if tof_ok else 'WAIT'}"
    )


def wait_for_chassis_and_tof(timeout_s: float = 6.0) -> None:
    """
    Do NOT wait for gimbal feedback here.

    The old version waited for chassis + gimbal + ToF before issuing the first
    gimbal command. If gimbal feedback had not started yet, the program timed
    out and exited before the turret ever moved.
    """
    deadline = time.monotonic() + timeout_s
    next_print = 0.0
    while time.monotonic() < deadline:
        stop_requested()
        s = snapshot()
        t = now()
        pose_ok = s.pose_time > 0 and (t - s.pose_time) < 1.0
        tof_ok = s.tof_time > 0 and s.tof_mm is not None and (t - s.tof_time) < 1.0

        if pose_ok and tof_ok:
            print(f"[STREAM] {stream_status_text()}")
            return

        if time.monotonic() >= next_print:
            print(f"[STREAM] waiting | {stream_status_text()}")
            next_print = time.monotonic() + 0.5
        time.sleep(0.05)

    raise RuntimeError(
        "Required chassis/ToF stream missing. "
        f"Final status: {stream_status_text()}"
    )


def wait_for_gimbal_feedback(timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        stop_requested()
        s = snapshot()
        if s.gimbal_time > 0 and (now() - s.gimbal_time) < 1.0:
            print(f"[STREAM] gimbal feedback OK yaw={s.gimbal_yaw:+.2f} deg")
            return
        time.sleep(0.05)

    raise RuntimeError(
        "Gimbal angle feedback is missing even after resume/recenter. "
        "The turret command was sent, but 360-degree mapping needs sub_angle() feedback."
    )


def latest_tof_m(max_age_s: float = TOF_STALE_S) -> Optional[float]:
    s = snapshot()
    if s.tof_mm is None or len(s.tof_mm) <= TOF_INDEX:
        return None
    if now() - s.tof_time > max_age_s:
        return None
    mm = s.tof_mm[TOF_INDEX]
    if mm <= 0:
        return None
    m = mm / 1000.0
    if not math.isfinite(m):
        return None
    if m < TOF_MIN_VALID_M or m > TOF_MAX_VALID_M:
        return None
    return m


def tof_median(samples: int = 5, delay_s: float = 0.02) -> Optional[float]:
    vals: List[float] = []
    for _ in range(samples):
        stop_requested()
        d = latest_tof_m()
        if d is not None:
            vals.append(d)
        time.sleep(delay_s)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=float)))


def wait_gimbal_target(
    gimbal,
    target_yaw: float,
    target_pitch: float = 0.0,
    timeout_s: float = 8.0,
    tol_deg: float = GIMBAL_CENTER_TOL_DEG,
) -> None:
    deadline = time.monotonic() + timeout_s
    stable = 0
    while time.monotonic() < deadline:
        stop_requested()
        s = snapshot()
        ey = abs(angle_diff(target_yaw, s.gimbal_yaw))
        ep = abs(target_pitch - s.gimbal_pitch)
        if ey <= tol_deg and ep <= tol_deg:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        time.sleep(0.03)

    safe_stop_gimbal(gimbal)
    s = snapshot()
    raise RuntimeError(
        f"Gimbal timeout target=({target_yaw:.1f},{target_pitch:.1f}) "
        f"actual=({s.gimbal_yaw:.1f},{s.gimbal_pitch:.1f})"
    )


def gimbal_moveto(
    gimbal,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    speed_dps: float = 60.0,
    timeout_s: float = 8.0,
    verify_yaw_tol_deg: Optional[float] = None,
) -> None:
    """
    Send a real RoboMaster gimbal action and bound the wait with a timeout.

    Action completion is used here instead of requiring the sub_angle callback
    to already exist before the first motor command.
    """
    stop_requested()
    action = gimbal.moveto(
        pitch=int(round(pitch_deg)),
        yaw=int(round(yaw_deg)),
        pitch_speed=int(round(speed_dps)),
        yaw_speed=int(round(speed_dps)),
    )
    ok = action.wait_for_completed(timeout=timeout_s)
    if not ok:
        safe_stop_gimbal(gimbal)
        raise RuntimeError(
            f"Gimbal action timeout target=({yaw_deg:+.1f},{pitch_deg:+.1f})"
        )

    # If angle telemetry exists, verify it. Do not make telemetry a prerequisite
    # for sending the first motion command.
    s = snapshot()
    if s.gimbal_time > 0 and (now() - s.gimbal_time) < 1.0:
        ey = abs(angle_diff(yaw_deg, s.gimbal_yaw))
        ep = abs(pitch_deg - s.gimbal_pitch)

        yaw_verify_tol = (
            GIMBAL_YAW_VERIFY_TOL_DEG
            if verify_yaw_tol_deg is None
            else float(verify_yaw_tol_deg)
        )
        if ey > yaw_verify_tol:
            raise RuntimeError(
                f"Gimbal yaw feedback differs: "
                f"target_yaw={yaw_deg:+.1f} actual_yaw={s.gimbal_yaw:+.1f} "
                f"tol={yaw_verify_tol:.1f}"
            )
        if ey > GIMBAL_YAW_VERIFY_TOL_DEG:
            print(
                f"[GIMBAL WARN] scan-start yaw target={yaw_deg:+.1f} "
                f"feedback={s.gimbal_yaw:+.1f}; using REAL feedback as sweep start"
            )

        if ep > GIMBAL_PITCH_HARD_TOL_DEG:
            raise RuntimeError(
                f"Gimbal pitch feedback differs too much: "
                f"target_pitch={pitch_deg:+.1f} actual_pitch={s.gimbal_pitch:+.1f}"
            )

        if ep > GIMBAL_PITCH_WARN_DEG:
            print(
                f"[GIMBAL WARN] pitch target={pitch_deg:+.1f} "
                f"feedback={s.gimbal_pitch:+.1f} (yaw is OK; continuing)"
            )


def recenter_gimbal(gimbal) -> None:
    safe_stop_gimbal(gimbal)
    stop_requested()
    action = gimbal.recenter(pitch_speed=90, yaw_speed=90)
    ok = action.wait_for_completed(timeout=7.0)
    if not ok:
        safe_stop_gimbal(gimbal)
        raise RuntimeError("Gimbal recenter timed out")
    time.sleep(GIMBAL_SETTLE_S)


def center_gimbal_closed_loop(
    gimbal,
    chassis=None,
    chassis_yaw_ref: Optional[float] = None,
    *,
    timeout_s: float = CENTER_GIMBAL_TIMEOUT_S,
) -> Tuple[float, float]:
    """Center gimbal with drive_speed() instead of an SDK Action.

    Intended for the end of a continuous speed-mode scan.  If ``chassis`` and
    ``chassis_yaw_ref`` are supplied, the base yaw is held at the same time so
    the return sweep does not rotate the whole robot.

    Returns final (yaw_error_deg, pitch_error_deg).  Yaw is safety-critical for
    navigation.  Moderate residual pitch is tolerated because some real EP
    gimbals report a few degrees of pitch offset at mechanical center.
    """
    deadline = time.monotonic() + timeout_s
    stable = 0
    last_print = 0.0
    last_motion_check_t = time.monotonic()
    last_motion_yaw: Optional[float] = None

    safe_stop_gimbal(gimbal)
    if chassis is not None:
        safe_stop_chassis(chassis)
    time.sleep(0.08)

    try:
        while time.monotonic() < deadline:
            stop_requested()
            st = snapshot()

            if st.gimbal_time <= 0 or (now() - st.gimbal_time) > CENTER_GIMBAL_TELEMETRY_STALE_S:
                raise RuntimeError("Gimbal telemetry stale while centering")

            # IMPORTANT: gimbal yaw is a mechanical-axis angle, not a wrapped
            # world heading. RoboMaster telemetry can legitimately report values
            # beyond +/-180 deg (for example +180.2 ... +255). Using angle_diff()
            # here causes a sign flip at 180 deg: +180.2 would incorrectly become
            # a +179.8 deg command and drive farther toward the mechanical stop.
            # For centering, the only correct target is the physical zero axis, so
            # use the RAW signed error directly.
            yaw_err = -float(st.gimbal_yaw)
            pitch_err = -float(st.gimbal_pitch)

            if abs(yaw_err) <= CENTER_GIMBAL_YAW_TOL_DEG:
                yaw_speed = 0.0
            else:
                yaw_mag = clamp(
                    abs(yaw_err) * CENTER_GIMBAL_YAW_KP,
                    CENTER_GIMBAL_MIN_DPS,
                    CENTER_GIMBAL_MAX_YAW_DPS,
                )
                yaw_speed = math.copysign(yaw_mag, yaw_err)

            if abs(pitch_err) <= CENTER_GIMBAL_PITCH_TOL_DEG:
                pitch_speed = 0.0
            else:
                pitch_mag = clamp(
                    abs(pitch_err) * CENTER_GIMBAL_PITCH_KP,
                    CENTER_GIMBAL_MIN_DPS,
                    CENTER_GIMBAL_MAX_PITCH_DPS,
                )
                pitch_speed = math.copysign(pitch_mag, pitch_err)

            # Continuous speed command: no ActionDispatcher object is created.
            gimbal.drive_speed(
                pitch_speed=pitch_speed,
                yaw_speed=yaw_speed,
            )

            chassis_err = 0.0
            z_cmd = 0.0
            if chassis is not None and chassis_yaw_ref is not None:
                chassis_err = angle_diff(chassis_yaw_ref, st.yaw)
                if abs(chassis_err) > SCAN_YAW_HOLD_DEADBAND_DEG:
                    z_mag = clamp(
                        abs(chassis_err) * SCAN_YAW_HOLD_KP,
                        SCAN_YAW_HOLD_MIN_DPS,
                        SCAN_YAW_HOLD_MAX_DPS,
                    )
                    z_cmd = YAW_CMD_SIGN * math.copysign(z_mag, chassis_err)
                chassis.drive_speed(x=0, y=0, z=z_cmd, timeout=0.15)

            if (
                abs(yaw_err) <= CENTER_GIMBAL_YAW_TOL_DEG
                and abs(pitch_err) <= CENTER_GIMBAL_PITCH_TOL_DEG
            ):
                stable += 1
                if stable >= CENTER_GIMBAL_STABLE_COUNT:
                    break
            else:
                stable = 0

            t = time.monotonic()
            if t - last_print >= 0.40:
                print(
                    f"[CENTER RAW] gyaw={st.gimbal_yaw:+6.1f} yaw_err={yaw_err:+6.1f} "
                    f"gpitch={st.gimbal_pitch:+5.1f} pitch_err={pitch_err:+5.1f} "
                    f"chassis_err={chassis_err:+4.1f}"
                )
                last_print = t

            # Mechanical-stall guard. If a substantial yaw command is being
            # requested but telemetry does not move for ~1 s, stop instead of
            # pushing continuously against a gimbal limit.
            if last_motion_yaw is None:
                last_motion_yaw = float(st.gimbal_yaw)
                last_motion_check_t = t
            elif t - last_motion_check_t >= 1.0:
                moved = abs(float(st.gimbal_yaw) - last_motion_yaw)
                if abs(yaw_err) > 8.0 and abs(yaw_speed) >= CENTER_GIMBAL_MIN_DPS and moved < 0.8:
                    raise RuntimeError(
                        f"Gimbal appears stalled while centering: yaw={st.gimbal_yaw:+.1f}, "
                        f"error={yaw_err:+.1f}, cmd={yaw_speed:+.1f} dps"
                    )
                last_motion_yaw = float(st.gimbal_yaw)
                last_motion_check_t = t

            time.sleep(CENTER_GIMBAL_LOOP_S)

    finally:
        safe_stop_gimbal(gimbal)
        if chassis is not None:
            safe_stop_chassis(chassis)
        time.sleep(0.10)

    final = snapshot()
    # Same rule as the control loop above: do not wrap gimbal mechanical yaw.
    yaw_err = -float(final.gimbal_yaw)
    pitch_err = -float(final.gimbal_pitch)

    if abs(yaw_err) > GIMBAL_YAW_VERIFY_TOL_DEG:
        raise RuntimeError(
            f"Closed-loop gimbal center failed: yaw={final.gimbal_yaw:+.1f} "
            f"(error={yaw_err:+.1f} deg)"
        )

    if abs(pitch_err) > CENTER_GIMBAL_PITCH_TOL_DEG:
        print(
            f"[GIMBAL WARN] centered yaw OK but pitch remains "
            f"{final.gimbal_pitch:+.1f} deg; continuing"
        )

    print(
        f"[CENTER DONE] gimbal_yaw={final.gimbal_yaw:+.2f} "
        f"gimbal_pitch={final.gimbal_pitch:+.2f}"
    )
    return yaw_err, pitch_err


def rotate_to_yaw(
    chassis,
    target_yaw_deg: float,
    timeout_s: float = 6.0,
    max_dps: float = TURN_MAX_DPS,
    tol_deg: float = TURN_TOL_DEG,
) -> float:
    """Closed-loop chassis yaw controller using the RoboMaster IMU."""
    global YAW_CMD_SIGN

    deadline = time.monotonic() + timeout_s
    stable = 0
    try:
        while time.monotonic() < deadline:
            stop_requested()
            s = snapshot()
            err = angle_diff(target_yaw_deg, s.yaw)

            if abs(err) <= tol_deg:
                stable += 1
                safe_stop_chassis(chassis)
                if stable >= TURN_STABLE_COUNT:
                    return err
                time.sleep(0.04)
                continue

            stable = 0
            mag = clamp(abs(err) * TURN_KP, TURN_MIN_DPS, max_dps)
            z_cmd = YAW_CMD_SIGN * math.copysign(mag, err)
            chassis.drive_speed(x=0, y=0, z=z_cmd, timeout=0.16)
            time.sleep(0.035)
    finally:
        safe_stop_chassis(chassis)

    return angle_diff(target_yaw_deg, snapshot().yaw)


def calibrate_yaw_sign(chassis) -> None:
    """Tiny +z pulse to determine RoboMaster command-vs-IMU yaw sign."""
    global YAW_CMD_SIGN

    safe_stop_chassis(chassis)
    time.sleep(0.12)
    y0 = snapshot().yaw

    chassis.drive_speed(x=0, y=0, z=6, timeout=0.25)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.16:
        stop_requested()
        time.sleep(0.02)
    safe_stop_chassis(chassis)
    time.sleep(0.15)

    y1 = snapshot().yaw
    delta = angle_diff(y1, y0)

    if abs(delta) >= 0.20:
        YAW_CMD_SIGN = 1.0 if delta > 0 else -1.0

    print(f"[CAL] +z pulse IMU delta={delta:+.2f} deg -> YAW_CMD_SIGN={YAW_CMD_SIGN:+.0f}")

    residual = rotate_to_yaw(chassis, y0, timeout_s=3.0, max_dps=12.0)
    if abs(residual) > RESTORE_YAW_HARD_FAIL_DEG:
        raise RuntimeError(f"Yaw sign calibration restore failed: residual={residual:+.2f} deg")


# ============================================================
# LOGGING
# ============================================================


class Logger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.tof_file = open(
            os.path.join(output_dir, "tof_log.csv"), "w", newline="", encoding="utf-8"
        )
        self.tof_csv = csv.writer(self.tof_file)
        self.tof_csv.writerow([
            "timestamp",
            "cell_row",
            "cell_col",
            "robot_x_m",
            "robot_y_m",
            "chassis_yaw_deg",
            "gimbal_yaw_deg",
            "corrected_relative_deg",
            "world_ray_deg",
            "tof_m",
        ])

        self.imu_file = open(
            os.path.join(output_dir, "imu_scan_log.csv"), "w", newline="", encoding="utf-8"
        )
        self.imu_csv = csv.writer(self.imu_file)
        self.imu_csv.writerow([
            "timestamp",
            "yaw_before_deg",
            "yaw_after_scan_deg",
            "scan_drift_deg",
            "yaw_after_restore_deg",
            "restore_residual_deg",
        ])

    def log_tof(
        self,
        cell: Tuple[int, int],
        s: RobotState,
        corrected_relative_deg: float,
        distance_m: float,
    ) -> None:
        world_ray = wrap_angle(s.yaw + s.gimbal_yaw)
        self.tof_csv.writerow([
            f"{now():.6f}",
            cell[0],
            cell[1],
            f"{s.x:.5f}",
            f"{s.y:.5f}",
            f"{s.yaw:.3f}",
            f"{s.gimbal_yaw:.3f}",
            f"{corrected_relative_deg:.3f}",
            f"{world_ray:.3f}",
            f"{distance_m:.4f}",
        ])
        self.tof_file.flush()

    def log_imu_scan(
        self,
        yaw_before: float,
        yaw_after_scan: float,
        yaw_after_restore: float,
    ) -> None:
        drift = angle_diff(yaw_after_scan, yaw_before)
        residual = angle_diff(yaw_before, yaw_after_restore)
        self.imu_csv.writerow([
            f"{now():.6f}",
            f"{yaw_before:.3f}",
            f"{yaw_after_scan:.3f}",
            f"{drift:.3f}",
            f"{yaw_after_restore:.3f}",
            f"{residual:.3f}",
        ])
        self.imu_file.flush()

    def save_robot_trajectory(self) -> None:
        path = os.path.join(self.output_dir, "trajectory.csv")
        with TRAJECTORY_LOCK:
            rows = list(TRAJECTORY)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "x_m", "y_m", "yaw_deg"])
            w.writerows(rows)

    def close(self) -> None:
        try:
            self.tof_file.close()
        except Exception:
            pass
        try:
            self.imu_file.close()
        except Exception:
            pass


# ============================================================
# 360-DEG GIMBAL SCAN
# ============================================================


# Each sample = corrected_relative_angle_deg, distance_m
ScanSample = Tuple[float, Optional[float]]


def scan_360(
    chassis,
    gimbal,
    logger: Logger,
    cell: Tuple[int, int],
) -> List[ScanSample]:
    """
    Smoothly sweep -180 -> +180 while ACTIVELY holding chassis yaw.

    Layer 1: low-gain IMU yaw hold counters turret reaction torque.
    Layer 2: each ToF ray still compensates any residual chassis yaw drift.

    After the sweep, the gimbal recenters first, then the chassis is fine-
    restored to the exact yaw that existed before scanning.
    """
    samples: List[ScanSample] = []

    wait_for_gimbal_feedback(timeout_s=2.0)

    safe_stop_chassis(chassis)
    safe_stop_gimbal(gimbal)
    time.sleep(0.10)

    yaw_ref = snapshot().yaw
    print(f"[SCAN] reference chassis yaw={yaw_ref:+.2f} deg")

    # Move turret to scan start.
    gimbal_moveto(
        gimbal, SCAN_START_DEG, 0.0, speed_dps=90.0,
        verify_yaw_tol_deg=SCAN_START_VERIFY_TOL_DEG,
    )
    actual_scan_start_yaw = snapshot().gimbal_yaw
    print(f"[SCAN] actual start gimbal yaw={actual_scan_start_yaw:+.2f} deg")

    # Positioning to -180 may nudge the base. Restore before the actual sweep.
    residual = rotate_to_yaw(
        chassis,
        yaw_ref,
        timeout_s=4.0,
        max_dps=28.0,
        tol_deg=RESTORE_YAW_TOL_DEG,
    )
    if abs(residual) > RESTORE_YAW_HARD_FAIL_DEG:
        raise RuntimeError(
            f"[SCAN] pre-scan yaw restore failed: {residual:+.2f} deg"
        )

    safe_stop_chassis(chassis)
    time.sleep(0.10)

    # One continuous smooth sweep using SPEED CONTROL, not a Gimbal Action.
    #
    # Why: moveto(+180) creates an SDK Action that can still be action_running
    # for a short time even after angle telemetry has already reached ~+180.
    # Starting recenter() during that window makes RoboMaster SDK throw:
    #   Robot is already performing 1 action(s)
    #
    # drive_speed() is an asynchronous speed command (not ActionDispatcher),
    # so when we stop it there is no outstanding gimbal Action to conflict
    # with the subsequent recenter().
    print(
        f"[SCAN] speed sweep {SCAN_START_DEG:+.0f} -> {SCAN_END_DEG:+.0f} deg "
        f"at {SCAN_SPEED_DPS:.1f} deg/s"
    )
    ok_speed = gimbal.drive_speed(pitch_speed=0, yaw_speed=SCAN_SPEED_DPS)
    if ok_speed is False:
        raise RuntimeError("[SCAN] gimbal.drive_speed() rejected sweep command")

    last_saved_angle: Optional[float] = None
    last_tof_time = -1.0
    started = time.monotonic()
    next_hold_command = time.monotonic()
    max_abs_drift = 0.0

    try:
        while True:
            stop_requested()
            s = snapshot()

            # current - reference
            drift = angle_diff(s.yaw, yaw_ref)
            max_abs_drift = max(max_abs_drift, abs(drift))

            # Active chassis yaw hold at 20 Hz.
            if SCAN_YAW_HOLD_ENABLED and time.monotonic() >= next_hold_command:
                yaw_error = angle_diff(yaw_ref, s.yaw)  # reference - current

                if abs(yaw_error) <= SCAN_YAW_HOLD_DEADBAND_DEG:
                    z_cmd = 0.0
                else:
                    magnitude = clamp(
                        abs(yaw_error) * SCAN_YAW_HOLD_KP,
                        SCAN_YAW_HOLD_MIN_DPS,
                        SCAN_YAW_HOLD_MAX_DPS,
                    )
                    z_cmd = YAW_CMD_SIGN * math.copysign(magnitude, yaw_error)

                chassis.drive_speed(
                    x=0,
                    y=0,
                    z=z_cmd,
                    timeout=max(0.12, SCAN_YAW_HOLD_COMMAND_PERIOD_S * 2.5),
                )
                next_hold_command = (
                    time.monotonic() + SCAN_YAW_HOLD_COMMAND_PERIOD_S
                )

            if abs(drift) > SCAN_YAW_HOLD_ABORT_DEG:
                safe_stop_gimbal(gimbal)
                safe_stop_chassis(chassis)
                raise RuntimeError(
                    f"[SCAN HOLD SAFETY] chassis drift={drift:+.2f} deg "
                    f"despite active yaw hold"
                )

            # Consume each fresh ToF callback once.
            if s.tof_time > last_tof_time:
                last_tof_time = s.tof_time

                # Residual base motion is still compensated in the ray angle.
                corrected_rel = wrap_angle(s.gimbal_yaw + drift)

                if last_saved_angle is None:
                    angular_advance = 999.0
                else:
                    angular_advance = abs(
                        angle_diff(corrected_rel, last_saved_angle)
                    )

                if angular_advance >= SCAN_SAMPLE_STEP_DEG:
                    distance_m: Optional[float] = None

                    if s.tof_mm is not None and len(s.tof_mm) > TOF_INDEX:
                        mm = s.tof_mm[TOF_INDEX]
                        if mm > 0:
                            d = mm / 1000.0
                            if (
                                math.isfinite(d)
                                and TOF_MIN_VALID_M <= d <= TOF_MAX_VALID_M
                            ):
                                distance_m = d

                    samples.append((corrected_rel, distance_m))
                    last_saved_angle = corrected_rel

                    if distance_m is not None:
                        logger.log_tof(cell, s, corrected_rel, distance_m)

                    if len(samples) % 20 == 0:
                        dtext = (
                            "INVALID"
                            if distance_m is None
                            else f"{distance_m:.2f}m"
                        )
                        hold_err = angle_diff(yaw_ref, s.yaw)
                        print(
                            f"[SCAN] g={s.gimbal_yaw:+6.1f} "
                            f"drift={drift:+5.2f} "
                            f"hold_err={hold_err:+5.2f} "
                            f"corr={corrected_rel:+6.1f} "
                            f"tof={dtext} n={len(samples)}"
                        )

            # End only from REAL gimbal angle feedback. Because this sweep is
            # drive_speed()-based, there is no SDK Action left running here.
            if s.gimbal_yaw >= SCAN_END_DEG - 1.0:
                print(
                    f"[SCAN] end angle reached: gimbal={s.gimbal_yaw:+.2f} deg"
                )
                break

            if time.monotonic() - started > SCAN_TIMEOUT_S:
                raise RuntimeError(
                    f"[SCAN] timeout at gimbal_yaw={s.gimbal_yaw:+.1f} deg"
                )

            time.sleep(SCAN_LOOP_S)

    finally:
        # Stop BOTH continuous controllers before starting any recenter action.
        # drive_speed() is not an SDK Action, so after this point the gimbal
        # ActionDispatcher is free for recenter().
        safe_stop_chassis(chassis)
        safe_stop_gimbal(gimbal)
        time.sleep(0.15)

    raw_scan_end_yaw = snapshot().gimbal_yaw
    raw_scan_span = raw_scan_end_yaw - actual_scan_start_yaw
    if raw_scan_span < SCAN_MIN_RAW_SPAN_DEG:
        raise RuntimeError(
            f"[SCAN] raw sweep span only {raw_scan_span:.1f} deg "
            f"(start={actual_scan_start_yaw:+.1f}, end={raw_scan_end_yaw:+.1f})"
        )

    yaw_after_scan = snapshot().yaw
    scan_drift = angle_diff(yaw_after_scan, yaw_ref)

    print(
        f"[SCAN DONE] samples={len(samples)} raw_span={raw_scan_span:.1f}deg "
        f"yaw_drift={scan_drift:+.2f} deg "
        f"max_abs_drift={max_abs_drift:.2f} deg"
    )

    # Return the turret to zero WITHOUT starting a new SDK Action.  Hold the
    # chassis heading at the same time because the return sweep can also impart
    # reaction torque to the base.
    print("[SCAN] closed-loop gimbal center -> yaw=0 / pitch=0")
    center_gimbal_closed_loop(
        gimbal,
        chassis=chassis,
        chassis_yaw_ref=yaw_ref,
    )

    after_recenter_yaw = snapshot().yaw
    print(
        f"[SCAN] after gimbal recenter chassis yaw={after_recenter_yaw:+.2f} "
        f"(ref={yaw_ref:+.2f})"
    )

    # Fine correction before DFS is allowed to choose/drive.
    residual = rotate_to_yaw(
        chassis,
        yaw_ref,
        timeout_s=5.0,
        max_dps=20.0,
        tol_deg=RESTORE_YAW_TOL_DEG,
    )

    yaw_after_restore = snapshot().yaw
    logger.log_imu_scan(
        yaw_ref,
        yaw_after_scan,
        yaw_after_restore,
    )

    print(
        f"[SCAN RESTORE] yaw={yaw_after_restore:+.2f} "
        f"residual={residual:+.2f} deg "
        f"gimbal={snapshot().gimbal_yaw:+.2f}"
    )

    if abs(residual) > RESTORE_YAW_HARD_FAIL_DEG:
        raise RuntimeError(
            f"[SCAN] final yaw restore failed: residual={residual:+.2f} deg"
        )

    if len(samples) < 80:
        raise RuntimeError(
            f"[SCAN] only {len(samples)} samples collected; "
            "360-degree scan is incomplete"
        )

    return samples


# ============================================================
# MAZE DIRECTION CLASSIFICATION
# ============================================================


EDGE_UNKNOWN = "UNKNOWN"
EDGE_OPEN = "OPEN"
EDGE_WALL = "WALL"
EDGE_OBSTACLE = "OBSTACLE"
EDGE_BOUNDARY = "BOUNDARY"

# Logical grid direction based on turn-command space.
DIR_DELTA = {
    0: (-1, 0),
    1: (0, +1),
    2: (+1, 0),
    3: (0, -1),
}

DIR_NAME = {
    0: "FORWARD_AXIS",
    1: "+90_AXIS",
    2: "BACK_AXIS",
    3: "-90_AXIS",
}


def in_bounds(cell: Tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < MAZE_ROWS and 0 <= c < MAZE_COLS


def neighbor(cell: Tuple[int, int], direction: int) -> Tuple[int, int]:
    dr, dc = DIR_DELTA[direction % 4]
    return cell[0] + dr, cell[1] + dc


def opposite(direction: int) -> int:
    return (direction + 2) % 4


def sector_distance(
    scan: Sequence[ScanSample],
    target_relative_deg: float,
    half_width_deg: float = SECTOR_HALF_WIDTH_DEG,
) -> Optional[float]:
    vals: List[float] = []
    for angle_deg, distance_m in scan:
        if distance_m is None:
            continue
        if not math.isfinite(distance_m):
            continue
        if distance_m < TOF_MIN_VALID_M or distance_m > TOF_MAX_VALID_M:
            continue
        if abs(angle_diff(target_relative_deg, angle_deg)) <= half_width_deg:
            vals.append(float(distance_m))

    if not vals:
        return None

    # Central-ray median is robust to one bad ToF sample and is less likely
    # than the old 25th percentile to classify a doorway edge as a wall.
    return float(np.median(np.asarray(vals, dtype=float)))


def classify_distance(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return EDGE_UNKNOWN
    if distance_m <= WALL_MAX_M:
        return EDGE_WALL
    if distance_m >= OPEN_MIN_M:
        return EDGE_OPEN
    return EDGE_OBSTACLE


# ============================================================
# SAFE ONE-CELL DRIVE
# ============================================================


def _rollback_to_cell_start(
    chassis,
    *,
    start_x: float,
    start_y: float,
    yaw_ref: float,
    forward_axis_cos: float,
    forward_axis_sin: float,
    initial_progress: float,
) -> Tuple[bool, str, float]:
    """Reverse along the path we just traversed until we are back at cell start.

    The path behind is the path the robot just drove through, so this is safer
    than leaving the robot physically between two cells while DFS still believes
    it is at the old logical cell.
    """
    if initial_progress < ROLLBACK_MIN_DISTANCE_M:
        return True, "ROLLBACK_NOT_NEEDED", 0.0

    print(f"[ROLLBACK] returning {initial_progress:.3f}m to previous cell center")
    rollback_started = time.monotonic()
    last_progress = initial_progress
    best_progress = initial_progress
    last_improvement_time = rollback_started
    try:
        while True:
            stop_requested()
            s = snapshot()

            dx = s.x - start_x
            dy = s.y - start_y
            forward_progress = (
                dx * forward_axis_cos + dy * forward_axis_sin
            )
            lateral = (
                -dx * forward_axis_sin + dy * forward_axis_cos
            )
            last_progress = forward_progress

            now_rb = time.monotonic()
            if forward_progress <= best_progress - ROLLBACK_PROGRESS_EPS_M:
                best_progress = forward_progress
                last_improvement_time = now_rb

            if forward_progress <= ROLLBACK_TOL_M:
                break

            if now_rb - rollback_started > ROLLBACK_TOTAL_TIMEOUT_S:
                return False, "ROLLBACK_TOTAL_TIMEOUT", forward_progress

            if now_rb - last_improvement_time > ROLLBACK_STALL_TIMEOUT_S:
                return False, "ROLLBACK_STALL_TIMEOUT", forward_progress

            if abs(lateral) > ROLLBACK_LATERAL_ABORT_M:
                return (
                    False,
                    f"ROLLBACK_LATERAL_{lateral:+.3f}m",
                    forward_progress,
                )

            yaw_err = angle_diff(yaw_ref, s.yaw)
            if abs(yaw_err) > DRIVE_YAW_ABORT_DEG:
                return (
                    False,
                    f"ROLLBACK_YAW_{yaw_err:+.2f}deg",
                    forward_progress,
                )

            if abs(yaw_err) <= DRIVE_YAW_DEADBAND_DEG:
                z_cmd = 0.0
            else:
                z_cmd = YAW_CMD_SIGN * clamp(
                    DRIVE_YAW_KP * yaw_err,
                    -DRIVE_MAX_Z_DPS,
                    DRIVE_MAX_Z_DPS,
                )

            chassis.drive_speed(
                x=-ROLLBACK_SPEED_MPS,
                y=0,
                z=z_cmd,
                timeout=0.18,
            )
            time.sleep(MOVE_LOOP_S)
    finally:
        safe_stop_chassis(chassis)
        time.sleep(0.12)

    # Restore the exact heading before the next scan.
    residual = rotate_to_yaw(
        chassis,
        yaw_ref,
        timeout_s=3.0,
        max_dps=15.0,
        tol_deg=RESTORE_YAW_TOL_DEG,
    )
    if abs(residual) > RESTORE_YAW_HARD_FAIL_DEG:
        return False, f"ROLLBACK_YAW_RESIDUAL_{residual:+.2f}deg", last_progress

    s = snapshot()
    pos_err = math.hypot(s.x - start_x, s.y - start_y)
    dx = s.x - start_x
    dy = s.y - start_y
    forward_progress = dx * forward_axis_cos + dy * forward_axis_sin

    # A few cm of odometry noise is acceptable; logical and physical cell are
    # again aligned enough for the next stop-and-scan cycle.
    if pos_err > 0.07 or abs(forward_progress) > 0.06:
        return (
            False,
            f"ROLLBACK_POSITION_ERR_{pos_err:.3f}m",
            forward_progress,
        )

    print(
        f"[ROLLBACK OK] pos_err={pos_err:.3f}m "
        f"forward_residual={forward_progress:+.3f}m "
        f"yaw_residual={residual:+.2f}deg"
    )
    return True, "ROLLBACK_OK", forward_progress


def drive_one_cell(chassis, gimbal, *, trusted_retrace: bool = False) -> Tuple[bool, str, float]:
    """Drive one 0.60-m cell and always keep physical/logical cell synchronized.

    Safety is based on predicted clearance AT THE DESTINATION instead of a fixed
    0.28-m stop threshold. In a 60-cm grid a wall in front of the destination
    cell can legitimately be ~0.2 m from the ToF when the robot reaches center.

    trusted_retrace=True is used only on an edge that was physically traversed
    successfully before. In that mode, the final part of the return move relaxes
    projected-clearance (which is noisy near walls) but retains a hard ToF stop,
    low speed, IMU heading hold and odometry distance verification.
    """
    # Make sure the ToF is pointing forward.
    st_center = snapshot()
    center_gimbal_closed_loop(
        gimbal,
        chassis=chassis,
        chassis_yaw_ref=st_center.yaw,
        timeout_s=5.0,
    )

    pre = tof_median(samples=6)
    if pre is None:
        return False, "TOF_INVALID_BEFORE_MOVE", 0.0

    # For a new passage, require the normal OPEN threshold. For a trusted
    # retrace, successful previous traversal is stronger evidence than a noisy
    # range classification, but still refuse to move if something is already
    # dangerously close in front.
    if trusted_retrace:
        if pre <= TRUSTED_RETRACE_HARD_STOP_M + 0.03:
            return False, f"TRUSTED_RETRACE_FRONT_BLOCKED_{pre:.2f}m", 0.0
    elif pre < OPEN_MIN_M:
        return False, f"PATH_NOT_OPEN_{pre:.2f}m", 0.0

    s0 = snapshot()
    x0, y0 = s0.x, s0.y
    yaw_ref = s0.yaw
    theta = math.radians(yaw_ref)
    c = math.cos(theta)
    sn = math.sin(theta)

    progress = 0.0
    lateral = 0.0
    started = time.monotonic()
    last_progress_watchdog = 0.0
    last_progress_time = started
    reason = "OK"
    last_front = pre
    last_projected = pre - CELL_SIZE_M

    mode_text = "TRUSTED_RETRACE" if trusted_retrace else "NORMAL"
    print(
        f"[MOVE] {CELL_SIZE_M:.2f}m forward | mode={mode_text} "
        f"| yaw_ref={yaw_ref:+.2f} deg | front={pre:.2f}m "
        f"| projected_end={last_projected:.2f}m"
    )

    try:
        while True:
            stop_requested()
            s = snapshot()

            dx = s.x - x0
            dy = s.y - y0
            progress = dx * c + dy * sn
            lateral = -dx * sn + dy * c
            remaining = max(0.0, CELL_SIZE_M - progress)

            if remaining <= MOVE_TOL_M:
                break

            # Progress watchdog: total duration may be longer while crawling near
            # a wall, but a chassis that is physically stuck must still abort.
            t_now = time.monotonic()
            if progress >= last_progress_watchdog + MOVE_PROGRESS_EPS_M:
                last_progress_watchdog = progress
                last_progress_time = t_now

            if t_now - started > MOVE_TOTAL_TIMEOUT_S:
                reason = "MOVE_TOTAL_TIMEOUT"
                break

            if t_now - last_progress_time > MOVE_STALL_TIMEOUT_S:
                reason = "MOVE_STALL_TIMEOUT"
                break

            yaw_err = angle_diff(yaw_ref, s.yaw)
            if abs(yaw_err) > DRIVE_YAW_ABORT_DEG:
                reason = f"YAW_ABORT_{yaw_err:+.2f}deg"
                break

            if abs(lateral) > DRIVE_LATERAL_ABORT_M:
                reason = f"LATERAL_ABORT_{lateral:+.3f}m"
                break

            front = latest_tof_m()
            if front is None:
                reason = "TOF_STALE_OR_OUT_OF_RANGE"
                break

            last_front = front

            # If we moved the entire remaining distance from THIS instant,
            # how much ToF clearance would remain at the destination center?
            projected_end_clearance = front - remaining
            last_projected = projected_end_clearance

            active_hard_stop = (
                TRUSTED_RETRACE_HARD_STOP_M
                if trusted_retrace
                else ABSOLUTE_EMERGENCY_STOP_M
            )

            # v12 near-target cell snap. On a confirmed passage, being only a
            # few cm short of nominal 0.60 m is preferable to reversing almost
            # the entire cell because the far wall has entered the ToF hard-stop
            # range. We stop immediately and accept the destination logical cell
            # without driving any closer to the wall.
            near_target_accept = (
                trusted_retrace
                and remaining <= NEAR_TARGET_ACCEPT_REMAINING_M
                and front >= NEAR_TARGET_ACCEPT_MIN_FRONT_M
                and abs(lateral) <= NEAR_TARGET_ACCEPT_MAX_LATERAL_M
                and abs(yaw_err) <= NEAR_TARGET_ACCEPT_MAX_YAW_ERR_DEG
            )
            if near_target_accept:
                print(
                    f"[CELL SNAP] trusted passage reached {progress:.3f}m/"
                    f"{CELL_SIZE_M:.3f}m; remaining={remaining:.3f}m "
                    f"front={front:.3f}m -> stop here and accept destination cell"
                )
                break

            if front <= active_hard_stop:
                prefix = "TRUSTED_RETRACE_HARD_STOP" if trusted_retrace else "HARD_STOP"
                reason = f"{prefix}_{front:.2f}m"
                break

            # Normal exploration always uses projected destination clearance.
            # On a confirmed retrace, keep that rule until the robot is within
            # the final ~22 cm. In that final zone, use odometry + IMU and crawl
            # while ToF remains above the hard-stop threshold. This prevents the
            # exact dead-end loop seen in v9 where the same known-open return
            # edge repeatedly aborted at 8-11 cm projected clearance.
            relax_projected = (
                trusted_retrace
                and remaining <= TRUSTED_RETRACE_RELAX_REMAINING_M
            )
            if (
                not relax_projected
                and projected_end_clearance < MIN_PROJECTED_FRONT_CLEARANCE_M
            ):
                reason = (
                    f"PROJECTED_CLEARANCE_{projected_end_clearance:.2f}m"
                )
                break

            if trusted_retrace:
                speed = TRUSTED_RETRACE_SLOW_MPS
                if (
                    remaining <= TRUSTED_RETRACE_RELAX_REMAINING_M
                    or front < 0.30
                ):
                    speed = TRUSTED_RETRACE_CRAWL_MPS
            else:
                speed = MOVE_SPEED_MPS
                if (
                    front < FRONT_SLOWDOWN_M
                    or remaining < 0.16
                    or projected_end_clearance < 0.20
                ):
                    speed = MOVE_SLOW_MPS
                if (
                    remaining < 0.14
                    and projected_end_clearance < 0.16
                ):
                    speed = MOVE_CRAWL_MPS

            if abs(yaw_err) <= DRIVE_YAW_DEADBAND_DEG:
                z_cmd = 0.0
            else:
                z_cmd = YAW_CMD_SIGN * clamp(
                    DRIVE_YAW_KP * yaw_err,
                    -DRIVE_MAX_Z_DPS,
                    DRIVE_MAX_Z_DPS,
                )

            chassis.drive_speed(x=speed, y=0, z=z_cmd, timeout=0.18)
            time.sleep(MOVE_LOOP_S)

    finally:
        safe_stop_chassis(chassis)
        time.sleep(0.12)

    if reason != "OK":
        print(
            f"[MOVE ABORT] {reason} progress={progress:.3f}m "
            f"remaining={max(0.0, CELL_SIZE_M-progress):.3f}m "
            f"front={last_front:.3f}m projected_end={last_projected:.3f}m "
            f"lateral={lateral:+.3f}m"
        )

        rollback_ok, rollback_reason, rb_residual = _rollback_to_cell_start(
            chassis,
            start_x=x0,
            start_y=y0,
            yaw_ref=yaw_ref,
            forward_axis_cos=c,
            forward_axis_sin=sn,
            initial_progress=max(0.0, progress),
        )

        if not rollback_ok:
            # This is a hard consistency failure: do not let DFS pretend the
            # robot is still centered in the old logical cell.
            raise RuntimeError(
                f"[POSE SAFETY] move aborted ({reason}) and rollback failed: "
                f"{rollback_reason}, residual={rb_residual:.3f}m"
            )

        return False, f"{reason};{rollback_reason}", max(0.0, progress)

    final = snapshot()
    residual = angle_diff(yaw_ref, final.yaw)
    if abs(residual) > RESTORE_YAW_TOL_DEG:
        rotate_residual = rotate_to_yaw(
            chassis,
            yaw_ref,
            timeout_s=3.0,
            max_dps=15.0,
            tol_deg=RESTORE_YAW_TOL_DEG,
        )
        if abs(rotate_residual) > RESTORE_YAW_HARD_FAIL_DEG:
            # We have physically entered the next cell. Stop rather than claim
            # a successful centered pose with a bad heading.
            raise RuntimeError(
                f"[POSE SAFETY] reached next cell but post-move yaw restore "
                f"failed: {rotate_residual:+.2f}deg"
            )

    print(
        f"[MOVE OK] progress={progress:.3f}m lateral={lateral:+.3f}m "
        f"yaw_err={residual:+.2f}deg "
        f"front={last_front:.3f}m projected_end={last_projected:.3f}m"
    )
    return True, "OK", progress


# ============================================================
# MAZE DFS
# ============================================================


class MazeExplorer:
    """Trémaux maze exploration on the known 4x5 logical cell lattice.

    Localization is deliberately split into two layers:
      - metric pose: RoboMaster odometry + IMU, used to verify each physical move
      - logical pose: (row, col, heading), updated ONLY after a successful 0.60 m move

    Each undirected OPEN passage has a Trémaux mark:
      0 = never traversed
      1 = traversed once
      2 = fully handled; normally do not choose again
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current = (START_ROW, START_COL)
        self.heading = START_HEADING % 4
        self.visited = {self.current}
        self.edges: Dict[Tuple[int, int, int], str] = {}

        # Successful physical traversal is stronger evidence than noisy ToF.
        self.confirmed_open_edges = set()

        # Canonical undirected edge -> Trémaux traversal count (0/1/2).
        self.edge_marks: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]], int
        ] = {}

        # Canonical edge -> consecutive/accumulated failed center-to-center
        # traversal attempts. Used as a loop guard; reset after success.
        self.edge_failures: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]], int
        ] = {}

        # v12: temporary edge quarantine. A failed passage is NOT treated as a
        # permanent wall. The robot remains centered, rescans, and searches for
        # another frontier while this edge cools down for a few scan cycles.
        self.scan_cycle = 0
        self.temp_blocked_until: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]], int
        ] = {}
        self.no_frontier_scan_streak = 0

        self.cell_log: List[Tuple[float, int, int, int, str]] = [
            (now(), self.current[0], self.current[1], self.heading, "START")
        ]

    def _edge_key(self, cell: Tuple[int, int], direction: int):
        d = direction % 4
        nb = neighbor(cell, d)
        if not in_bounds(nb):
            return None
        return tuple(sorted((tuple(cell), tuple(nb))))

    def edge_mark(self, cell: Tuple[int, int], direction: int) -> int:
        key = self._edge_key(cell, direction)
        if key is None:
            return 2
        return int(self.edge_marks.get(key, 0))

    def increment_edge_mark(self, cell: Tuple[int, int], direction: int) -> int:
        key = self._edge_key(cell, direction)
        if key is None:
            return 2
        old = int(self.edge_marks.get(key, 0))
        new = min(2, old + 1)
        self.edge_marks[key] = new
        nb = neighbor(cell, direction)
        print(f"[TREMAUX MARK] {cell}<->{nb}: {old} -> {new}")
        return new

    def edge_failure_count(self, cell: Tuple[int, int], direction: int) -> int:
        key = self._edge_key(cell, direction)
        if key is None:
            return MAX_EDGE_MOVE_FAILURES
        return int(self.edge_failures.get(key, 0))

    def record_edge_failure(self, cell: Tuple[int, int], direction: int) -> int:
        key = self._edge_key(cell, direction)
        if key is None:
            return MAX_EDGE_MOVE_FAILURES
        new = int(self.edge_failures.get(key, 0)) + 1
        self.edge_failures[key] = new
        nb = neighbor(cell, direction)
        print(f"[LOOP GUARD] move failure {cell}<->{nb}: {new}/{MAX_EDGE_MOVE_FAILURES}")
        return new

    def clear_edge_failures(self, cell: Tuple[int, int], direction: int) -> None:
        key = self._edge_key(cell, direction)
        if key is not None:
            self.edge_failures.pop(key, None)
            self.temp_blocked_until.pop(key, None)

    def quarantine_edge(self, cell: Tuple[int, int], direction: int, failures: int) -> None:
        key = self._edge_key(cell, direction)
        if key is None:
            return
        cycles = min(
            TEMP_BLOCK_MAX_RESCAN_CYCLES,
            TEMP_BLOCK_BASE_RESCAN_CYCLES + max(0, failures - 1),
        )
        until = self.scan_cycle + cycles
        self.temp_blocked_until[key] = max(
            int(self.temp_blocked_until.get(key, 0)),
            int(until),
        )
        nb = neighbor(cell, direction)
        print(
            f"[FRONTIER BLOCK] {cell}<->{nb} quarantined for {cycles} "
            f"full scan cycle(s); retry after scan_cycle>={until}"
        )

    def is_temporarily_blocked(self, cell: Tuple[int, int], direction: int) -> bool:
        key = self._edge_key(cell, direction)
        if key is None:
            return True
        return self.scan_cycle < int(self.temp_blocked_until.get(key, 0))

    def blocked_cycles_remaining(self, cell: Tuple[int, int], direction: int) -> int:
        key = self._edge_key(cell, direction)
        if key is None:
            return 0
        return max(0, int(self.temp_blocked_until.get(key, 0)) - self.scan_cycle)

    def is_confirmed_open(self, cell: Tuple[int, int], direction: int) -> bool:
        key = self._edge_key(cell, direction)
        return key is not None and key in self.confirmed_open_edges

    def set_edge(self, cell: Tuple[int, int], direction: int, status: str) -> None:
        d = direction % 4
        nb = neighbor(cell, d)
        key = self._edge_key(cell, d)

        # A passage physically traversed successfully can never be rewritten as
        # WALL by a later noisy sector scan.
        if key is not None and key in self.confirmed_open_edges and status != EDGE_OPEN:
            print(
                f"[EDGE LOCK] {cell}->{nb} already traversed OPEN; "
                f"ignoring new status={status}"
            )
            status = EDGE_OPEN

        self.edges[(cell[0], cell[1], d)] = status
        if in_bounds(nb):
            self.edges[(nb[0], nb[1], opposite(d))] = status

    def mark_traversed_open(self, cell: Tuple[int, int], direction: int) -> None:
        d = direction % 4
        nb = neighbor(cell, d)
        if not in_bounds(nb):
            return
        key = self._edge_key(cell, d)
        if key is not None:
            self.confirmed_open_edges.add(key)
        self.edges[(cell[0], cell[1], d)] = EDGE_OPEN
        self.edges[(nb[0], nb[1], opposite(d))] = EDGE_OPEN

    def get_edge(self, cell: Tuple[int, int], direction: int) -> str:
        return self.edges.get((cell[0], cell[1], direction % 4), EDGE_UNKNOWN)

    def update_from_scan(self, scan: Sequence[ScanSample]) -> None:
        relative_dirs = [
            (0.0, 0),
            (+90.0, +1),
            (180.0, +2),
            (-90.0, -1),
        ]

        for rel_angle, rel_step in relative_dirs:
            direction = (self.heading + rel_step) % 4
            nb = neighbor(self.current, direction)

            if not in_bounds(nb):
                self.set_edge(self.current, direction, EDGE_BOUNDARY)
                print(f"[EDGE] {self.current} {DIR_NAME[direction]} -> BOUNDARY")
                continue

            d = sector_distance(scan, rel_angle)
            observed_status = classify_distance(d)
            previous_status = self.get_edge(self.current, direction)

            # v12 persistent-map rule: an INVALID/UNKNOWN observation carries no
            # evidence that a previously mapped passage changed. Preserve the
            # last known OPEN/WALL state instead of erasing the map and losing
            # a frontier simply because one fast 360 scan missed samples.
            status = observed_status
            if (
                observed_status == EDGE_UNKNOWN
                and previous_status in (EDGE_OPEN, EDGE_WALL)
            ):
                status = previous_status
                print(
                    f"[MAP MEMORY] cell={self.current} dir={direction}: "
                    f"scan=UNKNOWN/INVALID -> keep previous {previous_status}"
                )

            self.set_edge(self.current, direction, status)
            dtext = "INVALID" if d is None else f"{d:.2f}m"
            print(
                f"[EDGE] cell={self.current} rel={rel_angle:+.0f} "
                f"dir={direction} -> {status} ({dtext}) "
                f"mark={self.edge_mark(self.current, direction)}"
            )

    def _turn_preference(self, direction: int) -> int:
        """Lower is preferred: straight, right, left, back."""
        rel = (direction - self.heading) % 4
        return {0: 0, 1: 1, 3: 2, 2: 3}.get(rel, 4)

    def _direction_between(self, a: Tuple[int, int], b: Tuple[int, int]) -> Optional[int]:
        for d in range(4):
            if neighbor(a, d) == b:
                return d
        return None

    def _local_new_passages(self):
        """OPEN mark-0 passages at the current cell, excluding quarantined edges."""
        candidates = []
        for direction in range(4):
            nb = neighbor(self.current, direction)
            if not in_bounds(nb):
                continue
            if self.get_edge(self.current, direction) != EDGE_OPEN:
                continue
            if self.is_temporarily_blocked(self.current, direction):
                continue
            mark = self.edge_mark(self.current, direction)
            if mark != 0:
                continue
            visited_rank = 0 if nb not in self.visited else 1
            turn_rank = self._turn_preference(direction)
            candidates.append((visited_rank, turn_rank, direction, nb, mark))
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates

    def _cell_has_frontier(self, cell: Tuple[int, int]) -> bool:
        """A frontier is a known OPEN passage that has never been traversed."""
        for d in range(4):
            nb = neighbor(cell, d)
            if not in_bounds(nb):
                continue
            if self.get_edge(cell, d) != EDGE_OPEN:
                continue
            if self.edge_mark(cell, d) != 0:
                continue
            if self.is_temporarily_blocked(cell, d):
                continue
            return True
        return False

    def _confirmed_graph_neighbors(self, cell: Tuple[int, int]):
        """Known-safe transit graph: only passages physically traversed before."""
        out = []
        for d in range(4):
            nb = neighbor(cell, d)
            if not in_bounds(nb):
                continue
            if not self.is_confirmed_open(cell, d):
                continue
            if self.is_temporarily_blocked(cell, d):
                continue
            out.append((d, nb))
        return out

    def route_to_nearest_frontier(self) -> Optional[List[Tuple[int, int]]]:
        """BFS through confirmed OPEN graph to nearest visited cell with a frontier.

        This is the SLAM/exploration part missing from strict local Trémaux:
        when the current junction has no new branch, use the map already built
        to navigate toward another known frontier rather than blindly forcing
        one specific back edge.
        """
        from collections import deque

        q = deque([self.current])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.current: None}
        target = None

        while q:
            cell = q.popleft()
            if cell != self.current and self._cell_has_frontier(cell):
                target = cell
                break
            for _d, nb in self._confirmed_graph_neighbors(cell):
                if nb in parent:
                    continue
                if nb not in self.visited:
                    continue
                parent[nb] = cell
                q.append(nb)

        if target is None:
            return None

        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def has_temporarily_blocked_passage(self) -> bool:
        for d in range(4):
            nb = neighbor(self.current, d)
            if not in_bounds(nb):
                continue
            if self.get_edge(self.current, d) == EDGE_OPEN and self.is_temporarily_blocked(self.current, d):
                return True
        # Also count a blocked edge in the known transit graph somewhere else;
        # another scan cycle may release it and make a frontier reachable again.
        return any(self.scan_cycle < until for until in self.temp_blocked_until.values())

    def choose_exploration_move(self) -> Optional[Tuple[int, Tuple[int, int], int, str]]:
        # 1) Highest priority: a new local passage (frontier immediately here).
        local_new = self._local_new_passages()
        if local_new:
            _visited_rank, _turn_rank, direction, nb, mark = local_new[0]
            print(
                f"[FRONTIER] local unexplored passage {self.current}->{nb} "
                f"mark={mark}"
            )
            return direction, nb, mark, "LOCAL_FRONTIER"

        # 2) No local new branch: route through the map to the nearest frontier.
        route = self.route_to_nearest_frontier()
        if route is not None and len(route) >= 2:
            nb = route[1]
            direction = self._direction_between(self.current, nb)
            if direction is not None:
                mark = self.edge_mark(self.current, direction)
                print(
                    f"[FRONTIER ROUTE] nearest frontier route={route}; "
                    f"next={self.current}->{nb} mark={mark}"
                )
                return direction, nb, mark, "ROUTE_TO_FRONTIER"

        # 3) Trémaux fallback: use a once-traversed OPEN edge to back out of a
        # dead end even if the target frontier is not yet visible in the graph.
        fallback = []
        for direction in range(4):
            nb = neighbor(self.current, direction)
            if not in_bounds(nb):
                continue
            if self.get_edge(self.current, direction) != EDGE_OPEN:
                continue
            if self.is_temporarily_blocked(self.current, direction):
                continue
            mark = self.edge_mark(self.current, direction)
            if mark != 1:
                continue
            failures = self.edge_failure_count(self.current, direction)
            fallback.append((failures, self._turn_preference(direction), direction, nb, mark))

        if fallback:
            fallback.sort(key=lambda x: (x[0], x[1]))
            failures, _turn_rank, direction, nb, mark = fallback[0]
            print(
                f"[TREMAUX BACKTRACK] {self.current}->{nb} mark=1 "
                f"failures={failures}"
            )
            return direction, nb, mark, "TREMAUX_BACKTRACK"

        return None

    def turn_to_heading(self, chassis, target_heading: int) -> None:
        target_heading %= 4
        delta_steps = (target_heading - self.heading) % 4
        if delta_steps == 3:
            delta_steps = -1
        elif delta_steps == 2:
            delta_steps = 2

        if delta_steps == 0:
            return

        before = snapshot().yaw
        cmd_deg = 90.0 * delta_steps
        target_imu = wrap_angle(before + cmd_deg * YAW_CMD_SIGN)
        print(
            f"[TURN] heading {self.heading}->{target_heading} "
            f"command-equivalent={cmd_deg:+.0f} deg IMU target={target_imu:+.2f}"
        )

        residual = rotate_to_yaw(
            chassis,
            target_imu,
            timeout_s=7.0,
            max_dps=TURN_MAX_DPS,
            tol_deg=TURN_TOL_DEG,
        )
        if abs(residual) > RESTORE_YAW_HARD_FAIL_DEG:
            raise RuntimeError(f"Turn failed, residual={residual:+.2f} deg")

        self.heading = target_heading

    def record(self, event: str) -> None:
        self.cell_log.append(
            (now(), self.current[0], self.current[1], self.heading, event)
        )

    def save(self) -> None:
        with open(
            os.path.join(self.output_dir, "maze_cell_trajectory.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "row", "col", "heading", "event"])
            w.writerows(self.cell_log)

        edge_rows = []
        for (r, c, d), status in sorted(self.edges.items()):
            nr, nc = neighbor((r, c), d)
            edge_rows.append({
                "row": r,
                "col": c,
                "direction": d,
                "direction_name": DIR_NAME[d],
                "neighbor_row": nr,
                "neighbor_col": nc,
                "status": status,
                "tremaux_mark": self.edge_mark((r, c), d) if in_bounds((nr, nc)) else None,
            })

        tremaux_rows = []
        for (a, b), mark in sorted(self.edge_marks.items()):
            tremaux_rows.append({
                "a": list(a),
                "b": list(b),
                "marks": int(mark),
            })

        st = snapshot()
        logical_offset_x = (self.current[1] - START_COL) * CELL_SIZE_M
        logical_offset_y = (START_ROW - self.current[0]) * CELL_SIZE_M

        payload = {
            "exploration_algorithm": "Frontier-Trémaux graph exploration with 360-degree rescanning",
            "localization": "logical cell (row,col,heading) + RoboMaster odometry/IMU verification",
            "scan_cycle": self.scan_cycle,
            "no_frontier_scan_streak": self.no_frontier_scan_streak,
            "maze_rows": MAZE_ROWS,
            "maze_cols": MAZE_COLS,
            "cell_size_m": CELL_SIZE_M,
            "start_cell": [START_ROW, START_COL],
            "current_cell": list(self.current),
            "current_heading": self.heading,
            "logical_offset_from_start_m": [logical_offset_x, logical_offset_y],
            "metric_pose": [st.x, st.y, st.yaw],
            "visited_cells": sorted([list(c) for c in self.visited]),
            "visited_count": len(self.visited),
            "total_cells": MAZE_ROWS * MAZE_COLS,
            "coverage_percent": 100.0 * len(self.visited) / (MAZE_ROWS * MAZE_COLS),
            "confirmed_open_edges": [
                [list(a), list(b)]
                for a, b in sorted(self.confirmed_open_edges)
            ],
            "tremaux_edges": tremaux_rows,
            "edge_move_failures": [
                {"a": list(a), "b": list(b), "failures": int(n)}
                for (a, b), n in sorted(self.edge_failures.items())
            ],
            "temporarily_blocked_edges": [
                {
                    "a": list(a),
                    "b": list(b),
                    "blocked_until_scan_cycle": int(until),
                    "remaining_scan_cycles": max(0, int(until) - self.scan_cycle),
                }
                for (a, b), until in sorted(self.temp_blocked_until.items())
                if self.scan_cycle < int(until)
            ],
            "edges": edge_rows,
        }

        with open(
            os.path.join(self.output_dir, "maze_state.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def run(self, chassis, gimbal, logger: Logger) -> None:
        print(
            f"[MAZE/FRONTIER] {MAZE_COLS}x{MAZE_ROWS}, cell={CELL_SIZE_M:.2f}m, "
            f"start={self.current}, heading={self.heading}"
        )
        print(
            "[EXPLORE] policy = scan 360 -> update map -> local frontier -> "
            "persistent-map local frontier -> route to nearest frontier -> Trémaux backtrack; INVALID scans preserve prior map; blocked edges trigger rescan/replan"
        )

        for action in range(1, MAX_ACTIONS + 1):
            stop_requested()
            self.scan_cycle += 1
            metric = snapshot()
            print(
                f"\n===== ACTION {action}/{MAX_ACTIONS} | scan_cycle={self.scan_cycle} | "
                f"cell={self.current} heading={self.heading} "
                f"visited={len(self.visited)}/{MAZE_ROWS * MAZE_COLS} "
                f"metric=({metric.x:.2f},{metric.y:.2f},{metric.yaw:+.1f}) ====="
            )

            # Exploration always stops at a cell center and observes before deciding.
            safe_stop_chassis(chassis)
            time.sleep(0.10)
            print("[EXPLORE] STOP -> 360 SCAN -> MAP UPDATE -> REPLAN")

            scan = scan_360(chassis, gimbal, logger, self.current)
            self.update_from_scan(scan)

            if len(self.visited) >= MAZE_ROWS * MAZE_COLS:
                print("[FRONTIER] all 20 logical cells visited -> exploration complete")
                self.record("COMPLETE_ALL_CELLS")
                break

            choice = self.choose_exploration_move()
            if choice is None:
                self.no_frontier_scan_streak += 1

                if self.has_temporarily_blocked_passage():
                    print(
                        "[EXPLORE WAIT] no usable route right now, but at least one "
                        "passage is temporarily quarantined. Staying at cell center "
                        "and rescanning 360 on the next cycle instead of stopping."
                    )
                    self.record("RESCAN_WAIT_TEMP_BLOCK")
                    self.save()
                    logger.save_robot_trajectory()
                    time.sleep(0.25)
                    continue

                if self.no_frontier_scan_streak < FRONTIER_COMPLETE_CONFIRM_SCANS:
                    print(
                        f"[FRONTIER VERIFY] no reachable frontier found "
                        f"({self.no_frontier_scan_streak}/{FRONTIER_COMPLETE_CONFIRM_SCANS}); "
                        "remain stopped and perform another full 360 scan to confirm."
                    )
                    self.record("RESCAN_VERIFY_NO_FRONTIER")
                    self.save()
                    logger.save_robot_trajectory()
                    time.sleep(0.20)
                    continue

                print(
                    "[FRONTIER COMPLETE] repeated 360 scans found no reachable "
                    "OPEN mark-0 frontier and no temporary blocked passage. "
                    "All currently reachable maze passages are explored."
                )
                self.record("COMPLETE_NO_REACHABLE_FRONTIER")
                break

            self.no_frontier_scan_streak = 0
            target_heading, nb, mark_before, plan_mode = choice
            old_cell = self.current

            trusted_retrace = (
                nb in self.visited
                and self.is_confirmed_open(old_cell, target_heading)
            )
            if trusted_retrace:
                print(
                    f"[MAP TRANSIT] {old_cell}->{nb}: confirmed OPEN; "
                    f"plan={plan_mode}, mark={mark_before}"
                )

            self.turn_to_heading(chassis, target_heading)

            ok, reason, travelled = drive_one_cell(
                chassis,
                gimbal,
                trusted_retrace=trusted_retrace,
            )
            if ok:
                self.clear_edge_failures(old_cell, target_heading)
                self.mark_traversed_open(old_cell, target_heading)
                mark_after = self.increment_edge_mark(old_cell, target_heading)
                self.current = nb
                self.visited.add(nb)
                self.record(
                    f"EXPLORE_MOVE:{plan_mode}:m{mark_before}->{mark_after}:"
                    f"from={old_cell}:to={nb}"
                )
                print(
                    f"[EXPLORE] entered {self.current}, odom={travelled:.3f}m, "
                    f"edge_mark={mark_after}; next action will stop and scan 360 again"
                )
            else:
                # Rollback already put the robot back at the exact old cell center.
                # Do NOT force the same return edge and do NOT terminate the whole
                # mission. Quarantine it briefly, then the next action rescans all
                # directions and replans from the updated map.
                failures = self.record_edge_failure(old_cell, target_heading)
                self.quarantine_edge(old_cell, target_heading, failures)

                # Only an edge never physically traversed before may be treated as
                # a suspected temporary obstacle. Confirmed passages remain OPEN in
                # the map and are simply unavailable during quarantine.
                if not self.is_confirmed_open(old_cell, target_heading):
                    self.set_edge(old_cell, target_heading, EDGE_OBSTACLE)

                self.record(
                    f"MOVE_ABORT_RESCAN:{reason}:plan={plan_mode}:failures={failures}"
                )
                print(
                    f"[EXPLORE REPLAN] move failed but rollback succeeded: {reason}. "
                    "Robot remains at the current cell center. Do not retry blindly; "
                    "next cycle performs a fresh 360 scan and searches another path."
                )

            self.save()
            logger.save_robot_trajectory()

        self.save()
        logger.save_robot_trajectory()
        print(
            f"[RESULT] visited={len(self.visited)}/{MAZE_ROWS * MAZE_COLS} "
            f"coverage={100.0 * len(self.visited)/(MAZE_ROWS*MAZE_COLS):.1f}% "
            f"mapped_edges={len(self.edge_marks)} scan_cycles={self.scan_cycle}"
        )


# ============================================================
# MAIN
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="RoboMaster 4x5 ToF gimbal frontier maze explorer")
    p.add_argument(
        "--conn-type",
        choices=["ap", "sta", "rndis"],
        default="ap",
        help="RoboMaster connection type (default: ap)",
    )
    return p.parse_args()


def make_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_ROOT, f"maze_4x5_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    STOP.clear()
    install_signal_handlers()
    args = parse_args()

    output_dir = make_output_dir()
    logger = Logger(output_dir)
    ep_robot = robot.Robot()
    initialized = False

    start_pose: Optional[Tuple[float, float, float]] = None

    try:
        print(f"[CONNECT] conn_type={args.conn_type}")
        ep_robot.initialize(conn_type=args.conn_type)
        initialized = True

        # FREE mode lets gimbal yaw independently of chassis.
        free_mode = getattr(robot, "FREE", "free")
        ep_robot.set_robot_mode(mode=free_mode)

        chassis = ep_robot.chassis
        gimbal = ep_robot.gimbal
        sensor = ep_robot.sensor

        # Wake the physical turret BEFORE waiting for gimbal telemetry.
        # This fixes the previous startup deadlock where the program could time
        # out before sending the first gimbal movement command.
        print("[GIMBAL] resume")
        try:
            gimbal.resume()
        except Exception as exc:
            print(f"[WARN] gimbal.resume() returned error: {exc}")
        time.sleep(0.25)

        print("[SUB] chassis position @20Hz")
        r_pos = chassis.sub_position(cs=1, freq=20, callback=on_position)
        print(f"[SUB RESULT] position={r_pos}")

        print("[SUB] chassis attitude @50Hz")
        r_att = chassis.sub_attitude(freq=50, callback=on_attitude)
        print(f"[SUB RESULT] attitude={r_att}")

        print("[SUB] gimbal angle @50Hz")
        r_gim = gimbal.sub_angle(freq=50, callback=on_gimbal_angle)
        print(f"[SUB RESULT] gimbal={r_gim}")

        print("[SUB] ToF distance @50Hz")
        r_tof = sensor.sub_distance(freq=50, callback=on_tof)
        print(f"[SUB RESULT] tof={r_tof}")

        # IMPORTANT: physically command the turret BEFORE waiting for ToF.
        # Therefore, even if the ToF stream is the missing stream, the program
        # can no longer sit still for six seconds without ever testing gimbal motion.
        print("[GIMBAL] startup recenter command")
        recenter_gimbal(gimbal)

        print("[GIMBAL] self-test -20 -> +20 -> center")
        gimbal_moveto(gimbal, -20.0, 0.0, speed_dps=55.0, timeout_s=3.0, verify_yaw_tol_deg=STARTUP_SELFTEST_YAW_TOL_DEG)
        gimbal_moveto(gimbal, +20.0, 0.0, speed_dps=55.0, timeout_s=3.0, verify_yaw_tol_deg=STARTUP_SELFTEST_YAW_TOL_DEG)
        recenter_gimbal(gimbal)

        # Now verify telemetry required by mapping/navigation.
        wait_for_gimbal_feedback(timeout_s=3.0)
        wait_for_chassis_and_tof(timeout_s=6.0)

        calibrate_yaw_sign(chassis)

        s = snapshot()
        start_pose = (s.x, s.y, s.yaw)
        print(
            f"[READY] x={s.x:.3f} y={s.y:.3f} yaw={s.yaw:+.2f} "
            f"gimbal={s.gimbal_yaw:+.2f}"
        )
        print(f"[OUTPUT] {output_dir}")
        print("[SAFETY] v12 Persistent-Frontier SLAM + map memory + near-target cell snap + rollback watchdog enabled; keep Ctrl+C ready")

        explorer = MazeExplorer(output_dir)
        explorer.run(chassis, gimbal, logger)

    except KeyboardInterrupt:
        print("\n[MAIN] interrupted by user")

    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")

    finally:
        if initialized:
            try:
                safe_stop_chassis(ep_robot.chassis)
            except Exception:
                pass
            try:
                safe_stop_gimbal(ep_robot.gimbal)
            except Exception:
                pass

        logger.save_robot_trajectory()

        end = snapshot()
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "maze_rows": MAZE_ROWS,
            "maze_cols": MAZE_COLS,
            "cell_size_m": CELL_SIZE_M,
            "physical_size_m": [MAZE_COLS * CELL_SIZE_M, MAZE_ROWS * CELL_SIZE_M],
            "start_pose": start_pose,
            "end_pose": [end.x, end.y, end.yaw],
            "connection": args.conn_type,
        }
        try:
            with open(
                os.path.join(output_dir, "summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[WARN] summary save failed: {exc}")

        logger.close()

        if initialized:
            try:
                ep_robot.chassis.unsub_position()
            except Exception:
                pass
            try:
                ep_robot.chassis.unsub_attitude()
            except Exception:
                pass
            try:
                ep_robot.gimbal.unsub_angle()
            except Exception:
                pass
            try:
                ep_robot.sensor.unsub_distance()
            except Exception:
                pass
            try:
                ep_robot.close()
            except Exception:
                pass

        print(f"[DONE] logs saved to {output_dir}")


if __name__ == "__main__":
    main()
