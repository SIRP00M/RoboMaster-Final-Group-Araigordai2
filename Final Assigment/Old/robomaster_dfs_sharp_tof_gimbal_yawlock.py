#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoboMaster EP - DFS Maze Explorer
---------------------------------
Sensors:
  Sharp LEFT  : Sensor Adapter ID 2, Port 1
  Sharp RIGHT : Sensor Adapter ID 3, Port 1
  ToF         : CAN bus distance_info[0], mounted on gimbal

Behavior:
  - DFS on a discrete cell graph.
  - Gimbal ToF scans LEFT / FRONT / RIGHT at each new cell.
  - Gimbal returns to FRONT and pitches DOWN 5 deg before chassis motion.
  - Corridor motion uses Sharp sensors for adaptive wall-following.
  - Normally follows one wall instead of forcing exact center.
  - If either side becomes too close, that side gets priority and the robot
    strafes away from it toward the safer part of the corridor.
  - Front ToF is always a collision stop while moving.
  - Chassis yaw hold reduces gradual Z-axis drift.

IMPORTANT:
  Tune CELL_LENGTH_M and TOF_OPEN_THRESHOLD_MM for the real maze geometry.
"""

from robomaster import robot
import math
import statistics
import threading
import time
from collections import deque


# ============================================================
# CONNECTION / SENSOR WIRING
# ============================================================

CONN_TYPE = "ap"

SENSOR_PORT = 1
SHARP_LEFT_ID = 2
SHARP_RIGHT_ID = 3

TOF_INDEX = 0
TOF_FREQ_HZ = 20

POSITION_FREQ_HZ = 20
ATTITUDE_FREQ_HZ = 50


# ============================================================
# SHARP GP2Y0A41SK0F CALIBRATION
# median ADC from user's 100-sample calibration
#
# Only the monotonic/reliable section is used for control.
# Beyond ~24 cm these sensors become noisy/non-monotonic in the
# measured dataset, so values below the last ADC are treated as
# "wall too far / unavailable for wall-follow".
# ============================================================

LEFT_CAL = [
    (4.0, 864.5),
    (6.0, 610.0),
    (8.0, 475.0),
    (10.0, 378.0),
    (12.0, 316.0),
    (14.0, 274.0),
    (16.0, 247.0),
    (18.0, 224.0),
    (20.0, 212.0),
    (22.0, 198.0),
    (24.0, 182.0),
]

RIGHT_CAL = [
    (4.0, 822.0),
    (6.0, 589.0),
    (8.0, 455.0),
    (10.0, 374.0),
    (12.0, 313.0),
    (14.0, 276.0),
    (16.0, 239.0),
    (18.0, 205.0),
    (20.0, 174.0),
    (22.0, 161.0),
    (24.0, 133.0),
]

SHARP_FILTER_SAMPLES = 5
SHARP_MIN_PLAUSIBLE_ADC = 20


# ============================================================
# CORRIDOR CONTROL
# ============================================================

# Normal wall-following distance.
WALL_TARGET_CM = 13.0

# If a wall enters this zone, it gets priority over normal following.
SIDE_PRIORITY_CM = 10.0

# Emergency side distance: strong escape strafe.
SIDE_HARD_CM = 7.0

# Follow controller.
WALL_FOLLOW_KP = 0.025          # m/s per cm error
MAX_FOLLOW_STRAFE_MPS = 0.10

# Priority / emergency escape.
SIDE_ESCAPE_KP = 0.040          # m/s per cm inside priority region
SIDE_ESCAPE_MIN_MPS = 0.055
SIDE_ESCAPE_MAX_MPS = 0.15

# Forward motion.
FORWARD_SPEED_MPS = 0.16
SLOW_FORWARD_SPEED_MPS = 0.10

# If ToF is this close while driving, stop immediately.
FRONT_HARD_STOP_MM = 160

# Begin slowing when front obstacle gets closer than this.
FRONT_SLOW_MM = 330


# ============================================================
# CELL / DFS GEOMETRY
# ============================================================

# Center-to-center distance between maze nodes/cells.
# *** TUNE THIS FOR THE REAL MAZE ***
CELL_LENGTH_M = 0.55

# Reaching this fraction counts as arriving at the next cell if ToF
# sees the end wall slightly earlier than odometry expects.
CELL_SUCCESS_FRACTION = 0.82

# ToF at cell center:
# <= threshold -> WALL
# > threshold  -> OPEN
#
# *** TUNE THIS FOR THE REAL MAZE ***
TOF_OPEN_THRESHOLD_MM = 400

TOF_SCAN_SAMPLES = 7
TOF_SCAN_INTERVAL_SEC = 0.055
GIMBAL_SETTLE_SEC = 0.12

# Gimbal pitch:
# negative = down on RoboMaster convention.
GIMBAL_PITCH_DEG = -5.0
GIMBAL_YAW_STEP_DEG = 45.0
GIMBAL_PITCH_SPEED = 60
GIMBAL_YAW_SPEED = 90

# Root is assumed to start at the maze entrance, with its back outside.
# Therefore root-back is not explored.
ROOT_BACK_IS_WALL = True


# ============================================================
# YAW HOLD / DRIFT CORRECTION
# ============================================================

YAW_HOLD_ENABLED = True

# Moving yaw lock: keep chassis on the logical DFS heading.
YAW_HOLD_KP = 1.8
YAW_HOLD_MAX_DPS = 22.0
YAW_HOLD_DEADBAND_DEG = 0.35

# Stronger yaw lock while the robot is stationary and the gimbal is moving.
# This directly counters reaction torque from the gimbal so the chassis does
# not slowly walk to the left/right during LEFT/FRONT/RIGHT ToF scans.
STATIONARY_YAW_HOLD_KP = 2.8
STATIONARY_YAW_HOLD_MAX_DPS = 28.0
STATIONARY_YAW_HOLD_HZ = 30.0
STATIONARY_SETTLE_SEC = 0.35

# RoboMaster attitude/move convention used by this program:
#   chassis.move(z=+90) = LEFT, chassis.move(z=-90) = RIGHT
# With target-current yaw error, +1.0 gives the corrective sign for that
# convention. If a real robot visibly corrects AWAY from the target, flip it.
YAW_DRIVE_SIGN = 1.0


# ============================================================
# TIMING / SAFETY
# ============================================================

CONTROL_DT = 0.05
DRIVE_COMMAND_TIMEOUT = 0.25
POSITION_WAIT_TIMEOUT = 3.0
MAX_CELL_TIME_SEC = max(5.0, (CELL_LENGTH_M / FORWARD_SPEED_MPS) * 2.5)

DEBUG_MOVE_PRINT_PERIOD_SEC = 0.25


# ============================================================
# DIRECTIONS
#
# 0=N, 1=E, 2=S, 3=W
# Coordinates are logical DFS coordinates only.
# ============================================================

DIR_NAMES = ("N", "E", "S", "W")
DIR_VEC = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}

REL_LEFT = -1
REL_FRONT = 0
REL_RIGHT = 1
REL_BACK = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    for d, (vx, vy) in DIR_VEC.items():
        if (dx, dy) == (vx, vy):
            return d

    raise ValueError(f"Cells are not adjacent: {a} -> {b}")


def neighbor(cell, direction):
    dx, dy = DIR_VEC[direction]
    return (cell[0] + dx, cell[1] + dy)


# ============================================================
# CALIBRATION / SENSOR HELPERS
# ============================================================

def adc_to_cm(adc, calibration):
    """
    Piecewise-linear interpolation using monotonic calibration points.

    Returns:
        float cm : within reliable calibrated region
        None     : ADC says wall is farther than reliable calibration region,
                   sensor value is invalid, or wall is effectively unavailable
                   for wall-follow.
    """
    if adc is None:
        return None

    try:
        adc = float(adc)
    except Exception:
        return None

    if not math.isfinite(adc) or adc < SHARP_MIN_PLAUSIBLE_ADC:
        return None

    # Calibration is near -> far, ADC high -> low.
    near_cm, near_adc = calibration[0]
    far_cm, far_adc = calibration[-1]

    # Closer than nearest calibration point.
    if adc >= near_adc:
        return near_cm

    # Farther than reliable calibrated range.
    if adc < far_adc:
        return None

    for i in range(len(calibration) - 1):
        d1, a1 = calibration[i]
        d2, a2 = calibration[i + 1]

        # a1 >= adc >= a2
        if a1 >= adc >= a2:
            if abs(a1 - a2) < 1e-9:
                return (d1 + d2) * 0.5

            t = (a1 - adc) / (a1 - a2)
            return d1 + t * (d2 - d1)

    return None


# ============================================================
# ROBOT STATE
# ============================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        self.tof_mm = None
        self.position = None          # (x, y, z)
        self.attitude = None          # (yaw, pitch, roll)

    def set_tof(self, value):
        with self.lock:
            self.tof_mm = value

    def get_tof(self):
        with self.lock:
            return self.tof_mm

    def set_position(self, value):
        with self.lock:
            self.position = value

    def get_position(self):
        with self.lock:
            return self.position

    def set_attitude(self, value):
        with self.lock:
            self.attitude = value

    def get_attitude(self):
        with self.lock:
            return self.attitude


# ============================================================
# DFS EXPLORER
# ============================================================

class DFSMazeExplorer:
    def __init__(self):
        self.ep_robot = robot.Robot()

        self.chassis = None
        self.gimbal = None
        self.sensor_adapter = None
        self.distance_sensor = None

        self.state = SharedState()

        self.left_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)
        self.right_adc_hist = deque(maxlen=SHARP_FILTER_SAMPLES)

        # Logical robot heading. Start direction = N.
        self.heading = 0

        # Absolute chassis yaw reference.
        # base_yaw_deg is captured ONCE at startup and is never replaced by
        # a yaw value that may have drifted because of gimbal motion.
        self.base_yaw_deg = None
        self.yaw_ref_deg = None

        # Prevent two yaw-hold loops from commanding the chassis at once.
        self.yaw_hold_lock = threading.Lock()

        # DFS state.
        self.root = (0, 0)
        self.current = self.root
        self.visited = set()
        self.parent = {}
        self.open_dirs = {}       # cell -> ordered list of absolute open dirs
        self.blocked_edges = set()

        self.running = True

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    def tof_callback(self, distance_info):
        try:
            if distance_info and len(distance_info) > TOF_INDEX:
                value = distance_info[TOF_INDEX]
                if value is not None:
                    self.state.set_tof(float(value))
        except Exception:
            pass

    def position_callback(self, position_info):
        try:
            if position_info and len(position_info) >= 3:
                x, y, z = position_info[:3]
                self.state.set_position((float(x), float(y), float(z)))
        except Exception:
            pass

    def attitude_callback(self, attitude_info):
        try:
            if attitude_info and len(attitude_info) >= 3:
                yaw, pitch, roll = attitude_info[:3]
                self.state.set_attitude(
                    (float(yaw), float(pitch), float(roll))
                )
        except Exception:
            pass

    # --------------------------------------------------------
    # CONNECT / CLEANUP
    # --------------------------------------------------------

    def connect(self):
        print("============================================================")
        print(" RoboMaster DFS + Sharp wall-follow + Gimbal ToF")
        print("============================================================")
        print(f" Sharp LEFT : Adapter ID {SHARP_LEFT_ID}, Port {SENSOR_PORT}")
        print(f" Sharp RIGHT: Adapter ID {SHARP_RIGHT_ID}, Port {SENSOR_PORT}")
        print(f" ToF        : CAN distance_info[{TOF_INDEX}]")
        print(f" Gimbal     : front, pitch {GIMBAL_PITCH_DEG:+.1f} deg")
        print("============================================================")

        print("[CONNECT] RoboMaster AP...")
        self.ep_robot.initialize(conn_type=CONN_TYPE)

        # Gimbal must be independent of chassis.
        self.ep_robot.set_robot_mode(mode=robot.FREE)

        self.chassis = self.ep_robot.chassis
        self.gimbal = self.ep_robot.gimbal
        self.sensor_adapter = self.ep_robot.sensor_adaptor
        self.distance_sensor = self.ep_robot.sensor

        print("[SUB] ToF...")
        print("      result =", self.distance_sensor.sub_distance(
            freq=TOF_FREQ_HZ,
            callback=self.tof_callback
        ))

        print("[SUB] chassis position...")
        print("      result =", self.chassis.sub_position(
            freq=POSITION_FREQ_HZ,
            callback=self.position_callback
        ))

        print("[SUB] chassis attitude...")
        print("      result =", self.chassis.sub_attitude(
            freq=ATTITUDE_FREQ_HZ,
            callback=self.attitude_callback
        ))

        # Give telemetry a moment to arrive.
        time.sleep(0.5)

        self.stop_chassis()

        # IMPORTANT: Capture the chassis reference BEFORE moving the gimbal.
        # If the turret kicks the base a little, that disturbed yaw must NOT
        # become the new target.
        t0 = time.monotonic()
        while self.current_yaw() is None and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)

        self.base_yaw_deg = self.current_yaw()
        if self.base_yaw_deg is None:
            raise RuntimeError("No chassis yaw telemetry; cannot initialize yaw lock")

        self.yaw_ref_deg = self.desired_yaw_for_heading(self.heading)
        print(
            f"[YAW LOCK] base={self.base_yaw_deg:+.2f} deg "
            f"heading={DIR_NAMES[self.heading]} "
            f"target={self.yaw_ref_deg:+.2f} deg"
        )

        self.gimbal_front_down()
        self.hold_heading_stationary(STATIONARY_SETTLE_SEC)

        print("[READY] Connected.")

    def cleanup(self):
        print("\n[CLEANUP] stopping robot...")
        self.running = False

        try:
            self.stop_chassis()
        except Exception:
            pass

        try:
            self.gimbal.recenter(
                pitch_speed=GIMBAL_PITCH_SPEED,
                yaw_speed=GIMBAL_YAW_SPEED
            ).wait_for_completed()
        except Exception:
            pass

        try:
            self.distance_sensor.unsub_distance()
        except Exception:
            pass

        try:
            self.chassis.unsub_position()
        except Exception:
            pass

        try:
            self.chassis.unsub_attitude()
        except Exception:
            pass

        try:
            self.ep_robot.close()
        except Exception:
            pass

        print("[CLEANUP] done.")

    # --------------------------------------------------------
    # BASIC MOTION
    # --------------------------------------------------------

    def stop_chassis(self):
        if self.chassis is not None:
            self.chassis.drive_speed(
                x=0.0,
                y=0.0,
                z=0.0,
                timeout=DRIVE_COMMAND_TIMEOUT
            )

    def wait_for_position(self, timeout=POSITION_WAIT_TIMEOUT):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            pos = self.state.get_position()
            if pos is not None:
                return pos
            time.sleep(0.05)
        return None

    def current_yaw(self):
        att = self.state.get_attitude()
        if att is None:
            return None
        return att[0]

    def desired_yaw_for_heading(self, heading=None):
        """
        Return the fixed absolute yaw target for a logical DFS heading.

        The existing turn convention in this program is:
            RIGHT 90 -> chassis.move(z=-90)
            LEFT  90 -> chassis.move(z=+90)

        Therefore, from the startup N reference:
            N = base
            E = base - 90
            S = base - 180
            W = base + 90

        This is calculated from base_yaw_deg every time, so turn/gimbal error
        cannot accumulate into the next reference.
        """
        if self.base_yaw_deg is None:
            return None

        if heading is None:
            heading = self.heading

        return wrap_deg(self.base_yaw_deg - 90.0 * (heading % 4))

    def yaw_error_deg(self, target_yaw=None):
        if target_yaw is None:
            target_yaw = self.yaw_ref_deg

        now = self.current_yaw()
        if target_yaw is None or now is None:
            return None

        return wrap_deg(target_yaw - now)

    def yaw_hold_command(self, target_yaw=None, stationary=False):
        """Return corrective chassis z command that drives yaw error to 0."""
        if not YAW_HOLD_ENABLED:
            return 0.0

        error = self.yaw_error_deg(target_yaw)
        if error is None:
            return 0.0

        if abs(error) <= YAW_HOLD_DEADBAND_DEG:
            return 0.0

        if stationary:
            kp = STATIONARY_YAW_HOLD_KP
            limit = STATIONARY_YAW_HOLD_MAX_DPS
        else:
            kp = YAW_HOLD_KP
            limit = YAW_HOLD_MAX_DPS

        z = YAW_DRIVE_SIGN * kp * error
        return clamp(z, -limit, limit)

    def hold_heading_stationary(self, duration=STATIONARY_SETTLE_SEC):
        """
        Keep x=y=0 and actively drive chassis yaw error toward zero.
        Used after turns and after gimbal movements.
        """
        if not YAW_HOLD_ENABLED or self.yaw_ref_deg is None:
            self.stop_chassis()
            return

        end_t = time.monotonic() + max(0.0, duration)
        dt = 1.0 / STATIONARY_YAW_HOLD_HZ

        with self.yaw_hold_lock:
            while self.running and time.monotonic() < end_t:
                z = self.yaw_hold_command(self.yaw_ref_deg, stationary=True)
                self.chassis.drive_speed(
                    x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT
                )
                time.sleep(dt)

            self.stop_chassis()

    def run_gimbal_action_with_yaw_lock(self, action):
        """
        Wait for a gimbal action while a background loop actively prevents
        gimbal reaction torque from rotating the chassis.
        """
        if not YAW_HOLD_ENABLED or self.yaw_ref_deg is None:
            action.wait_for_completed()
            return

        stop_event = threading.Event()

        def _holder():
            dt = 1.0 / STATIONARY_YAW_HOLD_HZ
            with self.yaw_hold_lock:
                while self.running and not stop_event.is_set():
                    z = self.yaw_hold_command(
                        self.yaw_ref_deg, stationary=True
                    )
                    self.chassis.drive_speed(
                        x=0.0, y=0.0, z=z, timeout=DRIVE_COMMAND_TIMEOUT
                    )
                    stop_event.wait(dt)

                self.stop_chassis()

        thread = threading.Thread(target=_holder, daemon=True)
        thread.start()

        try:
            action.wait_for_completed()
        finally:
            stop_event.set()
            thread.join(timeout=0.5)

        # Remove any small residual error after the turret stops accelerating.
        self.hold_heading_stationary(0.12)

    # --------------------------------------------------------
    # GIMBAL
    # --------------------------------------------------------

    def gimbal_front_down(self):
        """
        Recenter to chassis front, then pitch ToF down 5 deg.
        This is called before every chassis translation.
        """
        action = self.gimbal.recenter(
            pitch_speed=GIMBAL_PITCH_SPEED,
            yaw_speed=GIMBAL_YAW_SPEED
        )
        self.run_gimbal_action_with_yaw_lock(action)

        action = self.gimbal.move(
            pitch=GIMBAL_PITCH_DEG,
            yaw=0,
            pitch_speed=GIMBAL_PITCH_SPEED,
            yaw_speed=GIMBAL_YAW_SPEED
        )
        self.run_gimbal_action_with_yaw_lock(action)

        self.hold_heading_stationary(GIMBAL_SETTLE_SEC)

    def gimbal_point_relative_from_front(self, yaw_deg):
        """
        Robust relative pointing:
          1) recenter to actual chassis front
          2) pitch down
          3) yaw in <= 45 deg chunks

        We avoid relying on an old absolute gimbal yaw after chassis turns.
        """
        self.gimbal_front_down()

        remaining = float(yaw_deg)

        while abs(remaining) > 0.5:
            step = clamp(
                remaining,
                -GIMBAL_YAW_STEP_DEG,
                GIMBAL_YAW_STEP_DEG
            )

            action = self.gimbal.move(
                pitch=0,
                yaw=step,
                pitch_speed=GIMBAL_PITCH_SPEED,
                yaw_speed=GIMBAL_YAW_SPEED
            )
            self.run_gimbal_action_with_yaw_lock(action)

            remaining -= step

        self.hold_heading_stationary(GIMBAL_SETTLE_SEC)

    # --------------------------------------------------------
    # ToF
    # --------------------------------------------------------

    def sample_tof_median(self, samples=TOF_SCAN_SAMPLES):
        values = []

        for _ in range(samples):
            v = self.state.get_tof()

            if v is not None and math.isfinite(v) and v > 0:
                values.append(float(v))

            time.sleep(TOF_SCAN_INTERVAL_SEC)

        if not values:
            return None

        return float(statistics.median(values))

    def scan_tof_at_yaw(self, yaw_deg):
        self.gimbal_point_relative_from_front(yaw_deg)
        return self.sample_tof_median()

    # --------------------------------------------------------
    # SHARP
    # --------------------------------------------------------

    def read_sharp_adc(self):
        left = None
        right = None

        try:
            left = self.sensor_adapter.get_adc(
                id=SHARP_LEFT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        try:
            right = self.sensor_adapter.get_adc(
                id=SHARP_RIGHT_ID,
                port=SENSOR_PORT
            )
        except Exception:
            pass

        if left is not None:
            try:
                self.left_adc_hist.append(float(left))
            except Exception:
                pass

        if right is not None:
            try:
                self.right_adc_hist.append(float(right))
            except Exception:
                pass

        left_med = (
            statistics.median(self.left_adc_hist)
            if self.left_adc_hist else None
        )
        right_med = (
            statistics.median(self.right_adc_hist)
            if self.right_adc_hist else None
        )

        return left_med, right_med

    def read_sharp_cm(self):
        left_adc, right_adc = self.read_sharp_adc()

        left_cm = adc_to_cm(left_adc, LEFT_CAL)
        right_cm = adc_to_cm(right_adc, RIGHT_CAL)

        return left_cm, right_cm, left_adc, right_adc

    # --------------------------------------------------------
    # CORRIDOR CONTROL
    # --------------------------------------------------------

    def choose_follow_side(self, left_cm, right_cm):
        """
        Pick one wall and stick to it for the cell.

        Prefer the wall whose measured distance is closer to WALL_TARGET_CM,
        because it is generally in the sensor's more useful control region.
        """
        if left_cm is None and right_cm is None:
            return None

        if left_cm is None:
            return "RIGHT"

        if right_cm is None:
            return "LEFT"

        lerr = abs(left_cm - WALL_TARGET_CM)
        rerr = abs(right_cm - WALL_TARGET_CM)

        return "LEFT" if lerr <= rerr else "RIGHT"

    def corridor_lateral_command(self, left_cm, right_cm, follow_side):
        """
        RoboMaster SDK:
            y > 0 -> strafe RIGHT
            y < 0 -> strafe LEFT

        Priority rule:
          - If a side is too close, the nearest wall owns the command.
          - Otherwise follow one selected wall at WALL_TARGET_CM.
        """

        # ---------- HARD SAFETY ----------
        hard_candidates = []

        if left_cm is not None and left_cm <= SIDE_HARD_CM:
            hard_candidates.append(("LEFT", left_cm))

        if right_cm is not None and right_cm <= SIDE_HARD_CM:
            hard_candidates.append(("RIGHT", right_cm))

        if hard_candidates:
            side, dist = min(hard_candidates, key=lambda x: x[1])

            if side == "LEFT":
                return +SIDE_ESCAPE_MAX_MPS, "HARD_ESCAPE_LEFT"

            return -SIDE_ESCAPE_MAX_MPS, "HARD_ESCAPE_RIGHT"

        # ---------- PRIORITY SAFETY ----------
        priority_candidates = []

        if left_cm is not None and left_cm < SIDE_PRIORITY_CM:
            priority_candidates.append(("LEFT", left_cm))

        if right_cm is not None and right_cm < SIDE_PRIORITY_CM:
            priority_candidates.append(("RIGHT", right_cm))

        if priority_candidates:
            # Nearest wall gets priority.
            side, dist = min(priority_candidates, key=lambda x: x[1])

            penetration = SIDE_PRIORITY_CM - dist
            speed = SIDE_ESCAPE_MIN_MPS + SIDE_ESCAPE_KP * penetration
            speed = clamp(
                speed,
                SIDE_ESCAPE_MIN_MPS,
                SIDE_ESCAPE_MAX_MPS
            )

            if side == "LEFT":
                return +speed, "PRIORITY_LEFT"

            return -speed, "PRIORITY_RIGHT"

        # ---------- NORMAL WALL FOLLOW ----------
        if follow_side == "LEFT" and left_cm is not None:
            # Too close left -> positive y = move right.
            error = WALL_TARGET_CM - left_cm
            y = WALL_FOLLOW_KP * error
            return (
                clamp(y, -MAX_FOLLOW_STRAFE_MPS, MAX_FOLLOW_STRAFE_MPS),
                "FOLLOW_LEFT"
            )

        if follow_side == "RIGHT" and right_cm is not None:
            # Too close right -> negative y = move left.
            error = right_cm - WALL_TARGET_CM
            y = WALL_FOLLOW_KP * error
            return (
                clamp(y, -MAX_FOLLOW_STRAFE_MPS, MAX_FOLLOW_STRAFE_MPS),
                "FOLLOW_RIGHT"
            )

        # Selected wall disappeared. Use the other wall if possible.
        if left_cm is not None:
            error = WALL_TARGET_CM - left_cm
            y = WALL_FOLLOW_KP * error
            return (
                clamp(y, -MAX_FOLLOW_STRAFE_MPS, MAX_FOLLOW_STRAFE_MPS),
                "FALLBACK_LEFT"
            )

        if right_cm is not None:
            error = right_cm - WALL_TARGET_CM
            y = WALL_FOLLOW_KP * error
            return (
                clamp(y, -MAX_FOLLOW_STRAFE_MPS, MAX_FOLLOW_STRAFE_MPS),
                "FALLBACK_RIGHT"
            )

        return 0.0, "NO_SIDE_WALL"

    def move_one_cell(self):
        """
        Continuous corridor motion:
          - forward drive
          - Sharp wall-follow / nearest-wall priority
          - yaw hold
          - front ToF collision stop
          - stop after CELL_LENGTH_M odometry displacement
        """

        self.gimbal_front_down()

        start_pos = self.wait_for_position()
        if start_pos is None:
            print("[MOVE ERROR] no chassis position telemetry.")
            return False

        # NEVER capture a new target from current_yaw() here.
        # Current yaw may already have been disturbed by the gimbal.
        target_yaw = self.yaw_ref_deg

        # Prime Sharp filter before selecting a wall.
        for _ in range(SHARP_FILTER_SAMPLES):
            left_cm, right_cm, _, _ = self.read_sharp_cm()
            time.sleep(0.02)

        follow_side = self.choose_follow_side(left_cm, right_cm)

        print(
            f"[MOVE] start follow={follow_side} "
            f"target_yaw={target_yaw}"
        )

        t0 = time.monotonic()
        last_debug = 0.0

        while self.running:
            now_t = time.monotonic()

            pos = self.state.get_position()
            if pos is None:
                self.stop_chassis()
                print("[MOVE ERROR] position telemetry lost.")
                return False

            dx = pos[0] - start_pos[0]
            dy = pos[1] - start_pos[1]
            traveled = math.hypot(dx, dy)

            if traveled >= CELL_LENGTH_M:
                self.stop_chassis()
                print(f"[MOVE OK] reached cell: {traveled:.3f} m")
                return True

            if now_t - t0 > MAX_CELL_TIME_SEC:
                self.stop_chassis()

                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[MOVE WARN] timeout but close enough: "
                        f"{traveled:.3f} m -> accept"
                    )
                    return True

                print(
                    f"[MOVE FAIL] timeout: "
                    f"{traveled:.3f}/{CELL_LENGTH_M:.3f} m"
                )
                return False

            tof_mm = self.state.get_tof()

            if tof_mm is not None and tof_mm <= FRONT_HARD_STOP_MM:
                self.stop_chassis()

                if traveled >= CELL_LENGTH_M * CELL_SUCCESS_FRACTION:
                    print(
                        f"[MOVE WARN] front wall at {tof_mm:.0f} mm, "
                        f"but traveled {traveled:.3f} m -> accept cell"
                    )
                    return True

                print(
                    f"[MOVE BLOCKED] front ToF={tof_mm:.0f} mm at "
                    f"{traveled:.3f} m"
                )
                return False

            left_cm, right_cm, left_adc, right_adc = self.read_sharp_cm()

            # If selected wall disappears, hand over to the other wall.
            if follow_side == "LEFT" and left_cm is None and right_cm is not None:
                follow_side = "RIGHT"

            elif follow_side == "RIGHT" and right_cm is None and left_cm is not None:
                follow_side = "LEFT"

            elif follow_side is None:
                follow_side = self.choose_follow_side(left_cm, right_cm)

            y_cmd, side_mode = self.corridor_lateral_command(
                left_cm,
                right_cm,
                follow_side
            )

            z_cmd = self.yaw_hold_command(target_yaw)

            x_cmd = FORWARD_SPEED_MPS

            if tof_mm is not None and tof_mm < FRONT_SLOW_MM:
                x_cmd = SLOW_FORWARD_SPEED_MPS

            # If side is in HARD escape, also slow forward motion.
            if side_mode.startswith("HARD_"):
                x_cmd = min(x_cmd, SLOW_FORWARD_SPEED_MPS)

            self.chassis.drive_speed(
                x=x_cmd,
                y=y_cmd,
                z=z_cmd,
                timeout=DRIVE_COMMAND_TIMEOUT
            )

            if now_t - last_debug >= DEBUG_MOVE_PRINT_PERIOD_SEC:
                ltxt = "far" if left_cm is None else f"{left_cm:4.1f}"
                rtxt = "far" if right_cm is None else f"{right_cm:4.1f}"
                ttxt = "None" if tof_mm is None else f"{tof_mm:4.0f}"

                yaw_now = self.current_yaw()
                yaw_err = self.yaw_error_deg(target_yaw)
                yaw_txt = "None" if yaw_now is None else f"{yaw_now:+.2f}"
                err_txt = "None" if yaw_err is None else f"{yaw_err:+.2f}"

                print(
                    f"[CTRL] d={traveled:5.3f}m "
                    f"L={ltxt}cm({left_adc}) "
                    f"R={rtxt}cm({right_adc}) "
                    f"ToF={ttxt}mm "
                    f"mode={side_mode:<18} "
                    f"yaw={yaw_txt} err={err_txt} "
                    f"x={x_cmd:+.2f} y={y_cmd:+.2f} z={z_cmd:+.1f}"
                )

                last_debug = now_t

            time.sleep(CONTROL_DT)

        self.stop_chassis()
        return False

    # --------------------------------------------------------
    # TURNING
    # --------------------------------------------------------

    def turn_to_direction(self, target_dir):
        target_dir %= 4
        delta = (target_dir - self.heading) % 4

        if delta == 0:
            self.yaw_ref_deg = self.desired_yaw_for_heading(self.heading)
            self.hold_heading_stationary(0.15)
            self.gimbal_front_down()
            return

        self.stop_chassis()

        if delta == 1:
            print(
                f"[TURN] {DIR_NAMES[self.heading]} -> "
                f"{DIR_NAMES[target_dir]} : RIGHT 90"
            )
            self.chassis.move(
                x=0,
                y=0,
                z=-90,
                z_speed=45
            ).wait_for_completed()

        elif delta == 3:
            print(
                f"[TURN] {DIR_NAMES[self.heading]} -> "
                f"{DIR_NAMES[target_dir]} : LEFT 90"
            )
            self.chassis.move(
                x=0,
                y=0,
                z=+90,
                z_speed=45
            ).wait_for_completed()

        else:
            print(
                f"[TURN] {DIR_NAMES[self.heading]} -> "
                f"{DIR_NAMES[target_dir]} : 180"
            )
            self.chassis.move(
                x=0,
                y=0,
                z=180,
                z_speed=45
            ).wait_for_completed()

        self.heading = target_dir

        # Do NOT trust the post-turn measured yaw as the next reference.
        # Rebuild the target from the one startup base angle instead.
        self.yaw_ref_deg = self.desired_yaw_for_heading(self.heading)

        print(
            f"[YAW LOCK] logical={DIR_NAMES[self.heading]} "
            f"target={self.yaw_ref_deg:+.2f} "
            f"actual={self.current_yaw()}"
        )

        # Correct chassis turn error first; then move the turret while keeping
        # the same chassis target locked.
        self.hold_heading_stationary(STATIONARY_SETTLE_SEC)
        self.gimbal_front_down()

    # --------------------------------------------------------
    # CELL SCAN
    # --------------------------------------------------------

    def scan_cell(self, cell):
        """
        Scan only LEFT / FRONT / RIGHT.

        BACK is known from DFS parent because the robot just came through it.
        At root, back is treated as outside/closed by default.
        """
        print(
            f"\n[SCAN] cell={cell} heading={DIR_NAMES[self.heading]}"
        )

        relative_scans = [
            ("LEFT",  -90.0, REL_LEFT),
            ("FRONT",   0.0, REL_FRONT),
            ("RIGHT", +90.0, REL_RIGHT),
        ]

        ordered_open_dirs = []

        for label, yaw_deg, rel_dir in relative_scans:
            before_yaw = self.current_yaw()
            mm = self.scan_tof_at_yaw(yaw_deg)
            after_yaw = self.current_yaw()
            yaw_err = self.yaw_error_deg()

            print(
                f"  [YAW] before={before_yaw} after={after_yaw} "
                f"target={self.yaw_ref_deg} err={yaw_err}"
            )

            if mm is None:
                is_open = False
                print(f"  {label:<5}: NO DATA -> CLOSED for safety")
            else:
                is_open = mm > TOF_OPEN_THRESHOLD_MM
                state = "OPEN" if is_open else "WALL"

                print(
                    f"  {label:<5}: {mm:7.1f} mm -> {state}"
                )

            abs_dir = (self.heading + rel_dir) % 4

            if is_open:
                ordered_open_dirs.append(abs_dir)

        # Parent direction is a guaranteed known connection unless movement
        # later proves otherwise.
        p = self.parent.get(cell)

        if p is not None:
            back_dir = direction_between(cell, p)

            if back_dir not in ordered_open_dirs:
                ordered_open_dirs.append(back_dir)

        elif not ROOT_BACK_IS_WALL:
            # Optional root back scan if user explicitly enables it.
            mm = self.scan_tof_at_yaw(180.0)

            if mm is not None and mm > TOF_OPEN_THRESHOLD_MM:
                ordered_open_dirs.append(
                    (self.heading + REL_BACK) % 4
                )

        # Always return ToF to forward/down before anything moves.
        self.gimbal_front_down()

        print(
            "  open absolute dirs:",
            [DIR_NAMES[d] for d in ordered_open_dirs]
        )

        return ordered_open_dirs

    # --------------------------------------------------------
    # MAP HELPERS
    # --------------------------------------------------------

    def edge_key(self, a, b):
        return tuple(sorted((a, b)))

    def mark_blocked(self, a, b):
        self.blocked_edges.add(self.edge_key(a, b))

    def is_blocked(self, a, b):
        return self.edge_key(a, b) in self.blocked_edges

    def print_map_summary(self):
        print("\n================ DFS MAP SUMMARY ================")
        print(f"Visited cells: {len(self.visited)}")
        print("Cells:", sorted(self.visited, key=lambda p: (p[1], p[0])))

        for cell in sorted(self.open_dirs, key=lambda p: (p[1], p[0])):
            dirs = [DIR_NAMES[d] for d in self.open_dirs[cell]]
            print(f"  {cell}: open={dirs}")

        if self.blocked_edges:
            print("Blocked / failed edges:")
            for edge in sorted(self.blocked_edges):
                print(" ", edge)

        print("=================================================")

    # --------------------------------------------------------
    # DFS
    # --------------------------------------------------------

    def run_dfs(self):
        self.visited = {self.root}
        self.parent = {self.root: None}
        self.current = self.root

        stack = [self.root]

        print("\n[DFS] START")
        print(f"[DFS] root={self.root}, heading={DIR_NAMES[self.heading]}")

        while self.running and stack:
            cell = stack[-1]
            self.current = cell

            if cell not in self.open_dirs:
                self.open_dirs[cell] = self.scan_cell(cell)

            # Classic DFS:
            # choose the first open neighbor that has not been visited.
            next_dir = None
            next_cell = None

            for d in self.open_dirs[cell]:
                nb = neighbor(cell, d)

                if self.is_blocked(cell, nb):
                    continue

                if nb not in self.visited:
                    next_dir = d
                    next_cell = nb
                    break

            if next_cell is not None:
                print(
                    f"\n[DFS] EXPLORE {cell} -> {next_cell} "
                    f"dir={DIR_NAMES[next_dir]}"
                )

                self.turn_to_direction(next_dir)

                ok = self.move_one_cell()

                if not ok:
                    print(
                        f"[DFS] edge {cell}->{next_cell} failed; "
                        f"mark blocked and continue."
                    )
                    self.mark_blocked(cell, next_cell)

                    # Remove this false-positive opening from the cell map.
                    self.open_dirs[cell] = [
                        d for d in self.open_dirs[cell]
                        if neighbor(cell, d) != next_cell
                    ]
                    continue

                self.parent[next_cell] = cell
                self.visited.add(next_cell)
                stack.append(next_cell)

                print(
                    f"[DFS] ARRIVED {next_cell}; "
                    f"visited={len(self.visited)}"
                )

                continue

            # No unvisited open neighbor -> backtrack.
            parent = self.parent.get(cell)

            if parent is None:
                print("\n[DFS] Root has no unvisited neighbors.")
                print("[DFS] COMPLETE.")
                break

            back_dir = direction_between(cell, parent)

            print(
                f"\n[DFS] BACKTRACK {cell} -> {parent} "
                f"dir={DIR_NAMES[back_dir]}"
            )

            self.turn_to_direction(back_dir)

            ok = self.move_one_cell()

            if not ok:
                self.stop_chassis()
                raise RuntimeError(
                    f"Backtrack failed: {cell} -> {parent}. "
                    f"Stopping because DFS topology is no longer reliable."
                )

            stack.pop()
            self.current = parent

            print(f"[DFS] back at {parent}")

        self.stop_chassis()
        self.gimbal_front_down()
        self.print_map_summary()


# ============================================================
# MAIN
# ============================================================

def main():
    explorer = DFSMazeExplorer()

    try:
        explorer.connect()
        explorer.run_dfs()

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
