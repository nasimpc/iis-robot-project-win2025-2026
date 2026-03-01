"""
motion_control.py — A* grid-based path planning + PID differential-drive
controller for navigating a mobile robot to a goal while avoiding obstacles.

Pipeline:
  1. Build a 2-D occupancy grid from perceived obstacle / table-leg poses
     (inflated by a safety margin derived from the perception error threshold).
  2. Run A* on the grid to find the shortest collision-free path.
  3. Smooth the waypoints to remove staircase artifacts.
  4. Follow the waypoints with a PID controller driving the differential
     wheels in PyBullet.
"""

import heapq
import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from src.environment.world_builder import get_robot_pose


# ═══════════════════════════════════════════════════════════════════════════
#  PID controller (unchanged from original — linear + angular)
# ═══════════════════════════════════════════════════════════════════════════

class PIDController:
    def __init__(
        self,
        kp_lin=0.5, ki_lin=0.0, kd_lin=0.0,
        kp_ang=1.5, ki_ang=0.0, kd_ang=0.0,
        max_lin=0.5, max_ang=2.0,
        max_integral=5.0,
    ):
        self.kp_lin = kp_lin
        self.ki_lin = ki_lin
        self.kd_lin = kd_lin
        self.kp_ang = kp_ang
        self.ki_ang = ki_ang
        self.kd_ang = kd_ang
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.max_integral = max_integral
        self.reset()

    def reset(self):
        self._int_lin = 0.0
        self._int_ang = 0.0
        self._prev_lin = 0.0
        self._prev_ang = 0.0

    def compute(self, current_pose, target_pose, dt=1.0 / 240.0):
        """current_pose / target_pose: (x, y, yaw).

        Linear velocity is scaled by cos(angular_error) so the robot
        slows down (or reverses) when not facing the target, preventing
        the classic 180° oscillation problem.
        """
        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] - current_pose[1]
        dist = np.hypot(dx, dy)

        desired_yaw = np.arctan2(dy, dx)
        ang_err = np.arctan2(np.sin(desired_yaw - current_pose[2]),
                             np.cos(desired_yaw - current_pose[2]))

        # linear PID
        self._int_lin = np.clip(self._int_lin + dist * dt,
                                -self.max_integral, self.max_integral)
        d_lin = (dist - self._prev_lin) / dt if dt > 0 else 0.0
        self._prev_lin = dist
        v = (self.kp_lin * dist + self.ki_lin * self._int_lin
             + self.kd_lin * d_lin)

        # Scale linear velocity by cos(ang_err):
        #   facing target → cos≈1 → full speed
        #   perpendicular → cos≈0 → stop
        #   facing away   → cos<0 → slight reverse (helps turning)
        v *= np.cos(ang_err)

        # angular PID
        self._int_ang = np.clip(self._int_ang + ang_err * dt,
                                -self.max_integral, self.max_integral)
        d_ang = (ang_err - self._prev_ang) / dt if dt > 0 else 0.0
        self._prev_ang = ang_err
        omega = (self.kp_ang * ang_err + self.ki_ang * self._int_ang
                 + self.kd_ang * d_ang)

        v = np.clip(v, -self.max_lin, self.max_lin)
        omega = np.clip(omega, -self.max_ang, self.max_ang)
        return v, omega


# ═══════════════════════════════════════════════════════════════════════════
#  Occupancy grid from perceived poses
# ═══════════════════════════════════════════════════════════════════════════

class OccupancyGrid:
    """Axis-aligned 2-D grid with obstacle inflation."""

    def __init__(self, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0),
                 resolution=0.10):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = resolution
        self.nx = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.ny = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.grid = np.zeros((self.nx, self.ny), dtype=np.uint8)

    def world_to_grid(self, x, y) -> Tuple[int, int]:
        gx = int((x - self.x_min) / self.res)
        gy = int((y - self.y_min) / self.res)
        return (np.clip(gx, 0, self.nx - 1),
                np.clip(gy, 0, self.ny - 1))

    def grid_to_world(self, gx, gy) -> Tuple[float, float]:
        return (self.x_min + (gx + 0.5) * self.res,
                self.y_min + (gy + 0.5) * self.res)

    def mark_obstacle(self, cx, cy, half_size, inflate=0.0):
        """Mark a rectangular region as occupied (1) with optional inflation."""
        r = half_size + inflate
        gx0, gy0 = self.world_to_grid(cx - r, cy - r)
        gx1, gy1 = self.world_to_grid(cx + r, cy + r)
        self.grid[gx0:gx1 + 1, gy0:gy1 + 1] = 1

    def is_free(self, gx, gy) -> bool:
        if 0 <= gx < self.nx and 0 <= gy < self.ny:
            return self.grid[gx, gy] == 0
        return False

    def fill_table_footprint(self, leg_centroids, inflate: float = 0.15):
        """Fill the bounding rectangle of table-leg centroids as occupied.

        This prevents the robot from navigating *between* the legs,
        which would hit the table surface.
        """
        if len(leg_centroids) < 2:
            return
        pts = np.array(leg_centroids)
        x_min, y_min = pts.min(axis=0) - inflate
        x_max, y_max = pts.max(axis=0) + inflate
        gx0, gy0 = self.world_to_grid(x_min, y_min)
        gx1, gy1 = self.world_to_grid(x_max, y_max)
        self.grid[gx0:gx1 + 1, gy0:gy1 + 1] = 1
        print(f"  Filled table footprint: "
              f"({x_min:.2f},{y_min:.2f}) → ({x_max:.2f},{y_max:.2f})  "
              f"[{gx1 - gx0 + 1}×{gy1 - gy0 + 1} cells]")

    def populate_from_poses(self, poses: List[Dict],
                            obstacle_half_size: float = 0.25,
                            leg_half_size: float = 0.05,
                            inflate: float = 0.30,
                            leg_inflate: float = 0.15,
                            fill_table_hull: bool = False):
        """
        Fill the grid from perceived 2-D poses.

        inflate:         safety margin around obstacles.
        leg_inflate:     smaller inflation for table legs.
        fill_table_hull: if True, fill the full footprint between detected
                         table legs so the robot cannot pass between them.
        """
        leg_centroids = []
        for ps in poses:
            cx, cy = ps["centroid"]
            name = ps.get("name", "")
            if "leg" in name:
                hs = leg_half_size
                self.mark_obstacle(cx, cy, hs, inflate=leg_inflate)
                leg_centroids.append((cx, cy))
            elif "table_top" in name:
                # skip table top — robot drives towards it
                continue
            else:
                hs = obstacle_half_size
                self.mark_obstacle(cx, cy, hs, inflate=inflate)

        # Fill the area between table legs if requested
        if fill_table_hull and len(leg_centroids) >= 2:
            self.fill_table_footprint(leg_centroids, inflate=leg_inflate)


# ═══════════════════════════════════════════════════════════════════════════
#  A* path planner on 2-D grid
# ═══════════════════════════════════════════════════════════════════════════

def astar(grid: OccupancyGrid,
          start_xy: Tuple[float, float],
          goal_xy: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    """
    A* search on the occupancy grid.

    Returns a list of (x, y) world-frame waypoints from start to goal,
    or None if no path exists.
    """
    sx, sy = grid.world_to_grid(*start_xy)
    gx, gy = grid.world_to_grid(*goal_xy)

    # Clear a radius around start / goal so the planner can proceed
    # even when inflation touches these cells.
    clear_r = 4  # cells — 4 × 0.10 m = 0.40 m radius
    for cx, cy, label in [(sx, sy, "Start"), (gx, gy, "Goal")]:
        if not grid.is_free(cx, cy):
            print(f"[A*] {label} cell ({cx},{cy}) is occupied — clearing neighbourhood.")
            for dx in range(-clear_r, clear_r + 1):
                for dy in range(-clear_r, clear_r + 1):
                    nx_, ny_ = cx + dx, cy + dy
                    if 0 <= nx_ < grid.nx and 0 <= ny_ < grid.ny:
                        grid.grid[nx_, ny_] = 0

    # 8-connected neighbourhood
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1.0, 1.0, 1.0, 1.0,
             1.414, 1.414, 1.414, 1.414]

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set: List = []
    heapq.heappush(open_set, (0.0, (sx, sy)))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            # reconstruct
            path = []
            node = current
            while node in came_from:
                path.append(grid.grid_to_world(*node))
                node = came_from[node]
            path.append(grid.grid_to_world(sx, sy))
            path.reverse()
            return path

        for (dx, dy), c in zip(neighbours, costs):
            nx_, ny_ = current[0] + dx, current[1] + dy
            if not grid.is_free(nx_, ny_):
                continue
            tent_g = g_score[current] + c
            if tent_g < g_score.get((nx_, ny_), float("inf")):
                came_from[(nx_, ny_)] = current
                g_score[(nx_, ny_)] = tent_g
                f = tent_g + heuristic((nx_, ny_), (gx, gy))
                heapq.heappush(open_set, (f, (nx_, ny_)))

    return None  # no path


# ═══════════════════════════════════════════════════════════════════════════
#  Path smoothing (simple Douglas-Peucker style down-sampling)
# ═══════════════════════════════════════════════════════════════════════════

def smooth_path(path: List[Tuple[float, float]],
                grid: OccupancyGrid,
                step: int = 5) -> List[Tuple[float, float]]:
    """
    Reduce waypoints by skipping grid waypoints when the straight-line
    segment between kept points is collision-free.
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # try to skip ahead
        best = i + 1
        for j in range(min(i + step, len(path) - 1), i, -1):
            if _line_free(smoothed[-1], path[j], grid):
                best = j
                break
        smoothed.append(path[best])
        i = best
    return smoothed


def _line_free(a, b, grid: OccupancyGrid, samples: int = 20) -> bool:
    """Check if the straight line between a and b is collision-free."""
    for t in np.linspace(0, 1, samples):
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        gx, gy = grid.world_to_grid(x, y)
        if not grid.is_free(gx, gy):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Step-by-step drive controller (for use inside while p.isConnected() loop)
# ═══════════════════════════════════════════════════════════════════════════

class DriveController:
    """
    One-step-at-a-time waypoint follower.  Call ``step()`` once per
    simulation tick inside the ``while p.isConnected()`` main loop.
    Does **not** call ``p.stepSimulation()`` — the caller is responsible.
    """

    def __init__(self, robot_id: int,
                 waypoints: List[Tuple[float, float]],
                 pid: PIDController = None,
                 goal_tol: float = 0.30,
                 max_steps: int = 30000):
        self.robot_id   = robot_id
        self.waypoints  = waypoints
        self.goal_tol   = goal_tol
        self.max_steps  = max_steps
        self.pid        = pid or PIDController(
            kp_lin=0.8, kp_ang=3.0, max_lin=0.6, max_ang=4.0)

        self.wheel_radius = 0.1
        self.axle_half    = 0.225
        self.force        = 50

        # discover wheel joints
        self.wheels: Dict[str, int] = {}
        for j in range(p.getNumJoints(robot_id)):
            jn = p.getJointInfo(robot_id, j)[1].decode("utf-8")
            if jn.startswith("wheel_"):
                self.wheels[jn] = j

        self.wp_idx  = 0
        self._step   = 0
        self.done    = False
        self.reached = False

    def step(self, pos, yaw):
        """Execute one control tick.  ``pos`` = (x, y, z), ``yaw`` = float.

        Returns True while still driving, False when finished.
        """
        if self.done:
            return False

        if self.wp_idx >= len(self.waypoints) or self._step >= self.max_steps:
            self._stop_wheels()
            self.reached = self.wp_idx >= len(self.waypoints)
            self.done = True
            print(f"  Drive finished: step={self._step}, "
                  f"reached={self.reached}, "
                  f"final_pos=({pos[0]:+.2f}, {pos[1]:+.2f})")
            return False

        tx, ty = self.waypoints[self.wp_idx]
        dx = tx - pos[0]
        dy = ty - pos[1]
        dist = np.hypot(dx, dy)

        tol = (self.goal_tol if self.wp_idx < len(self.waypoints) - 1
               else self.goal_tol + 0.20)

        if dist < tol:
            if self.wp_idx % 3 == 0 or self.wp_idx == len(self.waypoints) - 1:
                print(f"    wp {self.wp_idx+1}/{len(self.waypoints)} reached  "
                      f"pos=({pos[0]:+.2f},{pos[1]:+.2f})")
            self.wp_idx += 1
            self.pid.reset()
            return True  # still driving, next tick will target next wp

        desired_yaw = np.arctan2(dy, dx)
        ang_err = np.arctan2(np.sin(desired_yaw - yaw),
                             np.cos(desired_yaw - yaw))

        v_raw = self.pid.kp_lin * min(dist, 1.5)
        cos_scale = max(np.cos(ang_err), 0.0)
        v = np.clip(v_raw * cos_scale, 0.0, self.pid.max_lin)
        omega = np.clip(self.pid.kp_ang * ang_err,
                        -self.pid.max_ang, self.pid.max_ang)

        v_left  = -((v - omega * self.axle_half) / self.wheel_radius)
        v_right = -((v + omega * self.axle_half) / self.wheel_radius)

        for name, jid in self.wheels.items():
            vel = v_left if ("_fl_" in name or "_bl_" in name) else v_right
            p.setJointMotorControl2(self.robot_id, jid, p.VELOCITY_CONTROL,
                                    targetVelocity=vel, force=self.force)
        self._step += 1

        if self._step % 4000 == 0:
            print(f"    step {self._step}: wp {self.wp_idx+1}/"
                  f"{len(self.waypoints)}  "
                  f"pos=({pos[0]:+.2f},{pos[1]:+.2f})  dist={dist:.2f}  "
                  f"ang={np.rad2deg(ang_err):+.1f}°  v={v:.2f}")
        return True

    def _stop_wheels(self):
        for jid in self.wheels.values():
            p.setJointMotorControl2(self.robot_id, jid, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=self.force)


# ═══════════════════════════════════════════════════════════════════════════
#  Plan path (pure computation — no p.stepSimulation)
# ═══════════════════════════════════════════════════════════════════════════

def plan_path(poses: List[Dict],
              start_xy: Tuple[float, float],
              goal_xy: Tuple[float, float],
              error_threshold: float = 0.22,
              inflate: float = 0.35,
              scene_map: dict = None,
              save_path: str = "astar_path.png",
              fill_table_hull: bool = False):
    """
    Build occupancy grid from perceived poses, run A*, smooth, plot.

    Returns the smoothed waypoint list or None if no path found.
    Does NOT call p.stepSimulation().
    """
    print("=" * 60)
    print("PLANNING: A* Path on Occupancy Grid")
    print("=" * 60)
    print(f"  Start : ({start_xy[0]:+.2f}, {start_xy[1]:+.2f})")
    print(f"  Goal  : ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})")

    robot_half = 0.30
    safe_inflate = max(inflate, error_threshold + robot_half)
    leg_inflate  = robot_half - 0.05   # robot half minus leg physical size
    print(f"  Safety inflation: obstacles={safe_inflate:.2f} m, "
          f"legs={leg_inflate:.2f} m, fill_table_hull={fill_table_hull}")

    grid = OccupancyGrid(x_range=(-5.5, 5.5), y_range=(-5.5, 5.5),
                         resolution=0.10)
    grid.populate_from_poses(poses, inflate=safe_inflate,
                             leg_inflate=leg_inflate,
                             fill_table_hull=fill_table_hull)
    occupied = int(grid.grid.sum())
    print(f"  Grid {grid.nx}×{grid.ny}, occupied cells: {occupied}")

    raw_path = astar(grid, start_xy, goal_xy)
    if raw_path is None:
        print("  A* FAILED — no path found!")
        return None
    print(f"  A* raw waypoints: {len(raw_path)}")

    s_path = smooth_path(raw_path, grid, step=8)
    print(f"  Smoothed waypoints: {len(s_path)}")

    # plot
    obs_color_map: Dict[int, str] = {}
    if scene_map:
        for oid, cname in zip(scene_map.get("obstacle_ids", []),
                              scene_map.get("obstacle_colors", [])):
            obs_color_map[oid] = cname

    plot_path(grid, raw_path, s_path, start_xy, goal_xy, poses,
              obstacle_color_map=obs_color_map, save_path=save_path)

    for i, wp in enumerate(s_path):
        print(f"    wp{i+1}: ({wp[0]:+.2f}, {wp[1]:+.2f})")

    return s_path


# ═══════════════════════════════════════════════════════════════════════════
#  Drive robot along waypoints using PID + differential drive
# ═══════════════════════════════════════════════════════════════════════════

def drive_to_waypoints(robot_id: int,
                       waypoints: List[Tuple[float, float]],
                       pid: PIDController = None,
                       goal_tol: float = 0.30,
                       max_steps: int = 40000,
                       dt: float = 1.0 / 240.0) -> bool:
    """
    Follow waypoints using a simple approach:
      - Always drive forward (v >= 0, clamped by cos(ang_err)).
      - Strong angular correction steers the robot toward the target.
      - No explicit turn-in-place; the robot arcs towards each waypoint.

    Returns True if the robot reached the final waypoint.
    """
    if pid is None:
        pid = PIDController(kp_lin=0.8, kp_ang=3.0, max_lin=0.6, max_ang=4.0)

    # discover wheel joints
    wheels: Dict[str, int] = {}
    for j in range(p.getNumJoints(robot_id)):
        jn = p.getJointInfo(robot_id, j)[1].decode("utf-8")
        if jn.startswith("wheel_"):
            wheels[jn] = j
    print(f"  Wheels found: {list(wheels.keys())}")

    wheel_radius = 0.1
    axle_half    = 0.225
    force        = 50

    wp_idx = 0
    step   = 0
    log_every = 4000
    while wp_idx < len(waypoints) and step < max_steps:
        pos, orn, (_, _, yaw) = get_robot_pose(robot_id)
        tx, ty = waypoints[wp_idx]
        dx = tx - pos[0]
        dy = ty - pos[1]
        dist = np.hypot(dx, dy)

        # Use a looser tolerance for the final waypoint
        tol = goal_tol if wp_idx < len(waypoints) - 1 else goal_tol + 0.20

        if dist < tol:
            if wp_idx % 3 == 0 or wp_idx == len(waypoints) - 1:
                print(f"    wp {wp_idx+1}/{len(waypoints)} reached  "
                      f"pos=({pos[0]:+.2f},{pos[1]:+.2f})")
            wp_idx += 1
            pid.reset()
            continue

        desired_yaw = np.arctan2(dy, dx)
        ang_err = np.arctan2(np.sin(desired_yaw - yaw),
                             np.cos(desired_yaw - yaw))

        # Forward speed: proportional to distance, scaled by max(cos(err), 0)
        # so robot slows/stops when not facing target, but NEVER reverses.
        v_raw = pid.kp_lin * min(dist, 1.5)
        cos_scale = max(np.cos(ang_err), 0.0)
        v = np.clip(v_raw * cos_scale, 0.0, pid.max_lin)

        # Angular correction: strong proportional
        omega = np.clip(pid.kp_ang * ang_err, -pid.max_ang, pid.max_ang)

        # Differential drive
        # Note: wheel joint axis is rotated 90° in URDF (rpy="1.5708 0 0"),
        # so positive joint velocity = backward motion. Negate to fix.
        v_left  = -((v - omega * axle_half) / wheel_radius)
        v_right = -((v + omega * axle_half) / wheel_radius)

        for name, jid in wheels.items():
            vel = v_left if ("_fl_" in name or "_bl_" in name) else v_right
            p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                    targetVelocity=vel, force=force)

        p.stepSimulation()
        step += 1

        if step % log_every == 0:
            print(f"    step {step}: wp {wp_idx+1}/{len(waypoints)}  "
                  f"pos=({pos[0]:+.2f},{pos[1]:+.2f})  dist={dist:.2f}  "
                  f"ang={np.rad2deg(ang_err):+.1f}°  v={v:.2f}")

    # stop wheels
    for jid in wheels.values():
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                targetVelocity=0, force=force)

    reached = wp_idx >= len(waypoints)
    pos, _, _ = get_robot_pose(robot_id)
    print(f"  Drive finished: step={step}, reached={reached}, "
          f"final_pos=({pos[0]:+.2f}, {pos[1]:+.2f})")
    return reached


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting helper — show grid, path, obstacles
# ═══════════════════════════════════════════════════════════════════════════

def plot_path(grid: OccupancyGrid,
              path: List[Tuple[float, float]],
              smoothed: List[Tuple[float, float]],
              start: Tuple[float, float],
              goal: Tuple[float, float],
              poses: List[Dict],
              obstacle_color_map: Dict[int, str] = None,
              save_path: str = "astar_path.png"):
    """Save a plot showing the occupancy grid, A* path, and obstacles."""
    if obstacle_color_map is None:
        obstacle_color_map = {}

    _OBS_RGB = {
        "Blue":   (0.0, 0.0, 1.0),
        "Pink":   (1.0, 0.4, 0.7),
        "Orange": (1.0, 0.5, 0.0),
        "Yellow": (1.0, 1.0, 0.0),
        "Black":  (0.1, 0.1, 0.1),
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    # occupancy grid as background
    extent = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    ax.imshow(grid.grid.T, origin="lower", extent=extent,
              cmap="Greys", alpha=0.3, aspect="equal")

    # obstacle centroids with their actual colours
    for ps in poses:
        cx, cy = ps["centroid"]
        bid = ps.get("body_id")
        cn = obstacle_color_map.get(bid)
        c = _OBS_RGB.get(cn, (0.5, 0.3, 0.1)) if cn else (0.5, 0.3, 0.1)
        nm = f"{cn} obstacle" if cn else ps.get("name", "")
        ax.plot(cx, cy, "s", color=c, ms=10, label=nm)

    # raw A* path (light)
    if path:
        px, py = zip(*path)
        ax.plot(px, py, "-", color="lightblue", lw=1, label="A* raw")

    # smoothed path (bold)
    if smoothed:
        sx, sy = zip(*smoothed)
        ax.plot(sx, sy, "-o", color="blue", lw=2, ms=4,
                label="Smoothed path")

    ax.plot(*start, "g^", ms=14, label="Start (robot)")
    ax.plot(*goal, "rD", ms=12, label="Goal (table)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("A* Path Plan — Robot → Table")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved path plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  High-level: plan_and_drive
# ═══════════════════════════════════════════════════════════════════════════

def plan_and_drive(robot_id: int,
                   poses: List[Dict],
                   table_pose: Dict,
                   scene_map: dict,
                   error_threshold: float = 0.20,
                   inflate: float = 0.35,
                   save_path: str = "astar_path.png") -> bool:
    """
    Full motion-control pipeline:
      1. Filter accepted poses by error threshold (uses extent > 0
         as a simple validity check — all poses with enough points are kept).
      2. Build occupancy grid from accepted obstacle & table-leg poses.
      3. A* path from current robot position to the table-top centroid.
      4. Smooth the path.
      5. Drive the robot along the path with PID.

    Parameters
    ----------
    error_threshold : float
        Maximum acceptable perception pose error (metres).  Used to set the
        safety inflation = max(inflate, error_threshold + robot_radius).
    inflate : float
        Base inflation radius around each obstacle cell (metres).

    Returns True if the robot reached the table.
    """
    print("=" * 60)
    print("MOTION CONTROL: A* Path Planning + PID Drive")
    print("=" * 60)

    # ── current robot pose ──
    pos, orn, (_, _, yaw) = get_robot_pose(robot_id)
    start_xy = (pos[0], pos[1])

    # ── goal = table-top centroid ──
    goal_xy = tuple(table_pose["centroid"][:2])
    print(f"  Start : ({start_xy[0]:+.2f}, {start_xy[1]:+.2f})")
    print(f"  Goal  : ({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})")

    # ── safety inflation = error_threshold + robot half-size ──
    robot_half = 0.30  # conservative: base is 0.6 × 0.4
    safe_inflate = max(inflate, error_threshold + robot_half)
    leg_inflate  = robot_half - 0.05   # robot half minus leg physical size
    print(f"  Safety inflation: obstacles={safe_inflate:.2f} m, "
          f"legs={leg_inflate:.2f} m  "
          f"(err_thresh={error_threshold:.2f}, robot_half={robot_half:.2f})")

    # ── 1. build occupancy grid ──
    grid = OccupancyGrid(x_range=(-5.5, 5.5), y_range=(-5.5, 5.5),
                         resolution=0.10)

    # populate grid — table_top is skipped inside populate_from_poses
    grid.populate_from_poses(poses, inflate=safe_inflate,
                             leg_inflate=leg_inflate)
    occupied = int(grid.grid.sum())
    print(f"  Grid {grid.nx}×{grid.ny}, occupied cells: {occupied}")

    # ── 2. A* ──
    raw_path = astar(grid, start_xy, goal_xy)
    if raw_path is None:
        print("  A* FAILED — no path found!")
        return False
    print(f"  A* raw waypoints: {len(raw_path)}")

    # ── 3. smooth ──
    s_path = smooth_path(raw_path, grid, step=8)
    print(f"  Smoothed waypoints: {len(s_path)}")

    # ── 4. plot ──
    obs_color_map: Dict[int, str] = {}
    for oid, cname in zip(scene_map.get("obstacle_ids", []),
                          scene_map.get("obstacle_colors", [])):
        obs_color_map[oid] = cname

    plot_path(grid, raw_path, s_path, start_xy, goal_xy, poses,
              obstacle_color_map=obs_color_map, save_path=save_path)

    # ── 5. Orient robot toward second waypoint before driving ──
    if len(s_path) >= 2:
        # Skip the first waypoint (robot's own position)
        first_target = s_path[1]
        dx = first_target[0] - start_xy[0]
        dy = first_target[1] - start_xy[1]
        target_yaw = np.arctan2(dy, dx)
        target_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
        p.resetBasePositionAndOrientation(
            robot_id, [pos[0], pos[1], pos[2]], target_orn)
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
        for _ in range(120):
            p.stepSimulation()
        print(f"  Oriented robot toward wp2: yaw={np.rad2deg(target_yaw):+.1f}°")

    # ── 6. drive ──
    print("  Driving …")
    # Print smoothed waypoints
    for i, wp in enumerate(s_path):
        print(f"    wp{i+1}: ({wp[0]:+.2f}, {wp[1]:+.2f})")

    reached = drive_to_waypoints(robot_id, s_path,
                                 goal_tol=0.30, max_steps=30000)

    final_pos, _, _ = get_robot_pose(robot_id)
    final_dist = np.hypot(final_pos[0] - goal_xy[0],
                          final_pos[1] - goal_xy[1])
    print(f"  Final distance to table: {final_dist:.3f} m")
    return reached


# ═══════════════════════════════════════════════════════════════════════════
#  IK-based Grasp stepper (generator for main-loop integration)
# ═══════════════════════════════════════════════════════════════════════════

# Joint layout (from URDF):
#   Movable joints in order:  0-3 (wheels), 5-11 (arm), 13-14 (gripper)
#   Fixed joints (skipped by IK): 4 (arm_base), 12 (gripper_base),
#                                  15-17 (camera, lidar, imu)
# p.calculateInverseKinematics returns one value per movable joint:
#   ik[0..3]  → wheels,  ik[4..10] → arm_joint_1..7,  ik[11..12] → fingers

_ARM_JOINT_IDS   = [5, 6, 7, 8, 9, 10, 11]   # arm_joint_1 … arm_joint_7
_GRIPPER_JOINT_IDS = [13, 14]                  # left / right finger
_EE_LINK_IDX     = 12                         # gripper_base
_IK_ARM_SLICE    = slice(4, 11)               # indices in IK solution


def _apply_arm_ik(robot_id, target_pos, target_orn,
                  max_vel=1.0, force=200):
    """Compute IK for gripper_base and command the 7 arm joints."""
    ik = p.calculateInverseKinematics(
        robot_id, _EE_LINK_IDX, target_pos, target_orn,
        maxNumIterations=200, residualThreshold=1e-4)
    arm_angles = ik[_IK_ARM_SLICE]
    for ji, angle in zip(_ARM_JOINT_IDS, arm_angles):
        p.setJointMotorControl2(robot_id, ji, p.POSITION_CONTROL,
                                targetPosition=angle,
                                force=force, maxVelocity=max_vel)
    return arm_angles


def grasp_target_stepper(robot_id, target_pos_3d):
    """Generator: open gripper → IK pre-grasp → lower → close → lift.

    Yields once per simulation tick.  Drive with ``next(gen)`` inside the
    ``while p.isConnected()`` loop;  ``StopIteration`` signals completion.

    Parameters
    ----------
    robot_id : int
        PyBullet body id of the robot.
    target_pos_3d : array-like, shape (3,)
        World-frame (x, y, z) of the target object centre.
    """
    tx, ty, tz = float(target_pos_3d[0]), float(target_pos_3d[1]), float(target_pos_3d[2])

    # Orientation: gripper pointing straight down (Z down)
    grasp_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    print("\n" + "=" * 60)
    print("GRASP: Starting IK-based grasp sequence")
    print("=" * 60)

    # ── 1. Open gripper ──
    print("  [1/5] Opening gripper …")
    for gj in _GRIPPER_JOINT_IDS:
        p.setJointMotorControl2(robot_id, gj, p.POSITION_CONTROL,
                                targetPosition=0.04, force=50)
    for _ in range(120):
        yield

    # ── 2. Pre-grasp: 10 cm above target ──
    pre_z = tz + 0.10
    print(f"  [2/5] Moving to pre-grasp ({tx:+.3f}, {ty:+.3f}, {pre_z:+.3f}) …")
    _apply_arm_ik(robot_id, [tx, ty, pre_z], grasp_orn)
    for _ in range(400):
        yield

    # ── 3. Lower to grasp height ──
    # PCA systematically over-estimates z by ~0.04 m (sees top surface).
    # Subtract 0.04 so the gripper straddles the cylinder centre.
    grasp_z = tz - 0.04
    print(f"  [3/5] Lowering to grasp ({tx:+.3f}, {ty:+.3f}, {grasp_z:+.3f}) …")
    _apply_arm_ik(robot_id, [tx, ty, grasp_z], grasp_orn)
    for _ in range(400):
        yield

    # ── 4. Close gripper ──
    print("  [4/5] Closing gripper …")
    for gj in _GRIPPER_JOINT_IDS:
        p.setJointMotorControl2(robot_id, gj, p.POSITION_CONTROL,
                                targetPosition=0.005, force=200)
    for _ in range(300):
        yield

    # ── 5. Lift object (keep gripper closed) ──
    lift_z = tz + 0.15
    print(f"  [5/5] Lifting to ({tx:+.3f}, {ty:+.3f}, {lift_z:+.3f}) …")
    _apply_arm_ik(robot_id, [tx, ty, lift_z], grasp_orn)
    # Keep re-commanding gripper closed during lift to maintain grip
    for _ in range(500):
        for gj in _GRIPPER_JOINT_IDS:
            p.setJointMotorControl2(robot_id, gj, p.POSITION_CONTROL,
                                    targetPosition=0.005, force=200)
        yield

    print("  Grasp sequence complete.")
    return True
