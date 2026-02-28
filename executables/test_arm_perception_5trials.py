#!/usr/bin/env python3
"""5-trial benchmark: arm-camera target perception with multi-angle sweep.

Each trial builds a fresh world (random target placement on table),
drives the robot near the table via the full pipeline
(spin→plan→drive→arm-sweep→PCA), and records positional error.

Run inside the Docker container:
    python3 test_arm_perception_5trials.py
"""

import os, sys, time
import numpy as np
import pybullet as p

from src.environment.world_builder import build_world, get_robot_pose, get_object_position
from src.modules.perception import (
    spin_and_capture_stepper, process_point_cloud,
    perceive_target_with_arm_camera, find_link_index,
    _capture_frame, depth_to_world_points,
)
from src.modules.motion_control import plan_path, DriveController

# ── constants (same as run_world_builder) ─────────────────────────────────
PERCEPTION_ERROR_MEAN = 0.1704
PERCEPTION_ERROR_STD  = 0.0180
ERROR_THRESHOLD = round(PERCEPTION_ERROR_MEAN + 3 * PERCEPTION_ERROR_STD, 3)
STOP_DISTANCE = 0.7
SWEEP_ANGLES = [-0.35, -0.17, 0.0, 0.17, 0.35]
SWEEP_SETTLE = 80          # ticks per angle
ARM_SETTLE   = 300         # initial arm settle ticks
NUM_TRIALS   = 5


def filter_poses(poses, min_pts=4):
    return [ps for ps in poses if ps["n_pts"] >= min_pts]


def run_one_trial(trial_idx):
    """Run one complete trial.  Returns result dict or None on failure."""
    cid = p.connect(p.DIRECT)
    scene = build_world(physics_client=cid)
    rid = scene["robot_id"]
    init_pos = scene.get("robot_position", [-3.0, -3.0, 0.2])

    print(f"\n{'#'*60}")
    print(f"#  TRIAL {trial_idx+1}/{NUM_TRIALS}")
    print(f"{'#'*60}")

    # ── 1. Settle ──────────────────────────────────────────────────────
    for _ in range(240):
        p.stepSimulation()

    # ── 2. Spin perception ─────────────────────────────────────────────
    gen = spin_and_capture_stepper(rid, scene, cam_link="lidar_link",
                                   spin_speed=2.0, capture_every_deg=10.0)
    while True:
        try:
            next(gen)
            p.stepSimulation()
        except StopIteration as e:
            pts, seg = e.value
            break

    # stop wheels, reset to start
    for j in range(p.getNumJoints(rid)):
        jn = p.getJointInfo(rid, j)[1].decode("utf-8")
        if jn.startswith("wheel_"):
            p.setJointMotorControl2(rid, j, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=50)
            p.resetJointState(rid, j, targetValue=0, targetVelocity=0)
    init_orn = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(rid, init_pos, init_orn)
    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
    for _ in range(480):
        p.stepSimulation()

    # ── 3. Process point cloud ─────────────────────────────────────────
    rp, _, _ = get_robot_pose(rid)
    poses = process_point_cloud(pts, seg, scene, list(rp),
                                save_path=f"perception_2d_map_trial{trial_idx+1}.png")
    accepted = filter_poses(poses)

    table_pose = None
    for ps in accepted:
        if "table_top" in ps.get("name", ""):
            table_pose = ps
            break
    if table_pose is None:
        print("  Table not detected — skipping trial")
        p.disconnect()
        return None

    table_centroid_xy = tuple(table_pose["centroid"][:2])
    start_xy = (init_pos[0], init_pos[1])

    dx = start_xy[0] - table_centroid_xy[0]
    dy = start_xy[1] - table_centroid_xy[1]
    dist_st = np.hypot(dx, dy)
    if dist_st < 1e-6:
        dx, dy, dist_st = 1.0, 0.0, 1.0
    ux, uy = dx / dist_st, dy / dist_st
    goal_xy = (table_centroid_xy[0] + ux * STOP_DISTANCE,
               table_centroid_xy[1] + uy * STOP_DISTANCE)

    # ── 4. Plan path ──────────────────────────────────────────────────
    waypoints = plan_path(accepted, start_xy, goal_xy,
                          error_threshold=ERROR_THRESHOLD, inflate=0.35,
                          scene_map=scene,
                          save_path=f"trial{trial_idx+1}_astar.png")
    if waypoints is None:
        print("  No path — skipping")
        p.disconnect()
        return None

    # orient robot
    if len(waypoints) >= 2:
        dyaw = np.arctan2(waypoints[1][1] - start_xy[1],
                           waypoints[1][0] - start_xy[0])
        p.resetBasePositionAndOrientation(
            rid, init_pos, p.getQuaternionFromEuler([0, 0, dyaw]))
        p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])

    # ── 5. Drive ───────────────────────────────────────────────────────
    dc = DriveController(rid, waypoints, goal_tol=0.30, max_steps=30000)
    while True:
        pos, orn, (_, _, yaw) = get_robot_pose(rid)
        if not dc.step(pos, yaw):
            break
        p.stepSimulation()
    if not dc.reached:
        print("  Drive failed — skipping")
        p.disconnect()
        return None

    # ── 6. Arm sweep perception ───────────────────────────────────────
    rpos, _, (_, _, ryaw) = get_robot_pose(rid)
    dx_t = table_centroid_xy[0] - rpos[0]
    dy_t = table_centroid_xy[1] - rpos[1]
    base_pan = np.arctan2(np.sin(np.arctan2(dy_t, dx_t) - ryaw),
                          np.cos(np.arctan2(dy_t, dx_t) - ryaw))

    arm_targets_base = {
        "arm_joint_2": 0.6, "arm_joint_3": 0.3,
        "arm_joint_4": 0.0, "arm_joint_5": 0.5,
        "arm_joint_6": 0.0, "arm_joint_7": 0.0,
    }

    pan_joint_idx = None
    cam_link_idx  = find_link_index(rid, "camera_link")
    for j in range(p.getNumJoints(rid)):
        jn = p.getJointInfo(rid, j)[1].decode("utf-8")
        if jn == "arm_joint_1":
            pan_joint_idx = j
            break

    # Set base arm pose + initial pan
    for j in range(p.getNumJoints(rid)):
        jn = p.getJointInfo(rid, j)[1].decode("utf-8")
        if jn in arm_targets_base:
            p.setJointMotorControl2(rid, j, p.POSITION_CONTROL,
                                    targetPosition=arm_targets_base[jn],
                                    force=200, maxVelocity=1.0)
    p.setJointMotorControl2(rid, pan_joint_idx, p.POSITION_CONTROL,
                            targetPosition=base_pan, force=200, maxVelocity=1.0)

    # initial settle
    for _ in range(ARM_SETTLE):
        p.stepSimulation()

    # sweep across angles
    target_id = scene["target_id"]
    all_target_pts = []

    for angle_offset in SWEEP_ANGLES:
        pan = base_pan + angle_offset
        p.setJointMotorControl2(rid, pan_joint_idx, p.POSITION_CONTROL,
                                targetPosition=pan, force=200, maxVelocity=1.0)
        for _ in range(SWEEP_SETTLE):
            p.stepSimulation()

        ls = p.getLinkState(rid, cam_link_idx, computeForwardKinematics=True)
        rgb, depth, seg_buf = _capture_frame(ls[0], ls[1])
        frame_pts, frame_seg = depth_to_world_points(depth, seg_buf, ls[0], ls[1])

        obj_ids = (frame_seg & 0xFFFFFF).astype(int)
        mask = obj_ids == target_id
        tgt_pts = frame_pts[mask]
        if len(tgt_pts) > 0:
            all_target_pts.append(tgt_pts)
        print(f"    sweep pan={np.rad2deg(pan):+.1f}°  target_pts={len(tgt_pts)}")

    if all_target_pts:
        merged = np.vstack(all_target_pts)
    else:
        merged = np.empty((0, 3))

    # ── 7. PCA ─────────────────────────────────────────────────────────
    result = perceive_target_with_arm_camera(
        rid, scene, cam_link="camera_link",
        all_target_pts=merged if len(merged) >= 3 else None,
        save_path=f"arm_trial_{trial_idx+1}.png",
    )

    p.disconnect()
    return result


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    errors_3d, errors_xy, n_pts_list = [], [], []
    successes = 0

    for i in range(NUM_TRIALS):
        res = run_one_trial(i)
        if res is not None:
            successes += 1
            errors_3d.append(res["position_error_3d"])
            errors_xy.append(res["position_error_xy"])
            n_pts_list.append(res["n_pts"])
            print(f"  Trial {i+1}: 3D={res['position_error_3d']:.4f}m "
                  f"XY={res['position_error_xy']:.4f}m "
                  f"pts={res['n_pts']}")
        else:
            print(f"  Trial {i+1}: FAILED (no perception)")

    print(f"\n{'='*60}")
    print(f"5-TRIAL ARM PERCEPTION BENCHMARK")
    print(f"{'='*60}")
    print(f"  Success: {successes}/{NUM_TRIALS}")
    if errors_3d:
        e3 = np.array(errors_3d)
        exy = np.array(errors_xy)
        pts = np.array(n_pts_list)
        print(f"  3D Error — Mean: {e3.mean():.4f}  Std: {e3.std():.4f}  "
              f"Min: {e3.min():.4f}  Max: {e3.max():.4f}")
        print(f"  XY Error — Mean: {exy.mean():.4f}  Std: {exy.std():.4f}  "
              f"Min: {exy.min():.4f}  Max: {exy.max():.4f}")
        print(f"  Points  — Mean: {pts.mean():.0f}  Min: {pts.min()}  Max: {pts.max()}")
    print(f"{'='*60}")
