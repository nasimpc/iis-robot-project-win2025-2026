"""Build the simulation world, spin the robot 360 deg using BOTH the base
camera (for obstacles / table legs) and the gripper camera (for the red
target cylinder), compute 2-D poses via PCA, fill the table footprint so
the robot avoids navigating between legs, plan an A* path toward the
target, drive there, refine the target pose with a close-range arm sweep,
move closer if needed, and grasp the object with IK.

All simulation stepping happens inside the required
``while p.isConnected()`` loop.

Usage:
    python3 run_world_builder.py [--headless] [--steps N]
"""
import os
import sys
import time
import argparse
import numpy as np
import pybullet as p
from src.environment.world_builder import build_world, get_robot_pose, get_object_position
from src.modules.perception import (
    spin_and_capture_stepper, process_point_cloud,
    perceive_target_with_arm_camera, find_link_index,
    _capture_frame, depth_to_world_points,
)
from src.modules.motion_control import (
    plan_path, DriveController, PIDController, grasp_target_stepper,
)

# -- perception error statistics from 5-trial benchmark --
PERCEPTION_ERROR_MEAN = 0.1704
PERCEPTION_ERROR_STD  = 0.0180
ERROR_THRESHOLD = round(PERCEPTION_ERROR_MEAN + 3 * PERCEPTION_ERROR_STD, 3)

# -- arm-camera sweep parameters --
SWEEP_ANGLES   = [-0.35, -0.17, 0.0, 0.17, 0.35]
SWEEP_SETTLE   = 80
ARM_SETTLE     = 300

# -- arm reach / approach parameters --
ARM_MAX_REACH  = 0.50
GRASP_STANDOFF = 0.30
STOP_DISTANCE  = 0.7


# -----------------------------------------------------------------------
def _goal_near_target_outside_hull(target_xy, leg_centroids,
                                   hull_inflate=0.25, margin=0.10):
    """Compute a goal just outside the nearest table-hull edge to target."""
    if len(leg_centroids) < 2:
        return (float(target_xy[0]), float(target_xy[1]))
    legs = np.array(leg_centroids)
    hx_min = float(legs[:, 0].min() - hull_inflate)
    hy_min = float(legs[:, 1].min() - hull_inflate)
    hx_max = float(legs[:, 0].max() + hull_inflate)
    hy_max = float(legs[:, 1].max() + hull_inflate)
    tx, ty = float(target_xy[0]), float(target_xy[1])
    edges = [
        ("left",   tx - hx_min, (hx_min - margin, ty)),
        ("right",  hx_max - tx, (hx_max + margin, ty)),
        ("bottom", ty - hy_min, (tx, hy_min - margin)),
        ("top",    hy_max - ty, (tx, hy_max + margin)),
    ]
    name, dist, goal = min(edges, key=lambda e: e[1])
    print(f"  Nearest hull edge to target: {name} "
          f"(dist={dist:.2f}m)  goal=({goal[0]:+.3f}, {goal[1]:+.3f})")
    return goal


def filter_poses(poses, min_pts=4):
    accepted = []
    for ps in poses:
        if ps["n_pts"] < min_pts:
            print(f"    dropped '{ps['name']}' -- only {ps['n_pts']} pts")
            continue
        accepted.append(ps)
    print(f"  Poses after filtering: {len(accepted)} / {len(poses)}")
    return accepted


# -- Phases --
SETTLE   = "settle"
SENSE    = "sense"
RESET    = "reset"
THINK    = "think"
ACT      = "act"
LOOK     = "look"
PERCEIVE = "perceive"
THINK2   = "think2"
ACT2     = "act2"
GRASP    = "grasp"
DONE     = "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    mode = p.DIRECT if (args.headless or not os.environ.get("DISPLAY")) else p.GUI
    cid = p.connect(mode)
    print(f"PyBullet connected ({'DIRECT' if mode == p.DIRECT else 'GUI'})")

    scene = build_world(physics_client=cid)
    rid = scene["robot_id"]
    init_pos = scene.get("robot_position", [-3.0, -3.0, 0.2])

    # -- state-machine variables --
    phase       = SETTLE
    tick        = 0
    spin_gen    = None
    spin_result = None
    poses       = None
    accepted    = None
    drive_ctrl  = None
    table_centroid_xy = None
    leg_centroids     = []

    initial_target_est_xy = None
    initial_target_est_3d = None

    sweep_idx       = 0
    sweep_settle    = 0
    arm_target_pts  = []
    pan_joint_idx   = None
    cam_link_idx    = None
    base_pan        = 0.0

    target_est_3d   = None
    grasp_gen       = None

    print("\nStarting main loop ...\n")

    while p.isConnected():  # DO NOT TOUCH

        # ============ SETTLE ============
        if phase == SETTLE:
            if tick == 0:
                # Position arm so gripper camera aims forward at table height
                arm_obs = {
                    "arm_joint_1": 0.0,
                    "arm_joint_2": 0.5,
                    "arm_joint_3": 0.3,
                    "arm_joint_4": 0.0,
                    "arm_joint_5": 0.4,
                    "arm_joint_6": 0.0,
                    "arm_joint_7": 0.0,
                }
                for j in range(p.getNumJoints(rid)):
                    jn = p.getJointInfo(rid, j)[1].decode("utf-8")
                    if jn in arm_obs:
                        p.setJointMotorControl2(
                            rid, j, p.POSITION_CONTROL,
                            targetPosition=arm_obs[jn],
                            force=200, maxVelocity=1.5)
            tick += 1
            if tick >= 300:
                print("Physics settled. Arm positioned for dual-camera scan.")
                phase = SENSE
                tick  = 0
                print("=" * 60)
                print("PERCEPTION: 360 deg dual-camera scan")
                print("  Base camera : lidar_link  (obstacles + table)")
                print("  Arm camera  : camera_link (target cylinder)")
                print("=" * 60)
                spin_gen = spin_and_capture_stepper(
                    rid, scene,
                    cam_link="lidar_link",
                    spin_speed=2.0,
                    capture_every_deg=10.0,
                    arm_cam_link="camera_link",
                    target_id=scene.get("target_id"),
                )

        # ============ SENSE ============
        elif phase == SENSE:
            try:
                next(spin_gen)
            except StopIteration as e:
                spin_result = e.value
                spin_gen = None
                phase = RESET
                tick  = 0

                for j in range(p.getNumJoints(rid)):
                    jn = p.getJointInfo(rid, j)[1].decode("utf-8")
                    if jn.startswith("wheel_"):
                        p.setJointMotorControl2(
                            rid, j, p.VELOCITY_CONTROL,
                            targetVelocity=0, force=50)
                        p.resetJointState(rid, j, targetValue=0,
                                          targetVelocity=0)
                    elif jn.startswith("arm_joint_"):
                        p.setJointMotorControl2(
                            rid, j, p.POSITION_CONTROL,
                            targetPosition=0.0, force=200, maxVelocity=1.5)

                init_orn = p.getQuaternionFromEuler([0, 0, 0])
                p.resetBasePositionAndOrientation(rid, init_pos, init_orn)
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                print(f"\nRobot reset to initial position {init_pos}")

        # ============ RESET ============
        elif phase == RESET:
            tick += 1
            if tick >= 480:
                phase = THINK
                tick  = 0

        # ============ THINK ============
        elif phase == THINK:
            pts, seg, arm_tgt_pts = spin_result
            rp, _, _ = get_robot_pose(rid)
            robot_pos = list(rp)

            poses = process_point_cloud(
                pts, seg, scene, robot_pos,
                save_path="perception_2d_map.png",
            )
            print(f"\nPerception complete -- {len(poses)} object parts detected.")

            if not poses:
                print("No objects detected. Disconnecting.")
                phase = DONE
                p.stepSimulation(); time.sleep(1./240.)
                continue

            accepted = filter_poses(poses)

            table_pose = None
            for ps in accepted:
                if "table_top" in ps.get("name", ""):
                    table_pose = ps
                    break

            if table_pose is None:
                print("Table not detected. Disconnecting.")
                phase = DONE
                p.stepSimulation(); time.sleep(1./240.)
                continue

            table_centroid_xy = tuple(table_pose["centroid"][:2])

            # Collect leg centroids (used by THINK and THINK2)
            leg_centroids = []
            for ps in accepted:
                if "leg" in ps.get("name", ""):
                    leg_centroids.append(ps["centroid"][:2])

            # -- Initial target estimate from arm camera --
            if len(arm_tgt_pts) >= 3:
                initial_target_est_3d = np.mean(arm_tgt_pts, axis=0)
                initial_target_est_xy = initial_target_est_3d[:2]
                gt = np.array(scene.get("target_position",
                              p.getBasePositionAndOrientation(
                                  scene["target_id"])[0]))
                err_3d = np.linalg.norm(initial_target_est_3d - gt)
                err_xy = np.linalg.norm(initial_target_est_xy - gt[:2])
                print(f"\n  INITIAL TARGET ESTIMATE (arm camera, "
                      f"{len(arm_tgt_pts)} pts):")
                print(f"    Estimated : ({initial_target_est_3d[0]:+.3f}, "
                      f"{initial_target_est_3d[1]:+.3f}, "
                      f"{initial_target_est_3d[2]:+.3f})")
                print(f"    Ground truth: ({gt[0]:+.3f}, "
                      f"{gt[1]:+.3f}, {gt[2]:+.3f})")
                print(f"    3D Error : {err_3d:.4f} m")
                print(f"    XY Error : {err_xy:.4f} m")
            else:
                print(f"\n  WARNING: Arm camera captured only "
                      f"{len(arm_tgt_pts)} target points -- "
                      f"using table centroid as fallback.")
                initial_target_est_xy = None
                initial_target_est_3d = None

            # -- Goal: nearest table edge to target --
            start_xy = (init_pos[0], init_pos[1])
            if initial_target_est_xy is not None:
                ref_xy = np.array(initial_target_est_xy, dtype=float)
            else:
                ref_xy = np.array(table_centroid_xy, dtype=float)

            goal_xy = _goal_near_target_outside_hull(
                ref_xy, leg_centroids, hull_inflate=0.25, margin=0.10)
            print(f"\n  Reference: ({ref_xy[0]:+.3f}, {ref_xy[1]:+.3f})"
                  f"  {'(target)' if initial_target_est_xy is not None else '(table)'}")
            print(f"  Goal: ({goal_xy[0]:+.3f}, {goal_xy[1]:+.3f})")

            # -- A* with table hull filled --
            waypoints = plan_path(
                accepted, start_xy, goal_xy,
                error_threshold=ERROR_THRESHOLD,
                inflate=0.35,
                scene_map=scene,
                save_path="astar_path.png",
                fill_table_hull=True,
            )
            if waypoints is None:
                print("No path with hull. Retrying without ...")
                waypoints = plan_path(
                    accepted, start_xy, goal_xy,
                    error_threshold=ERROR_THRESHOLD,
                    inflate=0.35,
                    scene_map=scene,
                    save_path="astar_path.png",
                    fill_table_hull=False,
                )
            if waypoints is None:
                print("No path found. Disconnecting.")
                phase = DONE
                p.stepSimulation(); time.sleep(1./240.)
                continue

            if len(waypoints) >= 2:
                dyaw = np.arctan2(waypoints[1][1] - start_xy[1],
                                  waypoints[1][0] - start_xy[0])
                p.resetBasePositionAndOrientation(
                    rid, init_pos,
                    p.getQuaternionFromEuler([0, 0, dyaw]))
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                print(f"  Oriented robot: yaw={np.rad2deg(dyaw):+.1f} deg")

            drive_ctrl = DriveController(rid, waypoints,
                                         goal_tol=0.15, max_steps=30000)
            print("  Driving toward target vicinity ...")
            phase = ACT
            tick  = 0

        # ============ ACT ============
        elif phase == ACT:
            pos, orn, (_, _, yaw) = get_robot_pose(rid)
            still_driving = drive_ctrl.step(pos, yaw)

            if not still_driving:
                if table_centroid_xy:
                    fd = np.hypot(pos[0] - table_centroid_xy[0],
                                  pos[1] - table_centroid_xy[1])
                    print(f"  Final dist to table centroid: {fd:.3f} m")
                if drive_ctrl.reached:
                    print("\n>>> Robot reached stop-point. <<<")

                    # Orient robot toward target before arm sweep
                    aim_xy = (initial_target_est_xy
                              if initial_target_est_xy is not None
                              else table_centroid_xy)
                    face_dx = aim_xy[0] - pos[0]
                    face_dy = aim_xy[1] - pos[1]
                    face_yaw = np.arctan2(face_dy, face_dx)
                    p.resetBasePositionAndOrientation(
                        rid, [pos[0], pos[1], pos[2]],
                        p.getQuaternionFromEuler([0, 0, face_yaw]))
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    print(f"  Oriented toward target: "
                          f"yaw={np.rad2deg(face_yaw):+.1f} deg")

                    phase = LOOK
                    tick  = 0
                    sweep_idx      = 0
                    sweep_settle   = 0
                    arm_target_pts = []
                    cam_link_idx   = find_link_index(rid, "camera_link")
                    for j in range(p.getNumJoints(rid)):
                        if p.getJointInfo(rid, j)[1].decode("utf-8") \
                                == "arm_joint_1":
                            pan_joint_idx = j
                            break
                else:
                    print("\n>>> Drive did NOT reach stop-point. <<<")
                    phase = DONE
                    tick  = 0

        # ============ LOOK ============
        elif phase == LOOK:
            if tick == 0:
                print("\n" + "=" * 60)
                print("LOOK: Multi-angle arm sweep for target perception")
                print("=" * 60)

                rpos, _, (_, _, ryaw) = get_robot_pose(rid)
                aim_xy = (initial_target_est_xy
                          if initial_target_est_xy is not None
                          else table_centroid_xy)
                dx_t = aim_xy[0] - rpos[0]
                dy_t = aim_xy[1] - rpos[1]
                yaw_to_target = np.arctan2(dy_t, dx_t)
                base_pan = np.arctan2(
                    np.sin(yaw_to_target - ryaw),
                    np.cos(yaw_to_target - ryaw))

                # Arm angles: adjust for closer range
                # arm_joint_2 higher to look more downward at nearby target
                arm_obs = {
                    "arm_joint_1": base_pan,
                    "arm_joint_2": 0.8,
                    "arm_joint_3": 0.4,
                    "arm_joint_4": 0.0,
                    "arm_joint_5": 0.6,
                    "arm_joint_6": 0.0,
                    "arm_joint_7": 0.0,
                }
                for j in range(p.getNumJoints(rid)):
                    jn = p.getJointInfo(rid, j)[1].decode("utf-8")
                    if jn in arm_obs:
                        p.setJointMotorControl2(
                            rid, j, p.POSITION_CONTROL,
                            targetPosition=arm_obs[jn],
                            force=200, maxVelocity=1.0)
                        print(f"    {jn} -> {arm_obs[jn]:+.3f} rad")

            tick += 1

            if tick <= ARM_SETTLE:
                pass
            elif sweep_idx < len(SWEEP_ANGLES):
                if sweep_settle == 0:
                    pan = base_pan + SWEEP_ANGLES[sweep_idx]
                    p.setJointMotorControl2(
                        rid, pan_joint_idx, p.POSITION_CONTROL,
                        targetPosition=pan, force=200, maxVelocity=1.0)
                sweep_settle += 1
                if sweep_settle >= SWEEP_SETTLE:
                    ls = p.getLinkState(rid, cam_link_idx,
                                       computeForwardKinematics=True)
                    rgb, depth, seg_buf = _capture_frame(ls[0], ls[1])
                    frame_pts, frame_seg = depth_to_world_points(
                        depth, seg_buf, ls[0], ls[1])
                    target_id = scene["target_id"]
                    obj_ids = (frame_seg & 0xFFFFFF).astype(int)
                    mask = obj_ids == target_id
                    tgt_pts = frame_pts[mask]
                    if len(tgt_pts) > 0:
                        arm_target_pts.append(tgt_pts)
                    pan = base_pan + SWEEP_ANGLES[sweep_idx]
                    print(f"    sweep {sweep_idx+1}/{len(SWEEP_ANGLES)} "
                          f"pan={np.rad2deg(pan):+.1f} deg "
                          f"target_pts={len(tgt_pts)}")
                    sweep_idx += 1
                    sweep_settle = 0
            else:
                total_pts = sum(len(a) for a in arm_target_pts)
                print(f"  Sweep complete -- {total_pts} target points "
                      f"across {len(arm_target_pts)} angles")
                phase = PERCEIVE
                tick  = 0

        # ============ PERCEIVE ============
        elif phase == PERCEIVE:
            if tick == 0:
                merged = (np.vstack(arm_target_pts)
                          if arm_target_pts else np.empty((0, 3)))

                result = perceive_target_with_arm_camera(
                    rid, scene,
                    cam_link="camera_link",
                    all_target_pts=merged if len(merged) >= 3 else None,
                    save_path="arm_target_perception.png",
                )

                if result is not None:
                    est = result["estimated_pos"]
                    gt  = result["ground_truth_pos"]
                    target_est_3d = np.array(est)
                    print(f"\n{'='*60}")
                    print("REFINED TARGET PERCEPTION RESULTS")
                    print(f"{'='*60}")
                    print(f"  Estimated   : ({est[0]:+.4f}, "
                          f"{est[1]:+.4f}, {est[2]:+.4f})")
                    print(f"  Ground Truth: ({gt[0]:+.4f}, "
                          f"{gt[1]:+.4f}, {gt[2]:+.4f})")
                    print(f"  3D Error    : "
                          f"{result['position_error_3d']:.4f} m")
                    print(f"  XY Error    : "
                          f"{result['position_error_xy']:.4f} m")
                    print(f"  Points used : {result['n_pts']}")
                    print(f"{'='*60}")
                else:
                    if initial_target_est_3d is not None:
                        target_est_3d = initial_target_est_3d.copy()
                        print("  Using initial arm-camera estimate.")
                    else:
                        print("  Target not visible -- cannot grasp.")
                        target_est_3d = None

            tick += 1
            if tick >= 10:
                if target_est_3d is not None:
                    phase = THINK2
                else:
                    phase = DONE
                tick = 0

        # ============ THINK2 ============
        elif phase == THINK2:
            rpos, _, _ = get_robot_pose(rid)
            tgt_xy = target_est_3d[:2]
            dist_to_target = np.hypot(rpos[0] - tgt_xy[0],
                                      rpos[1] - tgt_xy[1])
            print(f"\n{'='*60}")
            print("THINK2: Evaluating arm reach to target")
            print(f"{'='*60}")
            print(f"  Robot   : ({rpos[0]:+.3f}, {rpos[1]:+.3f})")
            print(f"  Target  : ({tgt_xy[0]:+.3f}, {tgt_xy[1]:+.3f})")
            print(f"  Distance: {dist_to_target:.3f} m")
            print(f"  Arm max reach: {ARM_MAX_REACH:.2f} m")

            if dist_to_target <= ARM_MAX_REACH:
                print("  Within arm reach -> GRASP")
                phase = GRASP
                tick  = 0
            else:
                print("  Out of reach -> planning closer approach")

                goal2_xy = _goal_near_target_outside_hull(
                    tgt_xy, leg_centroids,
                    hull_inflate=0.25, margin=0.10)
                print(f"  Approach goal: ({goal2_xy[0]:+.3f}, "
                      f"{goal2_xy[1]:+.3f})")

                for j in range(p.getNumJoints(rid)):
                    jn = p.getJointInfo(rid, j)[1].decode("utf-8")
                    if jn.startswith("arm_joint_"):
                        p.setJointMotorControl2(
                            rid, j, p.POSITION_CONTROL,
                            targetPosition=0.0, force=200, maxVelocity=1.5)

                start2_xy = (float(rpos[0]), float(rpos[1]))

                waypoints2 = plan_path(
                    accepted, start2_xy, goal2_xy,
                    error_threshold=ERROR_THRESHOLD,
                    inflate=0.35,
                    scene_map=scene,
                    save_path="astar_path2_to_target.png",
                    fill_table_hull=True,
                )
                if waypoints2 is None:
                    print("  No path with hull -- retrying ...")
                    waypoints2 = plan_path(
                        accepted, start2_xy, goal2_xy,
                        error_threshold=ERROR_THRESHOLD,
                        inflate=0.20,
                        scene_map=scene,
                        save_path="astar_path2_to_target.png",
                        fill_table_hull=False,
                    )
                if waypoints2 is None:
                    print("  No path -- attempting grasp from here.")
                    phase = GRASP
                    tick  = 0
                    p.stepSimulation(); time.sleep(1./240.)
                    continue

                if len(waypoints2) >= 2:
                    dyaw2 = np.arctan2(
                        waypoints2[1][1] - start2_xy[1],
                        waypoints2[1][0] - start2_xy[0])
                    p.resetBasePositionAndOrientation(
                        rid, [rpos[0], rpos[1], rpos[2]],
                        p.getQuaternionFromEuler([0, 0, dyaw2]))
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    print(f"  Oriented: yaw={np.rad2deg(dyaw2):+.1f} deg")

                drive_ctrl = DriveController(rid, waypoints2,
                                             goal_tol=0.08, max_steps=25000)
                print("  Driving closer to target ...")
                phase = ACT2
                tick  = 0

            p.stepSimulation(); time.sleep(1./240.)
            continue

        # ============ ACT2 ============
        elif phase == ACT2:
            pos, orn, (_, _, yaw) = get_robot_pose(rid)
            still = drive_ctrl.step(pos, yaw)

            if not still:
                tgt_xy = target_est_3d[:2]
                final_d = np.hypot(pos[0] - tgt_xy[0], pos[1] - tgt_xy[1])
                print(f"  Final distance to target: {final_d:.3f} m")

                if final_d <= ARM_MAX_REACH:
                    print(f"\n>>> Within arm reach ({final_d:.2f} m). <<<")
                    # Orient toward target
                    dx_f = float(tgt_xy[0] - pos[0])
                    dy_f = float(tgt_xy[1] - pos[1])
                    fy = np.arctan2(dy_f, dx_f)
                    p.resetBasePositionAndOrientation(
                        rid, [pos[0], pos[1], pos[2]],
                        p.getQuaternionFromEuler([0, 0, fy]))
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    phase = GRASP
                    tick  = 0
                else:
                    print(f"  Still too far ({final_d:.2f}m) -- nudging ...")
                    dx_n = float(tgt_xy[0] - pos[0])
                    dy_n = float(tgt_xy[1] - pos[1])
                    dn = np.hypot(dx_n, dy_n)
                    nudge_dist = final_d - GRASP_STANDOFF
                    if dn > 1e-6:
                        nudge_xy = (pos[0] + dx_n/dn * nudge_dist,
                                    pos[1] + dy_n/dn * nudge_dist)
                    else:
                        nudge_xy = (float(pos[0]), float(pos[1]))
                    nudge_wps = [(float(pos[0]), float(pos[1])), nudge_xy]
                    ny = np.arctan2(dy_n, dx_n)
                    p.resetBasePositionAndOrientation(
                        rid, [pos[0], pos[1], pos[2]],
                        p.getQuaternionFromEuler([0, 0, ny]))
                    p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
                    drive_ctrl = DriveController(
                        rid, nudge_wps, goal_tol=0.05, max_steps=15000)

        # ============ GRASP ============
        elif phase == GRASP:
            if tick == 0:
                # Orient toward target before grasping
                rpos, _, _ = get_robot_pose(rid)
                tgt_xy = target_est_3d[:2]
                face_yaw = np.arctan2(
                    tgt_xy[1] - rpos[1], tgt_xy[0] - rpos[0])
                p.resetBasePositionAndOrientation(
                    rid, [rpos[0], rpos[1], rpos[2]],
                    p.getQuaternionFromEuler([0, 0, face_yaw]))
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])

                print(f"\n{'='*60}")
                print("GRASP: IK grasp of target")
                print(f"{'='*60}")
                print(f"  Target 3D: ({target_est_3d[0]:+.4f}, "
                      f"{target_est_3d[1]:+.4f}, {target_est_3d[2]:+.4f})")
                print(f"  Robot facing target: "
                      f"yaw={np.rad2deg(face_yaw):+.1f} deg")
                grasp_gen = grasp_target_stepper(rid, target_est_3d)

            tick += 1
            try:
                next(grasp_gen)
            except StopIteration:
                print("\n>>> GRASP COMPLETE <<<")
                target_id = scene["target_id"]
                tgt_pos = list(
                    p.getBasePositionAndOrientation(target_id)[0])
                if tgt_pos[2] > 0.75:
                    print(f"  Target lifted to z={tgt_pos[2]:.3f} -- "
                          f"SUCCESS!")
                else:
                    print(f"  Target at z={tgt_pos[2]:.3f} -- "
                          f"may not be grasped.")
                phase = DONE
                tick  = 0

        # ============ DONE ============
        elif phase == DONE:
            tick += 1
            if tick >= args.steps:
                break

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH

    p.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
