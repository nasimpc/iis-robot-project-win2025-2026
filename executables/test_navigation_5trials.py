"""
Run the full pipeline (perceive → A* plan → drive to table) 5 times
using the required ``while p.isConnected()`` main loop and report
per-trial and aggregate statistics.

Usage:
    python3 test_navigation_5trials.py [--headless]
"""
import os, sys, time, argparse
import numpy as np
import pybullet as p

from src.environment.world_builder import build_world, get_robot_pose
from src.modules.perception import (
    spin_and_capture_stepper, process_point_cloud,
)
from src.modules.motion_control import (
    plan_path, DriveController, PIDController,
)

# ── perception error threshold (from prior 5-trial benchmark) ───────────
PERCEPTION_ERROR_MEAN = 0.1704
PERCEPTION_ERROR_STD  = 0.0180
ERROR_THRESHOLD = round(PERCEPTION_ERROR_MEAN + 3 * PERCEPTION_ERROR_STD, 3)

NUM_TRIALS = 5

# State machine phases
SETTLE = "settle"
SENSE  = "sense"
RESET  = "reset"
THINK  = "think"
ACT    = "act"
DONE   = "done"


def filter_poses(poses, min_pts=4):
    return [ps for ps in poses if ps["n_pts"] >= min_pts]


def run_one_trial(trial_idx, headless=True):
    """Run one full trial inside a while p.isConnected() loop."""
    print("\n" + "=" * 70)
    print(f"  TRIAL {trial_idx + 1} / {NUM_TRIALS}")
    print("=" * 70)

    mode = p.DIRECT if (headless or not os.environ.get("DISPLAY")) else p.GUI
    cid = p.connect(mode)

    scene = build_world(physics_client=cid)
    rid = scene["robot_id"]
    init_pos = scene.get("robot_position", [-3.0, -3.0, 0.2])

    # State machine variables
    phase       = SETTLE
    tick        = 0
    spin_gen    = None
    spin_result = None
    poses       = None
    table_pose  = None
    goal_xy     = None
    drive_ctrl  = None
    result      = None   # will be set when done

    while p.isConnected():  # DO NOT TOUCH
        if phase == SETTLE:
            tick += 1
            if tick >= 240:
                phase = SENSE
                tick  = 0
                spin_gen = spin_and_capture_stepper(
                    rid, scene,
                    cam_link="lidar_link",
                    spin_speed=2.0,
                    capture_every_deg=10.0,
                )

        elif phase == SENSE:
            try:
                next(spin_gen)
            except StopIteration as e:
                spin_result = e.value
                spin_gen = None
                phase = RESET
                tick  = 0

                # stop wheels
                for j in range(p.getNumJoints(rid)):
                    jn = p.getJointInfo(rid, j)[1].decode("utf-8")
                    if jn.startswith("wheel_"):
                        p.setJointMotorControl2(
                            rid, j, p.VELOCITY_CONTROL,
                            targetVelocity=0, force=50)
                        p.resetJointState(rid, j, targetValue=0,
                                          targetVelocity=0)

                init_orn = p.getQuaternionFromEuler([0, 0, 0])
                p.resetBasePositionAndOrientation(rid, init_pos, init_orn)
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])

        elif phase == RESET:
            tick += 1
            if tick >= 480:
                phase = THINK
                tick  = 0

        elif phase == THINK:
            pts, seg = spin_result
            rp, _, _ = get_robot_pose(rid)
            robot_pos = list(rp)

            poses = process_point_cloud(
                pts, seg, scene, robot_pos,
                save_path=f"trial{trial_idx+1}_perception.png",
            )

            if not poses:
                result = {"trial": trial_idx + 1, "reached": False,
                          "final_dist": float("inf"), "steps": 0,
                          "n_detected": 0, "table_detected": False}
                phase = DONE; tick = 0
                p.stepSimulation(); time.sleep(1./240.)
                continue

            accepted = filter_poses(poses)

            table_pose = None
            for ps in accepted:
                if "table_top" in ps.get("name", ""):
                    table_pose = ps
                    break

            if table_pose is None:
                result = {"trial": trial_idx + 1, "reached": False,
                          "final_dist": float("inf"), "steps": 0,
                          "n_detected": len(accepted),
                          "table_detected": False}
                phase = DONE; tick = 0
                p.stepSimulation(); time.sleep(1./240.)
                continue

            goal_xy = tuple(table_pose["centroid"][:2])
            start_xy = (init_pos[0], init_pos[1])

            waypoints = plan_path(
                accepted, start_xy, goal_xy,
                error_threshold=ERROR_THRESHOLD,
                inflate=0.35,
                scene_map=scene,
                save_path=f"trial{trial_idx+1}_astar.png",
            )

            if waypoints is None:
                fp, _, _ = get_robot_pose(rid)
                result = {"trial": trial_idx + 1, "reached": False,
                          "final_dist": np.hypot(fp[0] - goal_xy[0],
                                                 fp[1] - goal_xy[1]),
                          "n_detected": len(accepted),
                          "table_detected": True,
                          "table_est_err": 0, "goal_xy": goal_xy,
                          "final_xy": (fp[0], fp[1])}
                phase = DONE; tick = 0
                p.stepSimulation(); time.sleep(1./240.)
                continue

            # orient toward first real waypoint
            if len(waypoints) >= 2:
                dx = waypoints[1][0] - start_xy[0]
                dy = waypoints[1][1] - start_xy[1]
                tyaw = np.arctan2(dy, dx)
                p.resetBasePositionAndOrientation(
                    rid, init_pos,
                    p.getQuaternionFromEuler([0, 0, tyaw]))
                p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])

            drive_ctrl = DriveController(rid, waypoints,
                                         goal_tol=0.30, max_steps=30000)
            phase = ACT
            tick  = 0

        elif phase == ACT:
            pos, orn, (_, _, yaw) = get_robot_pose(rid)
            still = drive_ctrl.step(pos, yaw)

            if not still:
                fp, _, _ = get_robot_pose(rid)
                final_dist = np.hypot(fp[0] - goal_xy[0],
                                      fp[1] - goal_xy[1])
                gt_table = np.array(
                    scene.get("table_position", [0, 0, 0])[:2])
                est_table = np.array(table_pose["centroid"][:2])

                result = {
                    "trial": trial_idx + 1,
                    "reached": drive_ctrl.reached,
                    "final_dist": final_dist,
                    "n_detected": len(filter_poses(poses)),
                    "table_detected": True,
                    "table_est_err": np.linalg.norm(gt_table - est_table),
                    "goal_xy": goal_xy,
                    "final_xy": (fp[0], fp[1]),
                }
                phase = DONE
                tick  = 0

        elif phase == DONE:
            tick += 1
            if tick >= 60:
                break

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH

    p.disconnect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    results = []
    for i in range(NUM_TRIALS):
        r = run_one_trial(i, headless=args.headless)
        results.append(r)

    # ── summary ──
    print("\n" + "=" * 70)
    print("  NAVIGATION TEST — 5-TRIAL SUMMARY")
    print("=" * 70)
    print(f"{'Trial':>6} {'Reached':>8} {'Final Dist':>11} {'Detected':>9} "
          f"{'Table Err':>10} {'Goal':>16} {'Final Pos':>16}")
    print("-" * 80)

    dists = []
    successes = 0
    table_errs = []
    for r in results:
        tag = "YES" if r["reached"] else "NO"
        fd = f"{r['final_dist']:.3f}" if r["final_dist"] < 100 else "N/A"
        te = f"{r.get('table_est_err', 0):.3f}" if r["table_detected"] else "N/A"
        goal_s = (f"({r['goal_xy'][0]:+.2f},{r['goal_xy'][1]:+.2f})"
                  if "goal_xy" in r else "N/A")
        fin_s = (f"({r['final_xy'][0]:+.2f},{r['final_xy'][1]:+.2f})"
                 if "final_xy" in r else "N/A")
        print(f"{r['trial']:>6} {tag:>8} {fd:>11} {r['n_detected']:>9} "
              f"{te:>10} {goal_s:>16} {fin_s:>16}")
        if r["final_dist"] < 100:
            dists.append(r["final_dist"])
        if r["reached"]:
            successes += 1
        if r.get("table_est_err") is not None:
            table_errs.append(r["table_est_err"])

    print("-" * 80)
    if dists:
        print(f"  Success rate:       {successes}/{NUM_TRIALS} "
              f"({100*successes/NUM_TRIALS:.0f}%)")
        print(f"  Mean final dist:    {np.mean(dists):.3f} m  "
              f"(std={np.std(dists):.3f})")
        print(f"  Min  final dist:    {np.min(dists):.3f} m")
        print(f"  Max  final dist:    {np.max(dists):.3f} m")
    if table_errs:
        print(f"  Mean table est err: {np.mean(table_errs):.3f} m")
    print()


if __name__ == "__main__":
    main()
