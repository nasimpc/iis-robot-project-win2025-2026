"""
Run the perception pipeline N times (default 5), each time with a fresh
random world.  For every detected obstacle, find the closest ground-truth
obstacle and record the Euclidean position error.  Print per-trial and
overall average errors.

Usage:
    python3 test_perception_accuracy.py [--trials 5]
"""
import os, argparse, time
import numpy as np
import pybullet as p
from scipy.optimize import linear_sum_assignment

from src.environment.world_builder import build_world
from src.modules.perception import perceive_world


def match_poses_to_gt(detected_xy, gt_xy):
    """
    Use the Hungarian algorithm to optimally match detected 2-D centroids
    to ground-truth positions, minimising total Euclidean distance.

    Returns
    -------
    pairs   : list of (gt_index, det_index, distance)
    """
    if len(detected_xy) == 0 or len(gt_xy) == 0:
        return []

    cost = np.linalg.norm(
        np.array(gt_xy)[:, None, :] - np.array(detected_xy)[None, :, :],
        axis=2,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        pairs.append((r, c, cost[r, c]))
    return pairs


def run_one_trial(trial_idx):
    """Build world, run perception, return per-obstacle errors and counts."""
    cid = p.connect(p.DIRECT)

    scene = build_world(physics_client=cid)

    # settle physics
    for _ in range(240):
        p.stepSimulation()

    poses = perceive_world(
        scene["robot_id"],
        scene,
        cam_link="lidar_link",
        spin_speed=2.0,
        capture_every_deg=10.0,
        save_path=f"perception_2d_map_trial{trial_idx}.png",
    )

    # ground truth: obstacle 2-D positions
    gt_obs_positions = [pos[:2] for pos in scene["obstacle_positions"]]
    gt_obs_colors    = scene["obstacle_colors"]
    obs_ids_set      = set(scene["obstacle_ids"])

    # ground truth: table position
    gt_table_xy = scene["table_position"][:2]

    # detected obstacle centroids (only obstacle bodies)
    det_obs = [(ps["centroid"], ps["name"]) for ps in poses
               if ps.get("body_id") in obs_ids_set]

    # detected table top centroid
    det_table = [ps for ps in poses if "table_top" in ps.get("name", "")]

    # ── obstacle matching ──
    det_obs_xy = [d[0] for d in det_obs]
    gt_xy  = np.array(gt_obs_positions)
    det_xy = np.array(det_obs_xy) if det_obs_xy else np.empty((0, 2))

    pairs = match_poses_to_gt(det_xy, gt_xy)

    obstacle_errors = []
    print(f"\n  {'Ground Truth':30s}  {'Detected':30s}  Error (m)")
    print(f"  {'-'*30}  {'-'*30}  {'-'*10}")
    for gt_i, det_i, dist in pairs:
        gt_label = f"{gt_obs_colors[gt_i]} ({gt_xy[gt_i][0]:+.2f}, {gt_xy[gt_i][1]:+.2f})"
        d_label  = f"{det_obs[det_i][1]} ({det_xy[det_i][0]:+.2f}, {det_xy[det_i][1]:+.2f})"
        print(f"  {gt_label:30s}  {d_label:30s}  {dist:.4f}")
        obstacle_errors.append(dist)

    # unmatched ground truths
    matched_gt = {gt_i for gt_i, _, _ in pairs}
    for gi in range(len(gt_obs_positions)):
        if gi not in matched_gt:
            print(f"  {gt_obs_colors[gi]:30s}  {'<not detected>':30s}  —")

    # ── table matching ──
    table_error = None
    if det_table:
        tc = det_table[0]["centroid"]
        table_error = float(np.linalg.norm(np.array(tc) - np.array(gt_table_xy)))
        print(f"  {'Table top GT':30s}  "
              f"{'table_top det':30s}  {table_error:.4f}")

    p.disconnect()

    return {
        "obstacle_errors": obstacle_errors,
        "n_gt":            len(gt_obs_positions),
        "n_det":           len(det_obs),
        "table_error":     table_error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    all_obs_errors = []
    all_table_errors = []
    trial_summaries = []

    for t in range(args.trials):
        print(f"\n{'='*60}")
        print(f"  TRIAL {t+1} / {args.trials}")
        print(f"{'='*60}")

        result = run_one_trial(t + 1)
        trial_summaries.append(result)

        errs = result["obstacle_errors"]
        all_obs_errors.extend(errs)
        if result["table_error"] is not None:
            all_table_errors.append(result["table_error"])

        trial_avg = np.mean(errs) if errs else float("nan")
        print(f"\n  Trial {t+1}: detected {result['n_det']}/{result['n_gt']} "
              f"obstacles  |  avg error = {trial_avg:.4f} m")

    # ── overall summary ──
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS ({args.trials} trials)")
    print(f"{'='*60}")

    for i, r in enumerate(trial_summaries):
        errs = r["obstacle_errors"]
        tavg = np.mean(errs) if errs else float("nan")
        te = r["table_error"]
        te_str = f"{te:.4f}" if te is not None else "N/A"
        print(f"  Trial {i+1}: {r['n_det']}/{r['n_gt']} detected  "
              f"avg_err={tavg:.4f} m  table_err={te_str}")

    if all_obs_errors:
        arr = np.array(all_obs_errors)
        print(f"\n  Obstacle pose errors across all trials:")
        print(f"    Mean   = {np.mean(arr):.4f} m")
        print(f"    Std    = {np.std(arr):.4f} m")
        print(f"    Median = {np.median(arr):.4f} m")
        print(f"    Min    = {np.min(arr):.4f} m")
        print(f"    Max    = {np.max(arr):.4f} m")
        print(f"    Total matched pairs = {len(arr)}")
    else:
        print("\n  No obstacle matches across all trials.")

    if all_table_errors:
        tarr = np.array(all_table_errors)
        print(f"\n  Table top pose errors:")
        print(f"    Mean   = {np.mean(tarr):.4f} m")
        print(f"    Std    = {np.std(tarr):.4f} m")

    print()


if __name__ == "__main__":
    main()
