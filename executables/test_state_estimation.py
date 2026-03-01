"""Test the ParticleFilter in state_estimation.py.

Drives the robot in a straight line then a curve, feeding wheel-odometry
derived (v, omega) to predict(), and landmark distance observations to
update()/resample().  Compares PF estimate with ground-truth pose from
get_robot_pose() every 0.5 s and prints a summary table.

Usage (inside Docker):
    python3 test_state_estimation.py --headless
"""
import os, sys, math, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – no display needed
import matplotlib.pyplot as plt
import pybullet as p

from src.environment.world_builder import build_world, get_robot_pose, get_object_position
from src.robot.sensor_wrapper import get_joint_states, get_imu_data
from src.modules.state_estimation import ParticleFilter

# ── Robot geometry (from robot.urdf) ────────────────────────────────────
WHEEL_RADIUS = 0.10        # cylinder radius in URDF
WHEEL_TREAD  = 0.45        # 2 × 0.225 m (fl y=+0.225, fr y=-0.225)
DT           = 1.0 / 240.0

# ── Test parameters ─────────────────────────────────────────────────────
NUM_PARTICLES   = 500       # increased from 200 for better coverage
DRIVE_SPEED     = 3.0       # wheel rad/s for straight drive
TURN_SPEED      = 1.5       # differential for turning
STRAIGHT_TICKS  = 12000     # ~50 s straight  (100 report samples)
CURVE_TICKS     = 12000     # ~50 s curve     (100 report samples)
REPORT_EVERY    = 120       # every 0.5 s

# ── Landmark observation helper ─────────────────────────────────────────
OBS_NOISE_SIGMA = 0.05      # per-axis observation noise (m)
MAX_OBS_RANGE   = 6.0       # only observe landmarks within this range (m)

def observe_landmarks_relative(robot_pos, robot_yaw, landmark_map):
    """Return list of (landmark_id, (dx_local, dy_local)) observations.

    Simulates the robot observing known landmarks and computing their
    relative position in the robot body frame (with small noise).
    Only landmarks within MAX_OBS_RANGE are returned (realistic sensor).
    """
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    obs = []
    for lid, lpos in landmark_map.items():
        dx_w = lpos[0] - robot_pos[0]
        dy_w = lpos[1] - robot_pos[1]
        dist = math.sqrt(dx_w**2 + dy_w**2)
        if dist > MAX_OBS_RANGE:
            continue
        # world→robot frame
        dx_r =  dx_w * cos_y + dy_w * sin_y
        dy_r = -dx_w * sin_y + dy_w * cos_y
        # add sensor noise
        dx_r += np.random.normal(0, OBS_NOISE_SIGMA)
        dy_r += np.random.normal(0, OBS_NOISE_SIGMA)
        obs.append((lid, (dx_r, dy_r)))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    mode = p.DIRECT if (args.headless or not os.environ.get("DISPLAY")) else p.GUI
    cid = p.connect(mode)
    print(f"PyBullet connected ({'DIRECT' if mode == p.DIRECT else 'GUI'})")

    scene = build_world(physics_client=cid)
    rid   = scene["robot_id"]
    landmark_map = scene["landmark_map"]

    # ── Initial pose ────────────────────────────────────────────────────
    gt_pos, gt_orn, (_, _, gt_yaw) = get_robot_pose(rid)
    init_pose = (gt_pos[0], gt_pos[1], gt_yaw)

    # Convert landmark_map values to 2-D tuples
    lm_2d = {}
    for k, v in landmark_map.items():
        lm_2d[k] = (v[0], v[1])

    print(f"\nInitial robot pose: x={init_pose[0]:.3f}, y={init_pose[1]:.3f}, θ={math.degrees(init_pose[2]):.1f}°")
    print(f"Landmarks: {len(lm_2d)}")
    for k, v in lm_2d.items():
        print(f"  {k}: ({v[0]:+.2f}, {v[1]:+.2f})")

    # ── Initialise Particle Filter (tuned parameters) ───────────────────
    # motion_noise: reduced — single injection, no double-diffusion
    # measurement_noise: tighter (0.05) to make weight updates discriminative
    # mu=0.0 since odometry is from encoders (friction already accounted for)
    pf = ParticleFilter(
        num_particles=NUM_PARTICLES,
        initial_pose=init_pose,
        map_landmarks=lm_2d,
        motion_noise=(0.01, 0.01, 0.02),
        measurement_noise=0.03,
        mu=0.0,
    )

    # ── Wheel joint indices ─────────────────────────────────────────────
    wheel_names = ["wheel_fl_joint", "wheel_fr_joint",
                   "wheel_bl_joint", "wheel_br_joint"]
    wheel_ids = {}
    for j in range(p.getNumJoints(rid)):
        jn = p.getJointInfo(rid, j)[1].decode("utf-8")
        if jn in wheel_names:
            wheel_ids[jn] = j

    # Previous wheel angles (for odometry delta)
    prev_left  = p.getJointState(rid, wheel_ids["wheel_fl_joint"])[0]
    prev_right = p.getJointState(rid, wheel_ids["wheel_fr_joint"])[0]

    # ── Drive helpers ───────────────────────────────────────────────────
    def set_wheel_velocities(left_vel, right_vel):
        for wn in ["wheel_fl_joint", "wheel_bl_joint"]:
            p.setJointMotorControl2(rid, wheel_ids[wn],
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=left_vel, force=20)
        for wn in ["wheel_fr_joint", "wheel_br_joint"]:
            p.setJointMotorControl2(rid, wheel_ids[wn],
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=right_vel, force=20)

    # ── Logging ─────────────────────────────────────────────────────────
    errors_xy = []
    errors_theta = []
    log_rows = []

    total_ticks = STRAIGHT_TICKS + CURVE_TICKS
    tick = 0

    print(f"\n{'='*80}")
    print("  PARTICLE FILTER TEST (TUNED)  —  {0} particles, {1} landmarks, "
          "{2:.1f}s drive".format(NUM_PARTICLES, len(lm_2d), total_ticks * DT))
    print(f"{'='*80}")
    print(f"{'tick':>6}  {'phase':>8}  "
          f"{'GT_x':>7} {'GT_y':>7} {'GT_θ':>7}  "
          f"{'PF_x':>7} {'PF_y':>7} {'PF_θ':>7}  "
          f"{'err_xy':>7} {'err_θ':>7}  {'Neff':>5}")
    print("-" * 88)

    while p.isConnected() and tick < total_ticks:
        # ── Phase: straight vs curve ────────────────────────────────────
        if tick < STRAIGHT_TICKS:
            phase = "straight"
            set_wheel_velocities(DRIVE_SPEED, DRIVE_SPEED)
        else:
            phase = "curve"
            set_wheel_velocities(DRIVE_SPEED, DRIVE_SPEED - TURN_SPEED)

        p.stepSimulation()
        tick += 1

        # ── Wheel odometry → v, omega ──────────────────────────────────
        cur_left  = p.getJointState(rid, wheel_ids["wheel_fl_joint"])[0]
        cur_right = p.getJointState(rid, wheel_ids["wheel_fr_joint"])[0]

        # NOTE: This URDF's wheel axes are inverted — positive joint velocity
        # drives backward.  DriveController negates v, so we negate deltas here.
        d_left  = -(cur_left  - prev_left)  * WHEEL_RADIUS
        d_right = -(cur_right - prev_right) * WHEEL_RADIUS
        prev_left  = cur_left
        prev_right = cur_right

        d_centre = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / WHEEL_TREAD

        v     = d_centre / DT
        omega = d_theta  / DT

        # ── PF predict ──────────────────────────────────────────────────
        pf.predict(v, omega, DT)

        # ── PF update (every 10 ticks ≈ 24 Hz) ─────────────────────────
        if tick % 10 == 0:
            gt_pos, _, (_, _, gt_yaw) = get_robot_pose(rid)
            observations = observe_landmarks_relative(
                gt_pos, gt_yaw, lm_2d)
            for lid, rel in observations:
                pf.update(lid, rel)
            pf.resample(force=True)   # always resample for diversity

        # ── Report ──────────────────────────────────────────────────────
        if tick % REPORT_EVERY == 0 or tick == total_ticks:
            gt_pos, _, (_, _, gt_yaw) = get_robot_pose(rid)
            est_x, est_y, est_theta = pf.get_estimate()

            err_xy = math.sqrt((gt_pos[0] - est_x)**2 +
                               (gt_pos[1] - est_y)**2)
            err_th = abs(math.atan2(math.sin(gt_yaw - est_theta),
                                    math.cos(gt_yaw - est_theta)))
            neff = pf.neff()
            errors_xy.append(err_xy)
            errors_theta.append(err_th)
            log_rows.append((tick, phase, gt_pos[0], gt_pos[1], gt_yaw,
                             est_x, est_y, est_theta, err_xy, err_th))

            print(f"{tick:6d}  {phase:>8}  "
                  f"{gt_pos[0]:+7.3f} {gt_pos[1]:+7.3f} {math.degrees(gt_yaw):+7.1f}  "
                  f"{est_x:+7.3f} {est_y:+7.3f} {math.degrees(est_theta):+7.1f}  "
                  f"{err_xy:7.4f} {math.degrees(err_th):7.2f}°"
                  f"  {neff:5.0f}")

        # time.sleep(DT)  # skip real-time pacing for faster test

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SUMMARY  (tuned PF)")
    print("=" * 88)
    arr_xy = np.array(errors_xy)
    arr_th = np.array(errors_theta)
    print(f"  XY Error  — mean: {arr_xy.mean():.4f} m,  "
          f"std: {arr_xy.std():.4f} m,  max: {arr_xy.max():.4f} m")
    print(f"  θ  Error  — mean: {np.degrees(arr_th.mean()):.2f}°,  "
          f"std: {np.degrees(arr_th.std()):.2f}°,  "
          f"max: {np.degrees(arr_th.max()):.2f}°")
    print(f"  Samples   : {len(errors_xy)}")
    print("=" * 88)

    p.disconnect()
    print("Done.")

    # ── Plotting ────────────────────────────────────────────────────────
    times    = [r[0] * DT for r in log_rows]           # seconds
    gt_xs    = [r[2] for r in log_rows]
    gt_ys    = [r[3] for r in log_rows]
    pf_xs    = [r[5] for r in log_rows]
    pf_ys    = [r[6] for r in log_rows]
    phases   = [r[1] for r in log_rows]

    # Index where curve phase starts
    curve_start = next((i for i, ph in enumerate(phases) if ph == "curve"), len(phases))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Particle Filter – Tuned  ({} particles, {} landmarks, {:.0f}s)".format(
        NUM_PARTICLES, len(lm_2d), total_ticks * DT), fontsize=14, fontweight="bold")

    # ── 1) XY trajectory ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(gt_xs, gt_ys, "k-", linewidth=1.5, label="Ground Truth")
    ax.plot(pf_xs, pf_ys, "g--", linewidth=1.2, label="PF Estimate")
    # landmarks
    for lid, (lx, ly) in lm_2d.items():
        ax.plot(lx, ly, "r^", markersize=8)
    ax.plot([], [], "r^", markersize=8, label="Landmarks")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("XY Trajectory")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # ── 2) XY error over time ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(times[:curve_start], arr_xy[:curve_start] * 100, "b-",
            linewidth=0.8, label="Straight")
    ax.plot(times[curve_start:], arr_xy[curve_start:] * 100, "r-",
            linewidth=0.8, label="Curve")
    ax.axhline(arr_xy.mean() * 100, color="gray", ls="--", lw=1,
               label=f"Mean = {arr_xy.mean()*100:.1f} cm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("XY Error (cm)")
    ax.set_title("Position Error over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 3) θ error over time ───────────────────────────────────────────
    ax = axes[1, 0]
    arr_th_deg = np.degrees(arr_th)
    ax.plot(times[:curve_start], arr_th_deg[:curve_start], "b-",
            linewidth=0.8, label="Straight")
    ax.plot(times[curve_start:], arr_th_deg[curve_start:], "r-",
            linewidth=0.8, label="Curve")
    ax.axhline(arr_th_deg.mean(), color="gray", ls="--", lw=1,
               label=f"Mean = {arr_th_deg.mean():.2f}°")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("θ Error (°)")
    ax.set_title("Heading Error over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 4) Error histogram ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(arr_xy * 100, bins=25, alpha=0.7, color="steelblue",
            edgecolor="white", label="XY Error")
    ax.axvline(arr_xy.mean() * 100, color="red", ls="--", lw=1.5,
               label=f"Mean = {arr_xy.mean()*100:.1f} cm")
    ax.axvline(arr_xy.max() * 100, color="orange", ls=":", lw=1.5,
               label=f"Max  = {arr_xy.max()*100:.1f} cm")
    ax.set_xlabel("XY Error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("XY Error Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(os.path.dirname(__file__), "..", "imgs",
                             "particle_filter_results.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {os.path.abspath(plot_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
