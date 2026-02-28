"""
perception.py — Spin the robot 360° using differential wheel drive, capture
RGB-D frames from the base-mounted camera (``lidar_link``), accumulate a
world-frame point cloud, segment objects via the PyBullet segmentation mask
(including individual table legs), compute each object's 2-D pose via PCA,
and plot the result on a bird's-eye 2-D graph.
"""

import numpy as np
import pybullet as p
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
from typing import Dict, List, Set
from src.environment.world_builder import get_robot_pose

# ── camera settings (must match sensor_wrapper / getCameraImage call) ───────
CAM_W, CAM_H = 320, 240
CAM_FOV      = 60          # vertical, degrees
CAM_NEAR     = 0.1
CAM_FAR      = 10.0
CAM_ASPECT   = CAM_W / CAM_H

# ───────────────────────────── utility helpers ──────────────────────────────

def find_link_index(robot_id: int, link_name: str) -> int:
    """Return the joint/link index whose child link matches *link_name*."""
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == link_name:
            return i
    return -1


def _capture_frame(cam_pos, cam_orn):
    """Render one RGB-D + segmentation frame from the given camera pose."""
    rot = list(p.getMatrixFromQuaternion(cam_orn))
    fwd = [rot[0], rot[3], rot[6]]          # link X in world (forward)
    up  = [rot[2], rot[5], rot[8]]          # link Z in world (up)
    tgt = [cam_pos[i] + fwd[i] for i in range(3)]

    view = p.computeViewMatrix(cam_pos, tgt, up)
    proj = p.computeProjectionMatrixFOV(CAM_FOV, CAM_ASPECT, CAM_NEAR, CAM_FAR)
    _, _, rgb, depth, seg = p.getCameraImage(CAM_W, CAM_H, view, proj)
    return rgb, depth, seg


# ─────────────── depth buffer → world-frame 3-D point cloud ────────────────

def depth_to_world_points(depth_buf, seg_buf, cam_pos, cam_orn):
    """
    Back-project a PyBullet depth buffer into world-frame 3-D points.

    Parameters
    ----------
    depth_buf : array-like — raw PyBullet depth buffer (H × W values in [0,1])
    seg_buf   : array-like — raw segmentation mask (H × W)
    cam_pos   : (3,) camera world position
    cam_orn   : (4,) camera world quaternion

    Returns
    -------
    points     : (N, 3)  world XYZ
    seg_labels : (N,)    raw segmentation label (encodes body + link)
    """
    d2 = np.asarray(depth_buf, dtype=np.float64).reshape(CAM_H, CAM_W)

    # linearise the depth buffer
    z_lin = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * d2)

    # camera intrinsics (vertical focal length in pixels)
    f_px = CAM_H / (2.0 * np.tan(np.deg2rad(CAM_FOV) / 2.0))
    cx, cy = CAM_W / 2.0, CAM_H / 2.0
    u, v = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))

    # camera "image frame": z = forward (depth), x = right, y = up
    x_im = (u - cx) * z_lin / f_px
    y_im = -(v - cy) * z_lin / f_px        # v increases downward → negate
    z_im = z_lin

    pts_im = np.stack([x_im, y_im, z_im], axis=-1).reshape(-1, 3)

    # image frame → world  (link X = forward, −link Y = right, link Z = up)
    R = np.asarray(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    C2W = np.column_stack([-R[:, 1], R[:, 2], R[:, 0]])
    pts_world = (C2W @ pts_im.T).T + np.asarray(cam_pos)

    # keep only valid-depth points and valid mask values
    z_flat   = z_lin.reshape(-1)
    seg_flat = np.asarray(seg_buf, dtype=np.int64).reshape(-1)
    valid = (
        (z_flat > CAM_NEAR + 0.01)
        & (z_flat < CAM_FAR - 0.5)
        & (seg_flat >= 0)
    )

    return pts_world[valid], seg_flat[valid]


# ──────────────── spin robot 360° and accumulate point cloud ────────────────

def spin_and_capture(robot_id: int,
                     scene_map: dict,
                     cam_link: str = "lidar_link",
                     spin_speed: float = 2.0,
                     capture_every_deg: float = 10.0):
    """
    Spin the robot in place a full 360° while capturing RGB-D depth frames
    from the base-mounted camera at regular angular intervals.

    The robot is rotated using differential wheel drive.  A generous step
    budget is allocated; if the wheels cannot complete the full rotation in
    time (due to friction / inertia), the remaining viewing angles are
    filled in via controlled position resets so that the scan is always
    complete.

    Returns
    -------
    all_points     : (M, 3) world-frame XYZ
    all_seg_labels : (M,)   raw segmentation label per point
    """
    cam_idx = find_link_index(robot_id, cam_link)
    if cam_idx < 0:
        raise RuntimeError(f"Camera link '{cam_link}' not found on robot")

    # ── locate wheel joints ──
    wheels: Dict[str, int] = {}
    for j in range(p.getNumJoints(robot_id)):
        jn = p.getJointInfo(robot_id, j)[1].decode("utf-8")
        if jn.startswith("wheel_"):
            wheels[jn] = j

    # ── Phase 1: differential drive spin ──
    wvel = spin_speed * 8.0
    for name, jid in wheels.items():
        vel = -wvel if ("_fl_" in name or "_bl_" in name) else wvel
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                targetVelocity=vel, force=100)

    dt = 1.0 / 240.0
    pos0, orn0, (roll0, pitch0, yaw0) = get_robot_pose(robot_id)

    accum_deg = 0.0
    yaw_prev  = yaw0
    next_cap  = 0.0
    cap_step  = capture_every_deg
    max_steps = int(60 * np.pi / (spin_speed * dt))   # generous ceiling

    all_pts: List[np.ndarray] = []
    all_seg: List[np.ndarray] = []
    captured_indices: Set[int] = set()

    for _ in range(max_steps):
        p.stepSimulation()

        _, orn, (_, _, yaw) = get_robot_pose(robot_id)

        # unwrap yaw delta
        dy = yaw - yaw_prev
        if dy > np.pi:
            dy -= 2 * np.pi
        elif dy < -np.pi:
            dy += 2 * np.pi
        accum_deg += abs(np.rad2deg(dy))
        yaw_prev = yaw

        # capture at regular angular intervals
        if accum_deg >= next_cap:
            cap_idx = int(next_cap / cap_step)
            captured_indices.add(cap_idx)
            ls = p.getLinkState(robot_id, cam_idx,
                                computeForwardKinematics=True)
            _, depth, seg = _capture_frame(ls[0], ls[1])
            pts, slbl = depth_to_world_points(depth, seg, ls[0], ls[1])
            all_pts.append(pts)
            all_seg.append(slbl)
            print(f"  capture @ {accum_deg:6.1f}°  points={len(pts)}")
            next_cap += cap_step

        if accum_deg >= 360.0:
            break

    # stop wheels
    for jid in wheels.values():
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                targetVelocity=0, force=100)

    # ── Phase 2: fill any missed angles via position reset ──
    n_total = int(360.0 / cap_step)
    missed  = [i for i in range(n_total) if i not in captured_indices]
    if missed:
        cur_pos, _, _ = get_robot_pose(robot_id)
        print(f"  Filling {len(missed)} missed angles via position reset …")
        for idx in missed:
            angle_rad = yaw0 + np.deg2rad(idx * cap_step)
            orn_new = p.getQuaternionFromEuler([roll0, pitch0, angle_rad])
            p.resetBasePositionAndOrientation(robot_id, cur_pos, orn_new)
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            for _ in range(5):
                p.stepSimulation()

            ls = p.getLinkState(robot_id, cam_idx,
                                computeForwardKinematics=True)
            _, depth, seg = _capture_frame(ls[0], ls[1])
            pts, slbl = depth_to_world_points(depth, seg, ls[0], ls[1])
            all_pts.append(pts)
            all_seg.append(slbl)
            print(f"  capture @ {idx * cap_step:6.1f}° (fill)  "
                  f"points={len(pts)}")

        # restore original orientation
        p.resetBasePositionAndOrientation(robot_id, cur_pos, orn0)
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])

    # settle
    for _ in range(60):
        p.stepSimulation()

    if all_pts:
        return np.vstack(all_pts), np.concatenate(all_seg)
    return np.empty((0, 3)), np.empty(0, dtype=np.int64)


# ──────── generator-based spin+capture (for while p.isConnected loop) ───────

def spin_and_capture_stepper(robot_id: int,
                             scene_map: dict,
                             cam_link: str = "lidar_link",
                             spin_speed: float = 2.0,
                             capture_every_deg: float = 10.0,
                             arm_cam_link: str = None,
                             target_id: int = None):
    """
    Generator that yields once per simulation tick while spinning the robot
    360° and capturing RGB-D frames.  The **caller** must call
    ``p.stepSimulation()`` after each ``next()`` invocation.

    If *arm_cam_link* and *target_id* are given, the gripper camera is also
    sampled at each capture angle.  Points belonging to *target_id* are
    accumulated separately.

    Returns (via ``StopIteration.value``):
        ``(all_points, all_seg_labels, arm_target_pts)``
    where *arm_target_pts* is an (N,3) array of target-body world points
    captured by the arm camera (empty if arm_cam_link is None).
    """
    cam_idx = find_link_index(robot_id, cam_link)
    if cam_idx < 0:
        raise RuntimeError(f"Camera link '{cam_link}' not found on robot")

    # Optional arm (gripper) camera
    arm_cam_idx = None
    if arm_cam_link is not None:
        arm_cam_idx = find_link_index(robot_id, arm_cam_link)
        if arm_cam_idx < 0:
            print(f"  WARNING: arm camera link '{arm_cam_link}' not found — "
                  f"arm camera capture disabled.")
            arm_cam_idx = None
    arm_target_pts_list: List[np.ndarray] = []

    # locate wheel joints
    wheels: Dict[str, int] = {}
    for j in range(p.getNumJoints(robot_id)):
        jn = p.getJointInfo(robot_id, j)[1].decode("utf-8")
        if jn.startswith("wheel_"):
            wheels[jn] = j

    # start spinning
    wvel = spin_speed * 8.0
    for name, jid in wheels.items():
        vel = -wvel if ("_fl_" in name or "_bl_" in name) else wvel
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                targetVelocity=vel, force=100)

    dt = 1.0 / 240.0
    pos0, orn0, (roll0, pitch0, yaw0) = get_robot_pose(robot_id)

    accum_deg = 0.0
    yaw_prev  = yaw0
    next_cap  = 0.0
    cap_step  = capture_every_deg
    max_steps = int(60 * np.pi / (spin_speed * dt))

    all_pts: List[np.ndarray] = []
    all_seg: List[np.ndarray] = []
    captured_indices: Set[int] = set()

    # ── Phase 1: spin via differential drive ──
    for _ in range(max_steps):
        yield  # caller does p.stepSimulation() + time.sleep

        _, orn, (_, _, yaw) = get_robot_pose(robot_id)

        dy = yaw - yaw_prev
        if dy > np.pi:
            dy -= 2 * np.pi
        elif dy < -np.pi:
            dy += 2 * np.pi
        accum_deg += abs(np.rad2deg(dy))
        yaw_prev = yaw

        if accum_deg >= next_cap:
            cap_idx = int(next_cap / cap_step)
            captured_indices.add(cap_idx)
            ls = p.getLinkState(robot_id, cam_idx,
                                computeForwardKinematics=True)
            _, depth, seg = _capture_frame(ls[0], ls[1])
            pts, slbl = depth_to_world_points(depth, seg, ls[0], ls[1])
            all_pts.append(pts)
            all_seg.append(slbl)

            # ── arm (gripper) camera: extract target points ──
            n_tgt = 0
            if arm_cam_idx is not None and target_id is not None:
                arm_ls = p.getLinkState(robot_id, arm_cam_idx,
                                       computeForwardKinematics=True)
                _, a_depth, a_seg = _capture_frame(arm_ls[0], arm_ls[1])
                a_pts, a_slbl = depth_to_world_points(
                    a_depth, a_seg, arm_ls[0], arm_ls[1])
                a_obj = (a_slbl & 0xFFFFFF).astype(int)
                tgt_mask = a_obj == target_id
                tgt_pts = a_pts[tgt_mask]
                if len(tgt_pts) > 0:
                    arm_target_pts_list.append(tgt_pts)
                n_tgt = len(tgt_pts)

            extra = f"  arm_tgt={n_tgt}" if arm_cam_idx else ""
            print(f"  capture @ {accum_deg:6.1f}°  points={len(pts)}{extra}")
            next_cap += cap_step

        if accum_deg >= 360.0:
            break

    # stop wheels
    for jid in wheels.values():
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL,
                                targetVelocity=0, force=100)

    # ── Phase 2: fill missed angles via position reset ──
    n_total = int(360.0 / cap_step)
    missed  = [i for i in range(n_total) if i not in captured_indices]
    if missed:
        cur_pos, _, _ = get_robot_pose(robot_id)
        print(f"  Filling {len(missed)} missed angles via position reset …")
        for idx in missed:
            angle_rad = yaw0 + np.deg2rad(idx * cap_step)
            orn_new = p.getQuaternionFromEuler([roll0, pitch0, angle_rad])
            p.resetBasePositionAndOrientation(robot_id, cur_pos, orn_new)
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            for _ in range(5):
                yield  # 5 settle ticks per fill angle

            ls = p.getLinkState(robot_id, cam_idx,
                                computeForwardKinematics=True)
            _, depth, seg = _capture_frame(ls[0], ls[1])
            pts, slbl = depth_to_world_points(depth, seg, ls[0], ls[1])
            all_pts.append(pts)
            all_seg.append(slbl)

            # arm camera during fill
            n_tgt = 0
            if arm_cam_idx is not None and target_id is not None:
                arm_ls = p.getLinkState(robot_id, arm_cam_idx,
                                       computeForwardKinematics=True)
                _, a_depth, a_seg = _capture_frame(arm_ls[0], arm_ls[1])
                a_pts, a_slbl = depth_to_world_points(
                    a_depth, a_seg, arm_ls[0], arm_ls[1])
                a_obj = (a_slbl & 0xFFFFFF).astype(int)
                tgt_mask = a_obj == target_id
                tgt_pts = a_pts[tgt_mask]
                if len(tgt_pts) > 0:
                    arm_target_pts_list.append(tgt_pts)
                n_tgt = len(tgt_pts)

            extra = f"  arm_tgt={n_tgt}" if arm_cam_idx else ""
            print(f"  capture @ {idx * cap_step:6.1f}° (fill)  "
                  f"points={len(pts)}{extra}")

        p.resetBasePositionAndOrientation(robot_id, cur_pos, orn0)
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])

    # settle
    for _ in range(60):
        yield

    # ── assemble arm-camera target points ──
    arm_tgt_all = (np.vstack(arm_target_pts_list)
                   if arm_target_pts_list else np.empty((0, 3)))
    if arm_cam_idx is not None:
        print(f"  Arm camera: {len(arm_tgt_all)} target points "
              f"across {len(arm_target_pts_list)} captures")

    if all_pts:
        return (np.vstack(all_pts), np.concatenate(all_seg), arm_tgt_all)
    return (np.empty((0, 3)), np.empty(0, dtype=np.int64), arm_tgt_all)


# ─────────────── cluster by (body, link) → PCA 2-D pose ────────────────────

def _body_link_name(body_id: int, link_idx: int) -> str:
    """Human-readable label for a (body, link) pair."""
    try:
        bname = p.getBodyInfo(body_id)[1].decode("utf-8")
    except Exception:
        bname = f"body{body_id}"
    if link_idx < 0:
        return bname
    try:
        lname = p.getJointInfo(body_id, link_idx)[12].decode("utf-8")
        return f"{bname}/{lname}"
    except Exception:
        return f"{bname}/link{link_idx}"


def compute_2d_poses(points: np.ndarray,
                     seg_labels: np.ndarray,
                     ignore_bodies: Set[int],
                     min_points: int = 5) -> List[Dict]:
    """
    Group points by (body_id, link_id), project to XY, and run PCA to
    obtain each cluster's 2-D centroid, orientation angle, and extent.

    The segmentation mask from PyBullet encodes:
        objectUniqueId + ((linkIndex + 1) << 24)
    This lets us distinguish individual table legs from the table top.

    Returns a list of dicts with keys:
        body_id, link_id, name, centroid, angle_rad, extent, n_pts, xy
    """
    obj_ids  = (seg_labels & 0xFFFFFF).astype(int)
    link_ids = ((seg_labels >> 24) - 1).astype(int)

    keys   = np.stack([obj_ids, link_ids], axis=1)
    unique = np.unique(keys, axis=0)

    poses: List[Dict] = []
    for uid, lid in unique:
        uid, lid = int(uid), int(lid)
        if uid in ignore_bodies or uid < 0:
            continue

        mask    = (obj_ids == uid) & (link_ids == lid)
        cluster = points[mask]
        if len(cluster) < min_points:
            continue

        # project to 2-D (x, y)
        xy       = cluster[:, :2]
        centroid = np.mean(xy, axis=0)
        centered = xy - centroid

        pca = PCA(n_components=2)
        pca.fit(centered)
        angle = float(np.arctan2(pca.components_[0, 1],
                                 pca.components_[0, 0]))

        proj   = centered @ pca.components_.T
        extent = np.ptp(proj, axis=0)          # (length, width) along PCA axes

        poses.append({
            "body_id":   uid,
            "link_id":   lid,
            "name":      _body_link_name(uid, lid),
            "centroid":  centroid,
            "angle_rad": angle,
            "extent":    extent,
            "n_pts":     int(mask.sum()),
            "xy":        xy,
        })

    return poses


# ── table-specific: split cluster into top + individual legs ────────────────

def _split_table_parts(points_3d: np.ndarray,
                       table_body_id: int,
                       table_height_z: float = 0.625,
                       min_pts: int = 5) -> List[Dict]:
    """
    Given the full 3-D point cloud for a table body, separate the table top
    from the legs using a height threshold, then cluster the leg points in
    XY via DBSCAN to isolate each individual leg.

    Returns a list of pose dicts (same schema as compute_2d_poses output).
    """
    top_z      = table_height_z - 0.15     # below this → legs
    top_mask   = points_3d[:, 2] >= top_z
    leg_mask   = points_3d[:, 2] <  top_z

    parts: List[Dict] = []

    # ── table top ──
    top_pts = points_3d[top_mask]
    if len(top_pts) >= min_pts:
        xy       = top_pts[:, :2]
        centroid = np.mean(xy, axis=0)
        centered = xy - centroid
        pca = PCA(n_components=2)
        pca.fit(centered)
        angle  = float(np.arctan2(pca.components_[0, 1],
                                  pca.components_[0, 0]))
        extent = np.ptp(centered @ pca.components_.T, axis=0)
        parts.append({
            "body_id":   table_body_id,
            "link_id":   -1,
            "name":      "table/table_top",
            "centroid":  centroid,
            "angle_rad": angle,
            "extent":    extent,
            "n_pts":     len(top_pts),
            "xy":        xy,
        })

    # ── individual legs via DBSCAN ──
    leg_pts = points_3d[leg_mask]
    if len(leg_pts) >= min_pts:
        clustering = DBSCAN(eps=0.25, min_samples=min_pts).fit(leg_pts[:, :2])
        leg_labels = ["leg_fl", "leg_fr", "leg_bl", "leg_br"]

        for lab in sorted(set(clustering.labels_)):
            if lab == -1:                          # noise
                continue
            cmask   = clustering.labels_ == lab
            cluster = leg_pts[cmask]
            if len(cluster) < min_pts:
                continue

            xy       = cluster[:, :2]
            centroid = np.mean(xy, axis=0)
            centered = xy - centroid

            nc = min(2, centered.shape[0], centered.shape[1])
            if nc < 1:
                continue
            pca = PCA(n_components=nc)
            pca.fit(centered)
            angle  = float(np.arctan2(pca.components_[0, 1],
                                      pca.components_[0, 0]))
            proj   = centered @ pca.components_.T
            extent = np.ptp(proj, axis=0)
            if nc < 2:
                extent = np.array([float(extent), 0.0])

            lname = leg_labels[lab] if lab < len(leg_labels) \
                    else f"leg_{lab}"
            parts.append({
                "body_id":   table_body_id,
                "link_id":   lab,
                "name":      f"table/{lname}",
                "centroid":  centroid,
                "angle_rad": angle,
                "extent":    extent,
                "n_pts":     int(cmask.sum()),
                "xy":        xy,
            })

    return parts


# ───────────────────────── 2-D map plot ─────────────────────────────────────

# obstacle name → matplotlib-friendly RGB
_OBSTACLE_COLORS = {
    "Blue":   (0.0, 0.0, 1.0),
    "Pink":   (1.0, 0.4, 0.7),
    "Orange": (1.0, 0.5, 0.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Black":  (0.1, 0.1, 0.1),
}


def plot_2d_map(poses: List[Dict],
                robot_pos,
                obstacle_color_map: Dict[int, str] = None,
                save_path: str = "perception_2d_map.png"):
    """Draw a bird's-eye 2-D map with PCA orientation arrows.

    obstacle_color_map : {body_id: color_name} e.g. {3: 'Blue', 4: 'Pink'}
    """
    if obstacle_color_map is None:
        obstacle_color_map = {}

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(poses), 1)))

    for i, ps in enumerate(poses):
        xy       = ps["xy"]
        cx, cy   = ps["centroid"]
        a        = ps["angle_rad"]
        nm       = ps["name"]

        # use actual obstacle color if available, else fallback to colormap
        bid = ps.get("body_id")
        color_name = obstacle_color_map.get(bid)
        if color_name and color_name in _OBSTACLE_COLORS:
            c = _OBSTACLE_COLORS[color_name]
            nm = f"{color_name} obstacle"
        else:
            c = cmap[i % len(cmap)]

        # point cloud scatter (very light)
        ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.10, color=c)

        # centroid marker
        ax.plot(cx, cy, "o", color=c, ms=8,
                label=f"{nm} ({cx:.2f}, {cy:.2f})")

        # PCA principal-axis orientation arrow
        arrow_len = max(float(ps["extent"][0]) * 0.4, 0.15)
        ax.annotate("",
                     xy=(cx + arrow_len * np.cos(a),
                         cy + arrow_len * np.sin(a)),
                     xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2))

        # angle label next to centroid
        ax.annotate(f"{np.rad2deg(a):.0f}°", (cx, cy), fontsize=6,
                     xytext=(5, 5), textcoords="offset points")

    # robot marker
    ax.plot(robot_pos[0], robot_pos[1], "r*", ms=15, label="Robot")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("2-D Perception Map — Obstacle & Table-Leg Poses (PCA)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved 2-D map → {save_path}")


# ──────────────────────── main perception pipeline ──────────────────────────

def perceive_world(robot_id: int,
                   scene_map: dict,
                   cam_link: str = "lidar_link",
                   spin_speed: float = 2.0,
                   capture_every_deg: float = 10.0,
                   save_path: str = "perception_2d_map.png") -> List[Dict]:
    """
    Full perception pipeline:
      1. Spin robot 360° via differential drive
      2. Capture RGB-D frames from base-mounted camera at regular intervals
      3. Accumulate world-frame point cloud, segment by PyBullet mask
      4. PCA on each (body, link) cluster → 2-D pose (centroid + orientation)
      5. Plot all detected poses on a 2-D graph and save image

    Returns list of pose dicts.
    """
    print("=" * 60)
    print("PERCEPTION: Starting 360° RGB-D scan")
    print("=" * 60)

    # bodies to ignore: room/floor, robot itself, target cube on table
    ignore: Set[int] = {scene_map["room_id"], scene_map["robot_id"]}
    if "target_id" in scene_map:
        ignore.add(scene_map["target_id"])

    # 1. spin + capture
    pts, seg = spin_and_capture(
        robot_id, scene_map,
        cam_link=cam_link,
        spin_speed=spin_speed,
        capture_every_deg=capture_every_deg,
    )

    print(f"\nTotal accumulated points: {len(pts)}")
    if len(pts) == 0:
        print("No points captured — nothing to plot.")
        return []

    # 2. PCA 2-D poses
    poses = compute_2d_poses(pts, seg, ignore)

    # 3. post-process: split table body into top + individual legs
    table_id = scene_map.get("table_id")
    if table_id is not None:
        obj_ids_all = (seg & 0xFFFFFF).astype(int)
        table_mask  = obj_ids_all == table_id
        table_pts   = pts[table_mask]

        if len(table_pts) >= 5:
            table_z = scene_map.get("table_position", [0, 0, 0.625])[2]
            table_parts = _split_table_parts(table_pts, table_id, table_z)

            if table_parts:
                # replace the single table entry with detailed parts
                poses = [ps for ps in poses if ps.get("body_id") != table_id]
                poses.extend(table_parts)

    print(f"\nDetected {len(poses)} object parts:")
    for ps in poses:
        cx, cy = ps["centroid"]
        deg    = np.rad2deg(ps["angle_rad"])
        ex     = ps["extent"]
        print(f"  {ps['name']:30s}  pos=({cx:+7.2f}, {cy:+7.2f})  "
              f"θ={deg:+7.1f}°  extent=({ex[0]:.2f} × {ex[1]:.2f})  "
              f"pts={ps['n_pts']}")

    # 3. use actual current robot position for the plot
    robot_pos = list(scene_map.get("robot_position", [0, 0, 0]))
    try:
        curr_pos, _, _ = get_robot_pose(robot_id)
        robot_pos = list(curr_pos)
    except Exception:
        pass

    # 4. build obstacle body_id → color name mapping
    obs_color_map: Dict[int, str] = {}
    obs_ids    = scene_map.get("obstacle_ids", [])
    obs_colors = scene_map.get("obstacle_colors", [])
    for oid, cname in zip(obs_ids, obs_colors):
        obs_color_map[oid] = cname

    # 5. plot
    plot_2d_map(poses, robot_pos,
                obstacle_color_map=obs_color_map,
                save_path=save_path)

    return poses


# ──────── process raw point cloud into poses (no p.stepSimulation) ──────────

def process_point_cloud(pts, seg, scene_map, robot_pos,
                        save_path="perception_2d_map.png"):
    """
    Pure-computation post-processing of the raw point cloud produced by
    ``spin_and_capture_stepper``.  Computes PCA 2-D poses, splits table,
    and plots the map.

    Does **not** call ``p.stepSimulation()``.

    Returns the same list of pose dicts as ``perceive_world()``.
    """
    ignore: Set[int] = {scene_map["room_id"], scene_map["robot_id"]}
    if "target_id" in scene_map:
        ignore.add(scene_map["target_id"])

    print(f"\nTotal accumulated points: {len(pts)}")
    if len(pts) == 0:
        print("No points captured — nothing to plot.")
        return []

    poses = compute_2d_poses(pts, seg, ignore)

    table_id = scene_map.get("table_id")
    if table_id is not None:
        obj_ids_all = (seg & 0xFFFFFF).astype(int)
        table_mask  = obj_ids_all == table_id
        table_pts   = pts[table_mask]
        if len(table_pts) >= 5:
            table_z = scene_map.get("table_position", [0, 0, 0.625])[2]
            table_parts = _split_table_parts(table_pts, table_id, table_z)
            if table_parts:
                poses = [ps for ps in poses if ps.get("body_id") != table_id]
                poses.extend(table_parts)

    print(f"\nDetected {len(poses)} object parts:")
    for ps in poses:
        cx, cy = ps["centroid"]
        deg    = np.rad2deg(ps["angle_rad"])
        ex     = ps["extent"]
        print(f"  {ps['name']:30s}  pos=({cx:+7.2f}, {cy:+7.2f})  "
              f"θ={deg:+7.1f}°  extent=({ex[0]:.2f} × {ex[1]:.2f})  "
              f"pts={ps['n_pts']}")

    obs_color_map: Dict[int, str] = {}
    for oid, cname in zip(scene_map.get("obstacle_ids", []),
                          scene_map.get("obstacle_colors", [])):
        obs_color_map[oid] = cname

    plot_2d_map(poses, robot_pos,
                obstacle_color_map=obs_color_map,
                save_path=save_path)

    return poses


# ──────────── arm camera: perceive target cylinder via PCA ──────────────────

def perceive_target_with_arm_camera(robot_id: int,
                                    scene_map: dict,
                                    cam_link: str = "camera_link",
                                    all_target_pts: np.ndarray = None,
                                    save_path: str = "arm_target_perception.png"):
    """
    Use the RGB-D camera mounted on the gripper (``camera_link``) to
    perceive the red cylinder on the table.

    If *all_target_pts* is supplied (e.g. from a multi-angle sweep),
    the function skips its own capture and uses those points directly.
    Otherwise it captures a single frame and segments the target.

    Returns
    -------
    result : dict   or   None if too few points.
    """
    target_id = scene_map.get("target_id")
    if target_id is None:
        raise ValueError("No target_id in scene_map")

    print(f"\n{'='*60}")
    print("ARM CAMERA PERCEPTION: Target pose estimation")
    print(f"{'='*60}")

    # ── obtain target points ──
    if all_target_pts is not None and len(all_target_pts) >= 3:
        target_pts = all_target_pts
        print(f"  Using pre-accumulated target points: {len(target_pts)}")
    else:
        # Single-frame fallback
        cam_idx = find_link_index(robot_id, cam_link)
        if cam_idx < 0:
            raise RuntimeError(f"Camera link '{cam_link}' not found on robot")
        ls = p.getLinkState(robot_id, cam_idx, computeForwardKinematics=True)
        cam_pos, cam_orn = ls[0], ls[1]
        print(f"  Camera pos : ({cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f})")

        rgb, depth, seg = _capture_frame(cam_pos, cam_orn)
        pts, seg_labels = depth_to_world_points(depth, seg, cam_pos, cam_orn)
        print(f"  Total points captured: {len(pts)}")

        obj_ids  = (seg_labels & 0xFFFFFF).astype(int)
        target_mask = obj_ids == target_id
        target_pts  = pts[target_mask]
        print(f"  Target (body {target_id}) points: {len(target_pts)}")

    if len(target_pts) < 3:
        print("  WARNING: Too few target points for PCA!")
        return None

    # ── PCA pose estimation (3-D) ──
    centroid_3d = np.mean(target_pts, axis=0)

    # 2-D PCA on XY plane
    xy = target_pts[:, :2]
    centroid_2d = np.mean(xy, axis=0)
    centered_2d = xy - centroid_2d

    pca_2d = PCA(n_components=2)
    pca_2d.fit(centered_2d)
    angle_2d = float(np.arctan2(pca_2d.components_[0, 1],
                                pca_2d.components_[0, 0]))
    extent_2d = np.ptp(centered_2d @ pca_2d.components_.T, axis=0)

    # full 3-D PCA for orientation
    centered_3d = target_pts - centroid_3d
    n_components = min(3, centered_3d.shape[0], centered_3d.shape[1])
    pca_3d = PCA(n_components=n_components)
    pca_3d.fit(centered_3d)
    extent_3d = np.ptp(centered_3d @ pca_3d.components_.T, axis=0)

    print(f"\n  PCA Estimated Pose:")
    print(f"    Position  : ({centroid_3d[0]:+.4f}, {centroid_3d[1]:+.4f}, {centroid_3d[2]:+.4f})")
    print(f"    Angle (XY): {np.rad2deg(angle_2d):+.1f}°")
    print(f"    Extent 2D : ({extent_2d[0]:.4f} × {extent_2d[1]:.4f})")
    print(f"    Extent 3D : ({extent_3d[0]:.4f} × {extent_3d[1]:.4f}"
          + (f" × {extent_3d[2]:.4f})" if len(extent_3d) > 2 else ")"))

    # ── ground truth ──
    gt_pos = scene_map.get("target_position")
    if gt_pos is None:
        # fallback: query PyBullet
        gt_pos = list(p.getBasePositionAndOrientation(target_id)[0])
    gt_pos = np.array(gt_pos, dtype=float)

    pos_error = float(np.linalg.norm(centroid_3d - gt_pos))
    pos_error_xy = float(np.linalg.norm(centroid_3d[:2] - gt_pos[:2]))

    print(f"\n  Ground Truth Position:")
    print(f"    ({gt_pos[0]:+.4f}, {gt_pos[1]:+.4f}, {gt_pos[2]:+.4f})")
    print(f"\n  Position Error (3D): {pos_error:.4f} m")
    print(f"  Position Error (XY): {pos_error_xy:.4f} m")

    # ── plot the target point cloud + estimated vs ground truth ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- XY top-down view ---
    ax = axes[0]
    ax.scatter(target_pts[:, 0], target_pts[:, 1], s=2, alpha=0.4,
               color="red", label=f"Target pts (n={len(target_pts)})")
    ax.plot(*centroid_2d, "rx", ms=12, mew=3, label="PCA centroid")
    ax.plot(gt_pos[0], gt_pos[1], "g+", ms=14, mew=3,
            label="Ground truth")

    # PCA direction arrow
    arrow_len = max(float(extent_2d[0]) * 0.5, 0.05)
    ax.annotate("", xy=(centroid_2d[0] + arrow_len * np.cos(angle_2d),
                        centroid_2d[1] + arrow_len * np.sin(angle_2d)),
                xytext=(centroid_2d[0], centroid_2d[1]),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Target Cylinder — XY View\n"
                 f"Error(XY)={pos_error_xy:.4f}m  Error(3D)={pos_error:.4f}m")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- XZ side view ---
    ax2 = axes[1]
    ax2.scatter(target_pts[:, 0], target_pts[:, 2], s=2, alpha=0.4,
                color="red", label=f"Target pts (n={len(target_pts)})")
    ax2.plot(centroid_3d[0], centroid_3d[2], "rx", ms=12, mew=3,
             label="PCA centroid")
    ax2.plot(gt_pos[0], gt_pos[2], "g+", ms=14, mew=3,
             label="Ground truth")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Target Cylinder — XZ Side View")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Saved arm perception plot → {save_path}")

    result = {
        "estimated_pos":      centroid_3d,
        "estimated_angle_rad": angle_2d,
        "ground_truth_pos":   gt_pos,
        "position_error_3d":  pos_error,
        "position_error_xy":  pos_error_xy,
        "n_pts":              len(target_pts),
        "pca_extent_2d":      extent_2d,
        "pca_extent_3d":      extent_3d,
    }
    return result
