import pybullet as p
import numpy as np
from src.robot.sensor_preprocess import SensorPreprocessor, add_gaussian_noise
from src.environment.world_builder import get_robot_pose


NOISE_MU = 0.0
NOISE_SIGMA = 0.01


def add_noise(data, mu=NOISE_MU, sigma=NOISE_SIGMA):
    noise = np.random.normal(mu, sigma, np.shape(data))
    return data + noise


def _ensure_preprocessor(preprocessor: SensorPreprocessor = None):
    if preprocessor is None:
        return SensorPreprocessor()
    return preprocessor


def get_camera_image(robot_id, sensor_link_id=-1, preprocess=False, preprocessor: SensorPreprocessor = None, return_cov=False):
    if sensor_link_id == -1:
        pos, orn, _ = get_robot_pose(robot_id)
    else:
        state = p.getLinkState(robot_id, sensor_link_id)
        pos, orn = state[0], state[1]

    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up_vec = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

    target_pos = [pos[0] + forward_vec[0], pos[1] + forward_vec[1], pos[2] + forward_vec[2]]

    view_matrix = p.computeViewMatrix(pos, target_pos, up_vec)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)

    width, height, rgb, depth, mask = p.getCameraImage(320, 240, view_matrix, proj_matrix)

    noisy_depth = np.array(depth)
    noisy_depth = add_noise(noisy_depth, sigma=0.005)

    if not preprocess:
        return rgb, noisy_depth, mask

    pre = _ensure_preprocessor(preprocessor)
    # For now, return noisy depth and estimate per-frame variance (avoid expensive per-pixel ops)
    arr = np.asarray(noisy_depth)
    filtered_depth = arr
    cov = float(np.var(arr))

    if return_cov:
        return rgb, filtered_depth, mask, cov
    return rgb, filtered_depth, mask


def get_lidar_data(robot_id, sensor_link_id=-1, num_rays=36, preprocess=False, preprocessor: SensorPreprocessor = None, return_cov=False):
    if sensor_link_id == -1:
        pos, orn, _ = get_robot_pose(robot_id)
    else:
        state = p.getLinkState(robot_id, sensor_link_id)
        pos, orn = state[0], state[1]

    ray_start, ray_end = [], []
    ray_len = 5.0
    _, _, yaw = p.getEulerFromQuaternion(orn)

    for i in range(num_rays):
        angle = yaw + (2.0 * np.pi * i) / num_rays
        ray_start.append(pos)
        ray_end.append([pos[0] + ray_len * np.cos(angle), pos[1] + ray_len * np.sin(angle), pos[2]])

    results = p.rayTestBatch(ray_start, ray_end)

    raw_distances = np.array([res[2] * ray_len for res in results])
    noisy = add_noise(raw_distances, sigma=0.02)

    if not preprocess:
        return noisy.tolist()

    pre = _ensure_preprocessor(preprocessor)
    filtered, cov = pre.preprocess(noisy, method="moving_average", window=5)
    if return_cov:
        return filtered.tolist(), cov
    return filtered.tolist()


def get_joint_states(robot_id, preprocess=False, preprocessor: SensorPreprocessor = None, return_cov=False):
    joint_data = {}
    covs = {}
    num_joints = p.getNumJoints(robot_id)

    for i in range(num_joints):
        state = p.getJointState(robot_id, i)
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')

        pos_noisy = add_noise(state[0], sigma=0.002)
        vel_noisy = add_noise(state[1], sigma=0.005)

        joint_data[joint_name] = {
            "index": i,
            "position": pos_noisy,
            "velocity": vel_noisy,
            "applied_torque": state[3]
        }

    if not preprocess:
        return joint_data

    pre = _ensure_preprocessor(preprocessor)
    # build arrays per-joint for simple moving-average smoothing
    names = list(joint_data.keys())
    positions = np.array([joint_data[n]["position"] for n in names])
    velocities = np.array([joint_data[n]["velocity"] for n in names])

    # preprocess (single-shot) — using median to remove spikes
    filt_pos, pos_cov = pre.preprocess(positions, method="median", kernel_size=3)
    filt_vel, vel_cov = pre.preprocess(velocities, method="median", kernel_size=3)

    for idx, n in enumerate(names):
        joint_data[n]["position"] = float(filt_pos[idx])
        joint_data[n]["velocity"] = float(filt_vel[idx])

    if return_cov:
        covs = {"position_var": pos_cov, "velocity_var": vel_cov}
        return joint_data, covs

    return joint_data


def get_imu_data(robot_id, preprocess=False, preprocessor: SensorPreprocessor = None, return_cov=False):
    lin_vel, ang_vel = p.getBaseVelocity(robot_id)
    _, orn, _ = get_robot_pose(robot_id)
    _, inv_orn = p.invertTransform([0, 0, 0], orn)

    local_lin_vel, _ = p.multiplyTransforms([0, 0, 0], inv_orn, lin_vel, [0, 0, 0, 1])
    local_ang_vel, _ = p.multiplyTransforms([0, 0, 0], inv_orn, ang_vel, [0, 0, 0, 1])

    gyro = add_noise(np.array(local_ang_vel), sigma=0.01)
    accel = add_noise(np.array(local_lin_vel), sigma=0.05)

    if not preprocess:
        return {
            "gyroscope_data": gyro.tolist(),
            "accelerometer_data": accel.tolist(),
        }

    pre = _ensure_preprocessor(preprocessor)
    filt_gyro, gyro_cov = pre.preprocess(gyro, method="kalman", key="gyro", process_var=1e-5, meas_var=0.01)
    filt_accel, accel_cov = pre.preprocess(accel, method="kalman", key="accel", process_var=1e-4, meas_var=0.05)

    data = {
        "gyroscope_data": filt_gyro.tolist(),
        "accelerometer_data": filt_accel.tolist(),
    }
    if return_cov:
        cov = {"gyroscope_var": gyro_cov, "accelerometer_var": accel_cov}
        return data, cov

    return data
